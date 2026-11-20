"""Check that a user-supplied URL points to the public internet before the server calls it.

Users can connect their own vector database or MCP server, which makes the server send requests
to an address a stranger chose. Without this check, someone could aim it at the cloud metadata
service (169.254.169.254), at Postgres on the private network, or at localhost: a
server-side request forgery (SSRF).

Rules: HTTPS only, no user:password in the URL, and every address the host name resolves to
must be a public one. The caller must then connect to one of the returned `addresses`, not
resolve the name again: otherwise a DNS server could answer with a public address for this
check and a private one for the real request (DNS rebinding).
"""

import ipaddress
import socket
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from urllib.parse import urlsplit

IPAddress = ipaddress.IPv4Address | ipaddress.IPv6Address
Resolver = Callable[[str, int], Sequence[str]]

DEFAULT_HTTPS_PORT = 443


class UnsafeURLError(ValueError):
    """The URL is not allowed; the message says why and is safe to show the user."""


@dataclass(frozen=True, slots=True)
class SafeURL:
    url: str
    host: str
    port: int
    addresses: tuple[IPAddress, ...]


def system_resolver(host: str, port: int) -> list[str]:
    """Every IPv4 and IPv6 address the system resolver returns for `host`."""
    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except socket.gaierror as error:
        raise UnsafeURLError(f"cannot resolve host {host!r}") from error
    return [str(info[4][0]) for info in infos]


# IPv6 forms that carry an IPv4 address and can reach it, e.g. 64:ff9b::7f00:1 is 127.0.0.1
# through a NAT64 gateway. Python's `is_global` calls both of these ranges public.
_NAT64 = ipaddress.IPv6Network("64:ff9b::/96")
_IPV4_COMPATIBLE = ipaddress.IPv6Network("::/96")  # deprecated, never needed by a real server


def is_public(address: IPAddress) -> bool:
    """True only for globally routable unicast addresses.

    `is_global` already excludes private, loopback, link-local (which holds cloud metadata
    services), carrier-grade NAT, documentation and reserved ranges. IPv6 addresses that embed
    an IPv4 address are judged by the IPv4 address inside them.
    """
    if isinstance(address, ipaddress.IPv6Address):
        if address in _IPV4_COMPATIBLE:
            return False
        if address in _NAT64:
            return is_public(ipaddress.IPv4Address(int(address) & 0xFFFFFFFF))
        embedded = address.ipv4_mapped or address.sixtofour
        if embedded is not None:
            return is_public(embedded)
    return address.is_global and not address.is_multicast


def check_url(url: str, *, resolver: Resolver = system_resolver) -> SafeURL:
    """Return the URL with its vetted addresses, or raise `UnsafeURLError`."""
    try:
        parts = urlsplit(url.strip())
        port = parts.port or DEFAULT_HTTPS_PORT
    except ValueError as error:
        raise UnsafeURLError(f"not a valid URL: {error}") from error

    if parts.scheme != "https":
        raise UnsafeURLError("only https:// URLs are allowed")
    if parts.username is not None or parts.password is not None:
        raise UnsafeURLError("put credentials in the credential fields, not in the URL")
    host = parts.hostname
    if not host:
        raise UnsafeURLError("the URL has no host name")

    try:
        candidates = [ipaddress.ip_address(host)]
    except ValueError:
        resolved = resolver(host, port)
        if not resolved:
            raise UnsafeURLError(f"host {host!r} has no addresses") from None
        candidates = [ipaddress.ip_address(a.split("%", 1)[0]) for a in resolved]

    blocked = sorted({str(a) for a in candidates if not is_public(a)})
    if blocked:
        raise UnsafeURLError(f"host {host!r} points to a non-public address: {', '.join(blocked)}")
    return SafeURL(url.strip(), host, port, tuple(dict.fromkeys(candidates)))
