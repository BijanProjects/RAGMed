import ipaddress
from collections.abc import Sequence

import pytest

from ragmed_core.urlguard import Resolver, UnsafeURLError, check_url, is_public, system_resolver


def resolves_to(*addresses: str) -> Resolver:
    def resolver(host: str, port: int) -> Sequence[str]:
        return addresses

    return resolver


PUBLIC = ["8.8.8.8", "1.1.1.1", "2001:4860:4860::8888", "::ffff:8.8.8.8", "64:ff9b::808:808"]
NOT_PUBLIC = [
    "127.0.0.1",  # loopback
    "10.1.2.3", "172.16.0.1", "192.168.1.1",  # private networks
    "169.254.169.254",  # cloud metadata service (link-local)
    "100.64.0.1",  # carrier-grade NAT
    "0.0.0.0",  # noqa: S104 - the unspecified address, used here as a test value
    "255.255.255.255", "224.0.0.1", "192.0.2.1",  # broadcast, multicast, documentation
    "::1", "::", "fe80::1", "fc00::1", "ff02::1",
    "fd00:ec2::254",  # AWS metadata over IPv6
    "::ffff:127.0.0.1",  # IPv4-mapped loopback
    "::127.0.0.1",  # IPv4-compatible (deprecated) loopback
    "64:ff9b::7f00:1",  # NAT64 to loopback
    "64:ff9b::a9fe:a9fe",  # NAT64 to the metadata service
    "2002:7f00:1::",  # 6to4 wrapping loopback
    "2001::1",  # Teredo
]  # fmt: skip


@pytest.mark.parametrize("address", PUBLIC)
def test_public_addresses_are_allowed(address: str) -> None:
    assert is_public(ipaddress.ip_address(address))


@pytest.mark.parametrize("address", NOT_PUBLIC)
def test_internal_addresses_are_blocked(address: str) -> None:
    assert not is_public(ipaddress.ip_address(address))


def test_a_public_https_url_passes_with_its_addresses() -> None:
    resolver = resolves_to("8.8.8.8", "8.8.8.8")  # duplicates are collapsed

    safe = check_url(" https://db.example.com:6333/path ", resolver=resolver)

    assert safe.url == "https://db.example.com:6333/path"
    assert safe.host == "db.example.com"
    assert safe.port == 6333
    assert [str(a) for a in safe.addresses] == ["8.8.8.8"]


def test_default_port_is_443() -> None:
    assert check_url("https://db.example.com", resolver=resolves_to("8.8.8.8")).port == 443


@pytest.mark.parametrize(
    ("url", "reason"),
    [
        ("http://db.example.com", "only https"),
        ("ftp://db.example.com", "only https"),
        ("https://user:secret@db.example.com", "credentials"),
        ("https:///no-host", "no host"),
        ("https://db.example.com:99999", "not a valid URL"),
        ("https://127.0.0.1", "non-public"),
        ("https://[::1]:6333", "non-public"),
        ("https://169.254.169.254/latest/meta-data", "non-public"),
    ],
)
def test_unsafe_urls_are_rejected(url: str, reason: str) -> None:
    with pytest.raises(UnsafeURLError, match=reason):
        check_url(url, resolver=resolves_to("8.8.8.8"))


def test_one_private_address_among_public_ones_is_enough_to_block() -> None:
    with pytest.raises(UnsafeURLError, match=r"10\.0\.0\.5"):
        check_url("https://mixed.example.com", resolver=resolves_to("8.8.8.8", "10.0.0.5"))


def test_scoped_ipv6_addresses_are_judged_without_the_scope() -> None:
    with pytest.raises(UnsafeURLError, match="non-public"):
        check_url("https://scoped.example.com", resolver=resolves_to("fe80::1%en0"))


def test_host_without_addresses_is_rejected() -> None:
    with pytest.raises(UnsafeURLError, match="no addresses"):
        check_url("https://empty.example.com", resolver=resolves_to())


def test_unresolvable_host_is_rejected() -> None:
    with pytest.raises(UnsafeURLError, match="cannot resolve"):
        system_resolver("no-such-host.invalid", 443)


def test_the_real_resolver_blocks_localhost() -> None:
    assert system_resolver("localhost", 443)

    with pytest.raises(UnsafeURLError, match="non-public"):
        check_url("https://localhost:6333")
