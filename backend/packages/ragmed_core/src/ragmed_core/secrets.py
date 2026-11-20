"""Envelope encryption for credentials users give us, such as their database API keys.

Two layers of keys:

- One **master key** for the whole service, from the environment (`RAGMED_MASTER_KEY`). It never
  touches the database.
- One **data key per tenant**, random, stored in the database only in encrypted ("wrapped")
  form under the master key.

Each credential is encrypted with its tenant's data key using AES-256-GCM, which also detects
any tampering. The tenant id is bound into every ciphertext, so a value copied into another
tenant's row cannot be decrypted there. Rotating the master key only means re-wrapping the data
keys (`rewrap`), not re-encrypting every credential.
"""

import base64
import os

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

KEY_BYTES = 32  # AES-256
NONCE_BYTES = 12
VERSION = b"\x01"


class SecretError(ValueError):
    """Decryption failed: wrong key, wrong tenant, or the data was altered."""


def generate_master_key() -> str:
    """A new random master key, base64-encoded for an environment variable."""
    return base64.b64encode(AESGCM.generate_key(bit_length=KEY_BYTES * 8)).decode()


def _seal(key: bytes, plaintext: bytes, context: bytes) -> bytes:
    nonce = os.urandom(NONCE_BYTES)
    return VERSION + nonce + AESGCM(key).encrypt(nonce, plaintext, context)


def _open(key: bytes, sealed: bytes, context: bytes) -> bytes:
    if not sealed.startswith(VERSION) or len(sealed) <= 1 + NONCE_BYTES:
        raise SecretError("unknown or truncated ciphertext format")
    nonce, body = sealed[1 : 1 + NONCE_BYTES], sealed[1 + NONCE_BYTES :]
    try:
        return AESGCM(key).decrypt(nonce, body, context)
    except InvalidTag as error:
        raise SecretError("cannot decrypt: wrong key or tenant, or the data was altered") from error


def _context(kind: str, tenant_id: str) -> bytes:
    if not tenant_id:
        raise ValueError("tenant_id must not be empty")
    return f"ragmed:{kind}:{tenant_id}".encode()


class SecretBox:
    def __init__(self, master_key: str) -> None:
        try:
            key = base64.b64decode(master_key, validate=True)
        except ValueError as error:
            raise ValueError("master key must be base64") from error
        if len(key) != KEY_BYTES:
            raise ValueError(f"master key must be {KEY_BYTES} bytes, got {len(key)}")
        self._master = key

    def __repr__(self) -> str:
        return "SecretBox(<master key hidden>)"

    def new_tenant_key(self, tenant_id: str) -> bytes:
        """A new data key for a tenant, wrapped by the master key; store this with the tenant."""
        data_key = AESGCM.generate_key(bit_length=KEY_BYTES * 8)
        return _seal(self._master, data_key, _context("data-key", tenant_id))

    def _data_key(self, tenant_id: str, wrapped_key: bytes) -> bytes:
        return _open(self._master, wrapped_key, _context("data-key", tenant_id))

    def encrypt(self, tenant_id: str, wrapped_key: bytes, secret: str) -> bytes:
        data_key = self._data_key(tenant_id, wrapped_key)
        return _seal(data_key, secret.encode(), _context("secret", tenant_id))

    def decrypt(self, tenant_id: str, wrapped_key: bytes, sealed: bytes) -> str:
        data_key = self._data_key(tenant_id, wrapped_key)
        return _open(data_key, sealed, _context("secret", tenant_id)).decode()

    def rewrap(self, tenant_id: str, wrapped_key: bytes, new_box: "SecretBox") -> bytes:
        """Move a tenant's data key under a new master key; its secrets stay valid."""
        data_key = self._data_key(tenant_id, wrapped_key)
        return _seal(new_box._master, data_key, _context("data-key", tenant_id))
