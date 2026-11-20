import base64

import pytest

from ragmed_core.secrets import SecretBox, SecretError, generate_master_key


@pytest.fixture
def box() -> SecretBox:
    return SecretBox(generate_master_key())


def test_round_trip(box: SecretBox) -> None:
    key = box.new_tenant_key("clinic-a")

    sealed = box.encrypt("clinic-a", key, "qdrant-api-key-123")

    assert box.decrypt("clinic-a", key, sealed) == "qdrant-api-key-123"
    assert b"qdrant-api-key-123" not in sealed


def test_same_secret_encrypts_differently_each_time(box: SecretBox) -> None:
    key = box.new_tenant_key("clinic-a")

    assert box.encrypt("clinic-a", key, "same") != box.encrypt("clinic-a", key, "same")


def test_a_secret_cannot_be_read_as_another_tenant(box: SecretBox) -> None:
    key_a, key_b = box.new_tenant_key("clinic-a"), box.new_tenant_key("clinic-b")
    sealed = box.encrypt("clinic-a", key_a, "secret")

    with pytest.raises(SecretError):
        box.decrypt("clinic-b", key_b, sealed)  # copied into another tenant's row
    with pytest.raises(SecretError):
        box.decrypt("clinic-b", key_a, sealed)  # even with the right data key


def test_tampering_is_detected(box: SecretBox) -> None:
    key = box.new_tenant_key("clinic-a")
    sealed = bytearray(box.encrypt("clinic-a", key, "secret"))
    sealed[-1] ^= 1

    with pytest.raises(SecretError, match="altered"):
        box.decrypt("clinic-a", key, bytes(sealed))


def test_a_different_master_key_cannot_decrypt(box: SecretBox) -> None:
    key = box.new_tenant_key("clinic-a")
    sealed = box.encrypt("clinic-a", key, "secret")

    with pytest.raises(SecretError):
        SecretBox(generate_master_key()).decrypt("clinic-a", key, sealed)


def test_rewrap_moves_to_a_new_master_key_without_re_encrypting(box: SecretBox) -> None:
    key = box.new_tenant_key("clinic-a")
    sealed = box.encrypt("clinic-a", key, "secret")
    new_box = SecretBox(generate_master_key())

    new_key = box.rewrap("clinic-a", key, new_box)

    assert new_box.decrypt("clinic-a", new_key, sealed) == "secret"
    with pytest.raises(SecretError):
        box.decrypt("clinic-a", new_key, sealed)


@pytest.mark.parametrize("data", [b"", b"\x01", b"\x02" + bytes(40)])
def test_malformed_ciphertext_is_rejected(box: SecretBox, data: bytes) -> None:
    key = box.new_tenant_key("clinic-a")

    with pytest.raises(SecretError, match="format"):
        box.decrypt("clinic-a", key, data)


@pytest.mark.parametrize(
    ("master_key", "message"),
    [
        ("not base64!", "base64"),
        (base64.b64encode(bytes(16)).decode(), "32 bytes"),
    ],
)
def test_bad_master_keys_are_refused(master_key: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        SecretBox(master_key)


def test_empty_tenant_id_is_refused(box: SecretBox) -> None:
    with pytest.raises(ValueError, match="tenant_id"):
        box.new_tenant_key("")


def test_repr_hides_the_master_key() -> None:
    master = generate_master_key()

    assert master not in repr(SecretBox(master))
