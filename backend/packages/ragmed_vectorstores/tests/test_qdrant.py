from pathlib import Path

import pytest

from ragmed_vectorstores.models import Document
from ragmed_vectorstores.qdrant import QdrantStore, point_id

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def test_point_ids_are_stable_uuids() -> None:
    assert point_id("abc:0") == point_id("abc:0")
    assert point_id("abc:0") != point_id("abc:1")
    assert len(point_id("abc:0")) == 36


async def test_reserved_metadata_keys_are_rejected() -> None:
    store = QdrantStore("reserved", location=":memory:")

    with pytest.raises(ValueError, match="reserved"):
        await store.upsert([Document("a", "text", [1.0], {"_text": "clash"})])
    await store.close()


async def test_unreachable_server_is_unhealthy() -> None:
    store = QdrantStore(url="http://127.0.0.1:9", timeout=1, check_compatibility=False)

    health = await store.health()

    assert not health.ok
    assert health.detail
    await store.close()


async def test_an_embedded_store_on_disk_keeps_its_data(tmp_path: Path) -> None:
    first = QdrantStore("saved", path=str(tmp_path))
    await first.upsert([Document("a", "kept", [1.0, 0.0])])
    await first.close()

    second = QdrantStore("saved", path=str(tmp_path))
    info = await second.describe()
    await second.close()

    assert info.count == 1
