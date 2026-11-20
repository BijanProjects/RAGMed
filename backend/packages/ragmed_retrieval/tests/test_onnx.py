from pathlib import Path

import numpy as np
import pytest

from ragmed_retrieval.onnx import OnnxEmbedder, OnnxMetadata
from ragmed_retrieval.testing import TINY_DIMENSIONS, write_tiny_onnx_model


@pytest.fixture
def model_dir(tmp_path: Path) -> Path:
    write_tiny_onnx_model(tmp_path, "org/tiny", query_prefix="query ")
    return tmp_path


def test_embeddings_follow_the_first_token(model_dir: Path) -> None:
    embedder = OnnxEmbedder(model_dir, threads=1, batch_size=2)

    docs = embedder.embed_documents(["asthma wheezing", "diabetes", "asthma"])

    assert embedder.name == "org/tiny"
    assert docs.shape == (3, TINY_DIMENSIONS)
    assert docs.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(docs, axis=1), 1.0, rtol=1e-6)
    np.testing.assert_allclose(docs[0], docs[2], rtol=1e-6)  # padding does not change it
    assert not np.allclose(docs[0], docs[1])


def test_queries_get_the_prefix(model_dir: Path) -> None:
    embedder = OnnxEmbedder(model_dir, threads=1)

    query = embedder.embed_queries(["asthma"])
    document = embedder.embed_documents(["query asthma"])

    np.testing.assert_allclose(query, document, rtol=1e-6)


def test_no_texts_gives_an_empty_array(model_dir: Path) -> None:
    assert OnnxEmbedder(model_dir, threads=1).embed_documents([]).shape[0] == 0


def test_metadata_round_trips(tmp_path: Path) -> None:
    metadata = OnnxMetadata("org/model", "mean", 512, 1, "<pad>")

    metadata.save(tmp_path)

    assert OnnxMetadata.load(tmp_path) == metadata
