"""Tests for the artifacts bundle save/load round-trip.

Uses random embeddings (never a real SentenceTransformer) to build the index.
"""

import json
import os

import numpy as np

from semsearch.artifacts import load_artifacts, save_artifacts
from semsearch.index import DenseIndex


def _fixture(seed=0):
    rng = np.random.default_rng(seed)
    doc_ids = ["a", "b", "c", "d"]
    docs = {
        "a": {"title": "Apple", "text": "a red fruit"},
        "b": {"title": "", "text": "no title here"},
        "c": {"title": "Cat", "text": "a small animal"},
        "d": {"title": "Dog", "text": "a loyal animal"},
    }
    mat = rng.standard_normal((4, 8)).astype(np.float32)
    mat /= np.linalg.norm(mat, axis=1, keepdims=True)
    index = DenseIndex.build(mat)
    return doc_ids, docs, index, mat


def test_roundtrip_preserves_ids_docs_meta(tmp_path):
    doc_ids, docs, index, _ = _fixture()
    out = str(tmp_path / "bundle")
    save_artifacts(out, doc_ids, docs, index, {"model_name": "fake", "dataset": "toy"})

    loaded_ids, loaded_docs, loaded_index, meta = load_artifacts(out)

    assert loaded_ids == doc_ids  # order preserved (positional alignment)
    assert loaded_docs == docs
    assert meta["model_name"] == "fake"
    assert meta["dataset"] == "toy"
    assert meta["n_docs"] == 4
    assert meta["dim"] == 8
    assert meta["backend"] == index.backend


def test_docs_jsonl_order_matches_doc_ids(tmp_path):
    doc_ids, docs, index, _ = _fixture()
    out = str(tmp_path / "bundle")
    save_artifacts(out, doc_ids, docs, index, {"model_name": "fake", "dataset": "toy"})

    with open(os.path.join(out, "docs.jsonl"), "r", encoding="utf-8") as handle:
        line_ids = [json.loads(line)["_id"] for line in handle if line.strip()]
    assert line_ids == doc_ids


def test_roundtrip_preserves_search_results(tmp_path):
    doc_ids, docs, index, mat = _fixture(seed=3)
    out = str(tmp_path / "bundle")
    save_artifacts(out, doc_ids, docs, index, {"model_name": "fake", "dataset": "toy"})

    _, _, loaded_index, _ = load_artifacts(out)

    query = mat[2]
    s1, p1 = index.search(query, k=4)
    s2, p2 = loaded_index.search(query, k=4)
    assert np.array_equal(p1, p2)
    assert np.allclose(s1, s2, atol=1e-5)
