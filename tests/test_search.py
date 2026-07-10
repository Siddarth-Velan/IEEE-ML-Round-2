"""Offline tests for the hybrid/dense/bm25 SearchEngine.

No network, no model download: a deterministic ``FakeEncoder`` (seeded unit
vector per text, cached) and a ``FakeDenseIndex`` implement the SPEC component
interfaces and are injected into ``SearchEngine``.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from semsearch.bm25 import BM25
from semsearch.search import Result, SearchEngine

DIM = 24


class FakeEncoder:
    """Deterministic stand-in for the real ``Encoder`` (SPEC interface)."""

    def __init__(self, dim: int = DIM) -> None:
        self._dim = dim
        self._cache: dict[str, np.ndarray] = {}

    def _vec(self, text: str) -> np.ndarray:
        if text not in self._cache:
            seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16) % (2**32)
            rng = np.random.RandomState(seed)
            v = rng.randn(self._dim).astype(np.float32)
            v /= np.linalg.norm(v) + 1e-12
            self._cache[text] = v
        return self._cache[text]

    def encode_query(self, query: str) -> np.ndarray:
        return self._vec(query)

    def encode_docs(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        return np.vstack([self._vec(t) for t in texts]).astype(np.float32)

    @property
    def dim(self) -> int:
        return self._dim


class FakeDenseIndex:
    """Exact inner-product index over normalized rows (SPEC interface)."""

    def __init__(self, embeddings: np.ndarray) -> None:
        self._emb = np.asarray(embeddings, dtype=np.float32)

    @classmethod
    def build(cls, embeddings: np.ndarray) -> "FakeDenseIndex":
        return cls(embeddings)

    def search(self, query_vec: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        scores = self._emb @ np.asarray(query_vec, dtype=np.float32)
        k = max(0, min(k, len(scores)))
        order = np.argsort(-scores, kind="stable")[:k]
        return scores[order].astype(np.float32), order.astype(np.int64)

    @property
    def size(self) -> int:
        return len(self._emb)


def _doc_text(doc: dict) -> str:
    title = (doc.get("title") or "").strip()
    text = doc.get("text") or ""
    return f"{title} {text}" if title else text


def build_engine(doc_ids: list[str], docs: dict[str, dict]) -> SearchEngine:
    """Assemble a SearchEngine with the fakes over the given corpus."""
    encoder = FakeEncoder()
    texts = [_doc_text(docs[d]) for d in doc_ids]
    index = FakeDenseIndex.build(encoder.encode_docs(texts))
    bm25 = BM25(texts)
    return SearchEngine(doc_ids, docs, encoder, index, bm25)


@pytest.fixture
def engine() -> SearchEngine:
    doc_ids = [f"d{i}" for i in range(8)]
    docs = {}
    for i in range(8):
        # 'alpha' appears in every doc with a distinct term frequency, so BM25
        # scores are strictly distinct (no ties) across the corpus.
        body = ("alpha " * (i + 1)) + f"tag{i} filler content document number {i}"
        docs[f"d{i}"] = {"title": f"Title {i}", "text": body}
    return build_engine(doc_ids, docs)


def test_dense_sorted_desc_and_ranks(engine: SearchEngine) -> None:
    results = engine.search("alpha", k=5, mode="dense")
    assert [r.rank for r in results] == [1, 2, 3, 4, 5]
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_all_scores_are_python_float(engine: SearchEngine) -> None:
    for mode in ("dense", "bm25", "hybrid"):
        for r in engine.search("alpha", k=4, mode=mode):
            assert type(r.score) is float


def test_hybrid_alpha0_matches_dense_ordering(engine: SearchEngine) -> None:
    dense = engine.search("alpha", k=8, mode="dense")
    hybrid = engine.search("alpha", k=8, mode="hybrid", alpha=0.0)
    assert [r.doc_id for r in hybrid] == [r.doc_id for r in dense]


def test_hybrid_alpha1_matches_bm25_ordering(engine: SearchEngine) -> None:
    bm25 = engine.search("alpha", k=8, mode="bm25")
    hybrid = engine.search("alpha", k=8, mode="hybrid", alpha=1.0)
    assert [r.doc_id for r in hybrid] == [r.doc_id for r in bm25]


def test_hybrid_is_keyed_by_doc_id_not_text() -> None:
    # Two documents share identical TEXT but have distinct ids; both must appear.
    doc_ids = ["dup_a", "dup_b", "other"]
    shared = {"title": "Shared", "text": "alpha beta gamma identical body text"}
    docs = {
        "dup_a": dict(shared),
        "dup_b": dict(shared),
        "other": {"title": "Other", "text": "completely different lexical tokens here"},
    }
    eng = build_engine(doc_ids, docs)
    results = eng.search("alpha beta gamma", k=3, mode="hybrid", alpha=0.5)
    returned = {r.doc_id for r in results}
    assert {"dup_a", "dup_b"}.issubset(returned)
    assert len({r.doc_id for r in results}) == len(results)  # no id collisions


def test_ranks_are_1_to_k(engine: SearchEngine) -> None:
    results = engine.search("alpha", k=3, mode="hybrid", alpha=0.5)
    assert [r.rank for r in results] == [1, 2, 3]


def test_results_are_result_dataclass(engine: SearchEngine) -> None:
    results = engine.search("alpha", k=2, mode="dense")
    assert all(isinstance(r, Result) for r in results)
    assert results[0].title == docs_title(results[0], engine)


def docs_title(result: Result, engine: SearchEngine) -> str:
    return engine.docs[result.doc_id]["title"]


def test_invalid_mode_raises(engine: SearchEngine) -> None:
    with pytest.raises(ValueError):
        engine.search("alpha", mode="bogus")


def test_bm25_modes_require_bm25() -> None:
    encoder = FakeEncoder()
    doc_ids = ["d0", "d1"]
    docs = {"d0": {"title": "", "text": "one"}, "d1": {"title": "", "text": "two"}}
    index = FakeDenseIndex.build(encoder.encode_docs(["one", "two"]))
    eng = SearchEngine(doc_ids, docs, encoder, index, bm25=None)
    with pytest.raises(RuntimeError):
        eng.search("one", mode="bm25")
    with pytest.raises(RuntimeError):
        eng.search("one", mode="hybrid")


def test_run_for_eval_shape(engine: SearchEngine) -> None:
    run = engine.run_for_eval({"q1": "alpha", "q2": "content"}, k=3, mode="dense")
    assert set(run) == {"q1", "q2"}
    for hits in run.values():
        assert all(type(s) is float for s in hits.values())
        assert len(hits) == 3
