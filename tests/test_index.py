"""Tests for the dense index: backend equivalence, persistence, edge cases."""

import numpy as np
import pytest

from semsearch.index import _HAVE_FAISS, DenseIndex


def _normalized(n, d, seed):
    rng = np.random.default_rng(seed)
    mat = rng.standard_normal((n, d)).astype(np.float32)
    mat /= np.linalg.norm(mat, axis=1, keepdims=True)
    return mat


def test_numpy_search_sorted_and_positions():
    mat = _normalized(20, 8, seed=1)
    index = DenseIndex.build(mat, backend="numpy")
    query = mat[3]  # identical to row 3 → should rank first with score ~1
    scores, positions = index.search(query, k=5)

    assert scores.dtype == np.float32
    assert positions.dtype == np.int64
    assert positions[0] == 3
    assert np.isclose(scores[0], 1.0, atol=1e-5)
    # Scores strictly non-increasing.
    assert np.all(np.diff(scores) <= 1e-6)


@pytest.mark.skipif(not _HAVE_FAISS, reason="faiss not installed")
def test_faiss_and_numpy_identical():
    """The faiss and numpy backends must return identical top-k."""
    mat = _normalized(50, 16, seed=2)
    faiss_index = DenseIndex.build(mat, backend="faiss")
    numpy_index = DenseIndex.build(mat, backend="numpy")

    rng = np.random.default_rng(99)
    for _ in range(10):
        q = rng.standard_normal(16).astype(np.float32)
        q /= np.linalg.norm(q)
        fs, fp = faiss_index.search(q, k=10)
        ns, np_ = numpy_index.search(q, k=10)
        assert np.array_equal(fp, np_)
        assert np.allclose(fs, ns, atol=1e-5)


def test_k_larger_than_size_clamped():
    mat = _normalized(4, 8, seed=3)
    index = DenseIndex.build(mat, backend="numpy")
    scores, positions = index.search(mat[0], k=100)
    assert scores.shape == (4,)
    assert positions.shape == (4,)


@pytest.mark.skipif(not _HAVE_FAISS, reason="faiss not installed")
def test_save_load_roundtrip_faiss(tmp_path):
    mat = _normalized(30, 8, seed=4)
    index = DenseIndex.build(mat, backend="faiss")
    path = str(tmp_path / "idx")
    index.save(path)

    loaded = DenseIndex.load(path)
    assert loaded.backend == "faiss"
    assert loaded.size == 30
    q = mat[5]
    s1, p1 = index.search(q, k=7)
    s2, p2 = loaded.search(q, k=7)
    assert np.array_equal(p1, p2)
    assert np.allclose(s1, s2, atol=1e-5)


def test_save_load_roundtrip_numpy(tmp_path):
    mat = _normalized(30, 8, seed=5)
    index = DenseIndex.build(mat, backend="numpy")
    path = str(tmp_path / "idx")
    index.save(path)

    loaded = DenseIndex.load(path)
    assert loaded.backend == "numpy"
    assert loaded.size == 30
    q = mat[9]
    s1, p1 = index.search(q, k=7)
    s2, p2 = loaded.search(q, k=7)
    assert np.array_equal(p1, p2)
    assert np.allclose(s1, s2, atol=1e-5)


def test_size_and_dim():
    mat = _normalized(12, 5, seed=6)
    index = DenseIndex.build(mat, backend="numpy")
    assert index.size == 12
    assert index.dim == 5
