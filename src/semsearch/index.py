"""Dense vector index over L2-normalized embeddings (cosine via inner product).

Two interchangeable backends:

* **faiss** ``IndexFlatIP`` when ``faiss`` imports (the default here), and
* an exact **numpy** matmul + ``argpartition`` top-k fallback otherwise.

Both backends return identical results on the same data (verified in the tests).
The backend is an implementation detail invisible to callers; :meth:`save`
writes a faiss index file or a ``.npy`` array accordingly and :meth:`load`
sniffs which format is on disk.
"""

from __future__ import annotations

import os

import numpy as np

try:  # faiss is optional; the numpy backend is an exact substitute.
    import faiss

    _HAVE_FAISS = True
except ImportError:  # pragma: no cover - exercised only when faiss is absent
    faiss = None
    _HAVE_FAISS = False


def _topk_numpy(
    matrix: np.ndarray, query: np.ndarray, k: int
) -> tuple[np.ndarray, np.ndarray]:
    """Exact inner-product top-k via numpy.

    Returns ``(scores, positions)`` sorted by score descending, where
    ``positions`` are row indices into ``matrix``.
    """
    scores = matrix @ query  # (n,)
    n = scores.shape[0]
    k = min(k, n)
    if k == 0:
        return (
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
        )
    # argpartition gives the top-k unordered; then sort just those descending.
    part = np.argpartition(-scores, k - 1)[:k]
    order = part[np.argsort(-scores[part], kind="stable")]
    return scores[order].astype(np.float32), order.astype(np.int64)


class DenseIndex:
    """A flat inner-product index over normalized embedding vectors.

    Instances are created via :meth:`build`, not the constructor directly.
    """

    def __init__(self, matrix: np.ndarray, backend: str):
        # matrix: (n, d) float32, assumed L2-normalized by the caller.
        self._matrix = np.ascontiguousarray(matrix, dtype=np.float32)
        self._backend = backend
        self._faiss_index = None
        if backend == "faiss":
            self._faiss_index = faiss.IndexFlatIP(self._matrix.shape[1])
            if self._matrix.shape[0] > 0:
                self._faiss_index.add(self._matrix)

    @classmethod
    def build(cls, embeddings: np.ndarray, backend: str = "auto") -> "DenseIndex":
        """Build an index from an ``(n, d)`` matrix of normalized embeddings.

        Parameters
        ----------
        embeddings : numpy.ndarray
            ``(n, d)`` float32 matrix; rows are assumed already L2-normalized
            (inner product then equals cosine similarity).
        backend : {"auto", "faiss", "numpy"}, optional
            ``"auto"`` (default) uses faiss when available, else numpy. The
            explicit values exist mainly so tests can compare backends directly;
            they are not part of the caller-facing contract.

        Returns
        -------
        DenseIndex
        """
        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        if embeddings.ndim != 2:
            raise ValueError(f"embeddings must be 2-D, got shape {embeddings.shape}")
        if backend == "auto":
            backend = "faiss" if _HAVE_FAISS else "numpy"
        if backend == "faiss" and not _HAVE_FAISS:
            raise RuntimeError("faiss backend requested but faiss is not installed")
        if backend not in ("faiss", "numpy"):
            raise ValueError(f"unknown backend: {backend!r}")
        return cls(embeddings, backend)

    def search(
        self, query_vec: np.ndarray, k: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return the top-``k`` documents for a query vector.

        Parameters
        ----------
        query_vec : numpy.ndarray
            ``(d,)`` normalized query embedding.
        k : int
            Number of results. Clamped to the index size, so ``k`` larger than
            the corpus simply returns every document.

        Returns
        -------
        scores : numpy.ndarray
            ``(k,)`` float32 inner-product (cosine) scores, descending.
        positions : numpy.ndarray
            ``(k,)`` int64 row indices into the build matrix.
        """
        query_vec = np.ascontiguousarray(query_vec, dtype=np.float32).reshape(-1)
        k = min(k, self.size)
        if k <= 0:
            return (
                np.empty((0,), dtype=np.float32),
                np.empty((0,), dtype=np.int64),
            )
        if self._backend == "faiss":
            scores, positions = self._faiss_index.search(
                query_vec.reshape(1, -1), k
            )
            return (
                scores[0].astype(np.float32),
                positions[0].astype(np.int64),
            )
        return _topk_numpy(self._matrix, query_vec, k)

    def save(self, path: str) -> None:
        """Persist the index.

        Parameters
        ----------
        path : str
            Base path (with or without extension). The faiss backend writes
            ``<base>.faiss``; the numpy backend writes ``<base>.npy``.
        """
        base = _strip_index_ext(path)
        parent = os.path.dirname(base)
        if parent:
            os.makedirs(parent, exist_ok=True)
        if self._backend == "faiss":
            faiss.write_index(self._faiss_index, base + ".faiss")
        else:
            np.save(base + ".npy", self._matrix)

    @classmethod
    def load(cls, path: str) -> "DenseIndex":
        """Load an index, sniffing whether the on-disk format is faiss or npy.

        Parameters
        ----------
        path : str
            Base path (with or without extension) previously passed to
            :meth:`save`.

        Returns
        -------
        DenseIndex
        """
        base = _strip_index_ext(path)
        faiss_path = base + ".faiss"
        npy_path = base + ".npy"
        if os.path.exists(faiss_path):
            if not _HAVE_FAISS:  # pragma: no cover - faiss installed here
                raise RuntimeError(
                    f"{faiss_path} is a faiss index but faiss is not installed"
                )
            faiss_index = faiss.read_index(faiss_path)
            matrix = faiss_index.reconstruct_n(0, faiss_index.ntotal)
            obj = cls.__new__(cls)
            obj._matrix = np.ascontiguousarray(matrix, dtype=np.float32)
            obj._backend = "faiss"
            obj._faiss_index = faiss_index
            return obj
        if os.path.exists(npy_path):
            matrix = np.load(npy_path)
            return cls.build(matrix, backend="numpy")
        raise FileNotFoundError(
            f"no index found at {faiss_path!r} or {npy_path!r}"
        )

    @property
    def size(self) -> int:
        """Number of indexed vectors."""
        return int(self._matrix.shape[0])

    @property
    def dim(self) -> int:
        """Embedding dimensionality."""
        return int(self._matrix.shape[1])

    @property
    def backend(self) -> str:
        """Active backend, ``"faiss"`` or ``"numpy"``."""
        return self._backend


def _strip_index_ext(path: str) -> str:
    """Return ``path`` without a trailing ``.faiss`` or ``.npy`` extension."""
    for ext in (".faiss", ".npy"):
        if path.endswith(ext):
            return path[: -len(ext)]
    return path
