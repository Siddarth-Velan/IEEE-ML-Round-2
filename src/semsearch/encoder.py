"""SentenceTransformer wrapper producing L2-normalized float32 embeddings.

The model is constructed lazily (on first encode call, not in ``__init__``) so
that importing or instantiating :class:`Encoder` never triggers a model
download. This keeps the test suite fully offline: tests inject a fake encoder
and never touch this class's model path.
"""

from __future__ import annotations

import numpy as np


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """Return ``matrix`` with each row scaled to unit L2 norm (float32).

    A zero vector is left as zeros (its norm is clamped to 1 to avoid division
    by zero), so inner-product scores against it are 0 rather than NaN.
    """
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return (matrix / norms).astype(np.float32)


class Encoder:
    """Lazy SentenceTransformer wrapper.

    Parameters
    ----------
    model_name : str, optional
        HuggingFace / SentenceTransformers model id. The model itself is not
        loaded until the first :meth:`encode_docs` or :meth:`encode_query`
        call.

    Notes
    -----
    Document text convention: :meth:`encode_docs` expects callers to have
    already joined title and body. The recommended convention used throughout
    this project is ``title + " " + text`` when the title is non-empty, else
    just ``text`` (see :func:`semsearch.encoder.doc_text`).
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None  # constructed lazily on first encode
        self._dim: int | None = None

    def _ensure_model(self):
        """Construct the underlying SentenceTransformer on first use."""
        if self._model is None:
            # Imported here, not at module top, so importing this module never
            # pulls in torch / sentence-transformers or downloads a model.
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
            self._dim = int(self._model.get_sentence_embedding_dimension())
        return self._model

    def encode_docs(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        """Encode documents into an ``(n, d)`` L2-normalized float32 matrix.

        Parameters
        ----------
        texts : list of str
            Document texts (already title+body joined by the caller).
        batch_size : int, optional
            Encoding batch size.

        Returns
        -------
        numpy.ndarray
            Shape ``(len(texts), dim)``, dtype float32, each row unit norm. An
            empty ``texts`` list yields a ``(0, dim)`` array.
        """
        model = self._ensure_model()
        if len(texts) == 0:
            return np.zeros((0, self.dim), dtype=np.float32)
        # Progress bar only for reasonably large corpora, to keep small/test
        # encodes quiet.
        show_progress = len(texts) > 1000
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=show_progress,
        )
        return _l2_normalize(embeddings)

    def encode_query(self, query: str) -> np.ndarray:
        """Encode a single query into a ``(d,)`` L2-normalized float32 vector."""
        model = self._ensure_model()
        embedding = model.encode([query], convert_to_numpy=True)
        return _l2_normalize(embedding)[0]

    @property
    def dim(self) -> int:
        """Embedding dimensionality (loads the model if not yet loaded)."""
        if self._dim is None:
            self._ensure_model()
        assert self._dim is not None
        return self._dim


def doc_text(title: str, text: str) -> str:
    """Join a document title and body per the project convention.

    Returns ``title + " " + text`` when ``title`` is non-empty (after
    stripping), otherwise just ``text``.
    """
    title = (title or "").strip()
    text = text or ""
    if title:
        return f"{title} {text}"
    return text
