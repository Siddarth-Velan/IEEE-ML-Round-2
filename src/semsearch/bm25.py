"""Lexical retrieval via BM25 (Okapi), aligned with document order.

Thin wrapper over ``rank_bm25`` that exposes the same
``search(query, k) -> (scores, positions)`` contract as the dense index so the
two channels can be fused by the hybrid search engine.
"""

from __future__ import annotations

import re

import numpy as np
from rank_bm25 import BM25Okapi

_TOKEN_RE = re.compile(r"\w+")


def tokenize(text: str) -> list[str]:
    """Tokenize text into lowercase word tokens.

    Parameters
    ----------
    text : str
        Raw input text.

    Returns
    -------
    list[str]
        Lowercased ``\\w+`` tokens in order of appearance.
    """
    return _TOKEN_RE.findall(text.lower())


class BM25:
    """BM25-Okapi ranker over a fixed, order-aligned document collection.

    Parameters
    ----------
    texts : list[str]
        Document texts, positionally aligned with the caller's document order.
        Row index ``i`` in results corresponds to ``texts[i]``.
    """

    def __init__(self, texts: list[str]) -> None:
        self._n = len(texts)
        self._tokenized = [tokenize(t) for t in texts]
        # BM25Okapi requires a non-empty corpus with at least one token overall.
        # Guard the degenerate empty-corpus case to keep construction total.
        if self._n == 0:
            self._bm25: BM25Okapi | None = None
        else:
            self._bm25 = BM25Okapi(self._tokenized)

    def search(self, query: str, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Return the top-``k`` documents for ``query`` by BM25 score.

        Parameters
        ----------
        query : str
            Free-text query.
        k : int
            Number of results to return. Clamped to the collection size.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            ``(scores, positions)`` sorted by score descending. ``scores`` is
            ``float32`` of shape ``(min(k, n),)`` and ``positions`` is ``int64``
            row indices into the original ``texts`` order.
        """
        k = max(0, min(k, self._n))
        if self._bm25 is None or k == 0:
            return (
                np.empty(0, dtype=np.float32),
                np.empty(0, dtype=np.int64),
            )
        scores = np.asarray(
            self._bm25.get_scores(tokenize(query)), dtype=np.float32
        )
        # Stable sort so ties resolve by ascending position for determinism.
        order = np.argsort(-scores, kind="stable")[:k]
        return scores[order], order.astype(np.int64)
