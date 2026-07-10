"""Search engine: dense, lexical (BM25), and id-keyed hybrid retrieval.

This module fixes the original hybrid-search defects (see SPEC "Known defects"):
results are keyed by ``doc_id`` (never by document text), each channel is
min-max normalized over *its own* candidate list, and the fused results are
sorted before truncation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from .bm25 import BM25

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids runtime import
    from .encoder import Encoder
    from .index import DenseIndex

# Number of per-channel candidates pooled before fusion.
_CANDIDATE_POOL = 100


@dataclass
class Result:
    """A single ranked search hit.

    Attributes
    ----------
    rank : int
        1-based rank within the returned list.
    doc_id : str
        Stable document identifier.
    score : float
        Ranking score, always a native python ``float``.
    title : str
        Document title (may be empty).
    text : str
        Document body text.
    """

    rank: int
    doc_id: str
    score: float
    title: str
    text: str


def _minmax(scores: dict[str, float]) -> dict[str, float]:
    """Min-max normalize a channel's candidate scores to ``[0, 1]``.

    Parameters
    ----------
    scores : dict[str, float]
        ``{doc_id: raw_score}`` for one channel's candidates.

    Returns
    -------
    dict[str, float]
        ``{doc_id: normalized_score}``. A zero-range channel (all scores equal,
        including the single-candidate case) maps every candidate to ``0.5``.
    """
    if not scores:
        return {}
    values = list(scores.values())
    lo, hi = min(values), max(values)
    rng = hi - lo
    if rng <= 0.0:
        return {doc_id: 0.5 for doc_id in scores}
    return {doc_id: (v - lo) / rng for doc_id, v in scores.items()}


class SearchEngine:
    """Dense / BM25 / hybrid retrieval over a fixed document collection.

    Parameters
    ----------
    doc_ids : list[str]
        Document ids in index/build order; ``doc_ids[i]`` is the id at row ``i``.
    docs : dict[str, dict]
        ``{doc_id: {"title": str, "text": str}}``.
    encoder : Encoder
        Query encoder producing L2-normalized ``(d,)`` float32 vectors.
    dense_index : DenseIndex
        Dense index whose ``search`` returns ``(scores, positions)``.
    bm25 : BM25 | None, optional
        Lexical ranker; required for ``bm25`` and ``hybrid`` modes.
    """

    def __init__(
        self,
        doc_ids: list[str],
        docs: dict[str, dict],
        encoder: "Encoder",
        dense_index: "DenseIndex",
        bm25: BM25 | None = None,
    ) -> None:
        self.doc_ids = list(doc_ids)
        self.docs = docs
        self.encoder = encoder
        self.dense_index = dense_index
        self.bm25 = bm25

    @classmethod
    def from_artifacts(cls, artifacts_dir: str) -> "SearchEngine":
        """Build an engine from a saved artifact bundle.

        Loads ids/docs/index/meta, constructs an :class:`Encoder` from
        ``meta["model_name"]``, and builds a :class:`BM25` from the stored
        document texts (title + text, matching the dense encoding convention).

        Parameters
        ----------
        artifacts_dir : str
            Directory produced by ``artifacts.save_artifacts``.

        Returns
        -------
        SearchEngine
        """
        # Lazy imports so importing this module never pulls in the heavy /
        # optional encoder + artifacts dependencies (keeps tests offline).
        from .artifacts import load_artifacts
        from .encoder import Encoder

        doc_ids, docs, index, meta = load_artifacts(artifacts_dir)
        encoder = Encoder(meta["model_name"])
        texts = [_doc_text(docs[doc_id]) for doc_id in doc_ids]
        bm25 = BM25(texts)
        return cls(doc_ids, docs, encoder, index, bm25)

    def search(
        self,
        query: str,
        k: int = 10,
        mode: str = "dense",
        alpha: float = 0.5,
    ) -> list[Result]:
        """Retrieve the top-``k`` documents for ``query``.

        Parameters
        ----------
        query : str
            Free-text query.
        k : int, optional
            Number of results (default 10).
        mode : str, optional
            One of ``{"dense", "bm25", "hybrid"}``.
        alpha : float, optional
            Hybrid mixing weight; ``fused = alpha*bm25 + (1-alpha)*dense``.
            ``alpha=0`` reproduces dense ordering, ``alpha=1`` reproduces BM25.

        Returns
        -------
        list[Result]
            Ranked ``1..k`` by score descending. Every score is a python float.

        Raises
        ------
        ValueError
            If ``mode`` is not a recognized mode.
        RuntimeError
            If a lexical mode is requested but no BM25 index is available.
        """
        if mode == "dense":
            scored = self._dense_scores(query, k)
        elif mode == "bm25":
            scored = self._bm25_scores(query, k)
        elif mode == "hybrid":
            scored = self._hybrid_scores(query, alpha)
        else:
            raise ValueError(
                f"unknown mode {mode!r}; expected 'dense', 'bm25', or 'hybrid'"
            )
        ranked = sorted(scored, key=lambda item: -item[1])[:k]
        return [self._make_result(rank, doc_id, score)
                for rank, (doc_id, score) in enumerate(ranked, start=1)]

    def run_for_eval(
        self,
        queries: dict[str, str],
        k: int,
        mode: str,
        alpha: float = 0.5,
    ) -> dict:
        """Produce an evaluation run over many queries.

        Parameters
        ----------
        queries : dict[str, str]
            ``{qid: query_text}``.
        k : int
            Results per query.
        mode : str
            Retrieval mode (see :meth:`search`).
        alpha : float, optional
            Hybrid mixing weight.

        Returns
        -------
        dict
            ``{qid: {doc_id: score}}`` — the run format consumed by
            ``metrics.evaluate_run``. Scores are native python floats.
        """
        run: dict[str, dict[str, float]] = {}
        for qid, text in queries.items():
            hits = self.search(text, k=k, mode=mode, alpha=alpha)
            run[qid] = {hit.doc_id: hit.score for hit in hits}
        return run

    # -- internal scoring channels -----------------------------------------

    def _dense_scores(self, query: str, k: int) -> list[tuple[str, float]]:
        """Return ``[(doc_id, score)]`` from the dense channel (top-``k``)."""
        qv = self.encoder.encode_query(query)
        n = min(max(k, 1), self.dense_index.size)
        scores, positions = self.dense_index.search(qv, n)
        return [(self.doc_ids[int(p)], float(s))
                for s, p in zip(scores, positions)]

    def _bm25_scores(self, query: str, k: int) -> list[tuple[str, float]]:
        """Return ``[(doc_id, score)]`` from the BM25 channel (top-``k``)."""
        bm25 = self._require_bm25()
        scores, positions = bm25.search(query, k)
        return [(self.doc_ids[int(p)], float(s))
                for s, p in zip(scores, positions)]

    def _hybrid_scores(self, query: str, alpha: float) -> list[tuple[str, float]]:
        """Fuse dense + BM25 channels, keyed by ``doc_id`` (defect #3 fix).

        Pools the top-``_CANDIDATE_POOL`` of each channel, min-max normalizes
        each channel over its own pool, then fuses
        ``alpha*bm25 + (1-alpha)*dense`` with a missing channel contributing 0.
        """
        self._require_bm25()
        dense_raw = dict(self._dense_scores(query, _CANDIDATE_POOL))
        bm25_raw = dict(self._bm25_scores(query, _CANDIDATE_POOL))
        dense_norm = _minmax(dense_raw)
        bm25_norm = _minmax(bm25_raw)

        # Union ordered dense-candidates-first so that stable sorting preserves
        # dense ordering when alpha=0 (and analogously BM25 when alpha=1).
        ordered_ids: list[str] = list(dense_norm.keys())
        seen = set(ordered_ids)
        for doc_id in bm25_norm:
            if doc_id not in seen:
                ordered_ids.append(doc_id)
                seen.add(doc_id)

        fused: list[tuple[str, float]] = []
        for doc_id in ordered_ids:
            d = dense_norm.get(doc_id, 0.0)
            b = bm25_norm.get(doc_id, 0.0)
            fused.append((doc_id, float(alpha * b + (1.0 - alpha) * d)))
        return fused

    # -- helpers ------------------------------------------------------------

    def _require_bm25(self) -> BM25:
        """Return the BM25 index or raise if lexical retrieval is unavailable."""
        if self.bm25 is None:
            raise RuntimeError(
                "BM25 index is not available; this SearchEngine was built "
                "without a lexical channel (bm25/hybrid modes unsupported)."
            )
        return self.bm25

    def _make_result(self, rank: int, doc_id: str, score: float) -> Result:
        """Assemble a :class:`Result`, always with a native python float score."""
        doc = self.docs.get(doc_id, {})
        return Result(
            rank=rank,
            doc_id=doc_id,
            score=float(score),
            title=doc.get("title", ""),
            text=doc.get("text", ""),
        )


def _doc_text(doc: dict) -> str:
    """Join ``title`` and ``text`` the way documents are encoded for retrieval.

    Parameters
    ----------
    doc : dict
        Document with optional ``title`` and ``text`` fields.

    Returns
    -------
    str
        ``"title text"`` when a title is present, else just the text.
    """
    title = (doc.get("title") or "").strip()
    text = doc.get("text") or ""
    return f"{title} {text}" if title else text
