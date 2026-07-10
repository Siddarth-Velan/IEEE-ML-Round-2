"""Pure-numpy information-retrieval metrics.

Reimplements the handful of ranking metrics this project reports (there is no
``ranx`` dependency). All definitions are the standard IR ones:

* **MRR@10** - reciprocal of the rank of the first relevant document within the
  top 10 (0 if none), averaged over queries.
* **nDCG@k** - graded DCG ``sum(rel_i / log2(rank_i + 1))`` over the top ``k``
  ranks (rank starts at 1, so the rank-1 discount is ``log2(2) = 1``),
  normalized by the ideal DCG obtained by sorting the query's graded
  judgements descending.
* **recall@k** - ``|retrieved_topk ∩ relevant| / |relevant|`` (binary
  relevance: any positive judgement counts).
* **precision@k** - ``|retrieved_topk ∩ relevant| / k``.

A "run" is ``{qid: {doc_id: score}}`` and qrels are ``{qid: {doc_id: gain}}``.
"""

from __future__ import annotations

import numpy as np


def _rank_doc_ids(doc_scores: dict) -> list:
    """Return doc ids sorted by score descending (stable on insertion order)."""
    return sorted(doc_scores, key=lambda doc_id: doc_scores[doc_id], reverse=True)


def _reciprocal_rank(ranked: list, relevant: set, cutoff: int) -> float:
    """Reciprocal rank of the first relevant doc within ``cutoff`` (else 0)."""
    for rank, doc_id in enumerate(ranked[:cutoff], start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def _dcg(gains: np.ndarray) -> float:
    """Discounted cumulative gain of an ordered gain vector.

    ``gains[i]`` is the graded relevance at rank ``i + 1``; the discount for
    rank ``r`` is ``1 / log2(r + 1)``.
    """
    if gains.size == 0:
        return 0.0
    ranks = np.arange(1, gains.size + 1)
    discounts = 1.0 / np.log2(ranks + 1.0)
    return float(np.sum(gains * discounts))


def _ndcg_at_k(ranked: list, gains_by_doc: dict, k: int) -> float:
    """nDCG@k using graded gains from ``gains_by_doc``."""
    gains = np.array(
        [gains_by_doc.get(doc_id, 0) for doc_id in ranked[:k]], dtype=np.float64
    )
    dcg = _dcg(gains)
    ideal_gains = np.array(
        sorted(gains_by_doc.values(), reverse=True)[:k], dtype=np.float64
    )
    idcg = _dcg(ideal_gains)
    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def evaluate_run(
    run: dict, qrels: dict, ks: tuple = (1, 5, 10, 100)
) -> dict[str, float]:
    """Average IR metrics over a run.

    Parameters
    ----------
    run : dict
        ``{qid: {doc_id: score}}`` retrieval results.
    qrels : dict
        ``{qid: {doc_id: gain}}`` graded relevance judgements (gain > 0 means
        relevant).
    ks : tuple of int, optional
        Cutoffs for nDCG / recall / precision, by default ``(1, 5, 10, 100)``.

    Returns
    -------
    dict of str to float
        Keys: ``"mrr@10"`` plus ``f"ndcg@{k}"``, ``f"recall@{k}"``,
        ``f"precision@{k}"`` for each ``k`` in ``ks``. Averaged over queries
        present in BOTH ``run`` and ``qrels``; queries whose qrels are empty are
        skipped.
    """
    metric_names = ["mrr@10"]
    for k in ks:
        metric_names += [f"ndcg@{k}", f"recall@{k}", f"precision@{k}"]
    sums = {name: 0.0 for name in metric_names}

    n_queries = 0
    for qid, gains_by_doc in qrels.items():
        if not gains_by_doc:  # empty qrels → skip
            continue
        if qid not in run:  # query missing from run → skip (not a crash)
            continue

        n_queries += 1
        ranked = _rank_doc_ids(run[qid])
        relevant = {doc_id for doc_id, gain in gains_by_doc.items() if gain > 0}
        n_relevant = len(relevant)

        sums["mrr@10"] += _reciprocal_rank(ranked, relevant, cutoff=10)
        for k in ks:
            topk = ranked[:k]
            hits = sum(1 for doc_id in topk if doc_id in relevant)
            sums[f"ndcg@{k}"] += _ndcg_at_k(ranked, gains_by_doc, k)
            sums[f"recall@{k}"] += hits / n_relevant if n_relevant else 0.0
            sums[f"precision@{k}"] += hits / k

    if n_queries == 0:
        return {name: 0.0 for name in metric_names}
    return {name: sums[name] / n_queries for name in metric_names}
