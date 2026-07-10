"""Tests for the pure-numpy IR metrics, using hand-computed expected values."""

import math

from semsearch.metrics import evaluate_run


def test_single_relevant_at_rank_2():
    """One query, single relevant doc at rank 2 among 5 retrieved (k in {1,5}).

    Ranking by score: dA(.9) dB(.8) dC(.7) dD(.6) dE(.5); relevant = {dB}, gain 1.

    Hand computation
    ----------------
    MRR@10        : first relevant at rank 2               → 1/2      = 0.5
    recall@1      : top1={dA}, 0 hits / 1 relevant                    = 0.0
    recall@5      : top5 contains dB, 1 hit / 1 relevant              = 1.0
    precision@1   : 0 hits / 1                                        = 0.0
    precision@5   : 1 hit / 5                                         = 0.2
    nDCG@1        : gain@1 = 0 → DCG=0, IDCG=1 → 0/1                   = 0.0
    nDCG@5        : DCG = 1/log2(2+1) = 1/log2(3);  IDCG = 1/log2(2)=1
                    → 1/log2(3)                                       ≈ 0.6309298
    """
    run = {"q1": {"dA": 0.9, "dB": 0.8, "dC": 0.7, "dD": 0.6, "dE": 0.5}}
    qrels = {"q1": {"dB": 1}}

    m = evaluate_run(run, qrels, ks=(1, 5))

    assert m["mrr@10"] == 0.5
    assert m["recall@1"] == 0.0
    assert m["recall@5"] == 1.0
    assert m["precision@1"] == 0.0
    assert m["precision@5"] == 0.2
    assert m["ndcg@1"] == 0.0
    assert math.isclose(m["ndcg@5"], 1.0 / math.log2(3), rel_tol=1e-9)


def test_perfect_ranking_all_ones():
    """Three relevant docs ranked exactly top-3, k=3 → every metric = 1.0."""
    run = {"q1": {"dA": 0.9, "dB": 0.8, "dC": 0.7, "dX": 0.1}}
    qrels = {"q1": {"dA": 1, "dB": 1, "dC": 1}}

    m = evaluate_run(run, qrels, ks=(3,))

    assert m["mrr@10"] == 1.0
    assert math.isclose(m["ndcg@3"], 1.0)
    assert m["recall@3"] == 1.0
    assert m["precision@3"] == 1.0


def test_query_missing_from_run_is_skipped():
    """A qrel query absent from the run is skipped, not a crash.

    q1 present in both → its perfect result drives the average; q2 present only
    in qrels is skipped, so the average equals q1's metrics.
    """
    run = {"q1": {"dA": 0.9, "dB": 0.1}}
    qrels = {"q1": {"dA": 1}, "q2": {"dZ": 1}}

    m = evaluate_run(run, qrels, ks=(1,))

    assert m["mrr@10"] == 1.0
    assert m["recall@1"] == 1.0
    assert m["precision@1"] == 1.0
    assert m["ndcg@1"] == 1.0


def test_empty_qrels_query_skipped():
    """A query whose qrels are empty is skipped."""
    run = {"q1": {"dA": 0.9}, "q2": {"dB": 0.9}}
    qrels = {"q1": {"dA": 1}, "q2": {}}

    m = evaluate_run(run, qrels, ks=(1,))
    # Only q1 counts; perfect → 1.0.
    assert m["precision@1"] == 1.0


def test_no_overlap_returns_zeros():
    """No query in both run and qrels → all-zero metrics, no crash."""
    run = {"q9": {"dA": 0.9}}
    qrels = {"q1": {"dA": 1}}

    m = evaluate_run(run, qrels, ks=(1, 5))
    assert all(value == 0.0 for value in m.values())
