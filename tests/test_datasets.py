"""Tests for BEIR loading and qrel-aware subsetting.

All tests run against a fabricated tiny BEIR directory in ``tmp_path`` — no
network, no real dataset.
"""

import json
import os

import pytest

from semsearch.datasets import load_beir, qrel_aware_subset


def _write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _make_beir_dir(root):
    """Create a minimal BEIR dataset on disk and return its path."""
    _write_jsonl(
        os.path.join(root, "corpus.jsonl"),
        [
            {"_id": "d1", "title": "Alpha", "text": "first document"},
            {"_id": "d2", "title": "Beta", "text": "second document"},
            {"_id": "d3", "title": "", "text": "third document"},
            {"_id": "d4", "title": "Delta", "text": "fourth document"},
            {"_id": "d5", "title": "Epsilon", "text": "fifth document"},
            {"_id": "d6", "title": "Zeta", "text": "sixth document"},
        ],
    )
    _write_jsonl(
        os.path.join(root, "queries.jsonl"),
        [
            {"_id": "q1", "text": "query one"},
            {"_id": "q2", "text": "query two"},
            {"_id": "q3", "text": "query three with no judgements"},
        ],
    )
    os.makedirs(os.path.join(root, "qrels"), exist_ok=True)
    with open(os.path.join(root, "qrels", "test.tsv"), "w", encoding="utf-8") as h:
        h.write("query-id\tcorpus-id\tscore\n")
        h.write("q1\td1\t1\n")
        h.write("q1\td2\t2\n")
        h.write("q1\td4\t0\n")  # zero score → must be filtered out
        h.write("q2\td3\t1\n")
        # q3 has no qrels at all
    return root


def test_load_beir_shapes(tmp_path):
    data_dir = _make_beir_dir(str(tmp_path))
    corpus, queries, qrels = load_beir(data_dir, split="test")

    assert len(corpus) == 6
    assert corpus["d1"] == {"title": "Alpha", "text": "first document"}
    assert corpus["d3"]["title"] == ""  # empty title preserved

    # Queries filtered to those with positive qrels: q1, q2 (not q3).
    assert set(queries) == {"q1", "q2"}

    # Zero-score judgement dropped; graded scores preserved.
    assert qrels["q1"] == {"d1": 1, "d2": 2}
    assert qrels["q2"] == {"d3": 1}
    assert "d4" not in qrels["q1"]


def test_qrel_aware_subset_includes_all_judged(tmp_path):
    data_dir = _make_beir_dir(str(tmp_path))
    corpus, _, qrels = load_beir(data_dir)

    # Judged docs are d1, d2, d3 (three distinct).
    subset = qrel_aware_subset(corpus, qrels, n_docs=5, seed=0)
    assert len(subset) == 5
    for judged in ("d1", "d2", "d3"):
        assert judged in subset


def test_qrel_aware_subset_deterministic(tmp_path):
    data_dir = _make_beir_dir(str(tmp_path))
    corpus, _, qrels = load_beir(data_dir)

    a = qrel_aware_subset(corpus, qrels, n_docs=5, seed=7)
    b = qrel_aware_subset(corpus, qrels, n_docs=5, seed=7)
    assert set(a) == set(b)  # same seed → same subset


def test_qrel_aware_subset_exact_judged_only(tmp_path):
    data_dir = _make_beir_dir(str(tmp_path))
    corpus, _, qrels = load_beir(data_dir)

    # n_docs == n_judged → exactly the judged docs, no distractors.
    subset = qrel_aware_subset(corpus, qrels, n_docs=3, seed=0)
    assert set(subset) == {"d1", "d2", "d3"}


def test_qrel_aware_subset_too_small_raises(tmp_path):
    data_dir = _make_beir_dir(str(tmp_path))
    corpus, _, qrels = load_beir(data_dir)

    with pytest.raises(ValueError):
        qrel_aware_subset(corpus, qrels, n_docs=2, seed=0)
