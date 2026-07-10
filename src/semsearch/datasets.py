"""BEIR-format dataset download, loading, and qrel-aware subsetting.

This module reimplements the small slice of the ``beir`` package that this
project needs (there is no ``beir`` dependency). A BEIR dataset on disk has the
layout::

    <dir>/corpus.jsonl      # one JSON object per line: {"_id", "title", "text"}
    <dir>/queries.jsonl     # one JSON object per line: {"_id", "text"}
    <dir>/qrels/<split>.tsv # TSV with header: query-id  corpus-id  score

The headline function is :func:`qrel_aware_subset`, which fixes the original
project's meaningless evaluation: the old code sampled the corpus subset at
random, so most queries' relevant documents were absent from the index and the
reported metrics were near zero. Here every judged document is guaranteed to be
in the subset.
"""

from __future__ import annotations

import io
import json
import os
import random
import zipfile

import requests

BEIR_URL = (
    "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
)


def download_beir(dataset: str, dest: str = "data") -> str:
    """Download and unzip a BEIR dataset if it is not already present.

    Parameters
    ----------
    dataset : str
        BEIR dataset name, e.g. ``"scifact"``.
    dest : str, optional
        Destination directory into which ``<dataset>/`` is extracted.

    Returns
    -------
    str
        Path to the extracted dataset directory, ``<dest>/<dataset>``.

    Notes
    -----
    The download is streamed to avoid holding the whole zip in memory at once.
    If the target directory already exists it is returned unchanged (no
    re-download).
    """
    target = os.path.join(dest, dataset)
    if os.path.isdir(target):
        return target

    os.makedirs(dest, exist_ok=True)
    url = BEIR_URL.format(dataset=dataset)
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        buffer = io.BytesIO()
        for chunk in response.iter_content(chunk_size=1 << 20):
            if chunk:
                buffer.write(chunk)
    buffer.seek(0)
    with zipfile.ZipFile(buffer) as archive:
        archive.extractall(dest)
    return target


def _read_jsonl(path: str):
    """Yield parsed JSON objects from a JSON-lines file, skipping blank lines."""
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_beir(data_dir: str, split: str = "test") -> tuple[dict, dict, dict]:
    """Load a BEIR dataset from disk.

    Parameters
    ----------
    data_dir : str
        Path to the dataset directory (contains ``corpus.jsonl``,
        ``queries.jsonl``, and ``qrels/<split>.tsv``).
    split : str, optional
        Which qrels split to load, by default ``"test"``.

    Returns
    -------
    corpus : dict
        ``{doc_id: {"title": str, "text": str}}``.
    queries : dict
        ``{qid: str}``, filtered to only queries that have at least one
        positive judgement in this split.
    qrels : dict
        ``{qid: {doc_id: int}}`` containing only judgements with score > 0.
    """
    corpus: dict = {}
    for row in _read_jsonl(os.path.join(data_dir, "corpus.jsonl")):
        doc_id = str(row["_id"])
        corpus[doc_id] = {
            "title": row.get("title", "") or "",
            "text": row.get("text", "") or "",
        }

    all_queries: dict = {}
    for row in _read_jsonl(os.path.join(data_dir, "queries.jsonl")):
        all_queries[str(row["_id"])] = row.get("text", "") or ""

    qrels: dict = {}
    qrels_path = os.path.join(data_dir, "qrels", f"{split}.tsv")
    with open(qrels_path, "r", encoding="utf-8") as handle:
        header = handle.readline()  # discard the header row
        del header
        for line in handle:
            line = line.strip()
            if not line:
                continue
            qid, doc_id, score = line.split("\t")[:3]
            score = int(score)
            if score > 0:
                qrels.setdefault(str(qid), {})[str(doc_id)] = score

    # Keep only queries that actually have positive judgements in this split.
    queries = {qid: all_queries[qid] for qid in qrels if qid in all_queries}
    return corpus, queries, qrels


def qrel_aware_subset(
    corpus: dict, qrels: dict, n_docs: int, seed: int = 0
) -> dict:
    """Build a corpus subset that always contains every judged document.

    This is the fix for the original project's meaningless metrics. The subset
    is guaranteed to contain every document referenced by any qrel; the
    remainder is filled with seeded random distractor documents drawn from the
    rest of the corpus.

    Parameters
    ----------
    corpus : dict
        Full corpus, ``{doc_id: {"title", "text"}}``.
    qrels : dict
        Judgements, ``{qid: {doc_id: int}}``.
    n_docs : int
        Target subset size. Must be >= the number of distinct judged documents.
    seed : int, optional
        Seed for the distractor sampling, making the subset deterministic.

    Returns
    -------
    dict
        The sub-corpus, ``{doc_id: {"title", "text"}}``.

    Raises
    ------
    ValueError
        If ``n_docs`` is smaller than the number of distinct judged documents.
    """
    judged: set = set()
    for rels in qrels.values():
        for doc_id in rels:
            if doc_id in corpus:
                judged.add(doc_id)

    n_judged = len(judged)
    if n_docs < n_judged:
        raise ValueError(
            f"n_docs ({n_docs}) < number of judged docs ({n_judged}); "
            "cannot build a qrel-aware subset that drops judged documents"
        )

    # Deterministic fill with distractors drawn from the unjudged remainder.
    distractor_pool = sorted(set(corpus) - judged)
    rng = random.Random(seed)
    rng.shuffle(distractor_pool)
    n_fill = n_docs - n_judged
    selected = judged.union(distractor_pool[:n_fill])

    return {doc_id: corpus[doc_id] for doc_id in selected}
