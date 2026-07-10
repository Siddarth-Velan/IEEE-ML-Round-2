"""Build, save, and load the deployable search bundle.

An artifacts directory contains everything the API needs to serve queries
without recomputing embeddings::

    <dir>/docs.jsonl   # one {"_id", "title", "text"} per line, ORDER = doc_ids
    <dir>/index.faiss  # or index.npy, depending on the DenseIndex backend
    <dir>/meta.json    # model_name, dataset, n_docs, dim, backend, + extras

The positional alignment between ``docs.jsonl`` line order and the ``doc_ids``
list is load-bearing: index row ``i`` corresponds to ``doc_ids[i]`` and to the
``i``-th line of ``docs.jsonl``.
"""

from __future__ import annotations

import json
import os

from .index import DenseIndex

DOCS_FILE = "docs.jsonl"
INDEX_BASE = "index"
META_FILE = "meta.json"


def save_artifacts(
    out_dir: str,
    doc_ids: list[str],
    docs: dict,
    index: DenseIndex,
    meta: dict,
) -> None:
    """Write a deployable artifacts bundle to ``out_dir``.

    Parameters
    ----------
    out_dir : str
        Destination directory (created if missing).
    doc_ids : list of str
        Document ids in index-row order; ``doc_ids[i]`` aligns with index row
        ``i`` and with line ``i`` of ``docs.jsonl``.
    docs : dict
        ``{doc_id: {"title", "text"}}`` document contents.
    index : DenseIndex
        The dense index to persist.
    meta : dict
        Caller-supplied metadata (e.g. ``model_name``, ``dataset``). The fields
        ``n_docs``, ``dim``, and ``backend`` are filled in / overwritten from
        the index and doc list.
    """
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, DOCS_FILE), "w", encoding="utf-8") as handle:
        for doc_id in doc_ids:
            doc = docs[doc_id]
            record = {
                "_id": str(doc_id),
                "title": doc.get("title", "") or "",
                "text": doc.get("text", "") or "",
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    index.save(os.path.join(out_dir, INDEX_BASE))

    full_meta = dict(meta)
    full_meta["n_docs"] = len(doc_ids)
    full_meta["dim"] = index.dim
    full_meta["backend"] = index.backend
    with open(os.path.join(out_dir, META_FILE), "w", encoding="utf-8") as handle:
        json.dump(full_meta, handle, indent=2)


def load_artifacts(dir: str) -> tuple[list[str], dict, DenseIndex, dict]:
    """Load an artifacts bundle written by :func:`save_artifacts`.

    Parameters
    ----------
    dir : str
        Directory containing ``docs.jsonl``, the index file, and ``meta.json``.

    Returns
    -------
    doc_ids : list of str
        Document ids in index-row order.
    docs : dict
        ``{doc_id: {"title", "text"}}``.
    index : DenseIndex
        The reloaded dense index.
    meta : dict
        The metadata dictionary.
    """
    doc_ids: list[str] = []
    docs: dict = {}
    with open(os.path.join(dir, DOCS_FILE), "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            doc_id = str(record["_id"])
            doc_ids.append(doc_id)
            docs[doc_id] = {
                "title": record.get("title", "") or "",
                "text": record.get("text", "") or "",
            }

    index = DenseIndex.load(os.path.join(dir, INDEX_BASE))

    with open(os.path.join(dir, META_FILE), "r", encoding="utf-8") as handle:
        meta = json.load(handle)

    return doc_ids, docs, index, meta
