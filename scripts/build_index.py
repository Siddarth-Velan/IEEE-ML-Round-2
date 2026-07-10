"""Build a deployable semantic-search index from a BEIR dataset.

Pipeline: download -> load -> (optional qrel-aware subset) -> encode (title+text)
-> build a dense index -> save the artifacts bundle.

Example
-------
    python scripts/build_index.py --dataset scifact \
        --model sentence-transformers/all-MiniLM-L6-v2 \
        --n-docs 0 --data-dir data --out artifacts/scifact-minilm

``--n-docs 0`` indexes the full corpus; ``--n-docs N`` (N > 0) builds a
qrel-aware subset of N documents (every judged document is always included).
"""

from __future__ import annotations

import argparse
import sys
import time

# Windows consoles default to cp1252; reconfigure so output never crashes on
# encoding (model progress bars and paths can carry non-cp1252 characters).
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

from semsearch.artifacts import save_artifacts
from semsearch.datasets import download_beir, load_beir, qrel_aware_subset
from semsearch.encoder import Encoder, doc_text
from semsearch.index import DenseIndex


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="scifact")
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument(
        "--n-docs",
        type=int,
        default=0,
        help="0 = full corpus; >0 = qrel-aware subset of this many docs",
    )
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--out", default=None, help="output artifacts dir")
    parser.add_argument("--split", default="test")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    out_dir = args.out or f"artifacts/{args.dataset}-minilm"
    start = time.time()

    print(f"[build] downloading BEIR dataset '{args.dataset}' into {args.data_dir}/")
    data_dir = download_beir(args.dataset, args.data_dir)

    print(f"[build] loading corpus/queries/qrels (split={args.split})")
    corpus, queries, qrels = load_beir(data_dir, split=args.split)
    print(
        f"[build] loaded {len(corpus)} docs, {len(queries)} queries, "
        f"{len(qrels)} qrel sets"
    )

    if args.n_docs and args.n_docs > 0:
        corpus = qrel_aware_subset(corpus, qrels, args.n_docs, seed=args.seed)
        print(f"[build] qrel-aware subset -> {len(corpus)} docs")

    doc_ids = list(corpus.keys())
    texts = [doc_text(corpus[d]["title"], corpus[d]["text"]) for d in doc_ids]

    print(f"[build] encoding {len(texts)} docs with '{args.model}'")
    encoder = Encoder(args.model)
    embeddings = encoder.encode_docs(texts, batch_size=args.batch_size)

    print(f"[build] building dense index (dim={embeddings.shape[1]})")
    index = DenseIndex.build(embeddings)

    meta = {"model_name": args.model, "dataset": args.dataset}
    save_artifacts(out_dir, doc_ids, corpus, index, meta)

    elapsed = time.time() - start
    print(
        f"[build] done: n_docs={len(doc_ids)} dim={index.dim} "
        f"backend={index.backend} -> {out_dir} ({elapsed:.1f}s)"
    )


if __name__ == "__main__":
    main()
