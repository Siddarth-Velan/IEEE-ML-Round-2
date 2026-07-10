"""Evaluate bm25 / dense / hybrid retrieval on a BEIR split and write a report.

Produces ``reports/eval.md`` with an IR-metrics table (via
``metrics.evaluate_run``), dense-mode query latency (p50/p95 over the real
queries), and provenance, then prints the table.

Usage::

    python scripts/evaluate.py --artifacts artifacts/scifact-minilm \\
        --data-dir data/scifact --split test --k 100 --alphas 0.3 0.5 0.7
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date
from pathlib import Path

# Windows consoles default to cp1252, which cannot print characters like alpha;
# reconfigure stdout/stderr so script output never crashes on encoding.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

# Metrics reported in the table (subset of what evaluate_run returns).
_TABLE_METRICS = [
    ("nDCG@10", "ndcg@10"),
    ("Recall@10", "recall@10"),
    ("Recall@100", "recall@100"),
    ("P@10", "precision@10"),
    ("MRR@10", "mrr@10"),
]


def _percentile(values: list[float], pct: float) -> float:
    """Linear-interpolated percentile of ``values`` (empty → 0.0)."""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (pct / 100.0) * (len(ordered) - 1)
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    frac = rank - low
    return ordered[low] + (ordered[high] - ordered[low]) * frac


def _measure_dense_latency(engine, queries: dict, k: int) -> tuple[float, float]:
    """Return ``(p50_ms, p95_ms)`` for dense-mode single-query latency."""
    latencies: list[float] = []
    for text in queries.values():
        start = time.perf_counter()
        engine.search(text, k=k, mode="dense")
        latencies.append((time.perf_counter() - start) * 1000.0)
    return _percentile(latencies, 50), _percentile(latencies, 95)


def _read_meta(artifacts_dir: str) -> dict:
    """Read ``meta.json`` from the artifact bundle (``{}`` if missing)."""
    meta_path = Path(artifacts_dir) / "meta.json"
    if not meta_path.exists():
        return {}
    with meta_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _render_markdown(rows: list[tuple[str, dict]], latency: tuple[float, float],
                     meta: dict) -> str:
    """Render the full report markdown from evaluated rows and provenance."""
    header = "| Run | " + " | ".join(label for label, _ in _TABLE_METRICS) + " |"
    sep = "| --- | " + " | ".join("---" for _ in _TABLE_METRICS) + " |"
    lines = [header, sep]
    for name, scores in rows:
        cells = " | ".join(f"{scores.get(key, 0.0):.4f}" for _, key in _TABLE_METRICS)
        lines.append(f"| {name} | {cells} |")
    table = "\n".join(lines)

    p50, p95 = latency
    provenance = (
        f"- **Dataset:** {meta.get('dataset', '—')}\n"
        f"- **Model:** {meta.get('model_name', '—')}\n"
        f"- **Documents:** {meta.get('n_docs', '—')}\n"
        f"- **Backend:** {meta.get('backend', '—')}\n"
        f"- **Generated:** {date.today().isoformat()}\n"
    )
    return (
        "# semsearch evaluation\n\n"
        f"{provenance}\n"
        "## Retrieval metrics\n\n"
        f"{table}\n\n"
        "## Dense query latency\n\n"
        f"- p50: {p50:.2f} ms\n"
        f"- p95: {p95:.2f} ms\n"
    )


def main() -> None:
    """Parse args, evaluate every run configuration, and write the report."""
    parser = argparse.ArgumentParser(description="Evaluate semsearch retrieval.")
    parser.add_argument("--artifacts", default="artifacts/scifact-minilm")
    parser.add_argument("--data-dir", default="data/scifact")
    parser.add_argument("--split", default="test")
    parser.add_argument("--k", type=int, default=100)
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.3, 0.5, 0.7])
    parser.add_argument("--out", default="reports/eval.md")
    args = parser.parse_args()

    from semsearch.datasets import load_beir
    from semsearch.metrics import evaluate_run
    from semsearch.search import SearchEngine

    print(f"Loading engine from {args.artifacts} ...")
    engine = SearchEngine.from_artifacts(args.artifacts)
    meta = _read_meta(args.artifacts)

    print(f"Loading queries/qrels from {args.data_dir} [{args.split}] ...")
    _, queries, qrels = load_beir(args.data_dir, split=args.split)

    configs: list[tuple[str, str, float]] = [
        ("bm25", "bm25", 0.5),
        ("dense", "dense", 0.5),
    ]
    for alpha in args.alphas:
        configs.append((f"hybrid α={alpha:g}", "hybrid", alpha))

    rows: list[tuple[str, dict]] = []
    for name, mode, alpha in configs:
        print(f"Running {name} ...")
        run = engine.run_for_eval(queries, k=args.k, mode=mode, alpha=alpha)
        rows.append((name, evaluate_run(run, qrels)))

    print("Measuring dense latency ...")
    latency = _measure_dense_latency(engine, queries, args.k)

    report = _render_markdown(rows, latency, meta)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    print(f"\nWrote {out_path}\n")
    print(report)


if __name__ == "__main__":
    main()
