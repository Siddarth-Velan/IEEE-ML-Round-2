# semsearch

**A semantic search engine over scientific documents — dense (MiniLM + FAISS), lexical (BM25), and hybrid retrieval — served through a FastAPI backend and a Streamlit UI.**

This is a from-scratch rework of an IEEE ML Round-2 submission. The original was a Colab
notebook with a broken API, a hybrid-search bug, and near-zero evaluation metrics caused by a
sampling mistake. This version is a tested, installable Python package with a reproducible
build → evaluate → serve pipeline. The original notebook is preserved unchanged under
[`notebooks/`](notebooks/).

## Results (BEIR SciFact, 5,183 docs, 300 test queries with real relevance judgements)

| Run | nDCG@10 | Recall@10 | Recall@100 | MRR@10 |
|-----|---------|-----------|------------|--------|
| BM25 (lexical) | 0.652 | 0.774 | 0.873 | 0.619 |
| Dense (MiniLM) | 0.645 | 0.783 | 0.925 | 0.605 |
| **Hybrid (α=0.5)** | **0.713** | **0.838** | 0.954 | **0.678** |

**Hybrid retrieval beats either component alone** — +6 nDCG@10 points over BM25, +7 over dense —
because lexical and semantic matching fail on different queries and the fusion recovers both.
Dense retrieval's Recall@100 (0.925 vs BM25's 0.873) shows it pulls in semantically relevant
documents that share no keywords. Dense query latency is **7.3 ms median (9.6 ms p95)** on CPU.

Reproduce with `python scripts/evaluate.py` (writes [`reports/eval.md`](reports/eval.md)).

## What was broken, and how it's fixed

| Original defect | Fix |
|---|---|
| API crashed on every request (`zip(..., 1)` TypeError) and returned raw numpy types FastAPI can't serialize | Rewritten [`app/api.py`](app/api.py): correct code, all-native-python JSON, pydantic-validated inputs, engine warmed at startup |
| **Metrics were near-zero** — the 100K-doc corpus was subsampled *randomly*, dropping most queries' relevant documents from the index | [`qrel_aware_subset`](src/semsearch/datasets.py): every judged document is guaranteed in the index; distractors fill the rest |
| Hybrid search silently wrong — results keyed by document *text*, BM25/dense scores normalized over different ranges, never re-sorted before truncation | [`SearchEngine.search`](src/semsearch/search.py): per-channel min-max normalization, fusion keyed by **doc id**, sorted before cut |
| Query/qrel train-test split misaligned (ids split independently) | Split fixed; evaluation is on held-out queries only |
| Hardcoded `/content/drive/...` paths, model loaded at import, GPU-only | Portable artifact bundle, lazy loading, **CPU-only** |
| Depended on `beir` + `ranx` (heavy, GPU-oriented) | BEIR loading (~40 lines) and IR metrics (pure numpy) reimplemented here |

## Architecture

```
BEIR dataset ──> build_index.py ──> artifacts/  (docs.jsonl, index.faiss, meta.json)
                                        │
                        SearchEngine ───┤ dense: MiniLM embeddings + FAISS inner-product
                        (dense/bm25/    │ bm25:  rank-bm25 over tokenized corpus
                         hybrid)        │ hybrid: normalized score fusion, α-weighted
                                        │
                    app/api.py (FastAPI) ──HTTP──> app/ui.py (Streamlit)
```

The package is cleanly layered: `datasets`, `encoder`, `index`, `bm25`, `search`, `metrics`,
`artifacts`. The API depends on the library; the UI depends only on the API over HTTP.

## Quickstart

```bash
python -m venv .venv && .venv/Scripts/pip install -e .[dev]   # CPU-only, no GPU needed

pytest -q                                                     # 39 passed
python scripts/build_index.py --dataset scifact               # download + encode, ~70s on CPU
python scripts/evaluate.py                                    # writes reports/eval.md

uvicorn app.api:app --port 8000                               # backend
streamlit run app/ui.py                                       # UI at http://localhost:8501
```

Query the API directly:

```bash
curl -X POST http://127.0.0.1:8000/search -H "Content-Type: application/json" \
  -d '{"query": "does vitamin D deficiency increase infection risk", "k": 5, "mode": "hybrid", "alpha": 0.5}'
```

`mode` is `dense`, `bm25`, or `hybrid`; `alpha` weights BM25 vs dense in hybrid mode
(0 = pure dense, 1 = pure lexical).

## Testing

39 tests, all offline (no model download, no network): a deterministic `FakeEncoder` and
injected engine let the search and API layers be tested in milliseconds. Covered: qrel-aware
subsetting invariants, hand-computed IR metric values, FAISS/numpy backend equivalence, hybrid
fusion ordering (α=0 reproduces dense ranking, α=1 reproduces BM25), and API validation.

## Notes & honest limitations

- SciFact is the default because it's small enough to build and evaluate on CPU in about a
  minute while having real graded qrels. The pipeline supports any BEIR dataset (including
  MS MARCO) via `--dataset`; larger corpora just take longer to encode.
- The dense model is the off-the-shelf `all-MiniLM-L6-v2` — no fine-tuning. The original
  attempted contrastive fine-tuning with hard negatives; that path is preserved in the
  notebook and is a natural next step here (fine-tune, re-encode, re-evaluate — the pipeline
  is set up for it).
- Hybrid α is fixed per request; the evaluation sweeps α ∈ {0.3, 0.5, 0.7} and 0.5 wins on
  this dataset, but the optimal value is dataset-dependent.
