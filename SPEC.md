# semsearch — Rework Specification

**Project:** Rework of the IEEE-ML-Round-2 semantic search engine: from a Colab notebook with a broken API into a clean package with a FastAPI backend, Streamlit frontend, honest evaluation, and tests.
**Constraints:** CPU only. Python 3.14. Windows. Installed deps: torch (cpu), sentence-transformers, faiss-cpu, rank-bm25, fastapi, uvicorn, streamlit, requests, httpx, pytest. NO beir, NO ranx, NO numba — their jobs are reimplemented here (BEIR-format loading is ~40 lines; metrics are pure numpy).
**Default dataset:** BEIR **scifact** (5,183 docs, 300 test queries with qrels) — small enough to download + CPU-encode in ~1 minute, real qrels so metrics mean something. MS MARCO stays supported via the same loader with qrel-aware subsetting.

## Known defects in the original (fix ALL of these — they explain the failures in the old README)

1. `api.py`: `zip(s[0], idx[0], 1)` → TypeError at request time (the 500 error); numpy float32/int64 returned in JSON (FastAPI encoder rejects); hardcoded `/content/drive/...` paths; model loaded at import with no error handling.
2. Evaluation was meaningless: the 100K-doc corpus subset was sampled RANDOMLY, so most queries' relevant docs weren't in the index at all → near-zero metrics. Fix: **qrel-aware subsetting** (always include every judged doc, fill the rest with random distractors).
3. Hybrid search: results keyed by document TEXT instead of doc id; BM25 scores min-max scaled over the full corpus while dense scores scaled over top-k only (incomparable); fused results never sorted before truncation.
4. `train_test_split(qid, qrid, ...)` split query-ids and qrel-ids independently — train qrels didn't match train queries.

## Layout

```
pyproject.toml               # package name: semsearch, src layout, pytest config
src/semsearch/
  __init__.py
  datasets.py   # BEIR-format download/load + qrel-aware subsetting (no beir dep)
  encoder.py    # SentenceTransformer wrapper
  index.py      # dense index: faiss IndexFlatIP, numpy fallback
  bm25.py       # tokenization + BM25 wrapper (rank_bm25)
  search.py     # SearchEngine: dense / bm25 / hybrid, id-keyed, sorted
  metrics.py    # pure-numpy IR metrics
  artifacts.py  # build/save/load the deployable bundle
app/
  api.py        # FastAPI backend
  ui.py         # Streamlit frontend
scripts/
  build_index.py
  evaluate.py
tests/          # test_datasets.py test_metrics.py test_index.py test_search.py test_api.py test_artifacts.py
notebooks/IEEEML2.ipynb      # original notebook, moved here unchanged
artifacts/    reports/       # generated (gitignored)
```

## Module contracts (exact signatures)

### datasets.py
BEIR layout on disk: `<dir>/corpus.jsonl` (`_id`, `title`, `text`), `<dir>/queries.jsonl` (`_id`, `text`), `<dir>/qrels/{split}.tsv` (TSV: query-id, corpus-id, score; header row present).
```python
def download_beir(dataset: str, dest: str = "data") -> str
    # GET https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip
    # via requests (stream), unzip to dest/, return dest/{dataset}. Skip download if already present.
def load_beir(data_dir: str, split: str = "test") -> tuple[dict, dict, dict]
    # corpus: {doc_id: {"title": str, "text": str}}, queries: {qid: str},
    # qrels: {qid: {doc_id: int}} with score > 0 only; queries filtered to those with qrels.
def qrel_aware_subset(corpus: dict, qrels: dict, n_docs: int, seed: int = 0) -> dict
    # ALWAYS includes every doc referenced by any qrel; fills to n_docs with seeded random
    # distractors; returns the sub-corpus dict. n_docs >= n_judged or ValueError.
```

### encoder.py
```python
class Encoder:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2")
        # lazy: the SentenceTransformer is constructed on first use, not in __init__
    def encode_docs(self, texts: list[str], batch_size: int = 64) -> np.ndarray  # (n, d) float32 L2-NORMALIZED
    def encode_query(self, query: str) -> np.ndarray                             # (d,) float32 L2-normalized
    @property
    def dim(self) -> int
```
Doc text convention: encode `title + " " + text` when title is non-empty (document this).

### index.py
```python
class DenseIndex:
    @classmethod
    def build(cls, embeddings: np.ndarray) -> "DenseIndex"   # inner product on normalized vectors = cosine
    def search(self, query_vec: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]
        # (scores (k,), positions (k,)) — positions are ROW indices into the build matrix
    def save(self, path: str) -> None
    @classmethod
    def load(cls, path: str) -> "DenseIndex"
    @property
    def size(self) -> int
```
Backend: faiss.IndexFlatIP when faiss imports, else exact numpy matmul top-k (argpartition). Backend choice is an implementation detail invisible to callers; `save` writes faiss index file or .npy accordingly, `load` sniffs which. Both backends must return identical results on the same data (tested).

### bm25.py
```python
def tokenize(text: str) -> list[str]                      # lowercase \w+ regex
class BM25:
    def __init__(self, texts: list[str])                  # positional, aligned with doc order
    def search(self, query: str, k: int) -> tuple[np.ndarray, np.ndarray]  # (scores, positions), sorted desc
```

### search.py
```python
@dataclass
class Result:
    rank: int; doc_id: str; score: float; title: str; text: str   # score ALWAYS python float

class SearchEngine:
    def __init__(self, doc_ids: list[str], docs: dict[str, dict], encoder: Encoder,
                 dense_index: DenseIndex, bm25: BM25 | None = None)
    @classmethod
    def from_artifacts(cls, artifacts_dir: str) -> "SearchEngine"
    def search(self, query: str, k: int = 10, mode: str = "dense", alpha: float = 0.5) -> list[Result]
        # mode in {"dense", "bm25", "hybrid"}; ValueError otherwise. bm25/hybrid raise
        # RuntimeError if bm25 is None. Results ranked 1..k, sorted by score desc.
    def run_for_eval(self, queries: dict[str, str], k: int, mode: str, alpha: float = 0.5) -> dict
        # {qid: {doc_id: score}} — the evaluation "run" format
```
Hybrid fusion (fix of defect #3): retrieve top-100 dense and top-100 bm25 candidates; min-max normalize EACH list over its own candidates (guard zero range → all 0.5); missing candidate gets 0 in that channel; fused = alpha*bm25 + (1-alpha)*dense **keyed by doc_id**; SORT desc; return top k. alpha=1 → pure bm25 ordering, alpha=0 → pure dense ordering (tested).

### metrics.py (pure numpy, no deps)
```python
def evaluate_run(run: dict, qrels: dict, ks: tuple = (1, 5, 10, 100)) -> dict[str, float]
    # keys: "mrr@10", and f"ndcg@{k}", f"recall@{k}", f"precision@{k}" for each k.
    # Standard defs: MRR@10 over first relevant rank; nDCG with graded gains rel/log2(rank+1),
    # ideal DCG from sorted qrel gains; recall@k = |retrieved∩relevant|/|relevant|.
    # Averaged over queries present in BOTH run and qrels. Queries with empty qrels skipped.
```

### artifacts.py
```python
def save_artifacts(out_dir, doc_ids, docs, index: DenseIndex, meta: dict) -> None
    # writes: docs.jsonl (one {"_id","title","text"} per line, ORDER = doc_ids order),
    #         index file (index.faiss or index.npy per backend), meta.json
    #         (model_name, dataset, n_docs, dim, backend, plus caller-supplied fields)
def load_artifacts(dir) -> tuple[list[str], dict, DenseIndex, dict]   # doc_ids, docs, index, meta
```
`SearchEngine.from_artifacts` uses load_artifacts + meta["model_name"] for the Encoder, and builds BM25 from the stored texts (fast enough at this scale).

### app/api.py  (FastAPI — this is the deployable)
- Engine loaded in a **lifespan** handler from env `SEMSEARCH_ARTIFACTS` (default `artifacts/scifact-minilm`); startup fails with a clear message pointing at scripts/build_index.py if missing.
- CORS middleware allow_origins ["*"].
- `GET /healthz` → `{"status": "ok", "n_docs": int, "model": str, "dataset": str}`
- `POST /search` body `{"query": str (min_length 1 after strip), "k": int 1..100 = 10, "mode": "dense"|"bm25"|"hybrid" = "dense", "alpha": float 0..1 = 0.5}` → `{"query", "mode", "latency_ms": float, "results": [{"rank", "doc_id", "score", "title", "text"}]}`
- Every number in JSON is a native python type. Validation errors → 422 (pydantic) or 400 with detail. Engine accessed via app.state (tests can inject a fake).
- Runnable: `uvicorn app.api:app --port 8000`.

### app/ui.py  (Streamlit — talks to the API over HTTP, never imports semsearch)
- `SEMSEARCH_API` env, default `http://127.0.0.1:8000`.
- Sidebar: API health (calls /healthz; friendly error state if down), corpus size, model name, mode radio (dense/hybrid/bm25), k slider 1-50, alpha slider (only shown for hybrid).
- Main: search box + button; on search POST /search; show latency badge and results as cards: rank, score (3 decimals), bold title, expander with full text.
- Handles API errors (connection refused, 4xx) with st.error, never a stack trace.

### scripts/build_index.py
`--dataset scifact --model sentence-transformers/all-MiniLM-L6-v2 --n-docs 0 (0 = full corpus; >0 = qrel-aware subset) --data-dir data --out artifacts/{dataset}-minilm --batch-size 64`
download → load → (subset) → encode (title+text) → DenseIndex.build → save_artifacts. Prints progress + final summary (n_docs, dim, seconds). `if __name__ == "__main__"` guard.

### scripts/evaluate.py
`--artifacts artifacts/scifact-minilm --data-dir data/scifact --split test --k 100 --alphas 0.3 0.5 0.7`
Loads engine, runs bm25 / dense / hybrid(each alpha) over the split's queries, evaluate_run each, writes `reports/eval.md` with a markdown metrics table + p50/p95 query latency (dense mode, measured over the real queries) + provenance (dataset, model, n_docs, date from meta). Prints the table.

## Tests (no network, no model download — every test runs offline in seconds)
The key enabler: a `FakeEncoder` (deterministic seeded random unit vectors per text, cached) used everywhere an Encoder is needed; SearchEngine takes injected components by design.
- **test_datasets**: load_beir on a FABRICATED tiny BEIR dir (write corpus.jsonl/queries.jsonl/qrels tsv in tmp_path); qrel_aware_subset: all judged docs present, exact n_docs, deterministic per seed, ValueError when n_docs < n_judged.
- **test_metrics**: hand-computed cases — single query where ndcg@k/mrr/recall/precision are known exactly (e.g. relevant at rank 2 among k=5); perfect ranking → all 1.0; query missing from run → skipped not crash.
- **test_index**: numpy fallback vs faiss backend (skipif faiss missing) identical top-k on random normalized data; save/load round-trip both backends; k > size handled.
- **test_search**: tiny 8-doc corpus + FakeEncoder: dense results sorted desc + ranks 1..k; hybrid alpha=0 ordering == dense ordering, alpha=1 == bm25 ordering; hybrid keyed by doc_id (a duplicated TEXT in two docs must still yield two distinct doc_ids); all scores python float; mode validation errors.
- **test_api**: FastAPI TestClient with a fake engine injected into app.state (no lifespan model load): /healthz shape; /search happy path JSON-serializable; empty query → 4xx; k=0 → 422; bad mode → 4xx.
- **test_artifacts**: save/load round-trip with FakeEncoder-built index preserves ids, docs, meta, and search results.

## Quality bar
Same as prior projects: numpy-style docstrings stating method + units, type hints, no prints in src/ (scripts/app only), seeds explicit. README.md rewritten LAST by the architect from real deploy + eval results — do NOT write it. Original notebook preserved unchanged under notebooks/.
