"""FastAPI backend for semsearch — the deployable service.

Fixes the original ``api.py`` defects (SPEC "Known defects" #1): no import-time
model load, no hardcoded Drive paths, native-python JSON numbers, and a
lifespan-managed engine loaded from the ``SEMSEARCH_ARTIFACTS`` bundle.

Run with::

    uvicorn app.api:app --port 8000
"""

from __future__ import annotations

import json
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

DEFAULT_ARTIFACTS = "artifacts/scifact-minilm"


def _read_meta(artifacts_dir: str) -> dict:
    """Read ``meta.json`` from an artifact bundle, returning ``{}`` if absent."""
    meta_path = Path(artifacts_dir) / "meta.json"
    if not meta_path.exists():
        return {}
    with meta_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _load_engine_into_state(app: FastAPI) -> None:
    """Load the search engine + meta into ``app.state`` from env config.

    Reads ``SEMSEARCH_ARTIFACTS`` (default ``artifacts/scifact-minilm``) and
    builds the engine. Raises a clear ``RuntimeError`` pointing at the build
    script when the bundle is missing.
    """
    artifacts_dir = os.environ.get("SEMSEARCH_ARTIFACTS", DEFAULT_ARTIFACTS)
    if not Path(artifacts_dir).exists():
        raise RuntimeError(
            f"Artifacts not found at {artifacts_dir!r}. Build them first:\n"
            f"    python scripts/build_index.py --dataset scifact "
            f"--out {artifacts_dir}\n"
            f"or set SEMSEARCH_ARTIFACTS to an existing bundle."
        )
    # Imported lazily so importing this module never triggers heavy loads.
    from semsearch.search import SearchEngine

    engine = SearchEngine.from_artifacts(artifacts_dir)
    # Warm up the lazily-constructed encoder (and any faiss threading) with a
    # throwaway query so the first real request does not pay the model-load cost.
    try:
        engine.search("warmup", k=1, mode="dense")
    except Exception:  # pragma: no cover - warmup must never block startup
        pass
    app.state.engine = engine
    app.state.meta = _read_meta(artifacts_dir)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the engine on startup unless one was already injected (tests)."""
    if getattr(app.state, "engine", None) is None:
        _load_engine_into_state(app)
    yield


app = FastAPI(title="semsearch", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SearchRequest(BaseModel):
    """Request body for ``POST /search``."""

    query: str = Field(..., min_length=1)
    k: int = Field(default=10, ge=1, le=100)
    mode: Literal["dense", "bm25", "hybrid"] = "dense"
    alpha: float = Field(default=0.5, ge=0.0, le=1.0)

    @field_validator("query")
    @classmethod
    def _strip_non_empty(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("query must be non-empty after stripping whitespace")
        return stripped


class ResultItem(BaseModel):
    """One ranked hit in a search response."""

    rank: int
    doc_id: str
    score: float
    title: str
    text: str


class SearchResponse(BaseModel):
    """Response body for ``POST /search``."""

    query: str
    mode: str
    latency_ms: float
    results: list[ResultItem]


def _get_engine(request: Request):
    """Return the engine from ``app.state`` or 503 if it failed to load."""
    engine = getattr(request.app.state, "engine", None)
    if engine is None:
        raise HTTPException(status_code=503, detail="search engine not loaded")
    return engine


@app.get("/")
def root() -> dict:
    """Landing endpoint: point callers at the real routes and the docs."""
    return {
        "service": "semsearch",
        "endpoints": {
            "health": "/healthz",
            "search": "POST /search",
            "docs": "/docs",
        },
    }


@app.get("/healthz")
def healthz(request: Request) -> dict:
    """Liveness + corpus metadata.

    Returns
    -------
    dict
        ``{"status": "ok", "n_docs": int, "model": str, "dataset": str}``.
    """
    engine = _get_engine(request)
    meta = getattr(request.app.state, "meta", {}) or {}
    return {
        "status": "ok",
        "n_docs": len(engine.doc_ids),
        "model": meta.get("model_name", ""),
        "dataset": meta.get("dataset", ""),
    }


@app.post("/search", response_model=SearchResponse)
def search(payload: SearchRequest, request: Request) -> SearchResponse:
    """Run a search and return ranked results with measured latency.

    Latency is wall-clock milliseconds around the engine call only.
    """
    engine = _get_engine(request)
    start = time.perf_counter()
    try:
        hits = engine.search(
            payload.query, k=payload.k, mode=payload.mode, alpha=payload.alpha
        )
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    latency_ms = (time.perf_counter() - start) * 1000.0
    return SearchResponse(
        query=payload.query,
        mode=payload.mode,
        latency_ms=float(latency_ms),
        results=[
            ResultItem(
                rank=int(h.rank),
                doc_id=str(h.doc_id),
                score=float(h.score),
                title=h.title,
                text=h.text,
            )
            for h in hits
        ],
    )
