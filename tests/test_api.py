"""Offline tests for the FastAPI backend.

A fake engine is injected into ``app.state`` so the real lifespan never loads
artifacts or a model. Endpoints are exercised via the FastAPI TestClient.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.api import app
from semsearch.search import Result


class FakeEngine:
    """Minimal engine exposing the attributes/behavior the API depends on."""

    def __init__(self) -> None:
        self.doc_ids = ["d1", "d2"]

    def search(self, query: str, k: int = 10, mode: str = "dense", alpha: float = 0.5):
        hits = [
            Result(1, "d1", 0.987654, "First Title", "first document text"),
            Result(2, "d2", 0.123456, "Second Title", "second document text"),
        ]
        return hits[:k]


@pytest.fixture
def client() -> TestClient:
    # Inject the fake BEFORE any request so the lifespan skips real loading.
    app.state.engine = FakeEngine()
    app.state.meta = {"model_name": "fake-model", "dataset": "faketest"}
    return TestClient(app)


def test_healthz_shape(client: TestClient) -> None:
    resp = client.get("/healthz")
    assert resp.status_code == 200
    body = resp.json()
    assert body == {
        "status": "ok",
        "n_docs": 2,
        "model": "fake-model",
        "dataset": "faketest",
    }


def test_search_happy_path(client: TestClient) -> None:
    resp = client.post("/search", json={"query": "hello", "k": 2, "mode": "dense"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["query"] == "hello"
    assert body["mode"] == "dense"
    assert isinstance(body["latency_ms"], float)
    assert len(body["results"]) == 2
    first = body["results"][0]
    assert first["rank"] == 1
    assert first["doc_id"] == "d1"
    assert isinstance(first["score"], float)
    assert first["title"] == "First Title"


def test_search_strips_query(client: TestClient) -> None:
    resp = client.post("/search", json={"query": "  hello  "})
    assert resp.status_code == 200
    assert resp.json()["query"] == "hello"


def test_empty_query_rejected(client: TestClient) -> None:
    resp = client.post("/search", json={"query": ""})
    assert resp.status_code == 422


def test_whitespace_query_rejected(client: TestClient) -> None:
    resp = client.post("/search", json={"query": "   "})
    assert resp.status_code == 422


def test_k_zero_rejected(client: TestClient) -> None:
    resp = client.post("/search", json={"query": "hello", "k": 0})
    assert resp.status_code == 422


def test_k_too_large_rejected(client: TestClient) -> None:
    resp = client.post("/search", json={"query": "hello", "k": 101})
    assert resp.status_code == 422


def test_bad_mode_rejected(client: TestClient) -> None:
    resp = client.post("/search", json={"query": "hello", "mode": "sparse"})
    assert resp.status_code == 422


def test_alpha_out_of_range_rejected(client: TestClient) -> None:
    resp = client.post("/search", json={"query": "hello", "alpha": 1.5})
    assert resp.status_code == 422


def test_default_k_and_mode(client: TestClient) -> None:
    resp = client.post("/search", json={"query": "hello"})
    assert resp.status_code == 200
    assert resp.json()["mode"] == "dense"
