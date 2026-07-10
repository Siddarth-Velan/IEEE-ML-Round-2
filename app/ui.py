"""Streamlit frontend for semsearch.

Talks to the FastAPI backend over HTTP only — it never imports ``semsearch``.
Run with::

    streamlit run app/ui.py
"""

from __future__ import annotations

import os

import requests
import streamlit as st

API_URL = os.environ.get("SEMSEARCH_API", "http://127.0.0.1:8000").rstrip("/")
_REQUEST_TIMEOUT = 30
_START_HINT = "API not reachable — start it with: uvicorn app.api:app"


def fetch_health() -> tuple[dict | None, str | None]:
    """Query ``/healthz``.

    Returns
    -------
    tuple[dict | None, str | None]
        ``(payload, None)`` on success, or ``(None, error_message)`` if the
        API is unreachable or returns an error.
    """
    try:
        resp = requests.get(f"{API_URL}/healthz", timeout=_REQUEST_TIMEOUT)
        resp.raise_for_status()
        return resp.json(), None
    except requests.exceptions.RequestException:
        return None, _START_HINT


def run_search(query: str, k: int, mode: str, alpha: float) -> tuple[dict | None, str | None]:
    """POST to ``/search``.

    Returns
    -------
    tuple[dict | None, str | None]
        ``(payload, None)`` on success, or ``(None, error_message)`` otherwise.
    """
    body = {"query": query, "k": k, "mode": mode, "alpha": alpha}
    try:
        resp = requests.post(
            f"{API_URL}/search", json=body, timeout=_REQUEST_TIMEOUT
        )
    except requests.exceptions.RequestException:
        return None, _START_HINT
    if resp.status_code >= 400:
        detail = ""
        try:
            detail = resp.json().get("detail", "")
        except ValueError:
            detail = resp.text
        if isinstance(detail, list):  # pydantic validation error list
            detail = "; ".join(str(d.get("msg", d)) for d in detail)
        return None, f"Search failed ({resp.status_code}): {detail}"
    return resp.json(), None


def render_sidebar() -> dict:
    """Render the sidebar (health panel + controls) and return control values."""
    st.sidebar.title("semsearch")
    health, err = fetch_health()
    if err is not None:
        st.sidebar.error(err)
    else:
        st.sidebar.success("API online")
        st.sidebar.metric("Documents", health.get("n_docs", 0))
        st.sidebar.caption(f"Model: {health.get('model', '—')}")
        st.sidebar.caption(f"Dataset: {health.get('dataset', '—')}")

    st.sidebar.divider()
    mode = st.sidebar.radio("Mode", ["dense", "hybrid", "bm25"], index=0)
    k = st.sidebar.slider("Results (k)", min_value=1, max_value=50, value=10)
    alpha = 0.5
    if mode == "hybrid":
        alpha = st.sidebar.slider(
            "Alpha (bm25 weight)", min_value=0.0, max_value=1.0, value=0.5, step=0.05
        )
    return {"mode": mode, "k": k, "alpha": alpha, "api_online": err is None}


def render_results(payload: dict) -> None:
    """Render a latency badge and the result cards."""
    latency = payload.get("latency_ms", 0.0)
    results = payload.get("results", [])
    st.caption(f"⏱ {latency:.1f} ms · {len(results)} results")
    if not results:
        st.info("No results.")
        return
    for item in results:
        rank = item.get("rank", 0)
        score = item.get("score", 0.0)
        title = item.get("title") or "(untitled)"
        text = item.get("text", "")
        st.markdown(f"**{rank}. {title}**  \nscore: `{score:.3f}` · id: `{item.get('doc_id', '')}`")
        with st.expander("Full text"):
            st.write(text)
        st.divider()


def main() -> None:
    """Streamlit entry point."""
    st.set_page_config(page_title="semsearch", page_icon="🔎", layout="wide")
    controls = render_sidebar()

    st.title("Semantic Search")
    query = st.text_input("Query", placeholder="Search the corpus…")
    go = st.button("Search", type="primary")

    if go:
        if not query.strip():
            st.warning("Enter a query first.")
            return
        if not controls["api_online"]:
            st.error(_START_HINT)
            return
        payload, err = run_search(
            query.strip(), controls["k"], controls["mode"], controls["alpha"]
        )
        if err is not None:
            st.error(err)
            return
        render_results(payload)


if __name__ == "__main__":
    main()
