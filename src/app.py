from __future__ import annotations

import os

# PyTorch and FAISS both ship libomp.dylib on macOS; without this the second
# one to load aborts the process. Must be set before importing torch/faiss.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sys
from pathlib import Path

import streamlit as st
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.rag import answer  # noqa: E402
from src.retrieve import search  # noqa: E402


st.set_page_config(page_title="Image RAG", layout="wide")
st.title("Image RAG prototype")
st.caption("CLIP + FAISS retrieval over COCO, with optional Claude vision answer.")


@st.cache_resource(show_spinner="Loading index and CLIP model...")
def _warm() -> None:
    # Trigger a dummy query so the FAISS index and CLIP weights are loaded
    # once at startup rather than on the first user query.
    search("warmup", k=1)


_warm()

query = st.text_input(
    "Query",
    value="Give me 5 images with men wearing a hat",
)
col_a, col_b = st.columns([1, 3])
with col_a:
    k = st.slider("k", min_value=1, max_value=20, value=5)
with col_b:
    use_rag = st.toggle(
        "Generate grounded answer with Claude", value=False,
        help="Sends the retrieved images + query to claude-sonnet-4-6.",
    )

if st.button("Search", type="primary") and query.strip():
    with st.spinner("Retrieving..."):
        results = search(query, k=k)

    if not results:
        st.warning("No results.")
    else:
        st.subheader("Top results")
        cols = st.columns(min(len(results), 5))
        for i, (path, score) in enumerate(results):
            with cols[i % len(cols)]:
                try:
                    st.image(Image.open(path), use_container_width=True)
                except Exception as e:
                    st.error(f"Failed to load {path}: {e}")
                st.caption(f"Image {i + 1} · score {score:.3f}")
                st.caption(path.name)

        if use_rag:
            st.subheader("Claude's grounded answer")
            with st.spinner("Asking Claude..."):
                try:
                    text = answer(query, [p for p, _ in results])
                    st.write(text)
                except Exception as e:
                    st.error(f"RAG call failed: {e}")
