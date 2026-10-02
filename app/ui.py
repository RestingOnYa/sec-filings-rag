"""Streamlit demo that talks to the FastAPI service.

    uvicorn app.api:app &        # start the API first
    streamlit run app/ui.py
"""
import os

import requests
import streamlit as st

API = os.environ.get("FINRAG_API", "http://localhost:8000")

st.set_page_config(page_title="FinRAG", layout="wide")
st.title("FinRAG: ask questions about SEC filings")
st.caption("Fully local: Qwen3 via Ollama, BGE-M3 embeddings, hybrid BM25 + dense retrieval, cross-encoder reranking.")

examples = [
    "What is the FY2018 capital expenditure amount (in USD millions) for 3M?",
    "Does Adobe have an improving operating margin profile as of FY2022?",
    "What drove the change in Amazon's operating income in FY2017?",
]
question = st.text_input("Question", value=examples[0])
cols = st.columns(len(examples))
for col, ex in zip(cols, examples):
    if col.button(ex[:48] + "...", use_container_width=True):
        question = ex

if st.button("Ask", type="primary") and question:
    with st.spinner("Retrieving and generating..."):
        r = requests.post(f"{API}/ask", json={"question": question}, timeout=300)
        r.raise_for_status()
        data = r.json()
    st.markdown(f"### Answer\n{data['answer']}")
    st.caption(f"{data['latency_s']:.1f}s")
    st.markdown("### Sources")
    for s in data["sources"]:
        pages = ", ".join(map(str, s["pages"]))
        with st.expander(f"[{s['n']}] {s['doc_name']} - p. {pages}  (score {s['score']:.3f})"):
            st.text(s["text"])
