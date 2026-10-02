"""FastAPI service.  Run:  uvicorn app.api:app --reload"""
from __future__ import annotations

import os

from fastapi import FastAPI
from pydantic import BaseModel

from finrag.config import load_config
from finrag.pipeline import RAGPipeline

cfg = load_config(os.environ.get("FINRAG_CONFIG", "configs/hybrid_rerank.yaml"))
pipeline = RAGPipeline.from_config(cfg)
app = FastAPI(title="FinRAG", version="0.1.0")


class AskRequest(BaseModel):
    question: str
    doc_name: str | None = None


class Source(BaseModel):
    n: int
    doc_name: str
    pages: list[int]  # 1-indexed for display
    score: float
    text: str


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]
    latency_s: float


@app.get("/health")
def health():
    return {"status": "ok", "config": cfg.name, "chunks": len(pipeline.retriever.index.chunks)}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    res = pipeline.run(req.question, req.doc_name)
    sources = [
        Source(n=i, doc_name=h.chunk.doc_name, pages=[p + 1 for p in h.chunk.pages], score=h.score, text=h.chunk.text)
        for i, h in enumerate(res.hits, start=1)
    ]
    return AskResponse(answer=res.answer, sources=sources, latency_s=res.timings["total_s"])
