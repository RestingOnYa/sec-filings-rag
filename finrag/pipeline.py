"""Wires config -> index, retriever, generator. The one object the CLI, eval and app use."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from .embed import make_embedder
from .generate import Generator, OllamaClient
from .index import Index
from .retrieve import CrossEncoderReranker, Hit, Retriever


@dataclass
class RAGResult:
    question: str
    answer: str
    hits: list[Hit]
    timings: dict = field(default_factory=dict)


class RAGPipeline:
    def __init__(self, retriever: Retriever, generator: Generator | None):
        self.retriever = retriever
        self.generator = generator

    @classmethod
    def from_config(cls, cfg, with_generator: bool = True) -> "RAGPipeline":
        index = Index.load(cfg.index_dir, use_qdrant=cfg.get("vector_store") == "qdrant")
        embedder = make_embedder(cfg.embed)
        reranker = None
        if cfg.rerank.enabled:
            reranker = CrossEncoderReranker(cfg.rerank.model, cfg.embed.get("device"))
        retriever = Retriever(index, embedder, cfg.retrieve, reranker)
        retriever.rerank_candidates = cfg.rerank.get("candidates", 50)
        generator = None
        if with_generator:
            g = cfg.generate
            llm = OllamaClient(g.ollama_url, g.model, g.temperature, think=g.get("think", False))
            generator = Generator(llm, g.max_context_chars)
        return cls(retriever, generator)

    def run(self, question: str, doc_name: str | None = None) -> RAGResult:
        t0 = time.perf_counter()
        hits = self.retriever.retrieve(question, doc_name)
        t1 = time.perf_counter()
        answer = self.generator.answer(question, hits) if self.generator else ""
        t2 = time.perf_counter()
        return RAGResult(question, answer, hits, {"retrieve_s": t1 - t0, "generate_s": t2 - t1, "total_s": t2 - t0})
