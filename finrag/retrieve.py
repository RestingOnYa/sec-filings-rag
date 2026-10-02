"""Hybrid retrieval: dense + BM25, fused with Reciprocal Rank Fusion, optionally reranked."""
from __future__ import annotations

from dataclasses import dataclass

from .chunk import Chunk
from .embed import Embedder
from .index import Index


@dataclass
class Hit:
    chunk: Chunk
    score: float
    idx: int


def rrf(rankings: list[list[int]], k: int = 60) -> list[tuple[int, float]]:
    """Reciprocal Rank Fusion (Cormack et al., 2009): score(d) = sum 1 / (k + rank_d)."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking, start=1):
            scores[doc] = scores.get(doc, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: -kv[1])


class CrossEncoderReranker:
    def __init__(self, model: str = "BAAI/bge-reranker-v2-m3", device: str | None = None):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model, device=device)

    def rerank(self, query: str, texts: list[str]) -> list[float]:
        return [float(s) for s in self.model.predict([(query, t) for t in texts])]


class Retriever:
    def __init__(self, index: Index, embedder: Embedder, cfg, reranker=None):
        self.index = index
        self.embedder = embedder
        self.cfg = cfg  # the `retrieve` config section
        self.reranker = reranker
        self.rerank_candidates = 50

    def retrieve(self, query: str, doc_name: str | None = None) -> list[Hit]:
        cfg = self.cfg
        doc = doc_name if cfg.get("filter_by_doc") else None
        rankings: list[list[int]] = []
        first_scores: dict[int, float] = {}

        if cfg.dense_k > 0:
            qvec = self.embedder.encode([query], is_query=True)[0]
            dense = self.index.dense_search(qvec, cfg.dense_k, doc)
            rankings.append([i for i, _ in dense])
            first_scores.update(dense)

        if cfg.bm25_k > 0:
            cand = self.index.by_doc.get(doc, []) if doc else None
            sparse = self.index.bm25.search(query, cfg.bm25_k, cand)
            rankings.append([i for i, _ in sparse])
            for i, s in sparse:
                first_scores.setdefault(i, s)

        if cfg.fusion == "rrf" and len(rankings) > 1:
            fused = rrf(rankings, cfg.get("rrf_k", 60))
        else:
            fused = [(i, first_scores[i]) for i in rankings[0]] if rankings else []

        if self.reranker is not None and fused:
            cand_ids = [i for i, _ in fused[: self.rerank_candidates]]
            texts = [self.index.chunks[i].text for i in cand_ids]
            scores = self.reranker.rerank(query, texts)
            fused = sorted(zip(cand_ids, scores), key=lambda kv: -kv[1])

        return [Hit(self.index.chunks[i], s, i) for i, s in fused[: cfg.final_k]]
