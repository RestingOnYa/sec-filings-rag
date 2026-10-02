"""Build and load the search indexes: chunk store + dense vectors + BM25.

Layout of an index directory:
    chunks.jsonl     one Chunk per line (row number == chunk index)
    vectors.npy      float32 [n_chunks, dim], L2-normalized
    bm25.pkl         pickled BM25 state
    qdrant/          (only when vector_store: qdrant) local Qdrant collection
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .bm25 import BM25
from .chunk import Chunk, chunk_pages
from .embed import Embedder
from .parse import parse_pdf


class Index:
    def __init__(self, chunks: list[Chunk], vectors: np.ndarray, bm25: BM25, qdrant=None):
        self.chunks = chunks
        self.vectors = vectors
        self.bm25 = bm25
        self.qdrant = qdrant  # optional QdrantClient
        self.by_doc: dict[str, list[int]] = {}
        for i, c in enumerate(chunks):
            self.by_doc.setdefault(c.doc_name, []).append(i)

    # ---- dense search -------------------------------------------------
    def dense_search(self, qvec: np.ndarray, k: int, doc_name: str | None = None) -> list[tuple[int, float]]:
        if self.qdrant is not None:
            return self._qdrant_search(qvec, k, doc_name)
        if doc_name is not None:
            cand = np.array(self.by_doc.get(doc_name, []), dtype=int)
            if cand.size == 0:
                return []
            sims = self.vectors[cand] @ qvec
            order = np.argsort(-sims)[:k]
            return [(int(cand[i]), float(sims[i])) for i in order]
        sims = self.vectors @ qvec
        order = np.argsort(-sims)[:k]
        return [(int(i), float(sims[i])) for i in order]

    def _qdrant_search(self, qvec, k, doc_name):
        from qdrant_client import models

        flt = None
        if doc_name is not None:
            flt = models.Filter(
                must=[models.FieldCondition(key="doc_name", match=models.MatchValue(value=doc_name))]
            )
        hits = self.qdrant.query_points("chunks", query=qvec.tolist(), limit=k, query_filter=flt).points
        return [(int(h.id), float(h.score)) for h in hits]

    # ---- persistence --------------------------------------------------
    def save(self, out_dir: str | Path) -> None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "chunks.jsonl", "w") as f:
            for c in self.chunks:
                f.write(json.dumps(asdict(c)) + "\n")
        np.save(out / "vectors.npy", self.vectors)
        self.bm25.save(out / "bm25.pkl")

    @classmethod
    def load(cls, index_dir: str | Path, use_qdrant: bool = False) -> "Index":
        d = Path(index_dir)
        with open(d / "chunks.jsonl") as f:
            chunks = [Chunk(**json.loads(line)) for line in f]
        vectors = np.load(d / "vectors.npy")
        bm25 = BM25.load(d / "bm25.pkl")
        qdrant = None
        if use_qdrant:
            from qdrant_client import QdrantClient

            qdrant = QdrantClient(path=str(d / "qdrant"))
        return cls(chunks, vectors, bm25, qdrant)


def build_index(
    pdf_paths: list[Path],
    embedder: Embedder,
    out_dir: str | Path,
    parse_backend: str = "pymupdf",
    chunk_strategy: str = "fixed",
    size_tokens: int = 512,
    overlap_tokens: int = 64,
    use_qdrant: bool = False,
) -> Index:
    chunks: list[Chunk] = []
    for path in tqdm(pdf_paths, desc="parse+chunk"):
        pages = parse_pdf(path, parse_backend)
        chunks.extend(chunk_pages(pages, chunk_strategy, size_tokens, overlap_tokens))

    texts = [_index_text(c) for c in chunks]
    vectors = embedder.encode(texts)
    bm25 = BM25().fit(texts)
    index = Index(chunks, vectors, bm25)
    index.save(out_dir)

    if use_qdrant:
        index.qdrant = _write_qdrant(Path(out_dir) / "qdrant", chunks, vectors)
    return index


def _index_text(c: Chunk) -> str:
    # Prefixing the filing name helps both BM25 and dense retrieval match
    # "Apple's 2022 10-K" style questions.
    prefix = c.doc_name.replace("_", " ")
    if c.section:
        prefix += f" | {c.section}"
    return f"{prefix}\n{c.text}"


def _write_qdrant(path: Path, chunks: list[Chunk], vectors: np.ndarray):
    from qdrant_client import QdrantClient, models

    client = QdrantClient(path=str(path))
    if client.collection_exists("chunks"):
        client.delete_collection("chunks")
    client.create_collection(
        "chunks", vectors_config=models.VectorParams(size=vectors.shape[1], distance=models.Distance.COSINE)
    )
    client.upload_points(
        "chunks",
        points=[
            models.PointStruct(id=i, vector=v.tolist(), payload={"doc_name": c.doc_name, "pages": c.pages})
            for i, (c, v) in enumerate(zip(chunks, vectors))
        ],
        batch_size=256,
    )
    client.create_payload_index("chunks", "doc_name", models.PayloadSchemaType.KEYWORD)
    return client
