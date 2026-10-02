"""Embedding models. Anything with `.encode(list[str]) -> np.ndarray` (L2-normalized) works."""
from __future__ import annotations

from typing import Protocol

import numpy as np


class Embedder(Protocol):
    def encode(self, texts: list[str], is_query: bool = False) -> np.ndarray: ...


class SentenceTransformerEmbedder:
    def __init__(self, model: str = "BAAI/bge-m3", batch_size: int = 16, device: str | None = None):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model, device=device)
        self.batch_size = batch_size

    def encode(self, texts: list[str], is_query: bool = False) -> np.ndarray:
        # BGE-M3 needs no query instruction; add one here if you swap in a model that does.
        vecs = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 64,
            convert_to_numpy=True,
        )
        return vecs.astype(np.float32)


class HashEmbedder:
    """Deterministic bag-of-words hashing embedder. For tests and offline smoke runs only."""

    def __init__(self, dim: int = 256):
        self.dim = dim

    def encode(self, texts: list[str], is_query: bool = False) -> np.ndarray:
        import zlib

        from .bm25 import tokenize

        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for tok in tokenize(t):
                out[i, zlib.crc32(tok.encode()) % self.dim] += 1.0  # stable across processes
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.maximum(norms, 1e-9)


def make_embedder(cfg) -> Embedder:
    if cfg.model == "hash":
        return HashEmbedder()
    return SentenceTransformerEmbedder(cfg.model, cfg.get("batch_size", 16), cfg.get("device"))
