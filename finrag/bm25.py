"""A small, dependency-free BM25 (Okapi) index.

Written by hand on purpose: it is ~60 lines, easy to explain in an interview, and
fast enough for a few hundred filings. Swap for `bm25s` if you need more speed.
"""
from __future__ import annotations

import math
import pickle
import re
from collections import Counter
from pathlib import Path

import numpy as np

TOKEN_RE = re.compile(r"[a-z0-9]+(?:[.,][0-9]+)*")
STOPWORDS = frozenset(
    "a an and are as at be by for from has have in is it its of on or that the to was were "
    "what which with".split()
)


def tokenize(text: str) -> list[str]:
    # Keeps numbers like 1,234.5 and 3.2 intact; finance questions hinge on them.
    return [t for t in TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]


class BM25:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.doc_tfs: list[Counter] = []
        self.doc_lens = np.zeros(0)
        self.idf: dict[str, float] = {}
        self.postings: dict[str, list[int]] = {}

    def fit(self, texts: list[str]) -> "BM25":
        self.doc_tfs = [Counter(tokenize(t)) for t in texts]
        self.doc_lens = np.array([sum(tf.values()) for tf in self.doc_tfs], dtype=float)
        self.avgdl = float(self.doc_lens.mean()) if len(texts) else 0.0
        df: Counter = Counter()
        self.postings = {}
        for i, tf in enumerate(self.doc_tfs):
            for term in tf:
                df[term] += 1
                self.postings.setdefault(term, []).append(i)
        n = len(texts)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        return self

    def scores(self, query: str, candidates: list[int] | None = None) -> np.ndarray:
        out = np.zeros(len(self.doc_tfs))
        allowed = None if candidates is None else set(candidates)
        for term in set(tokenize(query)):
            idf = self.idf.get(term)
            if idf is None:
                continue
            for i in self.postings[term]:
                if allowed is not None and i not in allowed:
                    continue
                tf = self.doc_tfs[i][term]
                norm = self.k1 * (1 - self.b + self.b * self.doc_lens[i] / self.avgdl)
                out[i] += idf * tf * (self.k1 + 1) / (tf + norm)
        return out

    def search(self, query: str, k: int, candidates: list[int] | None = None) -> list[tuple[int, float]]:
        s = self.scores(query, candidates)
        if candidates is not None:
            mask = np.full(len(s), -np.inf)
            mask[candidates] = s[candidates]
            s = mask
        top = np.argsort(-s)[:k]
        return [(int(i), float(s[i])) for i in top if s[i] > 0]

    def save(self, path: str | Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self.__dict__, f)

    @classmethod
    def load(cls, path: str | Path) -> "BM25":
        obj = cls()
        with open(path, "rb") as f:
            obj.__dict__.update(pickle.load(f))
        return obj
