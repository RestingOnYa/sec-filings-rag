"""Retrieval metrics against FinanceBench's gold evidence pages. No LLM needed."""
from __future__ import annotations

import statistics


def first_relevant_rank(retrieved: list[tuple[str, list[int]]], gold_doc: str, gold_pages: set[int]) -> int | None:
    """1-based rank of the first retrieved chunk from the gold filing that covers a gold page."""
    for rank, (doc, pages) in enumerate(retrieved, start=1):
        if doc == gold_doc and gold_pages.intersection(pages):
            return rank
    return None


def recall_at_k(rank: int | None, k: int) -> float:
    return 1.0 if rank is not None and rank <= k else 0.0


def reciprocal_rank(rank: int | None) -> float:
    return 0.0 if rank is None else 1.0 / rank


def doc_hit(retrieved: list[tuple[str, list[int]]], gold_doc: str) -> float:
    """Did we at least find the right filing? Separates 'wrong 10-K' from 'wrong page' misses."""
    return 1.0 if any(doc == gold_doc for doc, _ in retrieved) else 0.0


def percentile(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    if len(values) == 1:
        return values[0]
    return statistics.quantiles(values, n=100, method="inclusive")[int(p) - 1]
