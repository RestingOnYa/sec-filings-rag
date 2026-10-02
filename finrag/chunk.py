"""Split pages into chunks that remember which page(s) they came from.

Token counts are approximated by whitespace words (~0.75 words per token is close
enough for sizing chunks); swap in a real tokenizer if you want exact budgets.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .parse import Page

WORDS_PER_TOKEN = 0.75

# 10-K section headers, e.g. "Item 7. Management's Discussion and Analysis"
ITEM_RE = re.compile(r"^\s*item\s+(\d{1,2}[a-c]?)[.:\s]", re.IGNORECASE | re.MULTILINE)


@dataclass
class Chunk:
    id: str
    doc_name: str
    text: str
    pages: list[int]
    section: str = ""
    meta: dict = field(default_factory=dict)


def _words_with_pages(pages: list[Page]) -> list[tuple[str, int]]:
    out = []
    for p in pages:
        out.extend((w, p.page) for w in p.text.split())
    return out


def chunk_fixed(pages: list[Page], size_tokens: int = 512, overlap_tokens: int = 64) -> list[Chunk]:
    """Sliding window over the whole document; chunks may span page breaks."""
    if overlap_tokens >= size_tokens:
        raise ValueError("overlap must be smaller than chunk size")
    size = max(1, int(size_tokens * WORDS_PER_TOKEN))
    step = max(1, size - int(overlap_tokens * WORDS_PER_TOKEN))
    words = _words_with_pages(pages)
    if not words:
        return []
    doc = pages[0].doc_name
    chunks = []
    for i, start in enumerate(range(0, len(words), step)):
        window = words[start : start + size]
        chunks.append(
            Chunk(
                id=f"{doc}::{i}",
                doc_name=doc,
                text=" ".join(w for w, _ in window),
                pages=sorted({p for _, p in window}),
            )
        )
        if start + size >= len(words):
            break
    return chunks


def chunk_by_section(pages: list[Page], size_tokens: int = 512, overlap_tokens: int = 64) -> list[Chunk]:
    """Split on 10-K 'Item N.' headers first, then window within each section.

    Keeps a chunk from straddling e.g. Risk Factors and MD&A, and tags each chunk
    with its section so the generator (and you) can see where it came from.
    """
    sections: list[tuple[str, list[Page]]] = []
    current, buf = "front", []
    for p in pages:
        m = ITEM_RE.search(p.text)
        if m and buf:
            sections.append((current, buf))
            buf = []
        if m:
            current = f"Item {m.group(1).upper()}"
        buf.append(p)
    if buf:
        sections.append((current, buf))

    out: list[Chunk] = []
    for s_idx, (name, sec_pages) in enumerate(sections):
        for c in chunk_fixed(sec_pages, size_tokens, overlap_tokens):
            c.id = f"{c.doc_name}::s{s_idx}::{c.id.rsplit('::', 1)[1]}"
            c.section = name
            out.append(c)
    return out


def chunk_pages(pages: list[Page], strategy: str, size_tokens: int, overlap_tokens: int) -> list[Chunk]:
    if strategy == "fixed":
        return chunk_fixed(pages, size_tokens, overlap_tokens)
    if strategy == "section":
        return chunk_by_section(pages, size_tokens, overlap_tokens)
    raise ValueError(f"unknown chunk strategy: {strategy}")
