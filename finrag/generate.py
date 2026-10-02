"""Answer generation with a local model served by Ollama (https://ollama.com)."""
from __future__ import annotations

import re

import requests

from .retrieve import Hit

SYSTEM_PROMPT = """You are a financial analyst answering questions about SEC filings.
Use ONLY the numbered context passages. Cite the passages you used like [1] or [2][3].
If a calculation is needed, show the inputs and the arithmetic in one line.
If the context does not contain the answer, reply exactly: "I cannot find this in the provided filings."
Be concise: one to three sentences."""

THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def format_context(hits: list[Hit], max_chars: int = 12000) -> str:
    blocks, used = [], 0
    for n, h in enumerate(hits, start=1):
        pages = ", ".join(str(p + 1) for p in h.chunk.pages)  # show 1-indexed pages to humans
        header = f"[{n}] {h.chunk.doc_name} (pages {pages})"
        if h.chunk.section:
            header += f" - {h.chunk.section}"
        block = f"{header}\n{h.chunk.text}"
        if used + len(block) > max_chars:
            break
        blocks.append(block)
        used += len(block)
    return "\n\n".join(blocks)


class OllamaClient:
    def __init__(self, url: str = "http://localhost:11434", model: str = "qwen3:8b",
                 temperature: float = 0.0, timeout: int = 300, think: bool | None = False):
        self.url = url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.timeout = timeout
        self.think = think

    def chat(self, system: str, user: str, json_mode: bool = False) -> str:
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "stream": False,
            "options": {"temperature": self.temperature},
        }
        if self.think is not None:
            body["think"] = self.think  # turn off Qwen3's reasoning trace for speed
        if json_mode:
            body["format"] = "json"
        r = requests.post(f"{self.url}/api/chat", json=body, timeout=self.timeout)
        r.raise_for_status()
        return THINK_RE.sub("", r.json()["message"]["content"]).strip()


class Generator:
    def __init__(self, llm, max_context_chars: int = 12000):
        self.llm = llm
        self.max_context_chars = max_context_chars

    def answer(self, question: str, hits: list[Hit]) -> str:
        if not hits:
            context = "(no passages retrieved)"
        else:
            context = format_context(hits, self.max_context_chars)
        user = f"Context:\n{context}\n\nQuestion: {question}"
        return self.llm.chat(SYSTEM_PROMPT, user)
