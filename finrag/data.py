"""Download and load the FinanceBench open-source sample.

Source: https://github.com/patronus-ai/financebench (CC BY-NC 4.0)
150 questions, each with a gold answer, the filing it comes from, and evidence pages.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import requests
from tqdm import tqdm

REPO_RAW = "https://raw.githubusercontent.com/patronus-ai/financebench/main"
QUESTIONS_URL = f"{REPO_RAW}/data/financebench_open_source.jsonl"
PDF_URL = REPO_RAW + "/pdfs/{doc_name}.pdf"


@dataclass
class Question:
    id: str
    question: str
    answer: str
    doc_name: str
    company: str = ""
    question_type: str = ""
    evidence_pages: list[int] = field(default_factory=list)  # 0-indexed page numbers
    evidence_texts: list[str] = field(default_factory=list)


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        tmp = dest.with_suffix(dest.suffix + ".part")
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 16):
                f.write(chunk)
        tmp.rename(dest)


def download_financebench(data_dir: str | Path) -> None:
    """Fetch the question file and every PDF it references (skips files already present)."""
    data_dir = Path(data_dir)
    qfile = data_dir / "financebench_open_source.jsonl"
    _download(QUESTIONS_URL, qfile)
    docs = sorted({q.doc_name for q in load_questions(data_dir)})
    for name in tqdm(docs, desc="PDFs"):
        _download(PDF_URL.format(doc_name=name), data_dir / "pdfs" / f"{name}.pdf")


def parse_row(row: dict) -> Question:
    evidence = row.get("evidence") or []
    pages, texts = [], []
    for ev in evidence:
        if ev.get("evidence_page_num") is not None:
            pages.append(int(ev["evidence_page_num"]))
        if ev.get("evidence_text"):
            texts.append(ev["evidence_text"])
    return Question(
        id=row["financebench_id"],
        question=row["question"],
        answer=str(row["answer"]),
        doc_name=row["doc_name"],
        company=row.get("company", ""),
        question_type=row.get("question_type", ""),
        evidence_pages=pages,
        evidence_texts=texts,
    )


def load_questions(data_dir: str | Path, path: str | Path | None = None) -> list[Question]:
    """Load FinanceBench (default) or any JSONL in the same schema, e.g. your dev set."""
    path = Path(path) if path else Path(data_dir) / "financebench_open_source.jsonl"
    with open(path) as f:
        return [parse_row(json.loads(line)) for line in f if line.strip()]
