"""Run one config over a question set and log per-question rows + a summary row.

Outputs:
    results/<config name>/<timestamp>.jsonl   one row per question (answer, hits, label, timings)
    results/summary.csv                        one row per run - this becomes your README table
"""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path

from tqdm import tqdm

from ..data import Question
from ..pipeline import RAGPipeline
from .judge import judge
from .metrics import doc_hit, first_relevant_rank, percentile, recall_at_k, reciprocal_rank

SUMMARY_FIELDS = [
    "timestamp", "config", "questions", "n", "doc_hit", "recall@1", "recall@5", "mrr",
    "accuracy", "refusal_rate", "p50_s", "p95_s",
]


def evaluate(
    pipeline: RAGPipeline,
    questions: list[Question],
    config_name: str,
    judge_llm=None,
    out_dir: str | Path = "results",
    questions_label: str = "financebench",
) -> dict:
    out_dir = Path(out_dir)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    run_file = out_dir / config_name / f"{stamp}.jsonl"
    run_file.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for q in tqdm(questions, desc=config_name):
        res = pipeline.run(q.question, q.doc_name)
        retrieved = [(h.chunk.doc_name, h.chunk.pages) for h in res.hits]
        rank = first_relevant_rank(retrieved, q.doc_name, set(q.evidence_pages))
        row = {
            "id": q.id,
            "question": q.question,
            "gold": q.answer,
            "answer": res.answer,
            "doc_name": q.doc_name,
            "gold_pages": q.evidence_pages,
            "retrieved": [{"doc": d, "pages": p} for d, p in retrieved],
            "rank": rank,
            "doc_hit": doc_hit(retrieved, q.doc_name),
            "timings": res.timings,
            "failure_tag": "",  # fill in by hand: retrieval | table | arithmetic | generation
        }
        rows.append(row)

    # Second pass: judge after all generation is done, so the generator and the
    # (bigger) judge model are never loaded into VRAM at the same time.
    if judge_llm is not None and pipeline.generator is not None:
        for row in tqdm(rows, desc="judge"):
            row.update(judge(judge_llm, row["question"], row["gold"], row["answer"]))

    with open(run_file, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    summary = summarize(rows, config_name, questions_label, stamp)
    _append_summary(out_dir / "summary.csv", summary)
    return summary


def summarize(rows: list[dict], config_name: str, questions_label: str, stamp: str = "") -> dict:
    n = len(rows)
    ranks = [r["rank"] for r in rows]
    latencies = [r["timings"]["total_s"] for r in rows]
    labels = [r.get("label") for r in rows if r.get("label")]

    def mean(xs):
        xs = list(xs)
        return round(sum(xs) / len(xs), 4) if xs else ""

    return {
        "timestamp": stamp,
        "config": config_name,
        "questions": questions_label,
        "n": n,
        "doc_hit": mean(r["doc_hit"] for r in rows),
        "recall@1": mean(recall_at_k(k, 1) for k in ranks),
        "recall@5": mean(recall_at_k(k, 5) for k in ranks),
        "mrr": mean(reciprocal_rank(k) for k in ranks),
        "accuracy": mean(lab == "correct" for lab in labels),
        "refusal_rate": mean(lab == "refused" for lab in labels),
        "p50_s": round(percentile(latencies, 50), 3) if latencies else "",
        "p95_s": round(percentile(latencies, 95), 3) if latencies else "",
    }


def _append_summary(path: Path, summary: dict) -> None:
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        if new:
            w.writeheader()
        w.writerow(summary)
