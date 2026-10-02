"""LLM-as-judge for answer correctness.

Validate it before trusting it: hand-label ~50 answers, then run
`finrag judge-agreement` to report how often the judge agrees with you.
"""
from __future__ import annotations

import json

LABELS = ("correct", "incorrect", "refused")

JUDGE_SYSTEM = """You grade answers to financial questions against a gold answer.
Labels:
- "correct": the answer states the same key fact or number as the gold answer. Allow rounding
  (e.g. $1.58B vs $1,577M) and extra correct detail. Units and sign must match.
- "incorrect": the answer gives a different number or fact, or contradicts the gold answer.
- "refused": the answer says it cannot find or cannot answer.
Respond with JSON only: {"label": "...", "reason": "<one short sentence>"}"""

JUDGE_USER = """Question: {question}
Gold answer: {gold}
Model answer: {answer}"""

REFUSAL_MARKERS = ("cannot find", "can't find", "not contain", "unable to", "no information")


def judge(llm, question: str, gold: str, answer: str) -> dict:
    # Cheap, deterministic shortcut for the canned refusal string.
    if any(m in answer.lower() for m in REFUSAL_MARKERS) and len(answer) < 160:
        return {"label": "refused", "reason": "model declined"}
    raw = llm.chat(JUDGE_SYSTEM, JUDGE_USER.format(question=question, gold=gold, answer=answer), json_mode=True)
    try:
        out = json.loads(raw)
        label = str(out.get("label", "")).lower().strip()
    except json.JSONDecodeError:
        out, label = {}, ""
    if label not in LABELS:
        label = "incorrect"
        out["reason"] = f"unparseable judge output: {raw[:120]}"
    return {"label": label, "reason": out.get("reason", "")}


def agreement(pairs: list[tuple[str, str]]) -> dict:
    """Agreement between (human_label, judge_label) pairs, plus Cohen's kappa."""
    n = len(pairs)
    if n == 0:
        return {"n": 0, "agreement": float("nan"), "kappa": float("nan")}
    po = sum(h == j for h, j in pairs) / n
    pe = 0.0
    for lab in LABELS:
        ph = sum(h == lab for h, _ in pairs) / n
        pj = sum(j == lab for _, j in pairs) / n
        pe += ph * pj
    kappa = (po - pe) / (1 - pe) if pe < 1 else 1.0
    return {"n": n, "agreement": po, "kappa": kappa}
