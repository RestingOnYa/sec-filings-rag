"""Command-line entry point: `finrag <command> --config configs/baseline.yaml`."""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from .config import load_config
from .data import download_financebench, load_questions


def cmd_download(cfg, args):
    download_financebench(cfg.data_dir)
    print(f"Downloaded FinanceBench to {cfg.data_dir}/")


def cmd_check_pages(cfg, args):
    """Confirm the evidence page numbers line up with how we parse PDFs (0-indexed).

    Looks for each evidence snippet in the parsed pages and reports the offset between
    where it was found and the labelled page. Expect the most common offset to be 0.
    """
    from .parse import parse_pdf

    questions = load_questions(cfg.data_dir)
    offsets: Counter = Counter()
    cache: dict = {}
    for q in questions[: args.limit]:
        pdf = Path(cfg.data_dir) / "pdfs" / f"{q.doc_name}.pdf"
        if not pdf.exists():
            continue
        if q.doc_name not in cache:
            cache[q.doc_name] = parse_pdf(pdf, cfg.parse.backend)
        pages = cache[q.doc_name]
        for text, labelled in zip(q.evidence_texts, q.evidence_pages):
            probe = " ".join(text.split()[:12]).lower()
            found = [p.page for p in pages if probe and probe in " ".join(p.text.split()).lower()]
            offsets[(found[0] - labelled) if found else "not found"] += 1
    print("offset (found - labelled): count")
    for k, v in offsets.most_common():
        print(f"  {k}: {v}")


def cmd_ingest(cfg, args):
    from .embed import make_embedder
    from .index import build_index

    pdfs = sorted((Path(cfg.data_dir) / "pdfs").glob("*.pdf"))
    if args.limit:
        pdfs = pdfs[: args.limit]
    if not pdfs:
        raise SystemExit("No PDFs found. Run `finrag download` first.")
    index = build_index(
        pdfs,
        make_embedder(cfg.embed),
        cfg.index_dir,
        parse_backend=cfg.parse.backend,
        chunk_strategy=cfg.chunk.strategy,
        size_tokens=cfg.chunk.size_tokens,
        overlap_tokens=cfg.chunk.overlap_tokens,
        use_qdrant=cfg.get("vector_store") == "qdrant",
    )
    print(f"Indexed {len(index.chunks)} chunks from {len(pdfs)} filings -> {cfg.index_dir}")


def cmd_ask(cfg, args):
    from .pipeline import RAGPipeline

    pipe = RAGPipeline.from_config(cfg)
    res = pipe.run(args.question, args.doc)
    print(res.answer, "\n")
    for n, h in enumerate(res.hits, start=1):
        pages = ", ".join(str(p + 1) for p in h.chunk.pages)
        print(f"[{n}] {h.chunk.doc_name} p.{pages}  score={h.score:.3f}")
    print(f"\n{res.timings['total_s']:.2f}s")


def cmd_eval(cfg, args):
    from .eval.run_eval import evaluate
    from .generate import OllamaClient
    from .pipeline import RAGPipeline

    questions = load_questions(cfg.data_dir, args.questions)
    if args.limit:
        random.Random(0).shuffle(questions)  # fixed seed -> same smoke subset every run
        questions = questions[: args.limit]
    pipe = RAGPipeline.from_config(cfg, with_generator=not args.retrieval_only)
    judge_llm = None
    if not args.retrieval_only:
        judge_llm = OllamaClient(cfg.generate.ollama_url, cfg.judge.model, 0.0, think=False)
    label = Path(args.questions).stem if args.questions else "financebench"
    summary = evaluate(pipe, questions, cfg.name, judge_llm, questions_label=label)
    print(json.dumps(summary, indent=2))


def cmd_judge_agreement(cfg, args):
    """Compare judge labels with your hand labels.

    Input: a run .jsonl where you've added a "human_label" field to the rows you checked.
    """
    from .eval.judge import agreement

    pairs = []
    with open(args.run_file) as f:
        for line in f:
            row = json.loads(line)
            if row.get("human_label") and row.get("label"):
                pairs.append((row["human_label"], row["label"]))
    print(json.dumps(agreement(pairs), indent=2))


def main(argv=None):
    p = argparse.ArgumentParser(prog="finrag")
    p.add_argument("--config", default="configs/baseline.yaml")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("download", help="fetch FinanceBench questions + PDFs")

    s = sub.add_parser("check-pages", help="verify evidence page numbering")
    s.add_argument("--limit", type=int, default=40)

    s = sub.add_parser("ingest", help="parse, chunk, embed, index")
    s.add_argument("--limit", type=int, default=0, help="only the first N PDFs (for quick tests)")

    s = sub.add_parser("ask", help="answer one question")
    s.add_argument("question")
    s.add_argument("--doc", default=None, help="restrict to one filing (if filter_by_doc)")

    s = sub.add_parser("eval", help="run the eval harness")
    s.add_argument("--questions", default=None, help="JSONL in FinanceBench schema (default: FinanceBench)")
    s.add_argument("--limit", type=int, default=0, help="random N-question subset (seeded)")
    s.add_argument("--retrieval-only", action="store_true", help="skip generation + judge")

    s = sub.add_parser("judge-agreement", help="judge vs. human labels")
    s.add_argument("run_file")

    args = p.parse_args(argv)
    cfg = load_config(args.config)
    {
        "download": cmd_download,
        "check-pages": cmd_check_pages,
        "ingest": cmd_ingest,
        "ask": cmd_ask,
        "eval": cmd_eval,
        "judge-agreement": cmd_judge_agreement,
    }[args.cmd](cfg, args)


if __name__ == "__main__":
    main()
