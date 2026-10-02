# FinRAG

**Fully local question answering over SEC filings, measured on [FinanceBench](https://github.com/patronus-ai/financebench).**

> Headline result goes here once you have it, e.g.
> *Hybrid retrieval + reranking raised FinanceBench accuracy from X% to Y% on a single RTX 3070, no API calls.*

<!-- Add a demo GIF here: assets/demo.gif -->

## Results

| # | Config | Recall@5 | Accuracy | p95 latency (s) |
|---|--------|---------:|---------:|----------------:|
| 0 | No retrieval (LLM alone) | – | | |
| 1 | Fixed chunks + dense (`baseline.yaml`) | | | |
| 2 | + BM25 hybrid (RRF) | | | |
| 3 | + cross-encoder reranker (`hybrid_rerank.yaml`) | | | |
| 4 | + section-aware chunking | | | |
| 5 | + filter to the question's filing | | | |
| 6 | + table-aware parsing (Docling) | | | |

FinanceBench open sample: 150 questions. Tuning was done on a separate dev set; the test set was scored once with the final config.

## How it works

```
PDFs -> parse (PyMuPDF | Docling) -> chunk (page numbers kept) -> embed (BGE-M3) -> indexes (vectors + BM25)

question -> dense top-k  ─┐
         -> BM25 top-k   ─┴─ RRF fusion -> cross-encoder rerank -> top 5 -> Qwen3 8B (Ollama) -> answer + [n] citations

eval: recall@k on gold evidence pages (no LLM)  +  LLM judge (correct / incorrect / refused) vs gold answer
```

Every stage is a config flag, so each ablation is a new YAML file, not a code change.

## Quickstart

Requirements: Python 3.10+, [Ollama](https://ollama.com), and ideally an NVIDIA GPU with 8 GB+ VRAM.

```bash
python -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\activate
# GPU: install the CUDA build of PyTorch first (get the exact command from pytorch.org)
pip install -e ".[app,dev]"

ollama pull qwen3:8b                   # generator (~5 GB at 4-bit)
ollama pull qwen3:14b                  # judge (partially offloads to CPU on 8 GB cards)

finrag download                        # FinanceBench questions + PDFs -> data/
finrag check-pages                     # sanity-check evidence page numbering (expect offset 0)
finrag ingest                          # build data/index/baseline
finrag eval --limit 30                 # 30-question smoke run
finrag eval --retrieval-only           # retrieval metrics only, fast, no LLM
finrag --config configs/hybrid_rerank.yaml eval

finrag ask "What is the FY2018 capital expenditure amount (in USD millions) for 3M?"
```

Demo app:

```bash
uvicorn app.api:app --reload           # API on :8000  (FINRAG_CONFIG picks the config)
streamlit run app/ui.py                # UI
```

Every eval run writes `results/<config>/<timestamp>.jsonl` (one row per question) and appends one row to `results/summary.csv`.

## Repo layout

```
finrag/
  config.py        YAML configs with `extends:` inheritance
  data.py          FinanceBench download + loading
  parse.py         PDF -> pages (PyMuPDF, or Docling for tables)
  chunk.py         fixed-window and 10-K section-aware chunking
  bm25.py          hand-written BM25
  embed.py         sentence-transformers embedder (+ hash embedder for tests)
  index.py         chunk store, vectors, BM25, optional Qdrant
  retrieve.py      dense + BM25, RRF, cross-encoder rerank
  generate.py      Ollama client + cited-answer prompt
  pipeline.py      wires it together
  eval/            metrics, LLM judge, eval runner
  cli.py           `finrag` command
app/               FastAPI service + Streamlit UI
configs/           one YAML per experiment
tests/             unit + end-to-end smoke tests (no models needed)
```

## Evaluation notes

- **Retrieval** is scored against FinanceBench's gold evidence pages: recall@1, recall@5, MRR, and "doc hit" (found the right filing at all).
- **Answers** are scored by an LLM judge that is larger than the generator, run as a second pass so the two models never share VRAM. Judge agreement with hand labels on 50 answers: **TODO%** (Cohen's κ = TODO). Reproduce: add `"human_label"` to rows in a run file, then `finrag judge-agreement results/<config>/<run>.jsonl`.
- **Setting:** by default every question searches all filings at once (the harder "shared store" setting). `filter_by_doc: true` restricts search to the question's filing.

## Failure analysis

<!-- After each run, tag wrong answers in the run .jsonl (`failure_tag`: retrieval | table | arithmetic | generation), then chart the counts here with two or three worked examples. -->

## Acknowledgements

Questions and filings come from FinanceBench (Islam et al., 2023, [arXiv:2311.11944](https://arxiv.org/abs/2311.11944)), licensed CC BY-NC 4.0.
