# CLAUDE.md

FinRAG: fully local RAG over SEC 10-K filings, evaluated on the FinanceBench open sample
(150 questions over 84 filings). Target hardware is a single RTX 3070 (8 GB VRAM). No API calls:
Qwen3 8B generates and Qwen3 14B judges (both via Ollama), BGE-M3 embeds, bge-reranker-v2-m3 reranks.

The point of the project is the ablation table in README.md: each row is one YAML config, and each
eval run appends a row to `results/summary.csv`.

## Environment (Windows)

- Use the project venv: `.venv\Scripts\python` / `.venv\Scripts\finrag`. The global Python has a
  CPU-only torch, so running from it silently embeds on CPU.
- The venv has CUDA PyTorch from `https://download.pytorch.org/whl/cu128`. If you recreate it,
  install torch from that index **before** `pip install -e ".[app,dev]"`, or pip pulls the CPU build.
- Ollama must be running (`ollama list` starts it). Required models: `qwen3:8b`, `qwen3:14b`.

## Commands

```bash
.venv/Scripts/python -m pytest -q                          # unit + e2e smoke tests, no models needed
.venv/Scripts/ruff check .

.venv/Scripts/finrag download                              # questions + 84 PDFs -> data/
.venv/Scripts/finrag check-pages                           # evidence page offset; expect 0
.venv/Scripts/finrag ingest                                # -> data/index/baseline (config's index_dir)
.venv/Scripts/finrag eval --limit 30                       # seeded 30-question smoke subset
.venv/Scripts/finrag eval --retrieval-only                 # recall/MRR only, no LLM
.venv/Scripts/finrag --config configs/hybrid_rerank.yaml eval
.venv/Scripts/finrag ask "What is the FY2018 capital expenditure for 3M?"
```

`--config` is a top-level flag and goes **before** the subcommand. Default is `configs/baseline.yaml`.

Rough timings on the RTX 3070: `download` ~2 min (160 MB), `ingest` ~18 min (18,340 chunks,
GPU-bound on BGE-M3), `eval --limit 30` ~9 min (generation + 14B judge pass). Run long commands in
the background.

## Hardware gotchas (8 GB VRAM)

- Ollama picks `num_ctx=4096` on 8 GB cards and **silently truncates** longer prompts. With
  `max_context_chars: 12000`, ~7% of prompts measured over 4096 tokens (max ~4.5k), so
  `generate.num_ctx: 8192` is set in `baseline.yaml`. If you raise `max_context_chars` or `final_k`,
  re-check prompt sizes (`prompt_eval_count` from `/api/chat`) and raise `num_ctx` to match.
- With `num_ctx: 8192`, qwen3:8b (6.2 GB) plus BGE-M3 in the eval process uses ~7.9 of 8 GB and
  still runs 100% on GPU (check with `ollama ps`). Anything that adds VRAM (the reranker, a larger
  context) may push Ollama into partial CPU offload and inflate latency numbers.

## Architecture

`finrag/pipeline.py` (`RAGPipeline.from_config`) is the one object the CLI, eval harness and app use.

- `config.py`: YAML with single-parent `extends:` deep-merge; `Config` is a dict with attribute access.
- `data.py`: downloads FinanceBench from GitHub raw; `Question.evidence_pages` are 0-indexed.
- `parse.py`: PDF -> `Page`s (PyMuPDF default, Docling optional). Pages are **0-indexed everywhere**;
  only display code (`generate.format_context`, CLI `ask`, API) converts to 1-indexed.
- `chunk.py`: `fixed` sliding window or `section` (splits on 10-K "Item N." headers). Chunks keep
  their page list, which is what retrieval metrics score against.
- `index.py`: index dir = `chunks.jsonl` + `vectors.npy` + `bm25.pkl` (+ optional `qdrant/`).
  Row number in `chunks.jsonl` is the chunk id used everywhere. Indexed text is prefixed with the
  doc name (and section) — see `_index_text`.
- `bm25.py`: hand-written BM25; tokenizer keeps numbers like `1,234.5` intact.
- `retrieve.py`: dense top-k + BM25 top-k -> RRF -> optional cross-encoder rerank -> `final_k`.
- `generate.py`: Ollama `/api/chat` client (`think: false` by default for Qwen3) + cited-answer prompt.
- `eval/`: `metrics.py` (recall@k / MRR / doc_hit vs gold evidence pages), `judge.py` (LLM judge,
  labels correct/incorrect/refused, plus Cohen's kappa for judge-vs-human agreement),
  `run_eval.py` (generates all answers first, then judges in a second pass so the 8B and 14B are
  never in VRAM together).
- `app/`: FastAPI (`FINRAG_CONFIG` env var picks config) + Streamlit UI that calls it.

## Conventions

- New experiment = new YAML in `configs/` that `extends: baseline.yaml`, not a code change.
  Configs that share chunking/embedding should reuse the same `index_dir`; anything that changes
  parse/chunk/embed settings needs its own `index_dir` and a fresh `finrag ingest`.
- Default setting searches all filings at once ("shared store"); `retrieve.filter_by_doc: true`
  restricts to the question's filing. Say which setting a number came from.
- `eval --limit N` shuffles with a fixed seed, so smoke subsets are comparable across configs.
- Outputs: `results/<config name>/<timestamp>.jsonl` (gitignored) and `results/summary.csv` (committed).
  `failure_tag` in run rows is filled in by hand: retrieval | table | arithmetic | generation.
- `data/` is gitignored (PDFs, indexes).
- Tests must not need models or network: use `HashEmbedder` and a fake LLM (see
  `tests/test_pipeline_smoke.py`).
- Ruff line length is 100. Ruff 0.16's defaults flag 7 auto-fixable style nits (UP037, SIM905,
  FURB122) in the original scaffold; they are style only, not bugs.
