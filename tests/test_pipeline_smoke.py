"""End-to-end smoke test with a hash embedder and a fake LLM: exercises index -> retrieve
-> generate -> eval without downloading any models."""
import json

from finrag.bm25 import BM25
from finrag.chunk import chunk_fixed
from finrag.config import Config
from finrag.data import Question
from finrag.embed import HashEmbedder
from finrag.eval.run_eval import evaluate
from finrag.generate import Generator
from finrag.index import Index
from finrag.parse import Page
from finrag.pipeline import RAGPipeline
from finrag.retrieve import Retriever


class FakeLLM:
    def chat(self, system, user, json_mode=False):
        if json_mode:  # judge
            return json.dumps({"label": "correct" if "1,577" in user.split("Model answer:")[1] else "incorrect",
                               "reason": "fake"})
        return "Capital expenditure was $1,577 million [1]." if "1,577" in user else "I cannot find this."


def _build(tmp_path):
    pages = [
        Page("3M_2018_10K", 0, "Cover page. 3M Company annual report 2018."),
        Page("3M_2018_10K", 1, "Purchases of property, plant and equipment (capital expenditure) were 1,577 million."),
        Page("3M_2018_10K", 2, "Dividends paid to shareholders totaled 3,193 million."),
        Page("ADBE_2022_10K", 0, "Adobe operating income grew; operating margin 34.6%."),
    ]
    chunks = chunk_fixed(pages[:3], 30, 5) + chunk_fixed(pages[3:], 30, 5)
    emb = HashEmbedder()
    texts = [c.text for c in chunks]
    index = Index(chunks, emb.encode(texts), BM25().fit(texts))
    index.save(tmp_path / "idx")
    index = Index.load(tmp_path / "idx")
    cfg = Config({"dense_k": 5, "bm25_k": 5, "fusion": "rrf", "rrf_k": 60, "final_k": 2, "filter_by_doc": False})
    return RAGPipeline(Retriever(index, emb, cfg), Generator(FakeLLM()))


def test_end_to_end(tmp_path):
    pipe = _build(tmp_path)
    res = pipe.run("What was 3M capital expenditure in 2018?")
    assert res.hits and res.hits[0].chunk.doc_name == "3M_2018_10K"
    assert "1,577" in res.answer

    q = Question(id="q1", question="What was 3M capital expenditure in 2018?", answer="$1,577 million",
                 doc_name="3M_2018_10K", evidence_pages=[1])
    summary = evaluate(pipe, [q], "smoke", judge_llm=FakeLLM(), out_dir=tmp_path / "results")
    assert summary["recall@5"] == 1.0
    assert summary["accuracy"] == 1.0
    assert (tmp_path / "results" / "summary.csv").exists()
