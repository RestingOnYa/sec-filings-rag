"""Fast unit tests: no models, no network. Run with `pytest -q`."""
from finrag.bm25 import BM25, tokenize
from finrag.chunk import chunk_by_section, chunk_fixed
from finrag.config import _deep_merge
from finrag.eval.judge import agreement
from finrag.eval.metrics import first_relevant_rank, recall_at_k, reciprocal_rank
from finrag.parse import Page
from finrag.retrieve import rrf


def _pages(n_pages=3, words_per_page=100, doc="ACME_2022_10K"):
    return [Page(doc, i, " ".join(f"p{i}w{j}" for j in range(words_per_page))) for i in range(n_pages)]


def test_fixed_chunks_cover_all_words_and_track_pages():
    pages = _pages()
    chunks = chunk_fixed(pages, size_tokens=80, overlap_tokens=16)  # 60 words, step 48
    all_words = {w for c in chunks for w in c.text.split()}
    assert all_words == {w for p in pages for w in p.text.split()}
    assert chunks[0].pages == [0]
    assert any(len(c.pages) == 2 for c in chunks), "some chunk should straddle a page break"
    assert chunks[-1].pages[-1] == 2


def test_section_chunks_split_on_items():
    pages = [
        Page("D", 0, "Cover page text"),
        Page("D", 1, "Item 1A. Risk Factors\nCompetition is intense."),
        Page("D", 2, "More risks here."),
        Page("D", 3, "Item 7. Management's Discussion\nRevenue grew 12%."),
    ]
    chunks = chunk_by_section(pages, size_tokens=512, overlap_tokens=0)
    assert [c.section for c in chunks] == ["front", "Item 1A", "Item 7"]
    assert chunks[1].pages == [1, 2]


def test_tokenize_keeps_numbers():
    assert tokenize("Revenue was $1,234.5 million in FY2022") == ["revenue", "1,234.5", "million", "fy2022"]


def test_bm25_ranks_matching_doc_first():
    docs = ["apple revenue iphone sales", "capital expenditure was 1,577 million", "dividends paid to shareholders"]
    bm = BM25().fit(docs)
    hits = bm.search("capital expenditure", k=3)
    assert hits[0][0] == 1
    assert bm.search("capital expenditure", k=3, candidates=[0, 2]) == []


def test_rrf_rewards_agreement():
    fused = rrf([[1, 2, 3], [2, 1, 4]], k=60)
    top_two = {d for d, _ in fused[:2]}
    assert top_two == {1, 2}
    assert dict(fused)[3] < dict(fused)[1]


def test_retrieval_metrics():
    retrieved = [("X", [4]), ("ACME", [9]), ("ACME", [11, 12])]
    rank = first_relevant_rank(retrieved, "ACME", {12})
    assert rank == 3
    assert recall_at_k(rank, 1) == 0.0 and recall_at_k(rank, 5) == 1.0
    assert abs(reciprocal_rank(rank) - 1 / 3) < 1e-9
    assert first_relevant_rank(retrieved, "ACME", {99}) is None


def test_judge_agreement_kappa():
    perfect = agreement([("correct", "correct"), ("incorrect", "incorrect"), ("refused", "refused")])
    assert perfect["agreement"] == 1.0 and perfect["kappa"] == 1.0
    half = agreement([("correct", "correct"), ("correct", "incorrect")])
    assert half["agreement"] == 0.5


def test_config_merge():
    base = {"retrieve": {"dense_k": 5, "bm25_k": 0}, "name": "a"}
    merged = _deep_merge(base, {"retrieve": {"bm25_k": 50}, "name": "b"})
    assert merged == {"retrieve": {"dense_k": 5, "bm25_k": 50}, "name": "b"}
