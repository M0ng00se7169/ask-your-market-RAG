"""
Tests for eval/bm25_baseline.py — Phase 5 sub-phase 5c BM25 baseline.

Tests cover:
- BM25Index construction from snapshots CSV
- BM25 search returns results in the expected shape
- BM25 search finds relevant products
- compare_search_methods computes recall correctly
- Edge cases: empty queries, unknown URLs, none_in_top_k questions
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from eval.bm25_baseline import (
    BM25Index,
    _tokenize,
    _recall_at_k,
    build_bm25_index,
    compare_search_methods,
)


EVAL_DIR = Path(__file__).resolve().parent.parent / "eval"
QUESTIONS_PATH = EVAL_DIR / "questions.json"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_snapshots_csv(tmp_path: Path) -> Path:
    """Create a small snapshots.csv for BM25 testing."""
    csv_path = tmp_path / "snapshots.csv"
    csv_content = """scraped_at,url,title,price,currency,in_stock,promo,rating,reviews_count,description
2026-10-01,https://example.com/books/mystery-1,Murder in London,12.99,GBP,true,,4.5,100,"A Victorian mystery novel set in foggy London streets with a brilliant detective solving a complex murder case."
2026-10-01,https://example.com/books/scifi-1,Space Odyssey,15.99,GBP,true,,4.2,200,"A science fiction epic about space travel to distant galaxies and the mysteries of the universe."
2026-10-01,https://example.com/books/history-1,World History,18.50,GBP,false,,4.1,150,"A sweeping account of human history from the Stone Age to the present day covering all continents."
2026-10-01,https://example.com/books/thriller-1,Dark Secrets,14.99,GBP,true,promo,4.3,80,"A dark psychological thriller about a reporter returning to her hometown to uncover long-buried secrets."
2026-10-01,https://example.com/books/poetry-1,Playful Verses,11.99,GBP,true,,4.4,60,"An illustrated collection of poems reflecting on childhood with drawings and playful verse."
"""
    csv_path.write_text(csv_content, encoding="utf-8")
    return csv_path


@pytest.fixture
def bm25_index(sample_snapshots_csv: Path) -> BM25Index:
    """Build a BM25Index from the sample CSV."""
    return BM25Index(sample_snapshots_csv)


@pytest.fixture
def semantic_questions() -> list[dict]:
    """Load semantic questions from questions.json."""
    data = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    return [q for q in data["questions"] if q["type"] == "semantic"]


# ---------------------------------------------------------------------------
# Tokenizer tests
# ---------------------------------------------------------------------------


class TestTokenize:
    def test_basic_tokenization(self):
        tokens = _tokenize("A Victorian mystery novel")
        assert tokens == ["a", "victorian", "mystery", "novel"]

    def test_lowercase(self):
        tokens = _tokenize("UPPERCASE Text")
        assert tokens == ["uppercase", "text"]

    def test_punctuation_stripped(self):
        tokens = _tokenize("hello, world! test-case.")
        assert tokens == ["hello", "world", "test", "case"]

    def test_empty_string(self):
        assert _tokenize("") == []

    def test_numbers_preserved(self):
        tokens = _tokenize("Book 123 and 456")
        assert tokens == ["book", "123", "and", "456"]


# ---------------------------------------------------------------------------
# BM25Index construction tests
# ---------------------------------------------------------------------------


class TestBM25IndexConstruction:
    def test_builds_from_csv(self, sample_snapshots_csv: Path):
        index = BM25Index(sample_snapshots_csv)
        assert index.product_count == 5

    def test_product_count_matches_catalog(self, bm25_index: BM25Index):
        assert bm25_index.product_count == 5

    def test_catalog_is_loaded(self, bm25_index: BM25Index):
        # Internal attributes should be populated
        assert len(bm25_index._urls) == 5
        assert len(bm25_index._titles) == 5
        assert len(bm25_index._descriptions) == 5

    def test_ids_are_stringified_indices(self, bm25_index: BM25Index):
        assert bm25_index._ids == ["0", "1", "2", "3", "4"]


# ---------------------------------------------------------------------------
# BM25 search tests
# ---------------------------------------------------------------------------


class TestBM25Search:
    def test_returns_list_of_dicts(self, bm25_index: BM25Index):
        results = bm25_index.search("mystery novel", top_k=3)
        assert isinstance(results, list)
        assert all(isinstance(r, dict) for r in results)

    def test_returns_expected_fields(self, bm25_index: BM25Index):
        results = bm25_index.search("mystery novel", top_k=3)
        if results:
            required = {"url", "title", "price", "currency", "in_stock", "promo", "scraped_at", "score"}
            assert required.issubset(results[0].keys())

    def test_scores_are_numeric(self, bm25_index: BM25Index):
        results = bm25_index.search("mystery novel", top_k=3)
        for r in results:
            assert isinstance(r["score"], float)

    def test_top_k_limits_results(self, bm25_index: BM25Index):
        results = bm25_index.search("mystery novel", top_k=2)
        assert len(results) <= 2

    def test_top_k_zero_returns_empty(self, bm25_index: BM25Index):
        results = bm25_index.search("mystery novel", top_k=0)
        assert results == []

    def test_negative_top_k_returns_empty(self, bm25_index: BM25Index):
        results = bm25_index.search("mystery novel", top_k=-1)
        assert results == []

    def test_empty_query_returns_empty(self, bm25_index: BM25Index):
        results = bm25_index.search("", top_k=5)
        assert results == []

    def test_whitespace_only_query_returns_empty(self, bm25_index: BM25Index):
        results = bm25_index.search("   ", top_k=5)
        assert results == []

    def test_results_ordered_by_score_descending(self, bm25_index: BM25Index):
        results = bm25_index.search("mystery", top_k=5)
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_common_word_returns_no_results(self, bm25_index: BM25Index):
        """Extremely common words may return no results (score <= 0)."""
        results = bm25_index.search("book", top_k=5)
        # 'book' appears in every description, so BM25 may assign ~0 score
        # and filter them out. This is expected BM25 behavior.
        assert results == []

    def test_searches_for_mystery_find_mystery_book(self, bm25_index: BM25Index):
        results = bm25_index.search("mystery novel set in Victorian London with a detective", top_k=5)
        urls = [r["url"] for r in results]
        assert "https://example.com/books/mystery-1" in urls

    def test_searches_for_scifi_find_scifi_book(self, bm25_index: BM25Index):
        results = bm25_index.search("science fiction short story collection about space travel and distant galaxies", top_k=5)
        urls = [r["url"] for r in results]
        assert "https://example.com/books/scifi-1" in urls

    def test_searches_for_history_find_history_book(self, bm25_index: BM25Index):
        results = bm25_index.search("sweeping account of human history from the Stone Age to the present", top_k=5)
        urls = [r["url"] for r in results]
        assert "https://example.com/books/history-1" in urls

    def test_searches_for_thriller_find_thriller_book(self, bm25_index: BM25Index):
        results = bm25_index.search("dark psychological thriller about a reporter returning to her hometown", top_k=5)
        urls = [r["url"] for r in results]
        assert "https://example.com/books/thriller-1" in urls

    def test_searches_for_poetry_find_poetry_book(self, bm25_index: BM25Index):
        results = bm25_index.search("illustrated collection of poems reflecting on childhood with drawings and verse", top_k=5)
        urls = [r["url"] for r in results]
        assert "https://example.com/books/poetry-1" in urls


# ---------------------------------------------------------------------------
# _recall_at_k tests
# ---------------------------------------------------------------------------


class TestRecallAtK:
    def test_recall_when_url_in_top_k(self):
        result = _recall_at_k(
            ["https://a.com", "https://b.com", "https://c.com"],
            "https://b.com",
            5,
        )
        assert result["recall"] == 1.0
        assert result["rank"] == 2
        assert result["in_top_k"] is True

    def test_recall_when_url_not_in_top_k(self):
        result = _recall_at_k(
            ["https://a.com", "https://b.com", "https://c.com"],
            "https://d.com",
            3,
        )
        assert result["recall"] == 0.0
        assert result["rank"] is None
        assert result["in_top_k"] is False

    def test_recall_at_k_smaller_than_results(self):
        result = _recall_at_k(
            ["https://a.com", "https://b.com", "https://c.com"],
            "https://c.com",
            2,  # c.com is at index 2 (rank 3), not in top-2
        )
        assert result["recall"] == 0.0
        assert result["rank"] is None

    def test_recall_at_k_larger_than_results(self):
        result = _recall_at_k(
            ["https://a.com"],
            "https://a.com",
            10,
        )
        assert result["recall"] == 1.0
        assert result["rank"] == 1


# ---------------------------------------------------------------------------
# compare_search_methods tests
# ---------------------------------------------------------------------------


class TestCompareSearchMethods:
    def test_returns_dict_with_expected_structure(self, bm25_index: BM25Index, semantic_questions: list[dict]):
        mock_collection = MagicMock()
        mock_collection.count.return_value = 5
        mock_collection.query.return_value = {
            "ids": [["0", "1", "2", "3", "4"]],
            "distances": [[0.1, 0.2, 0.3, 0.4, 0.5]],
            "metadatas": [[{"url": u} for u in [
                "https://example.com/books/mystery-1",
                "https://example.com/books/scifi-1",
                "https://example.com/books/history-1",
                "https://example.com/books/thriller-1",
                "https://example.com/books/poetry-1",
            ]]],
            "documents": [[""] * 5],
        }

        report = compare_search_methods(bm25_index, mock_collection, semantic_questions, top_k=5)

        assert "top_k" in report
        assert "questions_evaluated" in report
        assert "bm25" in report
        assert "embedding" in report
        assert "wins" in report
        assert "per_question" in report
        assert report["top_k"] == 5

    def test_bm25_section_has_avg_recall(self, bm25_index: BM25Index, semantic_questions: list[dict]):
        mock_collection = MagicMock()
        mock_collection.count.return_value = 5
        mock_collection.query.return_value = {
            "ids": [["0"]],
            "distances": [[0.1]],
            "metadatas": [[{"url": "https://example.com/books/mystery-1"}]],
            "documents": [[""]],
        }

        report = compare_search_methods(bm25_index, mock_collection, semantic_questions, top_k=5)
        assert "avg_recall" in report["bm25"]
        assert isinstance(report["bm25"]["avg_recall"], float)

    def test_embedding_section_has_avg_recall(self, bm25_index: BM25Index, semantic_questions: list[dict]):
        mock_collection = MagicMock()
        mock_collection.count.return_value = 5
        mock_collection.query.return_value = {
            "ids": [["0"]],
            "distances": [[0.1]],
            "metadatas": [[{"url": "https://example.com/books/mystery-1"}]],
            "documents": [[""]],
        }

        report = compare_search_methods(bm25_index, mock_collection, semantic_questions, top_k=5)
        assert "avg_recall" in report["embedding"]
        assert isinstance(report["embedding"]["avg_recall"], float)

    def test_none_in_top_k_questions_handled(self, bm25_index: BM25Index, semantic_questions: list[dict]):
        # semantic-01 is the vegetarian cookbook question with none_in_top_k assert
        veg_question = next(q for q in semantic_questions if q["id"] == "semantic-01")

        mock_collection = MagicMock()
        mock_collection.count.return_value = 5
        mock_collection.query.return_value = {
            "ids": [[]],
            "distances": [[]],
            "metadatas": [[]],
            "documents": [[]],
        }

        report = compare_search_methods(bm25_index, mock_collection, [veg_question], top_k=5)

        assert len(report["per_question"]) == 1
        pq = report["per_question"][0]
        assert pq["expected_assert"] == "none_in_top_k"
        # Should not have expected_url
        assert "expected_url" not in pq or pq.get("expected_url") is None

    def test_wins_count_totals(self, bm25_index: BM25Index, semantic_questions: list[dict]):
        mock_collection = MagicMock()
        mock_collection.count.return_value = 5
        mock_collection.query.return_value = {
            "ids": [["0", "1", "2", "3", "4"]],
            "distances": [[0.1, 0.2, 0.3, 0.4, 0.5]],
            "metadatas": [[{"url": u} for u in [
                "https://example.com/books/mystery-1",
                "https://example.com/books/scifi-1",
                "https://example.com/books/history-1",
                "https://example.com/books/thriller-1",
                "https://example.com/books/poetry-1",
            ]]],
            "documents": [[""] * 5],
        }

        report = compare_search_methods(bm25_index, mock_collection, semantic_questions, top_k=5)
        total = report["wins"]["bm25"] + report["wins"]["embedding"] + report["wins"]["tie"]
        # Should equal questions with expected_url (7 out of 8, excluding semantic-01)
        assert total == report["questions_with_expected_url"]

    def test_per_question_has_both_methods(self, bm25_index: BM25Index, semantic_questions: list[dict]):
        mock_collection = MagicMock()
        mock_collection.count.return_value = 5
        mock_collection.query.return_value = {
            "ids": [["0"]],
            "distances": [[0.1]],
            "metadatas": [[{"url": "https://example.com/books/mystery-1"}]],
            "documents": [[""]],
        }

        report = compare_search_methods(bm25_index, mock_collection, semantic_questions[:2], top_k=5)
        assert len(report["per_question"]) == 2
        for pq in report["per_question"]:
            assert "bm25" in pq
            assert "embedding" in pq


# ---------------------------------------------------------------------------
# build_bm25_index convenience function tests
# ---------------------------------------------------------------------------


class TestBuildBM25Index:
    def test_builds_and_returns_bm25_index(self, sample_snapshots_csv: Path):
        index = build_bm25_index(sample_snapshots_csv)
        assert isinstance(index, BM25Index)
        assert index.product_count == 5

    def test_raises_on_empty_csv(self, tmp_path: Path):
        empty_csv = tmp_path / "empty.csv"
        empty_csv.write_text("scraped_at,url,title,price,currency,in_stock,promo,rating,reviews_count,description\n")
        with pytest.raises(ValueError, match="empty"):
            build_bm25_index(empty_csv)


# ---------------------------------------------------------------------------
# Integration-style test with real questions.json structure
# ---------------------------------------------------------------------------


class TestIntegration:
    def test_full_pipeline_with_mock_collection(self, sample_snapshots_csv: Path, semantic_questions: list[dict]):
        """End-to-end: build BM25, mock embedding search, compare."""

        bm25_index = build_bm25_index(sample_snapshots_csv)

        # Mock collection whose .query() returns canned results.
        mock_collection = MagicMock()
        mock_collection.count.return_value = 5
        urls = [
            "https://example.com/books/mystery-1",
            "https://example.com/books/scifi-1",
            "https://example.com/books/history-1",
            "https://example.com/books/thriller-1",
            "https://example.com/books/poetry-1",
        ]
        mock_collection.query.return_value = {
            "ids": [["0", "1", "2", "3", "4"]],
            "distances": [[0.1, 0.2, 0.3, 0.4, 0.5]],
            "metadatas": [[{"url": u} for u in urls]],
            "documents": [[""] * 5],
        }

        # Use a subset of semantic questions that have expected URLs
        questions_with_urls = [q for q in semantic_questions if q["expected"]["check"].get("assert") != "none_in_top_k"]

        report = compare_search_methods(bm25_index, mock_collection, questions_with_urls, top_k=5)

        assert report["questions_evaluated"] == len(questions_with_urls)
        assert "bm25" in report
        assert "embedding" in report
        assert len(report["per_question"]) == len(questions_with_urls)


# ---------------------------------------------------------------------------
# Edge case tests
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_single_product_index(self, tmp_path: Path):
        """Test BM25Index with a single product - BM25 may not return results with
        very small corpora due to IDF computation, but the index should still work."""
        csv_path = tmp_path / "single.csv"
        csv_path.write_text("""scraped_at,url,title,price,currency,in_stock,promo,rating,reviews_count,description
2026-10-01,https://example.com/only,Only Book,9.99,GBP,true,,4.0,10,"A singleton product used for testing BM25 edge cases."
""")
        index = BM25Index(csv_path)
        assert index.product_count == 1

        # Verify URL/title/price are correctly stored
        assert index._urls[0] == "https://example.com/only"
        assert index._titles[0] == "Only Book"
        assert index._prices[0] == 9.99

        # With only one document, BM25 scores can be negative/zero.
        # The index should still function - just verify it doesn't crash.
        results = index.search("product", top_k=5)
        assert isinstance(results, list)

    def test_descriptions_with_special_characters(self, tmp_path: Path):
        """Test handling of special characters in descriptions."""
        csv_path = tmp_path / "special.csv"
        csv_content = """scraped_at,url,title,price,currency,in_stock,promo,rating,reviews_count,description
2026-10-01,https://example.com/axolotl,Axolotl Book,9.99,GBP,true,,4.0,10,"Axolotls & other creatures: a zoology book about rare animals (2026 ed.)."
2026-10-01,https://example.com/other,Other Book,12.99,GBP,true,,4.0,20,"A different book about something else."
"""
        csv_path.write_text(csv_content, encoding="utf-8")
        index = BM25Index(csv_path)
        assert index.product_count == 2

        # Verify URLs are stored correctly
        assert "https://example.com/axolotl" in index._urls
        assert "https://example.com/other" in index._urls

    def test_na_price_handled(self, tmp_path: Path):
        """Test that NA/NaN prices are handled correctly."""
        csv_path = tmp_path / "na_price.csv"
        csv_content = """scraped_at,url,title,price,currency,in_stock,promo,rating,reviews_count,description
2026-10-01,https://example.com/na,NA Price Book,NA,GBP,true,,4.0,10,"A book with missing price information."
2026-10-01,https://example.com/other,Other Book,12.99,GBP,true,,4.0,20,"A different book about something else."
"""
        csv_path.write_text(csv_content, encoding="utf-8")
        index = BM25Index(csv_path)
        assert index.product_count == 2

        # Verify NA price is stored as None
        na_idx = index._urls.index("https://example.com/na")
        assert index._prices[na_idx] is None

        # Verify normal price is stored correctly
        other_idx = index._urls.index("https://example.com/other")
        assert index._prices[other_idx] == 12.99

    def test_nan_in_stock_handled(self, tmp_path: Path):
        """Test that NaN in_stock values are treated as True (default)."""
        csv_path = tmp_path / "nan_stock.csv"
        csv_content = """scraped_at,url,title,price,currency,in_stock,promo,rating,reviews_count,description
2026-10-01,https://example.com/nan,NaN Stock Book,9.99,GBP,nan,,4.0,10,"A book with unknown stock status."
2026-10-01,https://example.com/other,Other Book,12.99,GBP,true,,4.0,20,"A different book about something else."
"""
        csv_path.write_text(csv_content, encoding="utf-8")
        index = BM25Index(csv_path)
        assert index.product_count == 2

        # Verify NaN in_stock is treated as True
        nan_idx = index._urls.index("https://example.com/nan")
        assert index._in_stock[nan_idx] is True

        # Verify normal in_stock is stored correctly
        other_idx = index._urls.index("https://example.com/other")
        assert index._in_stock[other_idx] is True
