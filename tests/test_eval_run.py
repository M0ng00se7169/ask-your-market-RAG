"""
Tests for eval/run.py — the Phase 5 sub-phase 5b evaluation runner.

Tests cover:
- Question loading from eval/questions.json
- Scoring functions for exact, semantic, mixed, and out-of-scope questions
- Summary computation
- Summary markdown writer
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from eval.run import (
    load_questions,
    score_exact_question,
    score_semantic_question,
    score_mixed_question,
    score_out_of_scope_question,
    score_question,
    _compute_summary,
    _write_summary,
    _url_from_source,
    _extract_urls_from_answer,
)

EVAL_DIR = Path(__file__).resolve().parent.parent / "eval"
QUESTIONS_PATH = EVAL_DIR / "questions.json"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_orchestrator_result():
    """A sample OrchestratorResult for testing."""
    return MagicMock(
        answer="The price is GBP 16.00 as of 2026-10-04.",
        sources=[
            {
                "url": "https://example.com/product-a",
                "title": "Product A",
                "price": 16.0,
                "currency": "GBP",
                "in_stock": True,
                "scraped_at": "2026-10-04T00:00:00",
            }
        ],
        tools_called=[
            {"name": "query_snapshots", "arguments": {"question_type": "price_changes", "params": {"days": 7}}}
        ],
        refusal=False,
        model="nvidia/nemotron-3.5-lightning:free",
        usage={"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
        latency_seconds=2.5,
    )


@pytest.fixture
def sample_collection():
    """A mock Chroma collection for semantic search tests."""
    coll = MagicMock()
    coll.count.return_value = 10
    coll.query.return_value = {
        "ids": [["0", "1", "2"]],
        "distances": [[0.1, 0.2, 0.3]],
        "metadatas": [[{"url": "https://example.com/a"}, {"url": "https://example.com/b"}, {"url": "https://example.com/c"}]],
        "documents": [["desc A", "desc B", "desc C"]],
    }
    return coll


# ---------------------------------------------------------------------------
# Question loading tests
# ---------------------------------------------------------------------------


class TestLoadQuestions:
    def test_loads_valid_json(self):
        questions = load_questions(QUESTIONS_PATH)
        assert isinstance(questions, list)
        assert len(questions) == 25

    def test_each_question_has_required_fields(self):
        questions = load_questions(QUESTIONS_PATH)
        required = {"id", "type", "question", "expected"}
        for q in questions:
            assert required.issubset(q.keys()), f"Missing fields in {q.get('id')}"

    def test_types_are_valid(self):
        questions = load_questions(QUESTIONS_PATH)
        valid_types = {"exact", "semantic", "mixed", "out_of_scope"}
        for q in questions:
            assert q["type"] in valid_types, f"Invalid type {q['type']} for {q['id']}"


# ---------------------------------------------------------------------------
# URL extraction tests
# ---------------------------------------------------------------------------


class TestUrlExtraction:
    def test_extracts_urls_from_answer(self):
        answer = "Product A is at https://books.toscrape.com/a and Product B at https://books.toscrape.com/b"
        urls = _extract_urls_from_answer(answer)
        assert len(urls) == 2
        assert "https://books.toscrape.com/a" in urls
        assert "https://books.toscrape.com/b" in urls

    def test_no_urls_returns_empty(self):
        answer = "No products found in the data."
        urls = _extract_urls_from_answer(answer)
        assert urls == []

    def test_handles_trailing_punctuation(self):
        answer = "See https://example.com/product for details."
        urls = _extract_urls_from_answer(answer)
        assert len(urls) == 1
        assert urls[0] == "https://example.com/product"


# ---------------------------------------------------------------------------
# URL from source tests
# ---------------------------------------------------------------------------


class TestUrlFromSource:
    def test_extracts_url(self):
        source = {"url": "https://example.com/product", "title": "Product"}
        assert _url_from_source(source) == "https://example.com/product"

    def test_missing_url_returns_empty(self):
        source = {"title": "Product"}
        assert _url_from_source(source) == ""


# ---------------------------------------------------------------------------
# Exact question scoring tests
# ---------------------------------------------------------------------------


class TestScoreExactQuestion:
    def test_snapshot_query_contains_url_passes(self, sample_orchestrator_result):
        question = {
            "id": "exact-01",
            "type": "exact",
            "question": "Which products got cheaper?",
            "expected": {
                "question_type": "query_snapshots",
                "params": {"days": 7},
                "check": {
                    "kind": "snapshot_query",
                    "assert": "contains_url",
                    "url": "https://example.com/product-a",
                },
            },
        }
        result = score_exact_question(question, sample_orchestrator_result, MagicMock())
        assert result["passed"] is True
        assert "found in sources" in result["details"]

    def test_snapshot_query_contains_url_fails_when_missing(self, sample_orchestrator_result):
        question = {
            "id": "exact-01",
            "type": "exact",
            "question": "Which products got cheaper?",
            "expected": {
                "question_type": "query_snapshots",
                "check": {
                    "kind": "snapshot_query",
                    "assert": "contains_url",
                    "url": "https://example.com/missing-product",
                },
            },
        }
        result = score_exact_question(question, sample_orchestrator_result, MagicMock())
        assert result["passed"] is False
        assert "not found" in result["details"]

    def test_snapshot_query_call_made_passes(self, sample_orchestrator_result):
        question = {
            "id": "exact-04",
            "type": "exact",
            "question": "Which products are out of stock?",
            "expected": {
                "question_type": "query_snapshots",
                "check": {
                    "kind": "snapshot_query",
                    "assert": "call_made",
                },
            },
        }
        result = score_exact_question(question, sample_orchestrator_result, MagicMock())
        assert result["passed"] is True
        assert "was called" in result["details"]

    def test_snapshot_query_call_made_fails_when_wrong_tool(self, sample_orchestrator_result):
        question = {
            "id": "exact-04",
            "type": "exact",
            "question": "Which products are out of stock?",
            "expected": {
                "question_type": "query_snapshots",
                "check": {
                    "kind": "snapshot_query",
                    "assert": "call_made",
                },
            },
        }
        # Modify the result to have wrong tool
        sample_orchestrator_result.tools_called = [{"name": "search_descriptions"}]
        result = score_exact_question(question, sample_orchestrator_result, MagicMock())
        assert result["passed"] is False

    def test_latest_snapshot_lookup_price_eq_passes(self, sample_orchestrator_result):
        question = {
            "id": "exact-02",
            "type": "exact",
            "question": "What is the price?",
            "expected": {
                "question_type": "current_product",
                "check": {
                    "kind": "latest_snapshot_lookup",
                    "assert": "price_eq",
                    "url": "https://example.com/product-a",
                    "field": "price",
                },
            },
        }
        result = score_exact_question(question, sample_orchestrator_result, MagicMock())
        assert result["passed"] is True
        assert "Price" in result["details"]

    def test_product_history_passes_with_url_in_sources(self, sample_orchestrator_result):
        question = {
            "id": "exact-03",
            "type": "exact",
            "question": "Product history?",
            "expected": {
                "question_type": "product_history",
                "check": {
                    "kind": "product_history",
                    "url": "https://example.com/product-a",
                },
            },
        }
        sample_orchestrator_result.tools_called = [
            {"name": "query_snapshots", "arguments": {"question_type": "product_history"}}
        ]
        sample_orchestrator_result.sources = [
            {"url": "https://example.com/product-a", "scraped_at": "2026-10-01"},
            {"url": "https://example.com/product-a", "scraped_at": "2026-10-04"},
        ]
        result = score_exact_question(question, sample_orchestrator_result, MagicMock())
        assert result["passed"] is True
        assert "snapshots" in result["details"]

    def test_data_summary_contains_columns_passes(self, sample_orchestrator_result):
        question = {
            "id": "exact-09",
            "type": "exact",
            "question": "What data do you have?",
            "expected": {
                "question_type": "available_data",
                "check": {
                    "kind": "data_summary",
                    "assert": "contains_columns",
                    "columns": ["scraped_at", "url", "title", "price", "currency"],
                },
            },
        }
        sample_orchestrator_result.answer = (
            "I have data with columns: scraped_at, url, title, price, currency, "
            "in_stock, promo, rating, reviews_count, description."
        )
        result = score_exact_question(question, sample_orchestrator_result, MagicMock())
        assert result["passed"] is True
        assert "columns" in result["details"]

    def test_refusal_fails_exact_question(self, sample_orchestrator_result):
        sample_orchestrator_result.refusal = True
        sample_orchestrator_result.answer = "This is not in the data."
        question = {
            "id": "exact-01",
            "type": "exact",
            "question": "Which products got cheaper?",
            "expected": {
                "question_type": "query_snapshots",
                "check": {"kind": "snapshot_query", "assert": "contains_url", "url": "https://example.com/x"},
            },
        }
        result = score_exact_question(question, sample_orchestrator_result, MagicMock())
        assert result["passed"] is False
        assert "Refused" in result["details"]


# ---------------------------------------------------------------------------
# Semantic question scoring tests
# ---------------------------------------------------------------------------


class TestScoreSemanticQuestion:
    def test_contains_url_passes_with_mock_collection(self, sample_orchestrator_result, sample_collection):
        question = {
            "id": "semantic-02",
            "type": "semantic",
            "question": "Mystery novels?",
            "expected": {
                "query": "mystery novel",
                "top_k": 5,
                "check": {
                    "kind": "semantic_search",
                    "assert": "contains_url",
                    "url": "https://example.com/a",
                },
            },
        }
        # Patch search_descriptions to return our expected URL
        with patch("src.index.search_descriptions") as mock_search:
            mock_search.return_value = [
                {"url": "https://example.com/a", "title": "Mystery Book", "score": 0.9},
                {"url": "https://example.com/b", "title": "Other Book", "score": 0.8},
            ]
            result = score_semantic_question(question, sample_orchestrator_result, sample_collection)
            assert result["passed"] is True
            assert "found at rank" in result["details"]
            assert result["actual_rank"] == 1

    def test_contains_url_fails_when_not_in_top_k(self, sample_orchestrator_result, sample_collection):
        question = {
            "id": "semantic-02",
            "type": "semantic",
            "question": "Mystery novels?",
            "expected": {
                "query": "mystery novel",
                "top_k": 5,
                "check": {
                    "kind": "semantic_search",
                    "assert": "contains_url",
                    "url": "https://example.com/missing",
                },
            },
        }
        with patch("src.index.search_descriptions") as mock_search:
            mock_search.return_value = [
                {"url": "https://example.com/a", "title": "Book A"},
                {"url": "https://example.com/b", "title": "Book B"},
            ]
            result = score_semantic_question(question, sample_orchestrator_result, sample_collection)
            assert result["passed"] is False
            assert "not in top" in result["details"]

    def test_none_in_top_k_passes(self, sample_orchestrator_result, sample_collection):
        question = {
            "id": "semantic-01",
            "type": "semantic",
            "question": "Vegetarian cookbooks?",
            "expected": {
                "query": "vegetarian cookbook",
                "top_k": 5,
                "check": {
                    "kind": "semantic_search",
                    "assert": "none_in_top_k",
                },
            },
        }
        with patch("src.index.search_descriptions") as mock_search:
            mock_search.return_value = []  # No results
            result = score_semantic_question(question, sample_orchestrator_result, sample_collection)
            assert result["passed"] is True
            assert "No matching product" in result["details"]


# ---------------------------------------------------------------------------
# Mixed question scoring tests
# ---------------------------------------------------------------------------


class TestScoreMixedQuestion:
    def test_both_tools_called_passes(self, sample_orchestrator_result):
        question = {
            "id": "mixed-01",
            "type": "mixed",
            "question": "Mixed question?",
            "expected": {
                "tools": ["query_snapshots", "search_descriptions"],
                "check": {"kind": "mixed"},
            },
        }
        sample_orchestrator_result.tools_called = [
            {"name": "query_snapshots"},
            {"name": "search_descriptions"},
        ]
        result = score_mixed_question(question, sample_orchestrator_result)
        assert result["passed"] is True
        assert "All expected tools called" in result["details"]

    def test_missing_tool_fails(self, sample_orchestrator_result):
        question = {
            "id": "mixed-01",
            "type": "mixed",
            "question": "Mixed question?",
            "expected": {
                "tools": ["query_snapshots", "search_descriptions"],
                "check": {"kind": "mixed"},
            },
        }
        sample_orchestrator_result.tools_called = [
            {"name": "query_snapshots"},
        ]
        result = score_mixed_question(question, sample_orchestrator_result)
        assert result["passed"] is False
        assert "Missing tools" in result["details"]


# ---------------------------------------------------------------------------
# Out-of-scope question scoring tests
# ---------------------------------------------------------------------------


class TestScoreOutOfScopeQuestion:
    def test_refusal_passes(self, sample_orchestrator_result):
        question = {
            "id": "outofscope-01",
            "type": "out_of_scope",
            "question": "Total revenue?",
            "expected": {"assert": "refusal"},
        }
        sample_orchestrator_result.refusal = True
        sample_orchestrator_result.answer = "This is not in the data."
        result = score_out_of_scope_question(question, sample_orchestrator_result)
        assert result["passed"] is True
        assert "Correctly refused" in result["details"]

    def test_no_refusal_fails(self, sample_orchestrator_result):
        question = {
            "id": "outofscope-01",
            "type": "out_of_scope",
            "question": "Total revenue?",
            "expected": {"assert": "refusal"},
        }
        sample_orchestrator_result.refusal = False
        sample_orchestrator_result.answer = "The total revenue was $10,000."
        result = score_out_of_scope_question(question, sample_orchestrator_result)
        assert result["passed"] is False
        assert "Did not refuse" in result["details"]


# ---------------------------------------------------------------------------
# Full question scoring tests
# ---------------------------------------------------------------------------


class TestScoreQuestion:
    def test_dispatch_to_correct_scorer_exact(self, sample_orchestrator_result, sample_collection):
        question = {
            "id": "exact-01",
            "type": "exact",
            "question": "Price changes?",
            "expected": {
                "question_type": "query_snapshots",
                "check": {"kind": "snapshot_query", "assert": "call_made"},
            },
        }
        sample_orchestrator_result.tools_called = [{"name": "query_snapshots"}]
        result = score_question(question, sample_orchestrator_result, MagicMock(), sample_collection)
        assert result["id"] == "exact-01"
        assert result["type"] == "exact"
        assert result["passed"] is True

    def test_dispatch_to_correct_scorer_semantic(self, sample_orchestrator_result, sample_collection):
        question = {
            "id": "semantic-02",
            "type": "semantic",
            "question": "Mystery novels?",
            "expected": {
                "query": "mystery novel",
                "top_k": 5,
                "check": {"kind": "semantic_search", "assert": "contains_url", "url": "https://example.com/a"},
            },
        }
        with patch("src.index.search_descriptions") as mock_search:
            mock_search.return_value = [{"url": "https://example.com/a", "title": "Mystery"}]
            result = score_question(question, sample_orchestrator_result, MagicMock(), sample_collection)
            assert result["id"] == "semantic-02"
            assert result["passed"] is True

    def test_dispatch_to_correct_scorer_mixed(self, sample_orchestrator_result):
        question = {
            "id": "mixed-01",
            "type": "mixed",
            "question": "Mixed?",
            "expected": {"tools": ["query_snapshots", "search_descriptions"], "check": {"kind": "mixed"}},
        }
        sample_orchestrator_result.tools_called = [
            {"name": "query_snapshots"},
            {"name": "search_descriptions"},
        ]
        result = score_question(question, sample_orchestrator_result, MagicMock(), MagicMock())
        assert result["id"] == "mixed-01"
        assert result["passed"] is True

    def test_dispatch_to_correct_scorer_out_of_scope(self, sample_orchestrator_result):
        question = {
            "id": "outofscope-01",
            "type": "out_of_scope",
            "question": "Revenue?",
            "expected": {"assert": "refusal"},
        }
        sample_orchestrator_result.refusal = True
        result = score_question(question, sample_orchestrator_result, MagicMock(), MagicMock())
        assert result["id"] == "outofscope-01"
        assert result["passed"] is True

    def test_includes_base_fields(self, sample_orchestrator_result, sample_collection):
        question = {
            "id": "exact-01",
            "type": "exact",
            "question": "Test question?",
            "expected": {"question_type": "query_snapshots", "check": {"kind": "snapshot_query", "assert": "call_made"}},
        }
        sample_orchestrator_result.tools_called = [{"name": "query_snapshots"}]
        result = score_question(question, sample_orchestrator_result, MagicMock(), sample_collection)
        assert result["latency_seconds"] == 2.5
        assert result["model"] == "nvidia/nemotron-3.5-lightning:free"
        assert result["usage"] == {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}
        assert result["answer_preview"] == "The price is GBP 16.00 as of 2026-10-04."
        assert result["sources_count"] == 1


# ---------------------------------------------------------------------------
# Summary computation tests
# ---------------------------------------------------------------------------


class TestComputeSummary:
    def test_empty_results(self):
        summary = _compute_summary([])
        assert summary["total"] == 0
        assert summary["passed"] == 0
        assert summary["accuracy"] == 0
        assert summary["by_type"] == {}

    def test_all_pass(self):
        results = [
            {"id": "q1", "type": "exact", "passed": True},
            {"id": "q2", "type": "exact", "passed": True},
            {"id": "q3", "type": "semantic", "passed": True},
            {"id": "q4", "type": "mixed", "passed": True},
            {"id": "q5", "type": "out_of_scope", "passed": True},
        ]
        summary = _compute_summary(results)
        assert summary["total"] == 5
        assert summary["passed"] == 5
        assert summary["accuracy"] == 100.0
        assert summary["by_type"]["exact"]["accuracy"] == 100.0
        assert summary["by_type"]["semantic"]["accuracy"] == 100.0
        assert summary["by_type"]["mixed"]["accuracy"] == 100.0
        assert summary["by_type"]["out_of_scope"]["accuracy"] == 100.0

    def test_mixed_results(self):
        results = [
            {"id": "q1", "type": "exact", "passed": True},
            {"id": "q2", "type": "exact", "passed": False},
            {"id": "q3", "type": "semantic", "passed": True},
            {"id": "q4", "type": "semantic", "passed": True},
            {"id": "q5", "type": "semantic", "passed": False},
        ]
        summary = _compute_summary(results)
        assert summary["total"] == 5
        assert summary["passed"] == 3
        assert summary["accuracy"] == 60.0
        assert summary["by_type"]["exact"]["total"] == 2
        assert summary["by_type"]["exact"]["passed"] == 1
        assert summary["by_type"]["exact"]["accuracy"] == 50.0
        assert summary["by_type"]["semantic"]["total"] == 3
        assert summary["by_type"]["semantic"]["passed"] == 2
        assert summary["by_type"]["semantic"]["accuracy"] == 66.7

    def test_unknown_types_ignored_in_by_type(self):
        results = [
            {"id": "q1", "type": "exact", "passed": True},
            {"id": "q2", "type": "unknown_type", "passed": True},
        ]
        summary = _compute_summary(results)
        assert "exact" in summary["by_type"]
        assert "unknown_type" not in summary["by_type"]


# ---------------------------------------------------------------------------
# Summary writer tests
# ---------------------------------------------------------------------------


class TestWriteSummary:
    def test_writes_markdown_file(self, tmp_path):
        output = {
            "metadata": {"eval_date": "2026-10-14T12:00:00"},
            "summary": {
                "total": 25,
                "passed": 20,
                "accuracy": 80.0,
                "by_type": {
                    "exact": {"total": 10, "passed": 8, "accuracy": 80.0},
                    "semantic": {"total": 8, "passed": 7, "accuracy": 87.5},
                    "mixed": {"total": 4, "passed": 3, "accuracy": 75.0},
                    "out_of_scope": {"total": 3, "passed": 2, "accuracy": 66.7},
                },
            },
            "results": [
                {
                    "id": "exact-01",
                    "type": "exact",
                    "question": "Test question?",
                    "passed": True,
                    "details": "Found in sources",
                    "tools_called": ["query_snapshots"],
                },
                {
                    "id": "outofscope-01",
                    "type": "out_of_scope",
                    "question": "Revenue?",
                    "passed": True,
                    "details": "Correctly refused",
                    "refusal": True,
                },
            ],
        }

        summary_path = tmp_path / "test_summary.md"
        _write_summary(output, summary_path)

        content = summary_path.read_text()
        assert "# Ask Your Market" in content and "Evaluation Results" in content
        assert "**Eval date:** 2026-10-14T12:00:00" in content
        assert "| Total questions | 25 |" in content
        assert "| Passed | 20 |" in content
        assert "| Accuracy | 80.0% |" in content
        assert "## Results by Type" in content
        assert "## Per-Question Results" in content
        assert "exact-01 (exact)" in content and "PASS" in content
        assert "outofscope-01 (out_of_scope)" in content and "PASS" in content
        assert "Correctly refused" in content


# ---------------------------------------------------------------------------
# Integration-style test for the scoring pipeline
# ---------------------------------------------------------------------------


class TestScoringPipeline:
    def test_full_pipeline_with_mock_orchestrator(self, tmp_path):
        """Test that the scoring functions work together end-to-end with mock data."""
        from eval.run import score_question

        # Create a mock result that simulates a successful exact query
        mock_result = MagicMock()
        mock_result.answer = "Product A dropped from GBP 20.00 to GBP 16.00."
        mock_result.sources = [
            {
                "url": "https://books.toscrape.com/catalogue/a-light-in-the-attic_1000/index.html",
                "title": "A Light in the Attic",
                "price": 16.0,
                "currency": "GBP",
                "in_stock": True,
                "scraped_at": "2026-10-04T00:00:00",
                "rating": 4.2,
            }
        ]
        mock_result.tools_called = [
            {"name": "query_snapshots", "arguments": {"question_type": "price_changes", "params": {"days": 7}}}
        ]
        mock_result.refusal = False
        mock_result.model = "test-model"
        mock_result.usage = None
        mock_result.latency_seconds = 1.5

        mock_collection = MagicMock()

        question = {
            "id": "exact-01",
            "type": "exact",
            "question": "Which products got cheaper?",
            "expected": {
                "question_type": "query_snapshots",
                "params": {"days": 7},
                "check": {
                    "kind": "snapshot_query",
                    "assert": "contains_url",
                    "url": "https://books.toscrape.com/catalogue/a-light-in-the-attic_1000/index.html",
                },
            },
        }

        result = score_question(question, mock_result, MagicMock(), mock_collection)

        assert result["id"] == "exact-01"
        assert result["passed"] is True
        assert "found in sources" in result["details"]
        assert result["tools_called"] == ["query_snapshots"]
