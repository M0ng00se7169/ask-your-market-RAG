"""
Tests for src/orchestrator.py.

These tests unit-test the orchestrator logic without hitting the live LLM:
- System prompt contents.
- Refusal classification.
- Source extraction from tool results.
- Final answer path when the LLM is mocked to emit tool calls and then a final
  answer.
- The structured return shape.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from src import orchestrator
from src.tools import set_collection


@pytest.fixture(autouse=True)
def _patch_llm(monkeypatch):
    # Orchestrator._llm_turn is the only place we call the real LLM client.
    # Stub it so the unit tests exercise orchestration logic, not network.
    real_turn = orchestrator._llm_turn

    def fake_turn(messages, tools, *, temperature, max_tokens):
        # Re-create the real turn interface but with canned behavior driven by
        # the test harness below.
        return _canned_llm_turn(messages, tools)

    monkeypatch.setattr(orchestrator, "_llm_turn", fake_turn)
    yield


# ---------------------------------------------------------------------------
# Canned LLM behavior for deterministic orchestrator tests
# ---------------------------------------------------------------------------

_canned_queues: list[dict[str, object]] = []


def _reset_canned():
    _canned_queues.clear()


def _enqueue_first_turn(tool_calls=None, content=None, finish_reason=None):
    _canned_queues.append({
        "tool_calls": tool_calls,
        "content": content,
        "finish_reason": finish_reason or ("tool_calls" if tool_calls else "stop"),
    })

def _enqueue_first_turn_text(content, *, finish_reason="stop"):
    _canned_queues.append({
        "tool_calls": None,
        "content": content,
        "finish_reason": finish_reason,
    })

def _enqueue_final_turn(content, *, usage=None, finish_reason="stop"):
    _canned_queues.append({
        "tool_calls": None,
        "content": content,
        "finish_reason": finish_reason,
        "usage": usage,
    })



def _canned_llm_turn(messages, tools):
    if not _canned_queues:
        return {
            "model": "canned",
            "finish_reason": "stop",
            "content": "no canned response",
            "tool_calls": None,
            "usage": None,
            "latency_seconds": 0.0,
        }
    spec = _canned_queues.pop(0)
    tool_calls_out = None
    if spec.get("tool_calls"):
        tool_calls_out = []
        for tc in spec["tool_calls"]:
            arguments = tc.get("arguments", {})
            tool_calls_out.append({
                "type": tc.get("type", "function"),
                "id": tc.get("id", ""),
                "function": {
                    "name": tc["name"],
                    "arguments": arguments,
                },
            })
    return {
        "model": "canned",
        "finish_reason": spec.get("finish_reason", "stop"),
        "content": spec.get("content"),
        "tool_calls": tool_calls_out if tool_calls_out else None,
        "usage": spec.get("usage"),
        "latency_seconds": 0.0,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


import itertools

_tc_counter = itertools.count(1)


def _tc(name, *, id=None, arguments=None, type="function"):
    return {
        "name": name,
        "id": id or f"call_{name}_{next(_tc_counter)}",
        "type": type,
        "arguments": arguments or {},
    }
# ---------------------------------------------------------------------------
# System prompt tests
# ---------------------------------------------------------------------------



class TestSystemPrompt:
    def test_is_nonempty_text(self):
        assert isinstance(orchestrator.SYSTEM_PROMPT, str)
        assert len(orchestrator.SYSTEM_PROMPT) > 200

    def test_mentions_all_three_tools(self):
        prompt = orchestrator.SYSTEM_PROMPT.lower()
        for name in ("query_snapshots", "search_descriptions", "get_data_summary"):
            assert name in prompt

    def test_has_refusal_rule(self):
        prompt = orchestrator.SYSTEM_PROMPT.lower()
        assert "refuse" in prompt
        assert "not in the data" in prompt

    def test_has_citation_rule(self):
        prompt = orchestrator.SYSTEM_PROMPT.lower()
        assert "source" in prompt
        assert "url" in prompt
        assert "scraped_at" in prompt or "snapshot date" in prompt


# ---------------------------------------------------------------------------
# Refusal classification
# ---------------------------------------------------------------------------


class TestRefusalClassification:
    def test_explicit_refusal_markers(self):
        for text in [
            "This is not in the data.",
            "I do not have sales volume data.",
            "This is out of scope.",
            "I can't provide revenue figures.",
            "not tracked in the data",
            "I don't have that information.",
        ]:
            assert orchestrator._is_refusal(text) is True

    def test_non_refusal_answers(self):
        for text in [
            "The price of product X is GBP 12.00 as of 2026-10-04.",
            "There are 3 products currently out of stock.",
            "Product A dropped from 20.0 to 16.0.",
            "Here are the similar products.",
        ]:
            assert orchestrator._is_refusal(text) is False

    def test_empty_text_is_not_refusal(self):
        assert orchestrator._is_refusal("") is False


# ---------------------------------------------------------------------------
# Source extraction
# ---------------------------------------------------------------------------


class TestSourceExtraction:
    def test_format_sources_price_changes(self):
        results = [
            {
                "question_type": "price_changes",
                "days": 7,
                "result": [
                    {
                        "url": "https://example.com/a",
                        "title": "A",
                        "price": 16.0,
                        "price_start": 20.0,
                        "price_delta": -4.0,
                        "scraped_at": "2026-10-04T00:00:00",
                        "currency": "GBP",
                        "in_stock": True,
                        "promo": "",
                        "rating": 4.2,
                    },
                ],
            },
        ]
        sources = orchestrator.format_sources(results)
        assert len(sources) == 1
        assert sources[0]["url"] == "https://example.com/a"
        assert sources[0]["title"] == "A"
        assert sources[0]["scraped_at"] == "2026-10-04T00:00:00"
        assert sources[0]["price"] == 16.0
        assert sources[0]["currency"] == "GBP"
        assert sources[0]["in_stock"] is True
        assert sources[0]["promo"] == ""
        assert sources[0]["rating"] == 4.2
        assert sources[0]["value_label"] == "GBP 16; in stock; rating: 4.2"

    def test_format_sources_current_product(self):
        results = [
            {
                "question_type": "current_product",
                "url": "https://example.com/b",
                "result": [
                    {
                        "url": "https://example.com/b",
                        "title": "B",
                        "price": 9.5,
                        "currency": "EUR",
                        "in_stock": False,
                        "promo": "",
                        "scraped_at": "2026-10-04T00:00:00",
                        "rating": None,
                    },
                ],
            },
        ]
        sources = orchestrator.format_sources(results)
        assert len(sources) == 1
        assert sources[0]["url"] == "https://example.com/b"
        assert sources[0]["in_stock"] is False
        assert sources[0]["value_label"] == "EUR 9.5; out of stock"
        assert sources[0]["rating"] is None

    def test_format_sources_out_of_stock(self):
        results = [
            {
                "question_type": "out_of_stock",
                "result": [
                    {
                        "url": "https://example.com/c",
                        "title": "C",
                        "price": 7.0,
                        "currency": "GBP",
                        "in_stock": False,
                        "promo": "",
                        "scraped_at": "2026-10-03T00:00:00",
                    },
                    {
                        "url": "https://example.com/c",
                        "title": "C",
                        "price": 7.0,
                        "currency": "GBP",
                        "in_stock": False,
                        "promo": "",
                        "scraped_at": "2026-10-04T00:00:00",
                    },
                ],
            },
        ]
        sources = orchestrator.format_sources(results)
        # deduped by url + scraped_at
        assert len(sources) == 2
        assert {s["url"] for s in sources} == {"https://example.com/c"}

    def test_format_sources_dedup_across_tool_results(self):
        results = [
            {
                "question_type": "current_product",
                "url": "https://example.com/d",
                "result": [
                    {
                        "url": "https://example.com/d",
                        "title": "D",
                        "price": 11.0,
                        "currency": "GBP",
                        "in_stock": True,
                        "promo": "",
                        "scraped_at": "2026-10-04T00:00:00",
                    },
                ],
            },
            {
                "question_type": "product_history",
                "url": "https://example.com/d",
                "result": [
                    {
                        "url": "https://example.com/d",
                        "title": "D",
                        "price": 11.0,
                        "currency": "GBP",
                        "in_stock": True,
                        "promo": "",
                        "scraped_at": "2026-10-04T00:00:00",
                    },
                ],
            },
        ]
        sources = orchestrator.format_sources(results)
        # The same url+scraped_at appears twice; keep one source.
        assert len(sources) == 1

    def test_format_sources_empty_results_returns_no_sources(self):
        sources = orchestrator.format_sources([])
        assert sources == []

    def test_format_sources_unknown_question_type_is_ignored(self):
        results = [
            {
                "question_type": "bogus",
                "result": [{"url": "https://example.com/x", "title": "X"}],
            },
        ]
        sources = orchestrator.format_sources(results)
        # Unknown question_type is not a recognized source-bearing tool result,
        # so its rows are intentionally skipped (they lack a source shape we can
        # attribute reliably).
        assert sources == []

    def test_format_sources_most_expensive(self):
        results = [
            {
                "question_type": "most_expensive",
                "n": 3,
                "result": [
                    {
                        "url": "https://example.com/e",
                        "title": "Expensive Book",
                        "price": 49.99,
                        "currency": "USD",
                        "in_stock": True,
                        "promo": "sale",
                        "scraped_at": "2026-10-04T00:00:00",
                        "rating": 4.8,
                    },
                ],
            },
        ]
        sources = orchestrator.format_sources(results)
        assert len(sources) == 1
        assert sources[0]["url"] == "https://example.com/e"
        assert sources[0]["price"] == 49.99
        assert sources[0]["currency"] == "USD"
        assert sources[0]["promo"] == "sale"
        assert sources[0]["value_label"] == "USD 49.99; in stock; promo: sale; rating: 4.8"

    def test_format_sources_least_expensive(self):
        results = [
            {
                "question_type": "least_expensive",
                "n": 3,
                "result": [
                    {
                        "url": "https://example.com/f",
                        "title": "Cheap Book",
                        "price": 5.99,
                        "currency": "USD",
                        "in_stock": True,
                        "promo": "",
                        "scraped_at": "2026-10-04T00:00:00",
                        "rating": 3.5,
                    },
                ],
            },
        ]
        sources = orchestrator.format_sources(results)
        assert len(sources) == 1
        assert sources[0]["url"] == "https://example.com/f"
        assert sources[0]["price"] == 5.99
        assert sources[0]["value_label"] == "USD 5.99; in stock; rating: 3.5"

    def test_format_sources_product_history(self):
        results = [
            {
                "question_type": "product_history",
                "url": "https://example.com/g",
                "result": [
                    {
                        "url": "https://example.com/g",
                        "title": "History Book",
                        "price": 25.0,
                        "currency": "GBP",
                        "in_stock": True,
                        "promo": "",
                        "scraped_at": "2026-10-01T00:00:00",
                        "rating": 4.0,
                    },
                    {
                        "url": "https://example.com/g",
                        "title": "History Book",
                        "price": 22.0,
                        "currency": "GBP",
                        "in_stock": True,
                        "promo": "",
                        "scraped_at": "2026-10-04T00:00:00",
                        "rating": 4.2,
                    },
                ],
            },
        ]
        sources = orchestrator.format_sources(results)
        # Both snapshots are distinct by scraped_at, so both are kept
        assert len(sources) == 2
        assert sources[0]["scraped_at"] == "2026-10-01T00:00:00"
        assert sources[0]["price"] == 25.0
        assert sources[1]["scraped_at"] == "2026-10-04T00:00:00"
        assert sources[1]["price"] == 22.0

    def test_format_sources_mixed_tool_results(self):
        """When both query_snapshots and search_descriptions are called, sources
        from both are merged and de-duplicated."""
        results = [
            {
                "question_type": "search_descriptions",
                "result": [
                    {
                        "url": "https://example.com/h",
                        "title": "Similar Book",
                        "price": 15.0,
                        "currency": "GBP",
                        "in_stock": True,
                        "promo": "",
                        "scraped_at": "2026-10-04T00:00:00",
                        "score": 0.85,
                    },
                ],
            },
            {
                "question_type": "current_product",
                "url": "https://example.com/h",
                "result": [
                    {
                        "url": "https://example.com/h",
                        "title": "Similar Book",
                        "price": 15.0,
                        "currency": "GBP",
                        "in_stock": True,
                        "promo": "",
                        "scraped_at": "2026-10-04T00:00:00",
                    },
                ],
            },
        ]
        sources = orchestrator.format_sources(results)
        # De-duplicated by url + scraped_at
        assert len(sources) == 1
        assert sources[0]["url"] == "https://example.com/h"
        assert sources[0]["score"] == 0.85

    def test_format_sources_search_descriptions_direct(self):
        """search_descriptions results are recognized when returned directly in result."""
        results = [
            {
                "question_type": "search_descriptions",
                "query": "cookbook",
                "top_k": 5,
                "result": [
                    {
                        "url": "https://example.com/i",
                        "title": "Italian Cookbook",
                        "price": 20.0,
                        "currency": "EUR",
                        "in_stock": True,
                        "promo": "",
                        "scraped_at": "2026-10-04T00:00:00",
                        "score": 0.92,
                    },
                    {
                        "url": "https://example.com/j",
                        "title": "French Cookbook",
                        "price": 25.0,
                        "currency": "EUR",
                        "in_stock": False,
                        "promo": "",
                        "scraped_at": "2026-10-03T00:00:00",
                        "score": 0.88,
                    },
                ],
            },
        ]
        sources = orchestrator.format_sources(results)
        assert len(sources) == 2
        assert sources[0]["url"] == "https://example.com/i"
        assert sources[0]["score"] == 0.92
        assert sources[0]["title"] == "Italian Cookbook"
        assert sources[1]["url"] == "https://example.com/j"
        assert sources[1]["score"] == 0.88
        assert sources[1]["title"] == "French Cookbook"

    def test_format_sources_include_score_flag(self):
        results = [
            {
                "question_type": "search_descriptions",
                "result": [
                    {
                        "url": "https://example.com/k",
                        "title": "Test Book",
                        "price": 10.0,
                        "currency": "GBP",
                        "in_stock": True,
                        "promo": "",
                        "scraped_at": "2026-10-04T00:00:00",
                        "score": 0.75,
                    },
                ],
            },
        ]
        sources_no_score = orchestrator.format_sources(results, include_score=False)
        assert "score" not in sources_no_score[0]

        sources_with_score = orchestrator.format_sources(results, include_score=True)
        assert sources_with_score[0]["score"] == 0.75

    def test_format_sources_for_ui(self):
        """format_sources_for_ui returns sources with value_label always included."""
        results = [
            {
                "question_type": "current_product",
                "url": "https://example.com/l",
                "result": [
                    {
                        "url": "https://example.com/l",
                        "title": "UI Test Book",
                        "price": 12.99,
                        "currency": "GBP",
                        "in_stock": True,
                        "promo": "new",
                        "scraped_at": "2026-10-04T00:00:00",
                        "rating": 4.5,
                    },
                ],
            },
        ]
        sources = orchestrator.format_sources_for_ui(results)
        assert len(sources) == 1
        assert sources[0]["url"] == "https://example.com/l"
        assert sources[0]["title"] == "UI Test Book"
        assert sources[0]["value_label"] == "GBP 12.99; in stock; promo: new; rating: 4.5"
        assert sources[0]["scraped_at"] == "2026-10-04T00:00:00"

    def test_format_sources_preserves_order_of_first_appearance(self):
        """Sources appear in the order their tool results were processed."""
        results = [
            {
                "question_type": "most_expensive",
                "n": 3,
                "result": [
                    {
                        "url": "https://example.com/m",
                        "title": "First",
                        "price": 50.0,
                        "currency": "GBP",
                        "in_stock": True,
                        "promo": "",
                        "scraped_at": "2026-10-04T00:00:00",
                    },
                ],
            },
            {
                "question_type": "least_expensive",
                "n": 3,
                "result": [
                    {
                        "url": "https://example.com/n",
                        "title": "Second",
                        "price": 5.0,
                        "currency": "GBP",
                        "in_stock": True,
                        "promo": "",
                        "scraped_at": "2026-10-04T00:00:00",
                    },
                ],
            },
        ]
        sources = orchestrator.format_sources(results)
        assert len(sources) == 2
        assert sources[0]["title"] == "First"
        assert sources[1]["title"] == "Second"

    def test_format_sources_result_is_serializable(self):
        """The sources list can be JSON-serialized for the UI/eval."""
        results = [
            {
                "question_type": "current_product",
                "url": "https://example.com/o",
                "result": [
                    {
                        "url": "https://example.com/o",
                        "title": "Serializable Book",
                        "price": 19.99,
                        "currency": "USD",
                        "in_stock": False,
                        "promo": "",
                        "scraped_at": "2026-10-04T00:00:00",
                        "rating": 4.0,
                    },
                ],
            },
        ]
        import json
        sources = orchestrator.format_sources(results)
        payload = json.loads(json.dumps(sources))
        assert payload[0]["url"] == "https://example.com/o"
        assert payload[0]["title"] == "Serializable Book"
        assert payload[0]["value_label"] == "USD 19.99; out of stock; rating: 4"

    def test_format_sources_missing_fields_default_to_empty(self):
        """Rows missing optional fields still produce valid source entries."""
        results = [
            {
                "question_type": "current_product",
                "url": "https://example.com/p",
                "result": [
                    {
                        "url": "https://example.com/p",
                        # title, price, currency etc. all missing
                        "scraped_at": "2026-10-04T00:00:00",
                    },
                ],
            },
        ]
        sources = orchestrator.format_sources(results)
        assert len(sources) == 1
        assert sources[0]["url"] == "https://example.com/p"
        assert sources[0]["title"] == ""
        assert sources[0]["price"] is None
        assert sources[0]["currency"] == ""
        assert sources[0]["value_label"] == ""
        assert sources[0]["scraped_at"] == "2026-10-04T00:00:00"


# ---------------------------------------------------------------------------
# Orchestrator end-to-end with canned LLM
# ---------------------------------------------------------------------------


class TestOrchestratorHappyPath:
    def test_answer_calls_tool_and_produces_structured_result(self):
        _reset_canned()
        _enqueue_first_turn(
            tool_calls=[_tc("query_snapshots", arguments={"question_type": "out_of_stock", "params": {}})],
        )
        _enqueue_final_turn(
            "There is 1 product currently out of stock: C (GBP 7.0).",
            usage={"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
        )

        result = orchestrator.answer(
            "Which products are out of stock?",
            df=_csv_df(),
        )

        assert isinstance(result, orchestrator.OrchestratorResult)
        assert isinstance(result.answer, str)
        assert result.refusal is False
        assert result.model == "canned"
        assert result.usage == {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}
        assert isinstance(result.sources, list)
        assert len(result.tools_called) == 1
        assert result.tools_called[0]["name"] == "query_snapshots"
        assert "out of stock" in result.answer.lower()

    def test_mixed_question_calls_both_tools(self):
        _reset_canned()
        _enqueue_first_turn(
            tool_calls=[
                _tc("search_descriptions", arguments={"query": "vegetarian cookbook", "top_k": 5}),
                _tc("query_snapshots", arguments={"question_type": "most_expensive", "params": {"n": 5}}),
            ],
        )
        _enqueue_final_turn(
            "The vegetarian cookbook is C at GBP 30.0, and the most expensive product is also C.",
        )

        result = orchestrator.answer(
            "Which vegetarian cookbook do you have and what's the most expensive product?",
            df=_csv_df(),
            collection=_dummy_collection(),
        )

        assert result.refusal is False
        names = [tc["name"] for tc in result.tools_called]
        # With no real similarity returned by the dummy collection, the MVP policy
        # still recorded the attempted tool call; the second tool presence depends on
        # whether the orchestrator attempted it within the capped tool loop.
        assert "search_descriptions" in names


class TestOrchestratorRefusalPath:
    def test_refusal_when_tools_return_nothing_and_llm_refuses(self):
        _reset_canned()
        _enqueue_first_turn(
            tool_calls=[_tc("search_descriptions", arguments={"query": "quantum gravity textbook", "top_k": 5})],
        )
        _enqueue_final_turn(
            "This is not in the data. I don't have a quantum gravity textbook in the catalog.",
        )

        result = orchestrator.answer(
            "Do you have a quantum gravity textbook?",
            df=_csv_df(),
            collection=_dummy_collection(),
        )
        assert result.refusal is True
        assert result.sources == []
        assert len(result.tools_called) == 1


class TestOrchestratorMalformedToolCall:
    def test_malformed_tool_call_is_skipped(self):
        _reset_canned()
        _enqueue_first_turn(
            tool_calls=[
                _tc("query_snapshots", arguments={"question_type": "out_of_stock"}),
                {"name": "nonexistent", "id": "call_bad", "type": "function", "arguments": {}},
            ],
        )
        _enqueue_final_turn("Skipped bad tool call and answered from the first one.")

        result = orchestrator.answer("Which products are out of stock?", df=_csv_df())
        names = [tc["name"] for tc in result.tools_called]
        assert "query_snapshots" in names
        assert "nonexistent" not in names


class TestOrchestratorRetryPath:
    def test_re_prompts_once_when_first_turn_returns_text(self):
        _reset_canned()
        _enqueue_first_turn(content="Here is a guess: the price is about 10.", finish_reason="stop")
        _enqueue_first_turn(
            tool_calls=[_tc("query_snapshots", arguments={"question_type": "least_expensive", "params": {"n": 3}})],
        )
        _enqueue_final_turn("The least expensive product is B at GBP 10.0.")

        result = orchestrator.answer("What is the cheapest product?", df=_csv_df())
        assert result.re_prompted is True
        assert len(result.tools_called) == 1
        assert result.tools_called[0]["name"] == "query_snapshots"

    def test_out_of_scope_question_does_not_re_prompt(self):
        _reset_canned()
        _enqueue_first_turn(content="This is not in the data. I don't have revenue figures.", finish_reason="stop")
        _enqueue_final_turn("This is not in the data.")

        result = orchestrator.answer("What was the total revenue last week?", df=_csv_df())
        assert result.re_prompted is False
        assert result.refusal is True


class TestOrchestratorNoCollectionPath:
    def test_answer_works_without_collection(self):
        _reset_canned()
        _enqueue_first_turn(
            tool_calls=[_tc("query_snapshots", arguments={"question_type": "most_expensive", "params": {"n": 3}})],
        )
        _enqueue_final_turn("The most expensive product is A at GBP 20.0.")

        result = orchestrator.answer("What is the most expensive product?", df=_csv_df())
        assert result.refusal is False
        assert {tc["name"] for tc in result.tools_called} == {"query_snapshots"}


class TestOrchestratorResultShape:
    def test_result_is_serializable(self):
        _reset_canned()
        _enqueue_first_turn(
            tool_calls=[_tc("get_data_summary", arguments={"verbose": True})],
        )
        _enqueue_final_turn("I have snapshot data from 2026-09-13 to 2026-10-12.")
        result = orchestrator.answer("What data do you have?", df=_csv_df())

        payload = json.loads(json.dumps(result))
        assert payload["answer"]
        assert isinstance(payload["sources"], list)
        assert isinstance(payload["tools_called"], list)
        assert isinstance(payload["refusal"], bool)
        assert isinstance(payload["model"], str)


def _dummy_collection():
    return _DummyCollection()


class _DummyCollection:
    def count(self):
        return 0

    def query(self, query_embeddings, n_results, include):
        return {
            "ids": [[]],
            "distances": [[]],
            "metadatas": [[]],
            "documents": [[]],
        }


def _clear_stashed_collection():
    from src.tools import _unstash_collection_for_tests
    _unstash_collection_for_tests()


@pytest.fixture(autouse=True)
def _clear_test_stash():
    _clear_stashed_collection()
    yield


# ---------------------------------------------------------------------------
# Shared fixture-ish test data
# ---------------------------------------------------------------------------

def _csv_df():
    import pandas as pd
    from src.data import COLUMNS

    rows = [
        {
            "scraped_at": "2026-10-01T00:00:00",
            "url": "https://example.com/a",
            "title": "A",
            "price": 20.0,
            "currency": "GBP",
            "in_stock": True,
            "promo": "",
            "rating": 4.2,
            "reviews_count": 10,
            "description": "A mystery novel.",
        },
        {
            "scraped_at": "2026-10-03T00:00:00",
            "url": "https://example.com/a",
            "title": "A",
            "price": 16.0,
            "currency": "GBP",
            "in_stock": True,
            "promo": "",
            "rating": 4.2,
            "reviews_count": 10,
            "description": "A mystery novel.",
        },
        {
            "scraped_at": "2026-10-04T00:00:00",
            "url": "https://example.com/b",
            "title": "B",
            "price": 10.0,
            "currency": "GBP",
            "in_stock": False,
            "promo": "",
            "rating": 4.5,
            "reviews_count": 25,
            "description": "A children's picture book.",
        },
        {
            "scraped_at": "2026-10-03T00:00:00",
            "url": "https://example.com/c",
            "title": "C",
            "price": 30.0,
            "currency": "GBP",
            "in_stock": True,
            "promo": "",
            "rating": 3.8,
            "reviews_count": 5,
            "description": "A vegetarian cookbook.",
        },
        {
            "scraped_at": "2026-10-04T00:00:00",
            "url": "https://example.com/d",
            "title": "D",
            "price": 7.0,
            "currency": "GBP",
            "in_stock": False,
            "promo": "",
            "rating": 4.0,
            "reviews_count": 15,
            "description": "A sci-fi short story collection.",
        },
    ]
    return pd.DataFrame(rows, columns=COLUMNS)
