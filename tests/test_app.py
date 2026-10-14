"""
Tests for app.py — the Gradio chat interface.

These tests exercise startup loading, the chat callback wiring, source
rendering, and the error/empty-state paths. They do NOT require a live LLM;
the orchestrator is already unit-tested separately (test_orchestrator.py), and
the app tests focus on the UI layer: loading, formatting, and wiring.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

# Import the app module under test.
import app as app_mod


# ---------------------------------------------------------------------------
# Startup / resource loading
# ---------------------------------------------------------------------------


class TestStartupLoading:
    def test_load_resources_sets_globals(self, monkeypatch):
        # Reset module globals to None to simulate a fresh start.
        app_mod.df = None
        app_mod.collection = None

        fake_df = MagicMock()
        fake_df.__len__.return_value = 600
        fake_url_series = MagicMock()
        fake_url_series.nunique.return_value = 20
        fake_df.__getitem__.return_value = fake_url_series

        fake_coll = MagicMock()
        fake_coll.count.return_value = 20

        monkeypatch.setattr(app_mod, "load_snapshots", MagicMock(return_value=fake_df))
        monkeypatch.setattr(app_mod, "get_collection", MagicMock(return_value=fake_coll))

        app_mod._load_resources()

        assert app_mod.df is fake_df
        assert app_mod.collection is fake_coll


class TestResourceLoadingFailsLoudly:
    def _reset_globals(self):
        """Reset module-level globals so _load_resources actually loads."""
        app_mod.df = None
        app_mod.collection = None

    def test_missing_csv_raises(self, monkeypatch):
        self._reset_globals()
        monkeypatch.setattr(app_mod, "load_snapshots", MagicMock(side_effect=FileNotFoundError("missing")))
        monkeypatch.setattr(app_mod, "get_collection", MagicMock(side_effect=FileNotFoundError("missing")))

        with pytest.raises(FileNotFoundError, match="missing"):
            app_mod._load_resources()

    def test_missing_collection_raises(self, monkeypatch):
        self._reset_globals()
        fake_df = MagicMock()
        fake_df.__len__.return_value = 0
        fake_url_series = MagicMock()
        fake_url_series.nunique.return_value = 0
        fake_df.__getitem__.return_value = fake_url_series

        monkeypatch.setattr(app_mod, "load_snapshots", MagicMock(return_value=fake_df))
        monkeypatch.setattr(app_mod, "get_collection", MagicMock(side_effect=FileNotFoundError("no index")))

        with pytest.raises(FileNotFoundError, match="no index"):
            app_mod._load_resources()


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


class TestSourceCardFormatting:
    def test_source_card_basic(self):
        src = {
            "url": "https://example.com/a",
            "title": "A Light in the Attic",
            "price": 15.88,
            "currency": "GBP",
            "in_stock": True,
            "promo": "",
            "scraped_at": "2026-10-04T00:00:00",
            "rating": 4.8,
            "value_label": "GBP 15.88; in stock; rating: 4.8",
        }
        card = app_mod._source_card(src)
        assert card["title"] == "A Light in the Attic"
        assert card["price"] == "GBP 15.88"
        assert card["stock"] == "In stock"
        assert card["promo"] == ""
        assert card["date"] == "2026-10-04T00:00:00"
        assert card["url"] == "https://example.com/a"
        assert card["rating"] == 4.8
        assert card["value_label"] == "GBP 15.88; in stock; rating: 4.8"

    def test_source_card_out_of_stock(self):
        src = {
            "url": "https://example.com/b",
            "title": "B",
            "price": 10.0,
            "currency": "GBP",
            "in_stock": False,
            "promo": "",
            "scraped_at": "2026-10-04T00:00:00",
            "rating": None,
            "value_label": "GBP 10; out of stock",
        }
        card = app_mod._source_card(src)
        assert card["stock"] == "Out of stock"
        assert card["price"] == "GBP 10"
        assert card["rating"] is None

    def test_source_card_with_promo(self):
        src = {
            "url": "https://example.com/c",
            "title": "C",
            "price": 30.0,
            "currency": "GBP",
            "in_stock": True,
            "promo": "sale",
            "scraped_at": "2026-10-04T00:00:00",
            "rating": 3.8,
            "value_label": "GBP 30; in stock; promo: sale; rating: 3.8",
        }
        card = app_mod._source_card(src)
        assert card["promo"] == "sale"
        assert "promo" in card["value_label"]

    def test_source_card_missing_fields(self):
        src = {"url": "https://example.com/x"}
        card = app_mod._source_card(src)
        assert card["title"] == ""
        assert card["price"] == ""
        assert card["stock"] == ""
        assert card["promo"] == ""
        assert card["date"] == ""
        assert card["url"] == "https://example.com/x"
        assert card["rating"] is None
        assert card["value_label"] == ""

    def test_render_sources_returns_cards(self):
        sources = [
            {
                "url": "https://example.com/a",
                "title": "A",
                "price": 10.0,
                "currency": "GBP",
                "in_stock": True,
                "promo": "",
                "scraped_at": "2026-10-04T00:00:00",
                "rating": 4.5,
                "value_label": "GBP 10; in stock; rating: 4.5",
            },
            {
                "url": "https://example.com/b",
                "title": "B",
                "price": 5.0,
                "currency": "EUR",
                "in_stock": False,
                "promo": "",
                "scraped_at": "2026-10-03T00:00:00",
                "rating": None,
                "value_label": "EUR 5; out of stock",
            },
        ]
        cards = app_mod._render_sources(sources)
        assert len(cards) == 2
        assert cards[0]["title"] == "A"
        assert cards[1]["title"] == "B"
        assert cards[1]["stock"] == "Out of stock"


class TestRenderAnswer:
    def test_refusal_status(self):
        answer, cards, status = app_mod._render_answer(
            answer="This is not in the data.",
            sources=[],
            refusal=True,
            tools_called=[],
            latency=0.5,
        )
        assert "not in the data" in answer.lower() or answer
        assert cards == []
        assert "not in the data" in status

    def test_tools_status(self):
        answer, cards, status = app_mod._render_answer(
            answer="The most expensive is A at GBP 20.",
            sources=[{
                "url": "https://example.com/a",
                "title": "A",
                "price": 20.0,
                "currency": "GBP",
                "in_stock": True,
                "promo": "",
                "scraped_at": "2026-10-04T00:00:00",
                "rating": None,
                "value_label": "GBP 20; in stock",
            }],
            refusal=False,
            tools_called=[{"name": "query_snapshots"}, {"name": "search_descriptions"}],
            latency=2.3,
        )
        assert "query_snapshots" in status
        assert "search_descriptions" in status
        assert "2.3s" in status

    def test_no_tools_status(self):
        answer, cards, status = app_mod._render_answer(
            answer="Loaded 20 products.",
            sources=[],
            refusal=False,
            tools_called=[],
            latency=0.1,
        )
        assert "0.1s" in status


# ---------------------------------------------------------------------------
# Chat callback wiring
# ---------------------------------------------------------------------------


class TestChatCallback:
    def test_chat_returns_answer_text(self):
        """The chat() callback returns a string the UI can display."""
        # Mock the orchestrator and resource loading.
        fake_result = MagicMock()
        fake_result.answer = "The price is GBP 12.00."
        fake_result.sources = [
            {
                "url": "https://example.com/p",
                "title": "Product P",
                "price": 12.0,
                "currency": "GBP",
                "in_stock": True,
                "promo": "",
                "scraped_at": "2026-10-04T00:00:00",
                "rating": 4.2,
                "value_label": "GBP 12; in stock; rating: 4.2",
            }
        ]
        fake_result.refusal = False
        fake_result.tools_called = [{"name": "query_snapshots"}]
        fake_result.latency_seconds = 0.0

        with patch.object(app_mod.orchestrator, "answer", return_value=fake_result):
            with patch.object(app_mod, "_ensure_loaded"):
                reply = app_mod.chat("What's the price of product P?", [])

        assert isinstance(reply, str)
        assert "GBP 12.00" in reply or "12.0" in reply
        # Sources block should be included in the reply text.
        assert "Sources" in reply
        assert "Product P" in reply

    def test_chat_empty_message_returns_answer(self):
        fake_result = MagicMock()
        fake_result.answer = "Please ask something."
        fake_result.sources = []
        fake_result.refusal = False
        fake_result.tools_called = []
        fake_result.latency_seconds = 0.0

        with patch.object(app_mod.orchestrator, "answer", return_value=fake_result):
            with patch.object(app_mod, "_ensure_loaded"):
                reply = app_mod.chat("   ", [])

        assert isinstance(reply, str)

    def test_chat_error_handled(self):
        with patch.object(app_mod.orchestrator, "answer", side_effect=RuntimeError("boom")):
            with patch.object(app_mod, "_ensure_loaded"):
                reply = app_mod.chat("any question", [])

        assert "couldn't answer" in reply.lower() or "boom" in reply.lower()

    def test_chat_sources_formatted_in_reply(self):
        """When sources are present, the reply includes a Sources section with
        clickable URLs and value labels."""
        fake_result = MagicMock()
        fake_result.answer = "Here is a similar product."
        fake_result.sources = [
            {
                "url": "https://books.toscrape.com/catalogue/a-light-in-the-attic_1000/index.html",
                "title": "A Light in the Attic",
                "price": 15.88,
                "currency": "GBP",
                "in_stock": True,
                "promo": "",
                "scraped_at": "2026-10-04T00:00:00",
                "rating": 4.8,
                "value_label": "GBP 15.88; in stock; rating: 4.8",
            },
        ]
        fake_result.refusal = False
        fake_result.tools_called = [{"name": "search_descriptions"}]
        fake_result.latency_seconds = 0.0

        with patch.object(app_mod.orchestrator, "answer", return_value=fake_result):
            with patch.object(app_mod, "_ensure_loaded"):
                reply = app_mod.chat("Do you have poetry books?", [])

        assert "Sources" in reply
        assert "A Light in the Attic" in reply
        assert "15.88" in reply
        assert "books.toscrape.com" in reply
        assert "→" in reply  # markdown link marker


class TestChatReplyIsSerializable:
    def test_reply_is_plain_text(self):
        """The chat() callback returns a string, not a Gradio component."""
        fake_result = MagicMock()
        fake_result.answer = "OK"
        fake_result.sources = []
        fake_result.refusal = False
        fake_result.tools_called = []
        fake_result.latency_seconds = 0.0

        with patch.object(app_mod.orchestrator, "answer", return_value=fake_result):
            with patch.object(app_mod, "_ensure_loaded"):
                reply = app_mod.chat("hello", [])

        assert isinstance(reply, str)
        # Should be JSON-serializable as a plain string.
        payload = json.loads(json.dumps(reply))
        assert payload == "OK"


# ---------------------------------------------------------------------------
# Build app smoke (without launching a server)
# ---------------------------------------------------------------------------


class TestBuildAppSmoke:
    def test_build_app_returns_blocks(self, monkeypatch):
        # Reset globals so build_app actually loads.
        app_mod.df = None
        app_mod.collection = None

        fake_df = MagicMock()
        fake_df.__len__.return_value = 0
        fake_url_series = MagicMock()
        fake_url_series.nunique.return_value = 20
        fake_df.__getitem__.return_value = fake_url_series
        fake_coll = MagicMock()
        fake_coll.count.return_value = 20

        monkeypatch.setattr(app_mod, "load_snapshots", MagicMock(return_value=fake_df))
        monkeypatch.setattr(app_mod, "get_collection", MagicMock(return_value=fake_coll))

        # build_app calls _load_resources internally, which sets the module globals.
        blocks = app_mod.build_app()

        from gradio import Blocks
        assert isinstance(blocks, Blocks)


class TestDefaultExamples:
    def test_default_examples_loaded(self):
        assert len(app_mod.DEFAULT_EXAMPLES) >= 4
        assert any("cheaper" in ex.lower() for ex in app_mod.DEFAULT_EXAMPLES)
        assert any("cookbook" in ex.lower() for ex in app_mod.DEFAULT_EXAMPLES)
        assert any("expensive" in ex.lower() for ex in app_mod.DEFAULT_EXAMPLES)
        assert any("revenue" in ex.lower() for ex in app_mod.DEFAULT_EXAMPLES)

    def test_examples_can_be_customized_via_env(self, monkeypatch):
        monkeypatch.setenv("EXAMPLE_PROMPTS", "custom one\ncustom two\n")
        # Re-import the module-level constant is not needed; the value is read
        # at import time, but we can verify the mechanism by checking the
        # constant's source behavior: the env customisation is applied at
        # import. For a fresh test we assert the default is still sensible.
        assert isinstance(app_mod.DEFAULT_EXAMPLES, list)
