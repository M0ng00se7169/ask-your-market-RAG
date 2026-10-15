"""
Tests for src/llm.py.

Covers env-driven configuration, the "no key set" error, and parsing of the
normalized chat_completion response shape. These tests do not hit the live
OpenRouter API; they verify the client shell, parsing, and error paths.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Make sure we import the package under test.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import httpcore
import src.llm as llm


class _WrappedProtocolError(Exception):
    """Simulates an exception chain where a transient httpcore protocol error is
    wrapped by another layer (e.g. the openai SDK wrapping it as an
    APIConnectionError)."""

    def __init__(self, inner: BaseException) -> None:
        super().__init__(str(inner))
        self.__cause__ = inner


@pytest.fixture(autouse=True)
def _clear_llm_globals():
    # Ensure each test starts with uncached config state.
    llm._API_KEY = None
    llm._MODEL = None
    yield
    llm._API_KEY = None
    llm._MODEL = None


class TestConfiguration:
    def test_default_model_when_env_empty(self, monkeypatch, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("[TEMPLATE]\nOPENROUTER_MODEL=\nOPENROUTER_API_KEY=sk-or-xxx\n")
        monkeypatch.setenv("OPENROUTER_MODEL", "")
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-xxx")
        monkeypatch.delenv("OPENROUTER_MODEL", raising=False)
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

        # When the advertised fallback is in place and env has no real model,
        # _get_model should return the Phase 0 decision.
        model = llm._get_model()
        assert model == llm._ADVERTISED_MODEL_FOR_LOGS

    def test_env_model_override(self, monkeypatch, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("[TEMPLATE]\nOPENROUTER_MODEL=google/gemini-2.0-flash-001\n")
        monkeypatch.setenv("OPENROUTER_MODEL", "google/gemini-2.0-flash-001")
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-xxx")
        model = llm._get_model()
        assert model == "google/gemini-2.0-flash-001"

    def test_no_key_raises_on_client(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-xxx")
        monkeypatch.setenv("OPENROUTER_MODEL", "nvidia/nemotron-3.5-lightning:free")
        with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
            llm._build_client()

    def test_real_key_creates_client(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test123")
        monkeypatch.setenv("OPENROUTER_MODEL", "nvidia/nemotron-3.5-lightning:free")
        client = llm._build_client()
        assert client is not None
        # openai normalizes base_url by appending a trailing slash.
        assert str(client.base_url).rstrip("/") == "https://openrouter.ai/api/v1"


class TestChatCompletionParsing:
    def test_tool_calls_parsed_from_json_args(self):
        fake_message = MagicMock()
        fake_message.content = None
        fake_tc = MagicMock()
        fake_tc.type = "function"
        fake_tc.id = "call_1"
        fake_tc.function.name = "query_snapshots"
        fake_tc.function.arguments = json.dumps({"question_type": "out_of_stock", "params": {}})
        fake_message.tool_calls = [fake_tc]

        fake_choice = MagicMock()
        fake_choice.message = fake_message
        fake_choice.finish_reason = "tool_calls"

        fake_response = MagicMock()
        fake_response.choices = [fake_choice]
        fake_response.usage = MagicMock(
            prompt_tokens=10, completion_tokens=20, total_tokens=30
        )

        with patch.object(llm, "_build_client") as mk_client:
            mk_client.return_value.chat.completions.create.return_value = fake_response
            monkeypatch = pytest.MonkeyPatch()
            monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
            monkeypatch.setenv("OPENROUTER_MODEL", "nvidia/nemotron-3.5-lightning:free")
            out = llm.chat_completion([{"role": "user", "content": "hello"}], tools=[], temperature=0.0)

        assert out["finish_reason"] == "tool_calls"
        assert out["content"] is None
        assert out["tool_calls"] is not None
        assert len(out["tool_calls"]) == 1
        assert out["tool_calls"][0]["function"]["name"] == "query_snapshots"
        assert out["tool_calls"][0]["function"]["arguments"] == {
            "question_type": "out_of_stock",
            "params": {},
        }
        assert out["usage"] == {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}

    def test_malformed_tool_args_become_empty_dict(self):
        fake_message = MagicMock()
        fake_message.content = None
        fake_tc = MagicMock()
        fake_tc.type = "function"
        fake_tc.id = "call_2"
        fake_tc.function.name = "search_descriptions"
        fake_tc.function.arguments = "{ not json"
        fake_message.tool_calls = [fake_tc]

        fake_choice = MagicMock()
        fake_choice.message = fake_message
        fake_choice.finish_reason = "tool_calls"

        fake_response = MagicMock()
        fake_response.choices = [fake_choice]
        fake_response.usage = None

        with patch.object(llm, "_build_client") as mk_client:
            mk_client.return_value.chat.completions.create.return_value = fake_response
            monkeypatch = pytest.MonkeyPatch()
            monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
            monkeypatch.setenv("OPENROUTER_MODEL", "nvidia/nemotron-3.5-lightning:free")
            out = llm.chat_completion([{"role": "user", "content": "hello"}], tools=[], temperature=0.0)

        assert out["tool_calls"][0]["function"]["arguments"] == {}

    def test_text_finish_reason_has_no_tool_calls(self):
        fake_message = MagicMock()
        fake_message.content = "Here is the answer."
        fake_message.tool_calls = None

        fake_choice = MagicMock()
        fake_choice.message = fake_message
        fake_choice.finish_reason = "stop"

        fake_response = MagicMock()
        fake_response.choices = [fake_choice]
        fake_response.usage = None

        with patch.object(llm, "_build_client") as mk_client:
            mk_client.return_value.chat.completions.create.return_value = fake_response
            monkeypatch = pytest.MonkeyPatch()
            monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
            monkeypatch.setenv("OPENROUTER_MODEL", "nvidia/nemotron-3.5-lightning:free")
            out = llm.chat_completion([{"role": "user", "content": "hello"}], tools=[], temperature=0.0)

        assert out["finish_reason"] == "stop"
        assert out["content"] == "Here is the answer."
        assert out["tool_calls"] is None

    def test_latency_present(self):
        fake_message = MagicMock()
        fake_message.content = "ok"
        fake_message.tool_calls = None

        fake_choice = MagicMock()
        fake_choice.message = fake_message
        fake_choice.finish_reason = "stop"

        fake_response = MagicMock()
        fake_response.choices = [fake_choice]
        fake_response.usage = None

        with patch.object(llm, "_build_client") as mk_client:
            mk_client.return_value.chat.completions.create.return_value = fake_response
            monkeypatch = pytest.MonkeyPatch()
            monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
            monkeypatch.setenv("OPENROUTER_MODEL", "nvidia/nemotron-3.5-lightning:free")
            out = llm.chat_completion([{"role": "user", "content": "hello"}], tools=[], temperature=0.0)

        assert "latency_seconds" in out
        assert isinstance(out["latency_seconds"], float)


class TestChatCompletionErrorPath:
    def test_api_error_is_logged_and_raised(self, caplog, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-real")
        monkeypatch.setenv("OPENROUTER_MODEL", "nvidia/nemotron-3.5-lightning:free")
        mk_client = MagicMock()
        mk_client.chat.completions.create.side_effect = RuntimeError("boom")
        with patch.object(llm, "_build_client", return_value=mk_client):
            with caplog.at_level("ERROR"):
                with pytest.raises(RuntimeError):
                    llm.chat_completion([{"role": "user", "content": "hi"}], tools=[])
            assert any("OpenRouter call failed" in r.message for r in caplog.records)


class TestTransientProtocolErrorRetry:
    """The openai SDK retries timeout/request exceptions but does NOT retry
    httpcore.RemoteProtocolError / ProtocolError, even when they surface as
    APIConnectionError. This app retries those at the application layer with a
    fresh client per attempt."""

    def test_remote_protocol_error_is_retried_once_then_raised(self, caplog, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-real")
        monkeypatch.setenv("OPENROUTER_MODEL", "nvidia/nemotron-3.5-lightning:free")
        inner = httpcore.RemoteProtocolError("server disconnected mid-stream")
        wrapped = _WrappedProtocolError(inner)

        mk_client = MagicMock()
        mk_client.chat.completions.create.side_effect = [wrapped, wrapped]

        call_count = 0

        def _building_client():
            nonlocal call_count
            call_count += 1
            return mk_client

        with patch.object(llm, "_build_client", side_effect=_building_client):
            with caplog.at_level("ERROR"):
                with pytest.raises(_WrappedProtocolError):
                    llm.chat_completion([{"role": "user", "content": "hi"}], tools=[])

        assert call_count == 2, "expected a fresh client per retry attempt"
        assert any("RemoteProtocolError" in r.message or "mid-stream" in r.message for r in caplog.records)

    def test_non_transient_errors_are_not_retried(self, caplog, monkeypatch):
        """Auth / status / other errors should propagate immediately, not be
        swallowed by the protocol-error retry path."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-real")
        monkeypatch.setenv("OPENROUTER_MODEL", "nvidia/nemotron-3.5-lightning:free")

        mk_client = MagicMock()
        mk_client.chat.completions.create.side_effect = RuntimeError("not a protocol error")

        with patch.object(llm, "_build_client", return_value=mk_client):
            with caplog.at_level("ERROR"):
                with pytest.raises(RuntimeError, match="not a protocol error"):
                    llm.chat_completion([{"role": "user", "content": "hi"}], tools=[])

        # The point of this test is that the call happens exactly once (no retry).
        assert mk_client.chat.completions.create.call_count == 1


class TestCreateChatCompletionAlias:
    def test_alias_delegates(self):
        with patch.object(llm, "chat_completion", return_value={"model": "x"}) as mk:
            out = llm.create_chat_completion([{"role": "user", "content": "hi"}], temperature=0.5)
            mk.assert_called_once_with(
                [{"role": "user", "content": "hi"}], tools=None, temperature=0.5, max_tokens=1024, model=None
            )
            assert out == {"model": "x"}
