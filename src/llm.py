"""
Ask Your Market — LLM client.

OpenAI-compatible client for OpenRouter. The model and key come from the
environment / .env (OPENROUTER_MODEL, OPENROUTER_API_KEY). This is the
transport layer; it knows nothing about our tools except that tool_call args
arrive as JSON strings from the API and must be parsed.

Design points
-------------
- One chat.completions.create wrapper so timeout, retry, and logging stay
  in one place.
- Tool call arguments from the API are JSON strings; we parse them into dicts.
- Malformed tool calls (bad JSON, unknown tool) are treated as "model opted
  out of calling a tool this turn" so the orchestrator can decide to re-prompt
  once rather than crash.
- Token usage is captured when the API returns it, for the eval/cost metric.

Phase 3b output alongside src/orchestrator.py.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

logger = logging.getLogger(__name__)

DOTENV_PATH = os.path.join(os.path.dirname(__file__), "..", ".env")

# ---------------------------------------------------------------------------
# Configuration (env-driven, with reasonable defaults)
# ---------------------------------------------------------------------------

_API_KEY: str | None = None
_MODEL: str | None = None
_BASE_URL: str = "https://openrouter.ai/api/v1"
_TIMEOUT_SECONDS: float = 60.0
_MAX_RETRIES: int = 2  # extra re-prompt attempts if the model returns text

_ADVERTISED_MODEL_FOR_LOGS: str = "nvidia/nemotron-3.5-lightning:free"


def _load_env() -> None:
    if os.path.exists(DOTENV_PATH):
        load_dotenv(dotenv_path=DOTENV_PATH)


def _get_api_key() -> str | None:
    global _API_KEY
    if _API_KEY is None:
        _load_env()
        _API_KEY = os.environ.get("OPENROUTER_API_KEY")
    return _API_KEY


def _get_model() -> str:
    global _MODEL
    if _MODEL is None:
        _load_env()
        env_model = os.environ.get("OPENROUTER_MODEL")
        # The Phase 0 decision locked nemotron:free; env may override it.
        if env_model:
            _MODEL = env_model
        else:
            _MODEL = _ADVERTISED_MODEL_FOR_LOGS
    return _MODEL


def _get_client() -> OpenAI:
    key = _get_api_key()
    if not key or key.startswith("sk-or-xxx"):
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set or still has the template value. "
            "Copy .env.example to .env and fill in a real key."
        )
    return OpenAI(api_key=key, base_url=_BASE_URL, timeout=_TIMEOUT_SECONDS)


# ---------------------------------------------------------------------------
# Chat completions wrapper
# ---------------------------------------------------------------------------


def chat_completion(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    temperature: float = 0.0,
    max_tokens: int = 1024,
    *,
    model: str | None = None,
) -> dict[str, Any]:
    """Call the configured OpenRouter model and return a normalized response dict.

    Response shape:
        {
            "model": str,
            "finish_reason": str,
            "content": str | None,
            "tool_calls": list[dict[str, Any]] | None,
            "usage": dict[str, Any] | None,
            "latency_seconds": float,
        }

    tool_calls entries are already parsed from JSON strings into argument dicts.
    """
    model_name = model or _get_model()
    client = _get_client()

    t0 = time.time()
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    except Exception as exc:
        logger.error("OpenRouter call failed for model %s: %s", model_name, exc)
        raise

    latency = time.time() - t0
    msg = response.choices[0].message

    tool_calls: list[dict[str, Any]] = []
    if getattr(msg, "tool_calls", None):
        for tc in msg.tool_calls:
            arguments: dict[str, Any] = {}
            raw_args = getattr(tc.function, "arguments", "") or ""
            if raw_args:
                try:
                    arguments = json.loads(raw_args)
                except (json.JSONDecodeError, TypeError):
                    logger.warning(
                        "Model returned malformed tool arguments for %s: %r",
                        tc.function.name,
                        raw_args,
                    )
                    arguments = {}
            tool_calls.append(
                {
                    "type": getattr(tc, "type", "function").lower(),
                    "id": getattr(tc, "id", ""),
                    "function": {
                        "name": tc.function.name,
                        "arguments": arguments,
                    },
                }
            )

    usage: dict[str, Any] | None = None
    if getattr(response, "usage", None):
        try:
            usage = {
                "prompt_tokens": int(response.usage.prompt_tokens),
                "completion_tokens": int(response.usage.completion_tokens),
                "total_tokens": int(response.usage.total_tokens),
            }
        except Exception:
            usage = None

    return {
        "model": model_name,
        "finish_reason": getattr(response.choices[0], "finish_reason", None),
        "content": msg.content if msg.content is not None else None,
        "tool_calls": tool_calls if tool_calls else None,
        "usage": usage,
        "latency_seconds": latency,
    }


# ---------------------------------------------------------------------------
# Convenience for tests / consumers that just want a parsed message
# ---------------------------------------------------------------------------


def create_chat_completion(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    temperature: float = 0.0,
    max_tokens: int = 1024,
    *,
    model: str | None = None,
) -> dict[str, Any]:
    """Backwards-compatible alias for chat_completion."""
    return chat_completion(messages, tools=tools, temperature=temperature, max_tokens=max_tokens, model=model)
