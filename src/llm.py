"""
Ask Your Market — LLM client.

OpenAI-compatible client for OpenRouter. The model and key come from the
environment / .env (OPENROUTER_API_KEY, OPENROUTER_MODEL). This is the
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
- The openai SDK's built-in retry only covers timeout and "request" exceptions
  (httpx2.RequestError family). Mid-stream disconnects surface as
  httpcore.RemoteProtocolError, which the SDK raises as APIConnectionError but
  does *not* retry. We therefore retry at the application level and build a
  fresh client per attempt.

Phase 3b output alongside src/orchestrator.py.
"""

from __future__ import annotations

import httpcore
import json
import logging
import os
import time
from typing import Any

import httpx
from httpx import Timeout
from openai import OpenAI
from dotenv import load_dotenv

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

# Timeouts tuned for OpenRouter free-tier latency + transient disconnects.
# httpx `Timeout(...)` here is converted by the openai SDK into httpx2 timeouts
# (connect / read / write / pool), so a flaky connection does not burn the full
# 60s with no retry.
_TIMEOUT = Timeout(connect=10.0, read=30.0, write=30.0, pool=10.0)

_ADVERTISED_MODEL_FOR_LOGS: str = "nvidia/nemotron-3.5-lightning:free"


# ---------------------------------------------------------------------------
# Retry taxonomy
# ---------------------------------------------------------------------------

# Exceptions that indicate a transient network/protocol failure worth retrying
# even though the openai SDK's built-in retry does not cover them.
#
# The SDK retries timeout_exceptions() (httpx2.TimeoutException) and
# request_exceptions() (httpx2.RequestError, ssl.SSLError, anyio.EndOfStream).
# httpcore.RemoteProtocolError / ProtocolError are *not* in either set, so a
# mid-stream disconnect from OpenRouter escapes the SDK retry and surfaces
# as an unhandled APIConnectionError. We treat those as retryable here.
#
# We intentionally do NOT retry on the first attempt's APIConnectionError when
# it came from a non-transient cause; the class is too broad. We only retry the
# protocol-error shapes we can identify as transient.
PROTOCOL_ERRORS = (httpcore.RemoteProtocolError, httpcore.ProtocolError)


def _is_transient_protocol_error(exc: BaseException) -> bool:
    """True for the httpcore protocol errors we want to retry."""
    e = exc
    while e is not None:
        if isinstance(e, PROTOCOL_ERRORS):
            return True
        # Some transports wrap the underlying httpcore error; walk up.
        e = getattr(e, "__cause__", None) or getattr(e, "__context__", None)
    return False


def _build_client() -> OpenAI:
    """Build a fresh OpenAI client for one attempt.

    We recreate the client per retry attempt so the SDK's per-client retry
    state (idempotency keys, request-body replay, etc.) does not interfere
    with our application-level retry on protocol errors.
    """
    key = _get_api_key()
    if not key or key.startswith("sk-or-xxx"):
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set or still has the template value. "
            "Copy .env.example to .env and fill in a real key."
        )
    return OpenAI(
        api_key=key,
        base_url=_BASE_URL,
        timeout=_TIMEOUT,
        max_retries=2,
        http_client=httpx.Client(
            timeout=_TIMEOUT,
            verify=True,
            follow_redirects=True,
        ),
    )


# ---------------------------------------------------------------------------
# Env helpers
# ---------------------------------------------------------------------------


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

    This wrapper retries on transient httpcore protocol errors (mid-stream
    disconnects) that the openai SDK does not retry, using a fresh client per
    attempt. It does not retry on streamed responses (this app never streams).
    """
    model_name = model or _get_model()

    t0 = time.time()
    last_exc: Exception | None = None

    # Application-level retry because the SDK's built-in retry does not cover
    # httpcore.RemoteProtocolError / ProtocolError.
    for attempt in range(1, 3):
        client = _build_client()

        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                tools=tools,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception as exc:
            logger.error(
                "OpenRouter call failed for model %s (attempt %d/2): %s: %s",
                model_name,
                attempt,
                type(exc).__name__,
                exc,
            )

            # If the SDK folded a transient protocol error into an
            # APIConnectionError, walk the causal chain and decide whether to
            # retry.
            if _is_transient_protocol_error(exc):
                last_exc = exc
                if attempt == 2:
                    raise
                time.sleep(1.0 * attempt)
                continue

            # Everything else (auth, 4xx, 5xx status, malformed request, etc.)
            # is not worth an application-level retry here — let it propagate so
            # the orchestrator can decide how to handle it.
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

    # Should be unreachable: if both attempts raise a transient protocol error,
    # the second attempt re-raises.
    if last_exc is not None:
        raise last_exc

    # Fallback: model/tool choice returned without a real response is not
    # expected; fail loudly rather than inventing one.
    raise RuntimeError("chat_completion exited the retry loop without returning.")


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
    return chat_completion(
        messages, tools=tools, temperature=temperature, max_tokens=max_tokens, model=model
    )
