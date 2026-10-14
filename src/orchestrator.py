"""
Ask Your Market — LLM orchestrator.

Combines the tool definitions (src/tools), the LLM client (src/llm), and a
system prompt into one chat turn with tool calling.

Public API (the shape the UI/eval will consume):
    answer(question, *, df=None, collection=None) -> OrchestratorResult

OrchestratorResult:
    {
        "answer": str,
        "sources": list[dict[str, Any]],
        "tools_called": list[dict[str, Any]],
        "refusal": bool,
        "model": str,
        "usage": dict[str, Any] | None,
        "latency_seconds": float,
    }

Tool calling policy (per the plan):
- Send messages + tool definitions.
- If the model requests tool calls, execute them once and append results.
- Re-prompt once if the model returned text instead of tool calls and the
  situation still seems tool-worthy (cheap retry, per Phase 0).
- After tool results are in (or after the retry), get the final answer.
- Never let the LLM invent figures or sources; citations come from tool results.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from src import llm
from src.tools import (
    TOOL_DEFS,
    run_tool,
    tool_definitions,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tool result conversion
# ---------------------------------------------------------------------------


def _tool_result_message(tool_call: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    """Build the OpenAI tool-role message for one tool call + its result."""
    return {
        "role": "tool",
        "tool_call_id": tool_call.get("id", ""),
        "name": tool_call.get("function", {}).get("name", ""),
        "content": _result_payload(result),
    }


def _result_payload(result: dict[str, Any]) -> str:
    """Serialize a tool result into a compact JSON string for the model."""
    # Keep the payload small; the orchestrator will render sources from this.
    return json.dumps(result, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Source extraction from tool results
# ---------------------------------------------------------------------------


def _sources_from_tool_results(tool_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build a flat sources list from dispatched tool results.

    This is the canonical source attribution path: sources come from tool
    results, not from the LLM's memory.
    """
    sources: list[dict[str, Any]] = []
    seen: set[str] = set()

    for result in tool_results:
        kind = result.get("question_type")
        payload = result.get("result")

        if kind == "product_history" and isinstance(payload, list):
            url = result.get("url", "")
            for row in payload:
                key = _source_key(url, row)
                if key in seen:
                    continue
                seen.add(key)
                sources.append(_source_entry(url, row))

        elif kind == "current_product" and isinstance(payload, list):
            for row in payload:
                url = row.get("url", result.get("url", ""))
                key = _source_key(url, row)
                if key in seen:
                    continue
                seen.add(key)
                sources.append(_source_entry(url, row))

        elif isinstance(payload, list):
            # price_changes / out_of_stock / most_expensive / least_expensive / current_product
            recognized = {"price_changes", "out_of_stock", "most_expensive", "least_expensive", "current_product", "product_history"}
            if kind not in recognized:
                continue
            for row in payload:
                url = row.get("url", "")
                key = _source_key(url, row)
                if key in seen:
                    continue
                seen.add(key)
                sources.append(_source_entry(url, row))

    return sources


def _source_key(url: str, row: dict[str, Any]) -> str:
    ts = row.get("scraped_at", "")
    return f"{url}\x00{ts}"


def _source_entry(url: str, row: dict[str, Any]) -> dict[str, Any]:
    return {
        "url": url or "",
        "title": row.get("title", ""),
        "scraped_at": row.get("scraped_at", ""),
        "price": row.get("price"),
        "currency": row.get("currency", ""),
        "in_stock": row.get("in_stock"),
        "promo": row.get("promo", ""),
        "rating": row.get("rating"),
        "value_label": _value_label(row),
    }


def _value_label(row: dict[str, Any]) -> str:
    """A short human label for the value shown in a source card."""
    price = row.get("price")
    currency = row.get("currency", "")
    in_stock = row.get("in_stock")
    promo = row.get("promo", "")
    rating = row.get("rating")

    parts: list[str] = []
    if price is not None and currency:
        parts.append(f"{currency} {price:g}")
    if in_stock is False:
        parts.append("out of stock")
    elif in_stock is True and not row.get("promo"):
        parts.append("in stock")
    if promo:
        parts.append(f"promo: {promo}")
    if rating is not None:
        parts.append(f"rating: {rating:g}")
    return "; ".join(parts)


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------


SYSTEM_PROMPT = (
    "You are 'Ask Your Market', a retail competitive-intelligence assistant. "
    "You answer ONLY from the results of the tools below. You do not have any "
    "other data, do not browse the web, and do not use your own training data "
    "to supply figures.\n\n"
    "Tools:\n"
    " - query_snapshots: exact numeric / availability questions (price changes "
    "over a period, out-of-stock products, product history by URL, most/least "
    "expensive products, look up a product by URL, or the available-data summary).\n"
    " - search_descriptions: semantic search over product descriptions. Use for "
    "'find products similar to X', 'what does the competitor have that is like "
    "our product', or any meaning-based question.\n"
    " - get_data_summary: what data is available (columns, date range, product "
    "count). Always useful when the user asks about the dataset or when a "
    "question looks out of scope.\n\n"
    "Rules:\n"
    "1. For exact numeric/availability questions, prefer query_snapshots.\n"
    "2. For 'similar to' / meaning questions, prefer search_descriptions.\n"
    "3. For mixed questions, call both.\n"
    "4. If a question asks for sales volume, revenue, forecasts, competitor "
    "internal info, or anything not in the data, refuse explicitly: say this is "
    "not in the data and give a one-line reason. Do not hedge with made-up data.\n"
    "5. Always cite sources from the tool results. A source is a product title, "
    "the snapshot date (scraped_at), and the URL. State the exact value (price, "
    "rating, stock status, promo) from that snapshot; copy figures verbatim, do "
    "not round them inventively.\n"
    "6. Keep the answer concise and factual. If multiple products match, list the "
    "most relevant ones and note the total count from the tool result.\n"
    "7. If the tools return nothing relevant, say so explicitly: 'This is not in "
    "the data.' with a short reason.\n\n"
    "When you call tools, return tool_calls with valid JSON arguments. If the user "
    "question is clearly out of scope and you already know from get_data_summary, "
    "you may refuse without calling more tools."
)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def answer(
    question: str,
    *,
    df: Any = None,
    collection: Any = None,
    temperature: float = 0.0,
    max_tokens: int = 1024,
    max_tool_calls: int = 1,
) -> OrchestratorResult:
    """Run one user question through the LLM with tool calling.

    Parameters
    ----------
    question : str
        The user's question.
    df : DataFrame, optional
        Snapshots DataFrame for query_snapshots / get_data_summary.
    collection : Chroma collection, optional
        Set into tools so search_descriptions can run.
    temperature : float
        Sampling temperature for the LLM.
    max_tokens : int
        Max tokens for the LLM responses.
    max_tool_calls : int
        Cap on distinct tool calls to execute per turn (kept at 1 for the MVP
        to bound latency; the policy documents that both tools may be called
        when the question is mixed, and that is fine under this cap because the
        model typically emits both in one turn).
    """
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    tools = tool_definitions()

    # Wire the runtime collection singleton so search_descriptions can resolve
    # it even when the orchestrator does not pass it through run_tool.
    if collection is not None:
        from src.tools import set_collection as _set
        _set(collection)
    # Also make the collection visible to run_tool's search_descriptions path via a
    # module-level stash so tests don't need a real Chroma collection.
    from src.tools import _stash_collection_for_tests
    _stash_collection_for_tests(collection)

    # First turn.
    resp = _llm_turn(messages, tools, temperature=temperature, max_tokens=max_tokens)
    tool_results: list[dict[str, Any]] = []
    tool_calls_emitted: list[dict[str, Any]] = []

    if resp["tool_calls"]:
        # Execute each requested tool once.
        executed = 0
        for tc in resp["tool_calls"]:
            if executed >= max_tool_calls:
                break
            name = tc.get("function", {}).get("name", "")
            args = tc.get("function", {}).get("arguments", {})
            if not _valid_tool_call(name, args):
                logger.warning("Ignoring malformed tool call: %s", tc)
                continue
            call = {
                "name": name,
                "arguments": args,
                "id": tc.get("id", ""),
                "type": tc.get("type", "function"),
            }
            result = run_tool(call, df=df)
            tool_results.append(result)
            tool_calls_emitted.append(call)
            messages.append({"role": "assistant", "content": None, "tool_calls": [tc]})
            messages.append(_tool_result_message(call, result))
            executed += 1

    # If the model emitted no tool calls and we still expect tools for a
    # non-trivial question, re-prompt once (cheap retry, documented in Phase 0).
    re_prompted = False
    if not tool_calls_emitted and _looks_tool_worthy(question):
        messages.append(
            {"role": "user", "content": "Use the tools above to answer the question. "
             "Call query_snapshots, search_descriptions, or get_data_summary as appropriate. "
             "Return tool_calls."}
        )
        resp = _llm_turn(messages, tools, temperature=temperature, max_tokens=max_tokens)
        re_prompted = True

        if resp["tool_calls"]:
            executed = 0
            for tc in resp["tool_calls"]:
                if executed >= max_tool_calls:
                    break
                name = tc.get("function", {}).get("name", "")
                args = tc.get("function", {}).get("arguments", {})
                if not _valid_tool_call(name, args):
                    logger.warning("Ignoring malformed tool call on retry: %s", tc)
                    continue
                call = {
                    "name": name,
                    "arguments": args,
                    "id": tc.get("id", ""),
                    "type": tc.get("type", "function"),
                }
                result = run_tool(call, df=df)
                tool_results.append(result)
                tool_calls_emitted.append(call)
                messages.append({"role": "assistant", "content": None, "tool_calls": [tc]})
                messages.append(_tool_result_message(call, result))
                executed += 1

    # Build the final answer message including any tool results we have.
    final_messages = list(messages)
    if tool_results:
        # Append a final reminder that ties tools to the answer we expect.
        final_messages.append(
            {"role": "user", "content": (
                "Based on the tool results above, write the final answer. Include the "
                "exact values and sources from the tool results. If the results are empty "
                "or irrelevant, refuse. Do not invent figures or sources."
            )}
        )
    else:
        final_messages.append(
            {"role": "user", "content": (
                "Answer the user's question from the data summary above if it helps, "
                "otherwise refuse. Do not invent data."
            )}
        )

    final_resp = _llm_turn(final_messages, tools=None, temperature=temperature, max_tokens=max_tokens)
    answer_text = final_resp["content"] or ""

    refusal = _is_refusal(answer_text)

    sources = _sources_from_tool_results(tool_results) if tool_results else []

    return OrchestratorResult(
        answer=answer_text,
        sources=sources,
        tools_called=list(tool_calls_emitted),
        refusal=refusal,
        model=resp.get("model", "unknown"),
        usage=final_resp.get("usage") or resp.get("usage"),
        latency_seconds=resp.get("latency_seconds", 0.0) + final_resp.get("latency_seconds", 0.0),
        re_prompted=re_prompted,
    )


def _llm_turn(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    *,
    temperature: float,
    max_tokens: int,
) -> dict[str, Any]:
    try:
        return llm.chat_completion(
            messages,
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    except Exception:
        logger.exception("LLM turn failed")
        return {
            "model": "error",
            "finish_reason": "error",
            "content": "I could not reach the language model right now.",
            "tool_calls": None,
            "usage": None,
            "latency_seconds": 0.0,
        }


def _valid_tool_call(name: str, args: Any) -> bool:
    if not name:
        return False
    if not isinstance(args, dict):
        return False
    # Only known tool names are executed; unknown names are skipped.
    if name not in {"query_snapshots", "search_descriptions", "get_data_summary"}:
        return False
    return True


def _looks_tool_worthy(question: str) -> bool:
    """Heuristic: is this a question where the LLM probably should call a tool?

    This is deliberately conservative; it only decides on re-prompting, not on
    whether a tool is strictly required.
    """
    q = question.strip().lower()
    if not q:
        return False
    # Out-of-scope keywords -> we probably do NOT want tools, we want a refusal.
    out_of_scope = ("sales", "revenue", "forecast", "predict", "estimate revenue",
                    "how many sold", "units sold", "competitor internal")
    if any(token in q for token in out_of_scope):
        return False
    # Anything that smells like a product/price/availability/semantic query is
    # tool-worthy.
    return any(token in q for token in (
        "price", "cost", "cheap", "expensive", "stock", "available", "out of stock",
        "history", "changed", "drop", "raise", "increase", "decrease",
        "similar", "like", "recommend", "alternatives", "products",
        "what does", "have", "rating", "review", "description",
    ))


def _is_refusal(text: str) -> bool:
    t = text.strip().lower()
    if not t:
        return False
    refusal_markers = (
        "not in the data", "not available", "i don't have", "i do not have",
        "not tracked", "cannot answer", "can't answer", "out of scope",
        "does not have", "no data", "no information", "cannot provide",
        "can't provide",
    )
    return any(marker in t for marker in refusal_markers)


# ---------------------------------------------------------------------------
# Structured return type (plain dict for simplicity; kept serializable)
# ---------------------------------------------------------------------------


class OrchestratorResult(dict):
    """Structured result: answer + sources + tool-calling trace."""

    answer: str
    sources: list[dict[str, Any]]
    tools_called: list[dict[str, Any]]
    refusal: bool
    model: str
    usage: dict[str, Any] | None
    latency_seconds: float
    re_prompted: bool

    def __init__(
        self,
        *,
        answer: str,
        sources: list[dict[str, Any]],
        tools_called: list[dict[str, Any]],
        refusal: bool,
        model: str,
        usage: dict[str, Any] | None,
        latency_seconds: float,
        re_prompted: bool = False,
    ) -> None:
        super().__init__(
            answer=answer,
            sources=sources,
            tools_called=tools_called,
            refusal=refusal,
            model=model,
            usage=usage,
            latency_seconds=latency_seconds,
            re_prompted=re_prompted,
        )
        # Stash typed attributes too for consumers that prefer attribute access.
        self.answer = answer
        self.sources = sources
        self.tools_called = tools_called
        self.refusal = refusal
        self.model = model
        self.usage = usage
        self.latency_seconds = latency_seconds
        self.re_prompted = re_prompted
