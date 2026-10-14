"""
Ask Your Market — tool definitions and dispatch.

Two core tools (query_snapshots, search_descriptions) plus an identity tool
(get_data_summary) so the LLM can see what's in scope.

Tool outputs are small, typed dicts — the LLM layer never sees raw 100-row
tables. The orchestrator wraps these into OpenAI tool-call format; this
module is the pure dispatch layer.
"""

from __future__ import annotations

from typing import Any, Callable

from src.data import (
    COLUMNS,
    available_data_descriptor,
    data_summary,
    find_by_url,
    least_expensive,
    load_snapshots,
    most_expensive,
    out_of_stock,
    price_change_over_period,
    product_history,
)
from src.index import search_descriptions


# ---------------------------------------------------------------------------
# Tool definitions (schema the LLM sees)
# ---------------------------------------------------------------------------

QUERY_SNAPSHOTS_DESCRIPTION = (
    "Query the snapshot history table. Use for exact numeric/availability "
    "questions: price changes over a period, what is out of stock, product "
    "history by URL, most/least expensive products, look up a product by URL, "
    "or the available data summary."
)

QUERY_SNAPSHOTS_ARGS = {
    "type": "object",
    "properties": {
        "question_type": {
            "type": "string",
            "description": (
                "One of: price_changes, out_of_stock, product_history, "
                "most_expensive, least_expensive, current_product, available_data."
            ),
            "enum": [
                "price_changes",
                "out_of_stock",
                "product_history",
                "most_expensive",
                "least_expensive",
                "current_product",
                "available_data",
            ],
        },
        "params": {
            "type": "object",
            "description": "Type-specific parameters.",
            "properties": {
                "days": {
                    "type": "integer",
                    "description": "Window in days for price_changes (e.g. 7).",
                },
                "url": {
                    "type": "string",
                    "description": "Product URL for product_history or current_product.",
                },
                "n": {
                    "type": "integer",
                    "description": "How many products for most_expensive / least_expensive (default 20).",
                },
            },
            "additionalProperties": False,
        },
    },
    "required": ["question_type"],
    "additionalProperties": False,
}

SEARCH_DESCRIPTIONS_DESCRIPTION = (
    "Semantic search over product descriptions. Use for 'find products similar "
    "to X', 'what does the competitor have that is like our product', or any "
    "question answered by meaning rather than exact numbers."
)

SEARCH_DESCRIPTIONS_ARGS = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": "Search query in natural language (e.g. 'vegetarian cookbook for busy weeknights').",
        },
        "top_k": {
            "type": "integer",
            "description": "Max results to return (default 10, cap 20).",
        },
    },
    "required": ["query"],
    "additionalProperties": False,
}

GET_DATA_SUMMARY_DESCRIPTION = (
    "Return a summary of what data is available: columns, date range, product "
    "count, and what the assistant can and cannot answer. Always call this "
    "first when the user asks about the dataset itself or an out-of-scope "
    "question."
)

GET_DATA_SUMMARY_ARGS = {
    "type": "object",
    "properties": {
        "verbose": {
            "type": "boolean",
            "description": "If True, include the full note about scope limits.",
        },
    },
    "required": [],
    "additionalProperties": False,
}


# Registry of tool definitions + their Python callables.
# The orchestrator builds the OpenAI tools list from TOOL_DEFS.
ToolDef = dict[str, Any]

TOOL_DEFS: list[ToolDef] = [
    {
        "name": "query_snapshots",
        "description": QUERY_SNAPSHOTS_DESCRIPTION,
        "parameters": QUERY_SNAPSHOTS_ARGS,
    },
    {
        "name": "search_descriptions",
        "description": SEARCH_DESCRIPTIONS_DESCRIPTION,
        "parameters": SEARCH_DESCRIPTIONS_ARGS,
    },
    {
        "name": "get_data_summary",
        "description": GET_DATA_SUMMARY_DESCRIPTION,
        "parameters": GET_DATA_SUMMARY_ARGS,
    },
]


def tool_definitions() -> list[dict[str, Any]]:
    """OpenAI-format tool list for the orchestrator / LLM client."""
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["parameters"],  # already in OpenAI format
            },
        }
        for t in TOOL_DEFS
    ]


# Required fields are now embedded in each tool's args schema.
# _required_for_tool kept for any downstream consumers that need the list.
def _required_for_tool(name: str) -> list[str]:
    """Required fields per tool name (mirrors the schema)."""
    if name == "query_snapshots":
        return ["question_type"]
    if name == "search_descriptions":
        return ["query"]
    if name == "get_data_summary":
        return []
    return []


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

# Default caps to keep prompts small.
_DEFAULT_TOP_K = 10
_MAX_TOP_K = 20
_DEFAULT_N = 20
_MAX_N = 50


def _cap_int(value: Any, default: int, maximum: int) -> int:
    if value is None:
        return default
    try:
        v = int(value)
    except (TypeError, ValueError):
        return default
    if v <= 0:
        return default
    return min(v, maximum)


def run_query_snapshots(
    question_type: str,
    params: dict[str, Any] | None = None,
    *,
    df: Any = None,
) -> dict[str, Any]:
    """Dispatch query_snapshots to the data module.

    Returns a compact dict: {"question_type": ..., "result": <rows or summary>,
    "note": optional human note}. Results are capped lists of dicts.
    """
    params = params or {}
    if df is None:
        df = load_snapshots()

    if question_type == "price_changes":
        days = _cap_int(params.get("days"), 7, 90)
        changed = price_change_over_period(df, days)
        rows = _df_to_records(changed[["url", "title", "price", "price_start", "price_delta"]])
        return {
            "question_type": "price_changes",
            "days": days,
            "result": rows,
            "note": f"{len(rows)} products changed price in the last {days} days." if rows else f"No products changed price in the last {days} days.",
        }

    if question_type == "out_of_stock":
        oos = out_of_stock(df)
        rows = _df_to_records(oos[["url", "title", "in_stock", "price", "promo"]])
        return {
            "question_type": "out_of_stock",
            "result": rows,
            "note": f"{len(rows)} products are currently out of stock." if rows else "No products are currently out of stock.",
        }

    if question_type == "product_history":
        url = str(params.get("url", "")).strip()
        if not url:
            return {"question_type": "product_history", "error": "url is required for product_history."}
        hist = product_history(df, url)
        if hist.empty:
            return {"question_type": "product_history", "error": f"No snapshots found for {url}.", "url": url}
        rows = _df_to_records(hist[["scraped_at", "price", "in_stock", "promo", "rating"]])
        return {
            "question_type": "product_history",
            "url": url,
            "result": rows,
            "note": f"{len(rows)} snapshots for {url}.",
        }

    if question_type == "most_expensive":
        n = _cap_int(params.get("n"), _DEFAULT_N, _MAX_N)
        top = most_expensive(df, n)
        rows = _df_to_records(top[["url", "title", "price", "currency", "in_stock"]])
        return {
            "question_type": "most_expensive",
            "n": n,
            "result": rows,
            "note": f"Top {len(rows)} most expensive products.",
        }

    if question_type == "least_expensive":
        n = _cap_int(params.get("n"), _DEFAULT_N, _MAX_N)
        bottom = least_expensive(df, n)
        rows = _df_to_records(bottom[["url", "title", "price", "currency", "in_stock"]])
        return {
            "question_type": "least_expensive",
            "n": n,
            "result": rows,
            "note": f"Top {len(rows)} least expensive products.",
        }

    if question_type == "current_product":
        url = str(params.get("url", "")).strip()
        if not url:
            return {"question_type": "current_product", "error": "url is required for current_product."}
        row = find_by_url(df, url)
        if row is None:
            return {"question_type": "current_product", "error": f"No product found for {url}.", "url": url}
        rec = _row_to_record(row)
        return {
            "question_type": "current_product",
            "url": url,
            "result": [rec],
            "note": f"Current snapshot for {url}.",
        }

    if question_type == "available_data":
        summary = data_summary(df)
        return {
            "question_type": "available_data",
            "result": summary,
            "note": "Data summary.",
        }

    return {"question_type": question_type, "error": f"Unknown question_type '{question_type}'."}


def run_search_descriptions(
    query: str,
    top_k: int | None = None,
    *,
    collection: Any = None,
) -> dict[str, Any]:
    """Dispatch search_descriptions.

    If collection is passed explicitly, uses it. Otherwise falls back to the
    singleton set via set_collection(). Raises if neither is available.
    Returns {"query": ..., "top_k": ..., "result": [records], "count": ..., "note": ...}.
    """
    top_k = _cap_int(top_k, _DEFAULT_TOP_K, _MAX_TOP_K)
    if not query or not query.strip():
        return {"query": query, "top_k": top_k, "result": [], "count": 0, "note": "Empty query."}
    coll = collection if collection is not None else _resolve_collection()
    results = search_descriptions(coll, query.strip(), top_k)
    return {
        "query": query.strip(),
        "top_k": top_k,
        "result": results,
        "count": len(results),
        "note": f"{len(results)} products found for '{query.strip()}'." if results else f"No products found for '{query.strip()}'.",
    }


def run_get_data_summary(
    verbose: bool | None = None,
    *,
    df: Any = None,
) -> dict[str, Any]:
    """Dispatch get_data_summary."""
    verbose = bool(verbose) if verbose is not None else False
    if df is None:
        from src.data import available_data_descriptor as fallback
        summary = fallback()
    else:
        summary = data_summary(df)
    out = {"verbose": verbose, "result": summary}
    if verbose:
        out["note"] = summary.get("note", "")
    else:
        out["note"] = "Data summary (use verbose=True to see scope limits)."
    return out


# ---------------------------------------------------------------------------
# Dependency injection for the index (so tests don't need a real model)
# ---------------------------------------------------------------------------

_collection_singleton: Any = None


def set_collection(collection: Any) -> None:
    """Register the Chroma collection for search_descriptions at runtime."""
    global _collection_singleton
    _collection_singleton = collection


def clear_collection() -> None:
    global _collection_singleton
    _collection_singleton = None


def _resolve_collection() -> Any:
    if _collection_singleton is not None:
        return _collection_singleton
    raise RuntimeError("No Chroma collection set. Call set_collection() before search_descriptions.")


# Monkey-patch the search path used by run_search_descriptions so the default
# collection comes from the singleton instead of requiring it as an arg every time.
# We keep the explicit collection= param for tests; the runtime path uses the singleton.
def _search_with_default(query: str, top_k: int | None = None) -> dict[str, Any]:
    return run_search_descriptions(query, top_k, collection=_resolve_collection())


# Expose a convenience that the orchestrator/runtime uses.
search_descriptions_tool = _search_with_default


# ---------------------------------------------------------------------------
# Format helpers (LLM-friendly structures)
# ---------------------------------------------------------------------------

def _df_to_records(df: Any) -> list[dict[str, Any]]:
    """Convert a DataFrame slice to a list of dicts, NaN-safe."""
    if df is None or df.empty:
        return []
    records = []
    for _, row in df.iterrows():
        rec: dict[str, Any] = {}
        for col in df.columns:
            val = row[col]
            if val is None:
                rec[col] = None
            elif hasattr(val, "item"):
                try:
                    rec[col] = val.item()
                except Exception:
                    rec[col] = str(val)
            elif isinstance(val, float) and val != val:  # NaN
                rec[col] = None
            else:
                rec[col] = val
        records.append(rec)
    return records


def _row_to_record(row: Any) -> dict[str, Any]:
    """Convert a single Series (latest snapshot row) to a dict."""
    rec: dict[str, Any] = {}
    for col in row.index:
        val = row[col]
        if val is None:
            rec[col] = None
        elif hasattr(val, "item"):
            try:
                rec[col] = val.item()
            except Exception:
                rec[col] = str(val)
        elif isinstance(val, float) and val != val:
            rec[col] = None
        else:
            rec[col] = val
    return rec


# ---------------------------------------------------------------------------
# Tool runner: given a tool call (name + args), execute and return a result dict
# ---------------------------------------------------------------------------

def run_tool(
    tool_call: dict[str, Any],
    *,
    df: Any = None,
) -> dict[str, Any]:
    """Execute one tool call from the LLM.

    tool_call: {"name": ..., "arguments": {...}}.
    Returns a dict the orchestrator can turn into a tool_result message.
    """
    name = tool_call.get("name")
    args = tool_call.get("arguments", {})
    if name == "query_snapshots":
        return run_query_snapshots(
            question_type=args.get("question_type", ""),
            params=args.get("params"),
            df=df,
        )
    if name == "search_descriptions":
        return run_search_descriptions(
            query=args.get("query", ""),
            top_k=args.get("top_k"),
        )
    if name == "get_data_summary":
        return run_get_data_summary(
            verbose=args.get("verbose", False),
            df=df,
        )
    return {"error": f"Unknown tool '{name}'."}
