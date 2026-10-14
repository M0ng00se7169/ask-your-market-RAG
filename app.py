"""
Ask Your Market — Gradio chat interface.

A RAG assistant that answers retail competitive-intelligence questions from
snapshot history (query_snapshots) and semantic product descriptions
(search_descriptions), with a first-class Sources block under every answer.

Run locally:  python app.py
On HF Spaces: same entrypoint; set OPENROUTER_API_KEY as a secret.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import gradio as gr

from src import orchestrator
from src.data import load_snapshots
from src.index import get_collection
from src.tools import set_collection

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths / config from env
# ---------------------------------------------------------------------------

HERE = Path(__file__).resolve().parent
CSV_PATH = os.environ.get("SNAPSHOTS_CSV", str(HERE / "data" / "snapshots.csv"))
PERSIST_DIR = os.environ.get("CHROMA_DIR", str(HERE / "chroma_db"))
DEFAULT_EXAMPLES = os.environ.get(
    "EXAMPLE_PROMPTS",
    (
        "Which products got cheaper in the last 7 days?\n"
        "Do you have any vegetarian cookbooks?\n"
        "What's the most expensive product and is it in stock?\n"
        "What was the total revenue last week?\n"
    ),
).split("\n")


# ---------------------------------------------------------------------------
# Startup: load data + index once, fail loudly if either is missing
# ---------------------------------------------------------------------------

df: Any = None
collection: Any = None


def _load_resources() -> None:
    global df, collection
    if df is not None and collection is not None:
        return

    # Data
    try:
        df = load_snapshots(CSV_PATH)
        logger.info("Loaded snapshots: %d products, %d rows", df["url"].nunique(), len(df))
    except Exception as exc:
        logger.error("Failed to load snapshots.csv: %s", exc)
        raise

    # Vector index
    try:
        collection = get_collection(PERSIST_DIR)
        logger.info("Loaded Chroma index: %d products", collection.count())
    except Exception as exc:
        logger.error("Failed to load Chroma index: %s", exc)
        raise

    # Wire the collection into the tools layer so search_descriptions can run.
    set_collection(collection)


def _ensure_loaded() -> None:
    """Public hook so tests can trigger loading without importing the module
    globals directly."""
    _load_resources()


# ---------------------------------------------------------------------------
# Formatting helpers for the UI
# ---------------------------------------------------------------------------


def _format_price(source: dict[str, Any]) -> str:
    price = source.get("price")
    currency = source.get("currency", "")
    if price is None or not currency:
        return ""
    return f"{currency} {price:g}"


def _source_card(source: dict[str, Any]) -> dict[str, Any]:
    """One source as a Gradio-consistent card dict for the Sources block."""
    return {
        "title": source.get("title", "") or "",
        "price": _format_price(source),
        "stock": _stock_label(source),
        "promo": source.get("promo", "") or "",
        "date": source.get("scraped_at", "") or "",
        "url": source.get("url", "") or "",
        "rating": source.get("rating"),
        "value_label": source.get("value_label", "") or "",
    }


def _stock_label(source: dict[str, Any]) -> str:
    in_stock = source.get("in_stock")
    if in_stock is False:
        return "Out of stock"
    if in_stock is True:
        return "In stock"
    return ""


def _render_sources(sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert orchestrator sources into UI card dicts."""
    return [_source_card(s) for s in sources]


def _render_answer(
    answer: str,
    sources: list[dict[str, Any]],
    refusal: bool,
    tools_called: list[dict[str, Any]],
    latency: float,
) -> tuple[str, list[dict[str, Any]], str]:
    """Return (answer_html, sources_cards, status_message)."""
    cards = _render_sources(sources)

    status = ""
    if refusal:
        status = "⚠️ This is not in the data."
    elif tools_called:
        names = [tc["name"] for tc in tools_called]
        status = f"Used: {', '.join(names)}. ({latency:.1f}s)"
    else:
        status = f"Answered ({latency:.1f}s)"

    return answer, cards, status


# ---------------------------------------------------------------------------
# Chat logic
# ---------------------------------------------------------------------------


def _chat(message: str, history: list[list[str]] | None) -> tuple[str, list[dict[str, Any]], str]:
    """Gradio chatbot callback: return (reply_text, sources_cards, status)."""
    _ensure_loaded()

    t0 = time.time()
    try:
        result = orchestrator.answer(message, df=df, collection=collection)
    except Exception as exc:
        logger.exception("Orchestrator call failed")
        return (
            f"I couldn't answer that right now: {exc}",
            [],
            "Error — check the logs.",
        )

    latency = time.time() - t0
    answer_text = result.answer or ""
    sources = result.sources or []
    refusal = result.refusal

    answer, cards, status = _render_answer(
        answer_text,
        sources,
        refusal,
        result.tools_called,
        latency,
    )
    return answer, cards, status


def chat(
    message: str,
    history: list[list[Any]] | None,
) -> str:
    """Gradio ChatInterface-compatible callback.

    Returns the text to append as the assistant message. The Sources block
    is rendered separately via a post-message badge using gr.Chatbot's
    additional columns / or a companion component.

    For simplicity with ChatInterface, we return the answer text and attach
    sources as a collapsible detail block below the answer text.
    """
    answer_text, cards, status = _chat(message, history)
    if cards:
        lines = [answer_text]
        lines.append("")
        lines.append("**Sources**")
        for card in cards:
            url = card["url"]
            url_markup = f"[→ {url}]({url})" if url.startswith("http") else url
            lines.append(
                f"- **{card['title']}** — {card['value_label']} "
                f"(scraped {card['date']}) {url_markup}"
            )
        return "\n".join(lines)
    return answer_text


# ---------------------------------------------------------------------------
# Build the UI
# ---------------------------------------------------------------------------


def build_app() -> gr.Blocks:
    _load_resources()

    with gr.Blocks(title="Ask Your Market") as app:
        gr.Markdown(
            "# Ask Your Market\n"
            "Retail competitive-intelligence assistant. Answers from snapshot "
            "history and product descriptions — every answer comes with a "
            "verifiable Sources block."
        )

        status_box = gr.Textbox(label="Status", interactive=False)

        chatbot = gr.Chatbot(
            label="Chat",
            height=420,
        )

        with gr.Row():
            msg_input = gr.Textbox(
                placeholder="Ask about prices, stock, similar products…",
                label="Your question",
                lines=1,
            )

        send_btn = gr.Button("Send", variant="primary")
        clear_btn = gr.Button("Clear", variant="secondary")

        # Example prompts as quick-click chips
        gr.Examples(
            examples=[DEFAULT_EXAMPLES],
            inputs=msg_input,
            label="Example questions",
        )

        def _send(message, hist):
            if not message or not message.strip():
                return (), hist or []
            reply, cards, status = _chat(message, hist)
            # Build a rich tuple: (user_msg, assistant_msg_with_sources)
            if cards:
                source_lines = ["**Sources**"]
                for card in cards:
                    url = card["url"]
                    url_markup = f"[→ {url}]({url})" if url.startswith("http") else url
                    source_lines.append(
                        f"- **{card['title']}** — {card['value_label']} "
                        f"(scraped {card['date']}) {url_markup}"
                    )
                assistant_msg = reply + "\n\n" + "\n".join(source_lines)
            else:
                assistant_msg = reply
            new_hist = list(hist or [])
            new_hist.append([message, assistant_msg])
            return (), new_hist, status

        def _clear():
            return (), [], "Cleared."

        send_btn.click(
            fn=_send,
            inputs=[msg_input, chatbot],
            outputs=[msg_input, chatbot, status_box],
        ).then(fn=lambda: "", inputs=None, outputs=[msg_input])

        msg_input.submit(
            fn=_send,
            inputs=[msg_input, chatbot],
            outputs=[msg_input, chatbot, status_box],
        ).then(fn=lambda: "", inputs=None, outputs=[msg_input])

        clear_btn.click(fn=_clear, inputs=None, outputs=[msg_input, chatbot, status_box])

        # Status on load
        status_box.value = f"Loaded {df['url'].nunique()} products, {collection.count()} indexed."

    return app


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main() -> None:
    app = build_app()
    share = os.environ.get("GRADIO_SHARE", "").lower() in ("1", "true", "yes")
    server_name = os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0")
    server_port = int(os.environ.get("GRADIO_SERVER_PORT", "7860"))
    logger.info("Starting Gradio on %s:%d (share=%s)", server_name, server_port, share)
    app.launch(
        server_name=server_name,
        server_port=server_port,
        share=share,
        theme=gr.themes.Soft(),
    )


if __name__ == "__main__":
    main()
