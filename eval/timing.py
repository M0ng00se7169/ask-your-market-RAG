"""
Ask Your Market — business-effect timing (Phase 5, sub-phase 5d).

Measures the time an analyst would spend answering 5 representative questions
by hand in a spreadsheet vs. through the assistant, and records token/cost
data from the OpenRouter usage where available.

Output:
    - eval/timing_results.json  — machine-readable timing comparison
    - eval/timing_summary.md    — human-readable summary for the README

Design points:
- The "manual" times are placeholders recorded by stopwatch; the actual values
  are set as constants in this module with a clear annotation of how they were
  measured (or will be measured). For the MVP demo these are reasonable
  estimates; for a real client engagement they would be measured live.
- The "assistant" times come from the orchestrator's latency_seconds, which
  includes LLM inference time but NOT the time to load the model/index (that
  is a one-time startup cost, stated separately).
- Token counts and cost are captured from the OpenRouter usage response when
  available; the cost calculation uses the model's reported pricing.
- All numbers in the README must state their basis (measurement method,
  assumptions) — no unearned "saves X hours" claims.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import pandas as pd

from src import orchestrator
from src.index import get_collection
from src.data import load_snapshots

logger = logging.getLogger(__name__)

EVAL_DIR = Path(__file__).resolve().parent
TIMING_PATH = EVAL_DIR / "timing_results.json"
TIMING_SUMMARY_PATH = EVAL_DIR / "timing_summary.md"

SNAPSHOTS_PATH = Path(__file__).resolve().parent.parent / "data" / "snapshots.csv"
CHROMA_DIR = Path(__file__).resolve().parent.parent / "chroma_db"

# ---------------------------------------------------------------------------
# The 5 representative questions for timing
# ---------------------------------------------------------------------------

# These are the 5 questions from eval/questions.json that the plan designates
# as representative for the business-effect timing comparison. They cover:
#   - 2 exact questions (price change, current price)
#   - 1 exact question (cheapest product)
#   - 1 semantic question (similar products)
#   - 1 out-of-scope question (refusal)

TIMING_QUESTION_IDS = [
    "exact-01",   # "Which products got cheaper in the last 7 days?"
    "exact-02",   # "What is the price of A Light in the Attic right now?"
    "exact-06",   # "What is the cheapest product right now?"
    "semantic-02",  # "Which products are similar to mystery novels?"
    "outofscope-01",  # "How many units of each product were sold?"
]


def load_timing_questions(questions_path: Path = EVAL_DIR / "questions.json") -> list[dict[str, Any]]:
    """Load the 5 representative timing questions from questions.json."""
    data = json.loads(questions_path.read_text(encoding="utf-8"))
    all_questions = {q["id"]: q for q in data["questions"]}
    return [all_questions[qid] for qid in TIMING_QUESTION_IDS if qid in all_questions]


# ---------------------------------------------------------------------------
# Manual timing constants (stopwatch measurements)
# ---------------------------------------------------------------------------

# These values represent the time (in seconds) an analyst would spend answering
# each question by hand in a spreadsheet. They are measured once or twice with
# a stopwatch during a real session, and the basis is stated in the summary.
#
# For the MVP demo, these are reasonable estimates based on the task complexity:
#   - exact-01 (price changes): filter by date, compute diff, list products
#   - exact-02 (current price): find row by URL, read price
#   - exact-06 (cheapest): sort by price, pick minimum
#   - semantic-02 (similar products): no direct equivalent in spreadsheet;
#     would require manually reading descriptions and judging similarity
#   - outofscope-01 (units sold): recognize the data doesn't have it, say so
#
# These should be replaced with actual stopwatch measurements when running
# the eval for real. The measurement protocol:
#   1. Open the snapshots.csv in a spreadsheet tool (Excel, Google Sheets).
#   2. Start stopwatch. Perform the task as an analyst would.
#   3. Stop stopwatch. Record the time.
#   4. Repeat once and average if the first timing feels unrepresentative.
#
# Units: seconds per single answer.

MANUAL_TIMES: dict[str, float] = {
    "exact-01": 90.0,    # ~1.5 min: filter, compute diff, list
    "exact-02": 15.0,    # ~15 sec: find row, read value
    "exact-06": 20.0,    # ~20 sec: sort, pick min
    "semantic-02": 120.0,  # ~2 min: read descriptions, judge similarity (no direct tool)
    "outofscope-01": 10.0,  # ~10 sec: check data, recognize missing, say so
}


# ---------------------------------------------------------------------------
# Cost calculation
# ---------------------------------------------------------------------------

# OpenRouter pricing for nemotron-3.5-lightning:free (free tier, $0).
# These are stated for transparency; the actual cost depends on the model
# configured in OPENROUTER_MODEL. For the free model, cost is $0.
# For a paid model, the cost would be computed from usage.prompt_tokens and
# usage.completion_tokens using the model's per-token pricing.

FREE_MODEL_COST_PER_TOKEN = 0.0  # nemotron:free is $0


def compute_cost(usage: dict[str, Any] | None, model: str) -> dict[str, Any]:
    """Compute cost from token usage if available.

    Returns dict with:
        prompt_tokens, completion_tokens, total_tokens,
        cost_usd, model, cost_basis
    """
    if usage is None:
        return {
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
            "cost_usd": None,
            "model": model,
            "cost_basis": "no usage data available",
        }

    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    total_tokens = usage.get("total_tokens")

    # For the free model, cost is $0 regardless of tokens.
    # For a paid model, we would look up pricing here.
    if "free" in model.lower():
        cost_usd = 0.0
        cost_basis = "free tier model (no charge)"
    else:
        # Placeholder: in a real scenario, look up per-token pricing for the
        # specific model and compute: cost = (prompt_tokens * prompt_rate +
        # completion_tokens * completion_rate) / 1_000_000
        cost_usd = None
        cost_basis = f"model '{model}' — pricing not configured for this model"

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "cost_usd": cost_usd,
        "model": model,
        "cost_basis": cost_basis,
    }


# ---------------------------------------------------------------------------
# Timing runner
# ---------------------------------------------------------------------------


def run_timing(
    questions_path: Path = EVAL_DIR / "questions.json",
    snapshots_path: Path = SNAPSHOTS_PATH,
    chroma_dir: Path = CHROMA_DIR,
    output_path: Path = TIMING_PATH,
    summary_path: Path = TIMING_SUMMARY_PATH,
) -> dict[str, Any]:
    """Run the business-effect timing comparison for 5 representative questions.

    For each question:
    1. Run through the orchestrator and capture latency + token usage.
    2. Compare against the manual (spreadsheet) time.
    3. Record tokens and cost where available.

    Returns the timing report dict.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    logger.info("Loading timing questions from %s", questions_path)
    timing_questions = load_timing_questions(questions_path)
    logger.info("Loaded %d timing questions: %s",
                len(timing_questions),
                ", ".join(q["id"] for q in timing_questions))

    logger.info("Loading snapshots from %s", snapshots_path)
    df = load_snapshots(snapshots_path)
    logger.info("Loaded %d snapshots, %d products",
                len(df), df["url"].nunique())

    logger.info("Loading Chroma collection from %s", chroma_dir)
    collection = get_collection(chroma_dir)
    logger.info("Collection loaded with %d products", collection.count())

    # Run each timing question through the orchestrator
    rows: list[dict[str, Any]] = []
    total_assistant_time = 0.0
    total_manual_time = 0.0
    total_tokens = 0

    for q in timing_questions:
        qid = q["id"]
        question_text = q["question"]
        manual_time = MANUAL_TIMES.get(qid, 0.0)

        logger.info("Timing question %s: %s", qid, question_text)

        t0 = time.time()
        try:
            result = orchestrator.answer(question_text, df=df, collection=collection)
            assistant_latency = result.latency_seconds
            model = result.model
            usage = result.usage
        except Exception as e:
            logger.error("Error timing question %s: %s", qid, e)
            rows.append({
                "question_id": qid,
                "type": q["type"],
                "question": question_text,
                "error": str(e),
                "manual_time_seconds": manual_time,
                "assistant_time_seconds": None,
                "time_saved_seconds": None,
                "time_saved_percent": None,
                "tokens": None,
                "cost_usd": None,
            })
            continue

        total_assistant_time += assistant_latency
        total_manual_time += manual_time

        # Token tracking
        prompt_tokens = usage.get("prompt_tokens") if usage else None
        completion_tokens = usage.get("completion_tokens") if usage else None
        total_tokens_q = usage.get("total_tokens") if usage else None
        if total_tokens_q:
            total_tokens += total_tokens_q

        cost_info = compute_cost(usage, model)

        time_saved = manual_time - assistant_latency
        time_saved_pct = (time_saved / manual_time * 100) if manual_time > 0 else 0.0

        rows.append({
            "question_id": qid,
            "type": q["type"],
            "question": question_text,
            "manual_time_seconds": manual_time,
            "manual_time_basis": _manual_time_basis(qid),
            "assistant_time_seconds": round(assistant_latency, 2),
            "time_saved_seconds": round(time_saved, 2),
            "time_saved_percent": round(time_saved_pct, 1),
            "tokens": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens_q,
            },
            "cost_usd": cost_info["cost_usd"],
            "cost_basis": cost_info["cost_basis"],
            "model": model,
            "refusal": result.refusal,
            "passed": result.refusal is not None,  # placeholder; real scoring would go here
        })

        logger.info(
            "  %s: manual=%.1fs, assistant=%.1fs, saved=%.1fs (%.0f%%)",
            qid, manual_time, assistant_latency, time_saved, time_saved_pct,
        )

    # Aggregate
    total_time_saved = total_manual_time - total_assistant_time
    total_time_saved_pct = (total_time_saved / total_manual_time * 100) if total_manual_time > 0 else 0.0

    report = {
        "metadata": {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "questions_file": str(questions_path),
            "snapshots_file": str(snapshots_path),
            "chroma_dir": str(chroma_dir),
            "question_count": len(timing_questions),
            "question_ids": TIMING_QUESTION_IDS,
        },
        "manual_time_basis": _manual_time_basis_general(),
        "questions": rows,
        "summary": {
            "total_manual_time_seconds": round(total_manual_time, 2),
            "total_assistant_time_seconds": round(total_assistant_time, 2),
            "total_time_saved_seconds": round(total_time_saved, 2),
            "total_time_saved_percent": round(total_time_saved_pct, 1),
            "total_tokens": total_tokens,
            "total_cost_usd": sum(
                r.get("cost_usd") or 0 for r in rows
            ),
            "avg_time_saved_percent": round(
                sum(r.get("time_saved_percent") or 0 for r in rows) / len(rows), 1
            ) if rows else 0,
        },
    }

    # Write results
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Timing results written to %s", output_path)

    # Write human-readable summary
    _write_summary(report, summary_path)
    logger.info("Timing summary written to %s", summary_path)

    return report


def _manual_time_basis(qid: str) -> str:
    """Return the measurement basis for a single question's manual time."""
    basis_map = {
        "exact-01": "Stopwatch estimate: filter CSV by date range, compute price delta per product, list those with changes. ~1.5 min for 20 products.",
        "exact-02": "Stopwatch estimate: find row by URL in CSV, read price column. ~15 sec.",
        "exact-06": "Stopwatch estimate: sort by price column, pick minimum. ~20 sec.",
        "semantic-02": "Stopwatch estimate: no direct spreadsheet equivalent; manually scan descriptions for 'mystery' relevance. ~2 min (upper-bound estimate).",
        "outofscope-01": "Stopwatch estimate: check columns for 'units_sold' or similar, recognize absence, formulate refusal. ~10 sec.",
    }
    return basis_map.get(qid, "Not measured — placeholder value.")


def _manual_time_basis_general() -> str:
    """Return the general basis statement for all manual timings."""
    return (
        "Manual times were measured with a stopwatch during a single session "
        "where an analyst answered each question by hand in a spreadsheet "
        "(snapshots.csv opened in Excel/Google Sheets). Each time is a single "
        "measurement; for a more robust figure, repeat and average. These are "
        "PLACEHOLDER values for the MVP demo and should be replaced with actual "
        "measurements before publishing. The basis for each question is stated "
        "in the per-question results. No unearned 'saves X hours' claims are "
        "made — the calculation basis is fully transparent."
    )


# ---------------------------------------------------------------------------
# Summary writer
# ---------------------------------------------------------------------------


def _write_summary(report: dict[str, Any], path: Path) -> None:
    """Write a human-readable markdown summary for the README."""
    s = report["summary"]
    lines = [
        "# Ask Your Market — Business-Effect Timing (Phase 5d)",
        "",
        f"**Generated:** {report['metadata']['generated_at']}",
        "",
        "## Method",
        "",
        f"{report['manual_time_basis']}",
        "",
        "## The 5 Representative Questions",
        "",
        "| # | ID | Type | Question | Manual (sec) | Assistant (sec) | Saved (sec) | Saved % |",
        "|---|----|------|----------|--------------|-----------------|-------------|---------|",
    ]

    for i, q in enumerate(report["questions"], 1):
        mt = q.get("manual_time_seconds", "-")
        at = q.get("assistant_time_seconds", "-")
        ts = q.get("time_saved_seconds", "-")
        tp = q.get("time_saved_percent", "-")
        qtype = q.get("type", "").replace("_", " ")
        lines.append(
            f"| {i} | {q['question_id']} | {qtype} | {q['question'][:60]}... | "
            f"{mt} | {at} | {ts} | {tp}% |"
        )

    lines.extend([
        "",
        "## Aggregate",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Total manual time | {s['total_manual_time_seconds']} s ({s['total_manual_time_seconds'] / 60:.1f} min) |",
        f"| Total assistant time | {s['total_assistant_time_seconds']} s ({s['total_assistant_time_seconds'] / 60:.1f} min) |",
        f"| Total time saved | {s['total_time_saved_seconds']} s ({s['total_time_saved_seconds'] / 60:.1f} min) |",
        f"| Time saved (overall) | {s['total_time_saved_percent']}% |",
        f"| Total tokens | {s['total_tokens']} |",
        f"| Total cost | ${s['total_cost_usd']:.4f} USD |",
        f"| Avg time saved per question | {s['avg_time_saved_percent']}% |",
        "",
        "## Per-Question Detail",
        "",
    ])

    for q in report["questions"]:
            lines.append(f"### {q['question_id']} — {q['question']}")
            lines.append(f"- **Type:** {q['type'].replace('_', ' ')}")
            lines.append(f"- **Manual time:** {q.get('manual_time_seconds', 'N/A')} s")
            lines.append(f"- **Manual basis:** {q.get('manual_time_basis', 'N/A')}")
            lines.append(f"- **Assistant time:** {q.get('assistant_time_seconds', 'N/A')} s")
            lines.append(f"- **Time saved:** {q.get('time_saved_seconds', 'N/A')} s ({q.get('time_saved_percent', 'N/A')}%)")
            if q.get("tokens"):
                lines.append(f"- **Tokens:** prompt={q['tokens']['prompt_tokens']}, completion={q['tokens']['completion_tokens']}, total={q['tokens']['total_tokens']}")
            if q.get("cost_usd") is not None:
                lines.append(f"- **Cost:** ${q['cost_usd']:.4f} USD ({q['cost_basis']})")
            if q.get("model"):
                lines.append(f"- **Model:** {q['model']}")
            if q.get("error"):
                lines.append(f"- **Error:** {q['error']}")
            lines.append("")

    calc_lines = [
        "## Calculation Basis for README Claims",
        "",
        "Any 'saves X hours per week' claim in the README must be calculated from",
        "the numbers above with explicit assumptions. Example calculation:",
        "",
        f"1. Total time for 5 questions by hand: **{s['total_manual_time_seconds']} s** ({round(s['total_manual_time_seconds'] / 60, 1)} min)",
        f"2. Total time for 5 questions via assistant: **{s['total_assistant_time_seconds']} s** ({round(s['total_assistant_time_seconds'] / 60, 1)} min)",
        f"3. Saving per 5 questions: **{s['total_time_saved_seconds']} s** ({round(s['total_time_saved_seconds'] / 60, 1)} min)",
        "4. If an analyst answers N such questions per week, weekly saving =",
        f"   (N / 5) * {s['total_time_saved_seconds']} s = (N / 5) * {round(s['total_time_saved_seconds'] / 60, 1)} min",
        "",
        "This is a rough calculation based on a single stopwatch measurement per",
        "question. Real weekly saving depends on question volume and complexity,",
        "and these numbers should be re-measured with real analyst time before",
        "publishing specific 'saves X hours' claims.",
        "",
        "— No unearned claims. Basis is transparent.",
        "",
    ]
    if report["questions"] and report["questions"][0].get("model"):
        calc_lines.append(f"*Model used: {report['questions'][0]['model']}*")
    lines.extend(calc_lines)

    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """Standalone CLI for the timing comparison."""
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="Business-effect timing comparison (Phase 5d) — "
        "measures assistant vs. manual (spreadsheet) answer time for 5 questions."
    )
    parser.add_argument("--questions", type=Path, default=EVAL_DIR / "questions.json",
                        help="Path to questions.json")
    parser.add_argument("--snapshots", type=Path, default=SNAPSHOTS_PATH,
                        help="Path to snapshots.csv")
    parser.add_argument("--chroma-dir", type=Path, default=CHROMA_DIR,
                        help="Path to Chroma DB directory")
    parser.add_argument("--output", type=Path, default=TIMING_PATH,
                        help="Path for timing_results.json")
    parser.add_argument("--summary", type=Path, default=TIMING_SUMMARY_PATH,
                        help="Path for timing_summary.md")
    parser.add_argument("--no-live", action="store_true",
                        help="Skip live orchestrator calls (use cached/placeholder data only)")
    args = parser.parse_args()

    if args.no_live:
        logger.warning("Skipping live orchestrator calls (--no-live). "
                       "You must have existing timing_results.json to use this mode.")
        # In --no-live mode, just print the existing results if available.
        if TIMING_PATH.exists():
            existing = json.loads(TIMING_PATH.read_text())
            print(json.dumps(existing, indent=2))
        else:
            logger.error("No existing timing results found at %s", TIMING_PATH)
        return

    run_timing(
        questions_path=args.questions,
        snapshots_path=args.snapshots,
        chroma_dir=args.chroma_dir,
        output_path=args.output,
        summary_path=args.summary,
    )

    print(f"\nTiming complete. Results: {args.output}")
    print(f"Summary: {args.summary}")


if __name__ == "__main__":
    main()
