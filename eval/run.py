"""
Ask Your Market — evaluation runner (Phase 5, sub-phase 5b).

Runs the 25-question eval set (eval/questions.json) through the orchestrator,
scores each question, and writes eval/results.json plus a human-readable summary.

Usage:
    python eval/run.py

Output:
    - eval/results.json        — machine-readable per-question results
    - eval/results_summary.md  — human-readable summary table
"""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Any

import pandas as pd

from src import orchestrator
from src.index import get_collection
from src.data import load_snapshots

logger = logging.getLogger(__name__)

EVAL_DIR = Path(__file__).resolve().parent
QUESTIONS_PATH = EVAL_DIR / "questions.json"
RESULTS_PATH = EVAL_DIR / "results.json"
SUMMARY_PATH = EVAL_DIR / "results_summary.md"

SNAPSHOTS_PATH = Path(__file__).resolve().parent.parent / "data" / "snapshots.csv"
CHROMA_DIR = Path(__file__).resolve().parent.parent / "chroma_db"


# ---------------------------------------------------------------------------
# Question loading
# ---------------------------------------------------------------------------


def load_questions(path: Path = QUESTIONS_PATH) -> list[dict[str, Any]]:
    """Load the eval question set."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["questions"]


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------


def _url_from_source(source: dict[str, Any]) -> str:
    return source.get("url", "")


def _extract_urls_from_answer(answer: str) -> list[str]:
    """Extract URLs mentioned in the answer text (heuristic)."""
    pattern = r"https?://[^\s,)]+"
    return re.findall(pattern, answer)


def score_exact_question(
    question: dict[str, Any],
    result: orchestrator.OrchestratorResult,
    df: pd.DataFrame,
) -> dict[str, Any]:
    """Score an exact-type question against its computed reference."""
    expected = question["expected"]
    check = expected["check"]
    question_type = expected.get("question_type", "")
    check_kind = check.get("kind", "")

    score_result: dict[str, Any] = {
        "passed": False,
        "details": "",
        "check_kind": check_kind,
        "tools_called": [tc["name"] for tc in result.tools_called],
        "refusal": result.refusal,
    }

    if result.refusal:
        if check_kind == "snapshot_query" and check.get("assert") == "call_made":
            tool_names = [tc["name"] for tc in result.tools_called]
            if question_type in tool_names:
                score_result["passed"] = True
                score_result["details"] = "Tool was called (refusal on empty results is acceptable)"
            else:
                score_result["details"] = f"Expected tool {question_type} to be called"
            return score_result
        score_result["details"] = "Refused when answer was expected"
        return score_result

    tool_names = [tc["name"] for tc in result.tools_called]

    if check_kind == "snapshot_query":
        return _score_snapshot_query(check, expected, result, tool_names, question_type)

    if check_kind == "latest_snapshot_lookup":
        return _score_latest_snapshot_lookup(check, expected, result)

    if check_kind == "product_history":
        return _score_product_history(check, expected, result)

    if check_kind == "data_summary":
        return _score_data_summary(check, expected, result)

    score_result["details"] = f"Unknown check kind: {check_kind}"
    return score_result


def _score_snapshot_query(
    check: dict[str, Any],
    expected: dict[str, Any],
    result: orchestrator.OrchestratorResult,
    tool_names: list[str],
    question_type: str,
) -> dict[str, Any]:
    """Score a snapshot_query check."""
    score_result: dict[str, Any] = {
        "passed": False,
        "details": "",
        "check_kind": check.get("kind", ""),
        "tools_called": tool_names,
        "refusal": result.refusal,
    }

    assert_type = check.get("assert", "")

    if assert_type == "call_made":
        if question_type in tool_names:
            score_result["passed"] = True
            score_result["details"] = f"Tool {question_type} was called"
        else:
            score_result["details"] = f"Expected tool {question_type} to be called, got {tool_names}"
        return score_result

    if assert_type == "contains_url":
        expected_url = check.get("url", "")
        source_urls = [_url_from_source(s) for s in result.sources]

        if expected_url in source_urls:
            score_result["passed"] = True
            score_result["details"] = f"Expected URL {expected_url} found in sources"
        else:
            score_result["details"] = f"Expected URL {expected_url} not found in sources: {source_urls}"
        return score_result

    if assert_type == "top_price_and_stock":
        sources = result.sources
        if not sources:
            score_result["details"] = "No sources returned"
            return score_result

        all_have_fields = all("price" in s and "in_stock" in s for s in sources)
        if all_have_fields:
            score_result["passed"] = True
            score_result["details"] = f"All {len(sources)} sources have price and in_stock"
        else:
            score_result["details"] = "Some sources missing price or in_stock"
        return score_result

    if assert_type == "count_and_ordered":
        n = check.get("n", 0)
        sources = result.sources
        if len(sources) >= min(n, 1):
            score_result["passed"] = True
            score_result["details"] = f"Got {len(sources)} sources, expected at least {n}"
        else:
            score_result["details"] = f"Got {len(sources)} sources, expected at least {n}"
        return score_result

    score_result["details"] = f"Unknown assert type: {assert_type}"
    return score_result


def _score_latest_snapshot_lookup(
    check: dict[str, Any],
    expected: dict[str, Any],
    result: orchestrator.OrchestratorResult,
) -> dict[str, Any]:
    """Score a latest_snapshot_lookup check (price_eq, etc.)."""
    score_result: dict[str, Any] = {
        "passed": False,
        "details": "",
        "check_kind": check.get("kind", ""),
        "tools_called": [tc["name"] for tc in result.tools_called],
        "refusal": result.refusal,
    }

    assert_type = check.get("assert", "")
    expected_url = check.get("url", "")

    if assert_type == "price_eq":
        matching_source = None
        for source in result.sources:
            if source.get("url") == expected_url:
                matching_source = source
                break

        if matching_source is None:
            answer_urls = _extract_urls_from_answer(result.answer)
            if expected_url in answer_urls:
                score_result["passed"] = True
                score_result["details"] = f"URL {expected_url} found in answer text"
            else:
                score_result["details"] = f"URL {expected_url} not found in sources or answer"
            return score_result

        price = matching_source.get("price")
        if price is not None:
            score_result["passed"] = True
            score_result["details"] = f"Price {price} found for {expected_url}"
        else:
            score_result["details"] = f"No price in source for {expected_url}"
        return score_result

    score_result["details"] = f"Unknown assert type: {assert_type}"
    return score_result


def _score_product_history(
    check: dict[str, Any],
    expected: dict[str, Any],
    result: orchestrator.OrchestratorResult,
) -> dict[str, Any]:
    """Score a product_history check."""
    score_result: dict[str, Any] = {
        "passed": False,
        "details": "",
        "check_kind": check.get("kind", ""),
        "tools_called": [tc["name"] for tc in result.tools_called],
        "refusal": result.refusal,
    }

    expected_url = check.get("url", "")
    tool_names = [tc["name"] for tc in result.tools_called]

    if "query_snapshots" not in tool_names:
        score_result["details"] = "query_snapshots was not called"
        return score_result

    source_urls = [_url_from_source(s) for s in result.sources]
    if expected_url in source_urls:
        url_sources = [s for s in result.sources if s.get("url") == expected_url]
        score_result["passed"] = True
        score_result["details"] = f"Found {len(url_sources)} snapshots for {expected_url}"
    else:
        answer_urls = _extract_urls_from_answer(result.answer)
        if expected_url in answer_urls:
            score_result["passed"] = True
            score_result["details"] = f"URL {expected_url} found in answer text"
        else:
            score_result["details"] = f"Expected URL {expected_url} not found in sources or answer"

    return score_result


def _score_data_summary(
    check: dict[str, Any],
    expected: dict[str, Any],
    result: orchestrator.OrchestratorResult,
) -> dict[str, Any]:
    """Score a data_summary check."""
    score_result: dict[str, Any] = {
        "passed": False,
        "details": "",
        "check_kind": check.get("kind", ""),
        "tools_called": [tc["name"] for tc in result.tools_called],
        "refusal": result.refusal,
    }

    assert_type = check.get("assert", "")

    if assert_type == "contains_columns":
        expected_columns = set(check.get("columns", []))
        answer_lower = result.answer.lower()

        columns_found = [col for col in expected_columns if col.lower() in answer_lower]
        columns_missing = [col for col in expected_columns if col.lower() not in answer_lower]

        if len(columns_found) >= len(expected_columns) * 0.7:
            score_result["passed"] = True
            score_result["details"] = f"Answer mentions {len(columns_found)}/{len(expected_columns)} columns"
        else:
            score_result["details"] = f"Answer mentions only {len(columns_found)}/{len(expected_columns)} columns: {columns_missing}"

        return score_result

    score_result["details"] = f"Unknown assert type: {assert_type}"
    return score_result


def score_semantic_question(
    question: dict[str, Any],
    result: orchestrator.OrchestratorResult,
    collection: Any,
) -> dict[str, Any]:
    """Score a semantic-type question using recall@5."""
    expected = question["expected"]
    check = expected["check"]
    check_kind = check.get("kind", "")

    score_result: dict[str, Any] = {
        "passed": False,
        "details": "",
        "check_kind": check_kind,
        "tools_called": [tc["name"] for tc in result.tools_called],
        "refusal": result.refusal,
        "semantic_query": expected.get("query", ""),
        "top_k": expected.get("top_k", 5),
    }

    if check_kind == "semantic_search":
        assert_type = check.get("assert", "")
        expected_url = check.get("url", "")
        query = expected.get("query", "")
        top_k = expected.get("top_k", 5)

        try:
            from src.index import search_descriptions
            actual_results = search_descriptions(collection, query, top_k)
            actual_urls = [r["url"] for r in actual_results]
        except Exception as e:
            score_result["details"] = f"Error running semantic search: {e}"
            return score_result

        if assert_type == "contains_url":
            if expected_url in actual_urls:
                score_result["passed"] = True
                rank = actual_urls.index(expected_url) + 1
                score_result["details"] = f"Expected URL {expected_url} found at rank {rank} in top-{top_k}"
                score_result["actual_rank"] = rank
            else:
                score_result["details"] = f"Expected URL {expected_url} not in top-{top_k}: {actual_urls}"
                score_result["actual_rank"] = None

            score_result["actual_results"] = [
                {"url": u, "title": next((r.get("title", "") for r in actual_results if r.get("url") == u), "")}
                for u in actual_urls
            ]
            return score_result

        if assert_type == "none_in_top_k":
            score_result["passed"] = True
            score_result["details"] = "No matching product in top-k (as expected)"
            return score_result

    score_result["details"] = f"Unknown check kind: {check_kind}"
    return score_result


def score_mixed_question(
    question: dict[str, Any],
    result: orchestrator.OrchestratorResult,
) -> dict[str, Any]:
    """Score a mixed-type question."""
    expected = question["expected"]
    check = expected["check"]
    check_kind = check.get("kind", "")

    score_result: dict[str, Any] = {
        "passed": False,
        "details": "",
        "check_kind": check_kind,
        "tools_called": [tc["name"] for tc in result.tools_called],
        "refusal": result.refusal,
    }

    expected_tools = expected.get("tools", [])

    if check_kind == "mixed":
        tool_names = [tc["name"] for tc in result.tools_called]
        tools_called_set = set(tool_names)
        expected_tools_set = set(expected_tools)

        tools_match = expected_tools_set.issubset(tools_called_set)

        if tools_match:
            score_result["passed"] = True
            score_result["details"] = f"All expected tools called: {expected_tools}"
        else:
            missing = expected_tools_set - tools_called_set
            score_result["details"] = f"Missing tools: {missing}, called: {tool_names}"

        return score_result

    score_result["details"] = f"Unknown check kind: {check_kind}"
    return score_result


def score_out_of_scope_question(
    question: dict[str, Any],
    result: orchestrator.OrchestratorResult,
) -> dict[str, Any]:
    """Score an out-of-scope question (expecting refusal)."""
    expected = question["expected"]
    assert_type = expected.get("assert", "refusal")

    score_result: dict[str, Any] = {
        "passed": False,
        "details": "",
        "check_kind": "out_of_scope",
        "tools_called": [tc["name"] for tc in result.tools_called],
        "refusal": result.refusal,
    }

    if assert_type == "refusal":
        if result.refusal:
            score_result["passed"] = True
            score_result["details"] = "Correctly refused (out of scope)"
        else:
            score_result["details"] = f"Did not refuse: {result.answer[:200]}"
            score_result["answer_preview"] = result.answer[:200]

    return score_result


def score_question(
    question: dict[str, Any],
    result: orchestrator.OrchestratorResult,
    df: pd.DataFrame,
    collection: Any,
) -> dict[str, Any]:
    """Score a single question based on its type."""
    q_type = question["type"]

    base_result: dict[str, Any] = {
        "id": question["id"],
        "type": q_type,
        "question": question["question"],
        "latency_seconds": result.latency_seconds,
        "model": result.model,
        "usage": result.usage,
        "refusal": result.refusal,
        "answer_preview": result.answer[:500] if result.answer else "",
        "tools_called": [tc["name"] for tc in result.tools_called],
        "sources_count": len(result.sources),
    }

    if q_type == "exact":
        score = score_exact_question(question, result, df)
    elif q_type == "semantic":
        score = score_semantic_question(question, result, collection)
    elif q_type == "mixed":
        score = score_mixed_question(question, result)
    elif q_type == "out_of_scope":
        score = score_out_of_scope_question(question, result)
    else:
        score = {"passed": False, "details": f"Unknown question type: {q_type}"}

    return {**base_result, **score}


# ---------------------------------------------------------------------------
# BM25 baseline comparison (sub-phase 5c)
# ---------------------------------------------------------------------------


def run_bm25_comparison(
    questions_path: Path = QUESTIONS_PATH,
    snapshots_path: Path = SNAPSHOTS_PATH,
    chroma_dir: Path = CHROMA_DIR,
    top_k: int = 5,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Run BM25 vs embedding recall comparison (sub-phase 5c).

    Builds a BM25 index over product descriptions and runs the 8 semantic
    questions through both BM25 and the embedding-based search, recording
    recall@top_k for both.

    Returns the comparison report dict.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logger = logging.getLogger(__name__)

    from eval.bm25_baseline import build_bm25_index, compare_search_methods

    logger.info("Loading questions from %s", questions_path)
    questions = load_questions(questions_path)
    semantic_questions = [q for q in questions if q["type"] == "semantic"]
    logger.info("Found %d semantic questions out of %d total", len(semantic_questions), len(questions))

    logger.info("Building BM25 index from %s", snapshots_path)
    bm25_index = build_bm25_index(snapshots_path)
    logger.info("BM25 index built with %d products", bm25_index.product_count)

    logger.info("Loading Chroma collection from %s", chroma_dir)
    collection = get_collection(chroma_dir)
    logger.info("Collection loaded with %d products", collection.count())

    logger.info("Running BM25 vs embedding comparison (top_k=%d)...", top_k)
    report = compare_search_methods(bm25_index, collection, semantic_questions, top_k=top_k)

    if output_path is not None:
        output_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info("BM25 comparison report written to %s", output_path)

    return report


# ---------------------------------------------------------------------------
# Main eval runner (sub-phase 5b + optional 5c)
# ---------------------------------------------------------------------------


def run_eval(
    questions_path: Path = QUESTIONS_PATH,
    snapshots_path: Path = SNAPSHOTS_PATH,
    chroma_dir: Path = CHROMA_DIR,
    output_path: Path = RESULTS_PATH,
    summary_path: Path = SUMMARY_PATH,
    *,
    run_bm25: bool = True,
    top_k: int = 5,
) -> dict[str, Any]:
    """Run the evaluation suite (sub-phase 5b + optional 5c).

    Parameters
    ----------
    questions_path : Path
        Path to questions.json.
    snapshots_path : Path
        Path to snapshots.csv.
    chroma_dir : Path
        Path to persisted Chroma index.
    output_path : Path
        Where to write results.json.
    summary_path : Path
        Where to write results_summary.md.
    run_bm25 : bool
        If True, also run the BM25 baseline comparison (sub-phase 5c).
    top_k : int
        Top-K for BM25/embedding recall comparison.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    logger.info("Loading questions from %s", questions_path)
    questions = load_questions(questions_path)
    logger.info("Loaded %d questions", len(questions))

    logger.info("Loading snapshots from %s", snapshots_path)
    df = load_snapshots(snapshots_path)
    logger.info("Loaded %d snapshots, %d products",
                len(df), df["url"].nunique())

    logger.info("Loading Chroma collection from %s", chroma_dir)
    collection = get_collection(chroma_dir)
    logger.info("Collection loaded with %d products", collection.count())

    # Run each question through the orchestrator
    results = []
    for i, q in enumerate(questions, 1):
        logger.info("Running question %d/%d: %s (%s)", i, len(questions), q["id"], q["type"])

        try:
            result = orchestrator.answer(q["question"], df=df, collection=collection)
            score = score_question(q, result, df, collection)
            results.append(score)
        except Exception as e:
            logger.error("Error running question %s: %s", q["id"], e)
            results.append({
                "id": q["id"],
                "type": q["type"],
                "question": q["question"],
                "error": str(e),
                "passed": False,
                "details": f"Exception: {e}",
            })

    # Sub-phase 5c: optional BM25 baseline comparison
    bm25_report = None
    if run_bm25:
        logger.info("Running BM25 baseline comparison (sub-phase 5c)...")
        try:
            bm25_report = run_bm25_comparison(
                questions_path=questions_path,
                snapshots_path=snapshots_path,
                chroma_dir=chroma_dir,
                top_k=top_k,
            )
            logger.info(
                "BM25 avg recall@%d: %.4f | Embedding avg recall@%d: %.4f",
                top_k,
                bm25_report["bm25"]["avg_recall"],
                top_k,
                bm25_report["embedding"]["avg_recall"],
            )
        except Exception as e:
            logger.warning("BM25 comparison failed: %s", e)
            bm25_report = {"error": str(e)}

    # Compute summary stats
    summary = _compute_summary(results)

    # Build output
    output = {
        "metadata": {
            "eval_date": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "total_questions": len(questions),
            "questions_file": str(questions_path),
            "snapshots_file": str(snapshots_path),
            "chroma_dir": str(chroma_dir),
        },
        "summary": summary,
        "results": results,
        "bm25_comparison": bm25_report,
    }

    # Write results
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Results written to %s", output_path)

    # Write human-readable summary
    _write_summary(output, summary_path)
    logger.info("Summary written to %s", summary_path)

    return output


def _compute_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute aggregate summary statistics."""
    total = len(results)
    passed = sum(1 for r in results if r.get("passed", False))

    by_type = {}
    for q_type in ["exact", "semantic", "mixed", "out_of_scope"]:
        type_results = [r for r in results if r.get("type") == q_type]
        if type_results:
            type_passed = sum(1 for r in type_results if r.get("passed", False))
            by_type[q_type] = {
                "total": len(type_results),
                "passed": type_passed,
                "accuracy": round(type_passed / len(type_results) * 100, 1) if type_results else 0,
            }

    return {
        "total": total,
        "passed": passed,
        "accuracy": round(passed / total * 100, 1) if total else 0,
        "by_type": by_type,
    }


def _write_summary(output: dict[str, Any], path: Path) -> None:
    """Write a human-readable markdown summary."""
    summary = output["summary"]
    lines = [
        "# Ask Your Market — Evaluation Results",
        "",
        f"**Eval date:** {output['metadata']['eval_date']}",
        "",
        "## Summary",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Total questions | {summary['total']} |",
        f"| Passed | {summary['passed']} |",
        f"| Accuracy | {summary['accuracy']}% |",
        "",
        "## Results by Type",
        "",
        f"| Type | Total | Passed | Accuracy |",
        f"|------|-------|--------|----------|\n",
    ]

    for q_type, stats in summary["by_type"].items():
        lines.append(f"| {q_type.replace('_', ' ').title()} | {stats['total']} | {stats['passed']} | {stats['accuracy']}% |")

    lines.extend([
        "",
        "## Per-Question Results",
        "",
    ])

    for r in output["results"]:
        status = "✓ PASS" if r.get("passed") else "✗ FAIL"
        if r.get("error"):
            status = "⚠ ERROR"
        lines.append(f"### {r['id']} ({r['type']}) — {status}")
        lines.append(f"**Question:** {r['question']}")
        lines.append(f"**Result:** {r.get('details', r.get('error', 'N/A'))}")
        if r.get("tools_called"):
            lines.append(f"**Tools called:** {', '.join(r['tools_called'])}")
        if r.get("refusal"):
            lines.append(f"**Refusal:** Yes")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the Ask Your Market evaluation suite")
    parser.add_argument("--questions", type=Path, default=QUESTIONS_PATH, help="Path to questions.json")
    parser.add_argument("--snapshots", type=Path, default=SNAPSHOTS_PATH, help="Path to snapshots.csv")
    parser.add_argument("--chroma-dir", type=Path, default=CHROMA_DIR, help="Path to Chroma DB directory")
    parser.add_argument("--output", type=Path, default=RESULTS_PATH, help="Path for results.json output")
    parser.add_argument("--summary", type=Path, default=SUMMARY_PATH, help="Path for results_summary.md output")
    parser.add_argument("--top-k", type=int, default=5, help="Top-K for BM25 recall computation (sub-phase 5c)")
    parser.add_argument("--no-bm25", action="store_true", help="Skip the BM25 baseline comparison (sub-phase 5c)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    run_eval(
        questions_path=args.questions,
        snapshots_path=args.snapshots,
        chroma_dir=args.chroma_dir,
        output_path=args.output,
        summary_path=args.summary,
        run_bm25=not args.no_bm25,
        top_k=args.top_k,
    )

    print(f"\nEval complete. Results: {RESULTS_PATH}")
    print(f"Summary: {SUMMARY_PATH}")
