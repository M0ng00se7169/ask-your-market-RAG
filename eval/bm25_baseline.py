"""
Ask Your Market — BM25 baseline for semantic search comparison (Phase 5, sub-phase 5c).

Implements a BM25 keyword baseline over product descriptions so the eval can
compare embedding-based semantic search against a classical IR baseline.

Public API:
    build_bm25_index(csv_path) -> BM25Index
    bm25_search(index, query, top_k) -> list[dict]
    compare_recall(bm25_index, collection, questions, ...) -> dict

Design points:
- Uses rank_bm25 (already in requirements.txt) with BM25Okapi.
- Tokenizes on whitespace + lowercase (simple, deterministic, comparable).
- One document per product description (same catalog as the embedding index).
- Returns results in the same shape as search_descriptions for easy comparison.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd
from rank_bm25 import BM25Okapi

from src.data import load_snapshots

# Simple tokenizer: lowercase + split on non-alphanumeric.
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    """Lowercase and extract alphanumeric tokens."""
    return _TOKEN_RE.findall(text.lower())


def _catalog_from_csv(csv_path: Path) -> pd.DataFrame:
    """One row per product (latest snapshot) with description text."""
    df = load_snapshots(csv_path)
    if df.empty:
        raise ValueError(f"snapshots.csv at {csv_path} is empty")

    latest = df.sort_values("scraped_at").groupby("url", as_index=False).last()
    return latest[["url", "title", "price", "currency", "in_stock", "promo", "scraped_at", "description"]].copy()


class BM25Index:
    """BM25 index over product descriptions.

    Wraps rank_bm25.BM25Okapi and keeps the catalog metadata so search results
    can be returned in the same shape as the embedding-based search.
    """

    def __init__(self, csv_path: str | Path) -> None:
        csv_path = Path(csv_path)
        catalog = _catalog_from_csv(csv_path)

        self._catalog = catalog.reset_index(drop=True)
        self._ids = [str(i) for i in range(len(catalog))]
        self._urls = catalog["url"].tolist()
        self._titles = catalog["title"].fillna("").tolist()
        # Convert prices to list, replacing NaN with None for cleaner handling.
        prices_raw = catalog["price"].tolist()
        self._prices = [None if pd.isna(p) else p for p in prices_raw]
        self._currencies = catalog["currency"].fillna("").tolist()
        self._in_stock = catalog["in_stock"].tolist()
        self._promos = catalog["promo"].fillna("").tolist()
        self._scraped_at = catalog["scraped_at"].tolist()
        self._descriptions = catalog["description"].fillna("").tolist()

        # Tokenize each description once at build time.
        self._tokenized_corpus = [_tokenize(desc) for desc in self._descriptions]

        self._bm25 = BM25Okapi(self._tokenized_corpus)

    @property
    def product_count(self) -> int:
        return len(self._catalog)

    def search(self, query: str, top_k: int = 10) -> list[dict[str, Any]]:
        """BM25 search returning results in the same shape as search_descriptions.

        Returns list of dicts with url/title/price/currency/in_stock/promo/
        scraped_at/score (BM25 raw score, higher = better match).
        """
        if top_k <= 0:
            return []

        tokens = _tokenize(query)
        if not tokens:
            return []

        scores = self._bm25.get_scores(tokens)
        # Pair (index, score), sort by score descending.
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        top_n = ranked[: top_k if top_k < len(ranked) else len(ranked)]

        out: list[dict[str, Any]] = []
        for idx, score in top_n:
            if score <= 0:
                continue
            out.append({
                "id": self._ids[idx],
                "url": self._urls[idx],
                "title": self._titles[idx],
                "price": self._prices[idx] if pd.notna(self._prices[idx]) else None,
                "currency": self._currencies[idx],
                "in_stock": bool(self._in_stock[idx]) if pd.notna(self._in_stock[idx]) else True,
                "promo": self._promos[idx],
                "scraped_at": str(self._scraped_at[idx]) if pd.notna(self._scraped_at[idx]) else "",
                "description": self._descriptions[idx],
                "score": float(score),
            })
        return out


def build_bm25_index(csv_path: str | Path) -> BM25Index:
    """Build a BM25 index from the snapshots CSV.

    One document per product (latest snapshot description).
    """
    return BM25Index(csv_path)


def bm25_search(index: BM25Index, query: str, top_k: int = 10) -> list[dict[str, Any]]:
    """Convenience wrapper: BM25 search over an existing index."""
    return index.search(query, top_k)


def _recall_at_k(actual_urls: list[str], expected_url: str, top_k: int) -> dict[str, Any]:
    """Compute recall@top_k for a single expected URL.

    Returns dict with:
        recall: 1.0 if expected_url in top_k, else 0.0
        rank: 1-based rank if found, None otherwise
        in_top_k: bool
    """
    top = actual_urls[:top_k]
    if expected_url in top:
        return {"recall": 1.0, "rank": top.index(expected_url) + 1, "in_top_k": True}
    return {"recall": 0.0, "rank": None, "in_top_k": False}


def compare_search_methods(
    bm25_index: BM25Index,
    collection: Any,
    semantic_questions: list[dict[str, Any]],
    *,
    top_k: int = 5,
) -> dict[str, Any]:
    """Run semantic questions through both BM25 and embedding search.

    Returns a comparison report with per-question results and aggregate
    recall@top_k for both methods.

    Parameters
    ----------
    bm25_index : BM25Index
        The BM25 index to query.
    collection : Chroma collection
        The embedding-based collection (for search_descriptions).
    semantic_questions : list[dict]
        The semantic-type questions from questions.json.
    top_k : int
        How many results to consider for recall (default 5).
    """
    from src.index import search_descriptions

    results: list[dict[str, Any]] = []
    bm25_recalls: list[float] = []
    embedding_recalls: list[float] = []

    for q in semantic_questions:
        expected = q["expected"]
        query_text = expected.get("query", "")
        expected_url = expected.get("check", {}).get("url", "")
        assert_type = expected.get("check", {}).get("assert", "")

        # Skip "none_in_top_k" questions — they don't have an expected URL.
        if assert_type == "none_in_top_k":
            # For these, we just verify both methods return nothing relevant.
            bm25_results = bm25_index.search(query_text, top_k)
            emb_results = search_descriptions(collection, query_text, top_k)
            results.append({
                "question_id": q["id"],
                "question": q["question"],
                "query": query_text,
                "top_k": top_k,
                "expected_assert": assert_type,
                "bm25": {
                    "results": bm25_results,
                    "count": len(bm25_results),
                    "note": "none_in_top_k — no expected URL to check",
                },
                "embedding": {
                    "results": emb_results,
                    "count": len(emb_results),
                    "note": "none_in_top_k — no expected URL to check",
                },
            })
            continue

        # BM25 search.
        bm25_results = bm25_index.search(query_text, top_k)
        bm25_urls = [r["url"] for r in bm25_results]
        bm25_recall_info = _recall_at_k(bm25_urls, expected_url, top_k)
        bm25_recalls.append(bm25_recall_info["recall"])

        # Embedding search.
        emb_results = search_descriptions(collection, query_text, top_k)
        emb_urls = [r["url"] for r in emb_results]
        emb_recall_info = _recall_at_k(emb_urls, expected_url, top_k)
        embedding_recalls.append(emb_recall_info["recall"])

        results.append({
            "question_id": q["id"],
            "question": q["question"],
            "query": query_text,
            "top_k": top_k,
            "expected_url": expected_url,
            "bm25": {
                "results": bm25_results,
                "count": len(bm25_results),
                "recall": bm25_recall_info["recall"],
                "rank": bm25_recall_info["rank"],
                "in_top_k": bm25_recall_info["in_top_k"],
            },
            "embedding": {
                "results": emb_results,
                "count": len(emb_results),
                "recall": emb_recall_info["recall"],
                "rank": emb_recall_info["rank"],
                "in_top_k": emb_recall_info["in_top_k"],
            },
            "bm25_beats_embedding": bm25_recall_info["recall"] > emb_recall_info["recall"],
            "embedding_beats_bm25": emb_recall_info["recall"] > bm25_recall_info["recall"],
            "tie": bm25_recall_info["recall"] == emb_recall_info["recall"],
        })

    # Aggregate stats.
    n_with_expected = sum(1 for r in results if r.get("expected_url"))
    bm25_avg_recall = sum(bm25_recalls) / len(bm25_recalls) if bm25_recalls else 0.0
    emb_avg_recall = sum(embedding_recalls) / len(embedding_recalls) if embedding_recalls else 0.0

    bm25_wins = sum(1 for r in results if r.get("bm25_beats_embedding"))
    emb_wins = sum(1 for r in results if r.get("embedding_beats_bm25"))
    ties = sum(1 for r in results if r.get("tie"))

    return {
        "top_k": top_k,
        "questions_evaluated": len(results),
        "questions_with_expected_url": n_with_expected,
        "bm25": {
            "avg_recall": round(bm25_avg_recall, 4),
            "wins": bm25_wins,
            "per_question_recall": [r.get("bm25", {}).get("recall", 0.0) for r in results],
        },
        "embedding": {
            "avg_recall": round(emb_avg_recall, 4),
            "wins": emb_wins,
            "per_question_recall": [r.get("embedding", {}).get("recall", 0.0) for r in results],
        },
        "wins": {
            "bm25": bm25_wins,
            "embedding": emb_wins,
            "tie": ties,
        },
        "per_question": results,
    }


# ---------------------------------------------------------------------------
# CLI for standalone BM25 comparison
# ---------------------------------------------------------------------------


def main() -> None:
    """Standalone CLI to run the BM25 vs embedding comparison."""
    import argparse
    import json
    import logging
    import sys
    import time

    from src.index import get_collection

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logger = logging.getLogger(__name__)

    EVAL_DIR = Path(__file__).resolve().parent
    QUESTIONS_PATH = EVAL_DIR / "questions.json"
    SNAPSHOTS_PATH = EVAL_DIR.parent / "data" / "snapshots.csv"

    parser = argparse.ArgumentParser(description="BM25 vs embedding recall comparison (Phase 5c)")
    parser.add_argument("--questions", type=Path, default=QUESTIONS_PATH, help="Path to questions.json")
    parser.add_argument("--snapshots", type=Path, default=SNAPSHOTS_PATH, help="Path to snapshots.csv")
    parser.add_argument("--chroma-dir", type=Path, default=EVAL_DIR.parent / "chroma_db", help="Path to Chroma DB")
    parser.add_argument("--top-k", type=int, default=5, help="Top-K for recall computation")
    parser.add_argument("--output", type=Path, default=None, help="Path to write JSON report (default: stdout)")
    args = parser.parse_args()

    logger.info("Loading questions from %s", args.questions)
    questions_data = json.loads(args.questions.read_text(encoding="utf-8"))
    all_questions = questions_data["questions"]
    semantic_questions = [q for q in all_questions if q["type"] == "semantic"]
    logger.info("Found %d semantic questions out of %d total", len(semantic_questions), len(all_questions))

    logger.info("Building BM25 index from %s", args.snapshots)
    t0 = time.time()
    bm25_index = build_bm25_index(args.snapshots)
    logger.info("BM25 index built in %.2fs (%d products)", time.time() - t0, bm25_index.product_count)

    logger.info("Loading Chroma collection from %s", args.chroma_dir)
    collection = get_collection(args.chroma_dir)
    logger.info("Chroma collection loaded (%d products)", collection.count())

    logger.info("Running comparison (top_k=%d)...", args.top_k)
    t0 = time.time()
    report = compare_search_methods(bm25_index, collection, semantic_questions, top_k=args.top_k)
    logger.info("Comparison complete in %.2fs", time.time() - t0)

    # Print summary.
    print("\n=== BM25 vs Embedding Search Comparison (recall@{}) ===".format(args.top_k))
    print(f"Questions evaluated: {report['questions_evaluated']}")
    print(f"Questions with expected URL: {report['questions_with_expected_url']}")
    print()
    print(f"BM25  avg recall@{args.top_k}: {report['bm25']['avg_recall']:.4f}  (wins: {report['wins']['bm25']})")
    print(f"Emb   avg recall@{args.top_k}: {report['embedding']['avg_recall']:.4f}  (wins: {report['wins']['embedding']})")
    print(f"Ties: {report['wins']['tie']}")
    print()

    for r in report["per_question"]:
        qid = r["question_id"]
        b = r.get("bm25", {})
        e = r.get("embedding", {})
        b_recall = b.get("recall", "-")
        e_recall = e.get("recall", "-")
        b_rank = b.get("rank", "-")
        e_rank = e.get("rank", "-")
        winner = "BM25" if r.get("bm25_beats_embedding") else ("Emb" if r.get("embedding_beats_bm25") else "tie")
        print(f"  {qid}: BM25 recall={b_recall} (rank={b_rank}) | Emb recall={e_recall} (rank={e_rank}) → {winner}")

    if args.output:
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("Full report written to %s", args.output)
    else:
        print("\n--- Full report (JSON) ---")
        print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
