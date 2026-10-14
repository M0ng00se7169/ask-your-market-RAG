"""
Ask Your Market — data module.

Pure pandas queries over snapshots.csv. No LLM, no vector DB.
All functions return DataFrames; the tool layer converts to LLM-friendly structures.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

import pandas as pd


# Columns present in snapshots.csv.
COLUMNS = [
    "scraped_at",
    "url",
    "title",
    "price",
    "currency",
    "in_stock",
    "promo",
    "rating",
    "reviews_count",
    "description",
]

# Columns used as a unique product key across snapshots.
PRODUCT_KEY_COLUMNS = ["url", "title"]


def _parse_snapshots(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize a raw snapshots DataFrame."""
    if df.empty:
        return df

    df = df.copy()
    required = set(COLUMNS)
    present = set(df.columns)
    missing = required - present
    if missing:
        raise ValueError(f"snapshots.csv is missing required columns: {sorted(missing)}")

    # Coerce types.
    df["scraped_at"] = pd.to_datetime(df["scraped_at"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df["reviews_count"] = pd.to_numeric(df["reviews_count"], errors="coerce")
    df["in_stock"] = df["in_stock"].astype(bool)
    df["promo"] = df["promo"].fillna("").astype(str)

    return df


@lru_cache(maxsize=1)
def load_snapshots(path: str | Path = "data/snapshots.csv") -> pd.DataFrame:
    """Load and normalize snapshots.csv. Results are cached by path."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"snapshots.csv not found at {path}")

    df = pd.read_csv(path)
    return _parse_snapshots(df)


def latest_snapshot(df: pd.DataFrame) -> pd.DataFrame:
    """Most recent snapshot per product (dedup by url).

    Returns one row per product, the row with the latest scraped_at.
    """
    if df.empty:
        return df

    idx = df.groupby("url")["scraped_at"].idxmax()
    return df.loc[idx].copy()


def _latest_time(df: pd.DataFrame) -> Optional[pd.Timestamp]:
    """Most recent scraped_at across the entire dataset."""
    if df.empty or df["scraped_at"].isna().all():
        return None
    return df["scraped_at"].max()


def product_history(df: pd.DataFrame, url: str) -> pd.DataFrame:
    """Time series for one product, ordered oldest -> newest."""
    mask = df["url"] == url
    sub = df.loc[mask].copy()
    if sub.empty:
        return sub
    sub = sub.sort_values("scraped_at", ascending=True)
    return sub


def price_change_over_period(
    df: pd.DataFrame,
    days: int,
) -> pd.DataFrame:
    """Products whose price changed within the last `days` days.

    Requires a latest snapshot and the price at the start of the period.
    Returns products where the two differ (NaN-aware).
    """
    if df.empty or days <= 0:
        return df.iloc[:0].copy()

    cutoff = _latest_time(df) - pd.Timedelta(days=days)
    if cutoff is None:
        return df.iloc[:0].copy()

    within = df[df["scraped_at"] >= cutoff]
    if within.empty:
        return within

    latest = latest_snapshot(within)

    # Price at the start: the snapshot nearest (>=) the cutoff for each product.
    start_candidates = df[df["scraped_at"] >= cutoff]
    if start_candidates.empty:
        return latest.iloc[:0].copy()

    start_idx = start_candidates.groupby("url")["scraped_at"].idxmin()
    start = df.loc[start_idx, ["url", "price"]].rename(columns={"price": "price_start"})

    changes = latest.merge(start, on="url", how="inner")
    changes["price_delta"] = changes["price"] - changes["price_start"]
    changes["changed"] = changes["price_delta"].notna() & (changes["price_delta"] != 0)

    changed = changes[changes["changed"]].copy()
    changed = changed.sort_values("price_delta")
    return changed


def out_of_stock(df: pd.DataFrame) -> pd.DataFrame:
    """Products currently out of stock (latest snapshot, in_stock == False)."""
    if df.empty:
        return df.iloc[:0].copy()
    latest = latest_snapshot(df)
    return latest[latest["in_stock"] == False].copy()


def _sorted_slice(df: pd.DataFrame, n: int, ascending: bool) -> pd.DataFrame:
    """Top/bottom N products by current price, ties broken by url."""
    if df.empty or n <= 0:
        return df.iloc[:0].copy()
    latest = latest_snapshot(df)
    valid = latest.dropna(subset=["price"])
    if valid.empty:
        return valid.iloc[:0].copy()
    valid = valid.sort_values(["price", "url"], ascending=[ascending, True])
    return valid.head(n).copy()


def most_expensive(df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    """Top N most expensive products by current price."""
    return _sorted_slice(df, n, ascending=False)


def least_expensive(df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    """Top N least expensive products by current price."""
    return _sorted_slice(df, n, ascending=True)


def find_by_url(df: pd.DataFrame, url: str) -> Optional[pd.Series]:
    """Latest snapshot row for a product URL, or None if not present."""
    if df.empty:
        return None
    latest = latest_snapshot(df)
    match = latest[latest["url"] == url]
    if match.empty:
        return None
    return match.iloc[0]


def data_summary(df: pd.DataFrame) -> dict:
    """Descriptor the LLM can reference: what's in the dataset.

    Returns a small dict describing columns, date range, and product count.
    """
    if df.empty:
        return {
            "columns": COLUMNS,
            "product_count": 0,
            "date_min": None,
            "date_max": None,
            "note": "No data loaded.",
        }

    parsed = _parse_snapshots(df)
    ts = parsed["scraped_at"].dropna()
    return {
        "columns": COLUMNS,
        "product_count": int(parsed["url"].nunique()),
        "snapshot_count": int(len(parsed)),
        "date_min": ts.min().isoformat() if len(ts) else None,
        "date_max": ts.max().isoformat() if len(ts) else None,
        "note": (
            "Prices, availability, promos, and ratings come from snapshots.csv. "
            "The assistant does not have sales volume, revenue, forecast, or "
            "competitor-internal data."
        ),
    }


def available_data_descriptor() -> dict:
    """Static descriptor of what *can* be answered, independent of loaded data.

    Useful as the fallback when the dataset itself is empty or missing.
    """
    return {
        "columns": COLUMNS,
        "product_count": None,
        "date_min": None,
        "date_max": None,
        "note": (
            "Queries available: price changes over a period, out-of-stock products, "
            "product history by URL, most/least expensive products, find a product by URL, "
            "and the data summary. "
            "The assistant does not track sales volume, revenue, forecasts, or "
            "competitor-internal information."
        ),
    }
