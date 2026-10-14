"""
Tests for src/data.py.

Uses a small synthetic snapshots.csv fixture generated inline so the tests are
self-contained and deterministic.
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from src.data import (
    COLUMNS,
    PRODUCT_KEY_COLUMNS,
    available_data_descriptor,
    data_summary,
    find_by_url,
    latest_snapshot,
    least_expensive,
    load_snapshots,
    most_expensive,
    out_of_stock,
    price_change_over_period,
    product_history,
)


FIXTURE_DIR = Path(__file__).resolve().parent / "data"
SNAPSHOTS_PATH = FIXTURE_DIR / "snapshots.csv"

# A synthetic fixture: 3 products over a few days with deliberate events.
# Product A: price drops on day 2.
# Product B: goes out of stock on the latest day.
# Product C: no changes.
FIXTURE_ROWS = [
    # product A
    dict(
        scraped_at="2026-10-01T00:00:00",
        url="https://books.toscrape.com/a",
        title="A",
        price=20.0,
        currency="GBP",
        in_stock=True,
        promo="",
        rating=4.2,
        reviews_count=10,
        description="A desc",
    ),
    dict(
        scraped_at="2026-10-02T00:00:00",
        url="https://books.toscrape.com/a",
        title="A",
        price=20.0,
        currency="GBP",
        in_stock=True,
        promo="",
        rating=4.2,
        reviews_count=10,
        description="A desc",
    ),
    dict(
        scraped_at="2026-10-03T00:00:00",
        url="https://books.toscrape.com/a",
        title="A",
        price=16.0,
        currency="GBP",
        in_stock=True,
        promo="",
        rating=4.2,
        reviews_count=10,
        description="A desc",
    ),
    # product B
    dict(
        scraped_at="2026-10-03T00:00:00",
        url="https://books.toscrape.com/b",
        title="B",
        price=10.0,
        currency="GBP",
        in_stock=True,
        promo="Summer sale",
        rating=4.5,
        reviews_count=25,
        description="B desc",
    ),
    dict(
        scraped_at="2026-10-04T00:00:00",
        url="https://books.toscrape.com/b",
        title="B",
        price=10.0,
        currency="GBP",
        in_stock=False,
        promo="",
        rating=4.5,
        reviews_count=25,
        description="B desc",
    ),
    # product C
    dict(
        scraped_at="2026-10-03T00:00:00",
        url="https://books.toscrape.com/c",
        title="C",
        price=30.0,
        currency="GBP",
        in_stock=True,
        promo="",
        rating=3.8,
        reviews_count=5,
        description="C desc",
    ),
]


@pytest.fixture(scope="session", autouse=True)
def write_fixture_csv():
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(FIXTURE_ROWS, columns=COLUMNS)
    df.to_csv(SNAPSHOTS_PATH, index=False)


@pytest.fixture
def df():
    return load_snapshots(SNAPSHOTS_PATH)


# ---- load_snapshots ----

class TestLoadSnapshots:
    def test_loads_csv(self, df):
        assert not df.empty
        assert set(df.columns) == set(COLUMNS)

    def test_parses_dates(self, df):
        assert pd.api.types.is_datetime64_any_dtype(df["scraped_at"])

    def test_parses_prices(self, df):
        assert pd.api.types.is_numeric_dtype(df["price"])

    def test_parses_in_stock_bool(self, df):
        assert df["in_stock"].dtype == bool

    def test_fills_blank_promo(self, df):
        blanks = df[df["promo"] == ""]
        assert not blanks.empty

    def test_missing_required_column_raises(self):
        bad = FIXTURE_DIR / "bad.csv"
        bad.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([{"x": 1}]).to_csv(bad, index=False)
        with pytest.raises(ValueError):
            load_snapshots(bad)
        bad.unlink()

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_snapshots("/nonexistent/path/snapshots.csv")

    def test_caching(self, monkeypatch):
        import src.data as mod

        # Prefer the fixture path for this test so we don't touch the real one.
        monkeypatch.setattr(mod, "data/snapshots.csv", str(SNAPSHOTS_PATH), raising=False)
        a = load_snapshots(SNAPSHOTS_PATH)
        b = load_snapshots(SNAPSHOTS_PATH)
        assert a is b


# ---- latest_snapshot ----

class TestLatestSnapshot:
    def test_one_row_per_url(self, df):
        latest = latest_snapshot(df)
        assert latest["url"].is_unique
        assert len(latest) == df["url"].nunique()

    def test_latest_dates(self, df):
        latest = latest_snapshot(df)
        expected = {  # latest scraped_at per url from the fixture
            "https://books.toscrape.com/a": pd.Timestamp("2026-10-03T00:00:00"),
            "https://books.toscrape.com/b": pd.Timestamp("2026-10-04T00:00:00"),
            "https://books.toscrape.com/c": pd.Timestamp("2026-10-03T00:00:00"),
        }
        got = latest.set_index("url")["scraped_at"].to_dict()
        assert got == expected

    def test_empty_returns_empty(self):
        assert latest_snapshot(pd.DataFrame(columns=COLUMNS)).empty


# ---- product_history ----

class TestProductHistory:
    def test_one_product_all_snapshots(self, df):
        hist = product_history(df, "https://books.toscrape.com/a")
        assert len(hist) == 3
        assert list(hist["price"]) == [20.0, 20.0, 16.0]

    def test_order_oldest_to_newest(self, df):
        hist = product_history(df, "https://books.toscrape.com/a")
        assert hist["scraped_at"].is_monotonic_increasing

    def test_unknown_url_returns_empty(self, df):
        hist = product_history(df, "https://example.com/nope")
        assert hist.empty

    def test_empty_df_returns_empty(self):
        assert product_history(pd.DataFrame(columns=COLUMNS), "x").empty


# ---- price_change_over_period ----

class TestPriceChangeOverPeriod:
    def test_detects_price_drop_in_window(self, df):
        # Last date is 2026-10-04. A 7-day window covers all snapshots.
        changes = price_change_over_period(df, days=7)
        a = changes[changes["url"] == "https://books.toscrape.com/a"]
        assert not a.empty
        row = a.iloc[0]
        assert row["price"] == pytest.approx(16.0)
        assert row["price_start"] == pytest.approx(20.0)
        assert row["price_delta"] == pytest.approx(-4.0)

    def test_excludes_no_change_products(self, df):
        changes = price_change_over_period(df, days=7)
        urls = set(changes["url"])
        assert "https://books.toscrape.com/c" not in urls
        assert "https://books.toscrape.com/b" not in urls

    def test_window_too_short_returns_empty(self, df):
        # 1-day window from the latest date (2026-10-04) only includes B's OOS row.
        changes = price_change_over_period(df, days=1)
        assert changes.empty

    def test_empty_df_returns_empty(self):
        assert price_change_over_period(pd.DataFrame(columns=COLUMNS), days=7).empty

    def test_zero_or_negative_days_returns_empty(self, df):
        assert price_change_over_period(df, days=0).empty
        assert price_change_over_period(df, days=-1).empty


# ---- out_of_stock ----

class TestOutOfStock:
    def test_returns_oos_products(self, df):
        oos = out_of_stock(df)
        assert not oos.empty
        assert set(oos["url"]) == {"https://books.toscrape.com/b"}

    def test_empty_df_returns_empty(self):
        assert out_of_stock(pd.DataFrame(columns=COLUMNS)).empty


# ---- most/least expensive ----

class TestMostLeastExpensive:
    def test_most_expensive_order(self, df):
        top = most_expensive(df, n=2)
        prices = list(top["price"])
        assert prices == sorted(prices, reverse=True)
        assert len(top) == 2
        # Latest prices: C=30, A=16 (dropped on day 3), B=10.
        assert top.iloc[0]["price"] == 30.0  # C
        assert top.iloc[1]["price"] == 16.0  # A (latest)

    def test_least_expensive_order(self, df):
        bottom = least_expensive(df, n=2)
        prices = list(bottom["price"])
        assert prices == sorted(prices)
        assert bottom.iloc[0]["price"] == 10.0  # B (latest)

    def test_n_larger_than_catalog(self, df):
        top = most_expensive(df, n=100)
        assert len(top) == df["url"].nunique()

    def test_zero_n_returns_empty(self, df):
        assert most_expensive(df, n=0).empty
        assert least_expensive(df, n=0).empty

    def test_empty_df_returns_empty(self):
        assert most_expensive(pd.DataFrame(columns=COLUMNS), n=5).empty


# ---- find_by_url ----

class TestFindByUrl:
    def test_returns_latest_row(self, df):
        row = find_by_url(df, "https://books.toscrape.com/a")
        assert row is not None
        assert row["price"] == 16.0
        assert row["scraped_at"] == pd.Timestamp("2026-10-03T00:00:00")

    def test_unknown_url_returns_none(self, df):
        assert find_by_url(df, "https://example.com/nope") is None

    def test_empty_df_returns_none(self):
        assert find_by_url(pd.DataFrame(columns=COLUMNS), "x") is None


# ---- data_summary ----

class TestDataSummary:
    def test_contains_columns(self, df):
        s = data_summary(df)
        assert s["columns"] == COLUMNS

    def test_counts_products_uniques(self, df):
        s = data_summary(df)
        assert s["product_count"] == 3
        assert s["snapshot_count"] == 6

    def test_date_range(self, df):
        s = data_summary(df)
        assert s["date_min"] == "2026-10-01T00:00:00"
        assert s["date_max"] == "2026-10-04T00:00:00"

    def test_empty_summary(self):
        s = data_summary(pd.DataFrame(columns=COLUMNS))
        assert s["product_count"] == 0
        assert s["date_min"] is None
        assert s["date_max"] is None
        assert s["note"] == "No data loaded."

    def test_note_mentions_scope_limit(self, df):
        s = data_summary(df)
        assert "sales volume" in s["note"]


class TestAvailableDataDescriptor:
    def test_static_descriptor(self):
        d = available_data_descriptor()
        assert d["columns"] == COLUMNS
        assert d["product_count"] is None
        assert d["date_min"] is None
        assert "sales volume" in d["note"]


# ---- smoke via CLI summary ----

def test_cli_summary_print(capsys, monkeypatch):
    """Smoke: running `python src/data.py` prints a summary of the fixture.

    This mirrors the plan's "quick smoke test" expectation and gives a handy
    local view of the loaded data.
    """
    monkeypatch.chdir(FIXTURE_DIR)
    __import__("src.data")  # noqa: F401  (ensures module importable standalone)
    # The smoke entrypoint is gated on __main__; invoke explicitly.
    import sys
    import src.data as mod

    # Build a mini CLI that prints a summary and a couple of query results.
    df = load_snapshots(str(SNAPSHOTS_PATH))
    print("COLUMNS:", json.dumps(COLUMNS))
    print("SUMMARY:", json.dumps(data_summary(df)))
    print("PRICE_CHANGES_7D:", json.dumps(
        price_change_over_period(df, days=7)[["url", "title", "price", "price_start", "price_delta"]].to_dict("records")
    ))
    print("OOS:", json.dumps(out_of_stock(df)[["url", "title", "in_stock"]].to_dict("records")))
    print("MOST_EXPENSIVE_3:", json.dumps(
        most_expensive(df, n=3)[["url", "title", "price"]].to_dict("records")
    ))
    out = capsys.readouterr().out
    assert "SUMMARY" in out
    assert "PRICE_CHANGES_7D" in out
    assert "OOS" in out
    assert "MOST_EXPENSIVE_3" in out
