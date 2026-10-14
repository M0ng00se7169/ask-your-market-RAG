"""
Tests for scripts/generate_history.py — the synthetic 30-day history generator.

Deterministic tests that assert on the fixed-seed output, the planned events,
and the general contract of the generator.
"""

from __future__ import annotations

import csv
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

# Make scripts importable without installing the repo.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import generate_history as gh  # noqa: E402


@pytest.fixture
def catalog():
    return gh.build_catalog()


@pytest.fixture
def rows(catalog):
    return gh.generate_snapshots(catalog=catalog, days=gh.DAYS, seed=gh.SEED)


@pytest.fixture
def date_range(rows):
    dates = sorted({r["scraped_at"] for r in rows})
    return dates[0], dates[-1]


snapshots_csv = REPO_ROOT / "data" / "snapshots.csv"
catalog_csv = REPO_ROOT / "data" / "catalog.csv"


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

class TestSchema:
    EXPECTED_SNAPSHOT_COLUMNS = [
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

    EXPECTED_CATALOG_COLUMNS = [
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

    def test_snapshots_csv_has_expected_columns(self):
        with snapshots_csv.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            assert reader.fieldnames == self.EXPECTED_SNAPSHOT_COLUMNS

    def test_catalog_csv_has_expected_columns(self):
        with catalog_csv.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            assert reader.fieldnames == self.EXPECTED_CATALOG_COLUMNS

    def test_snapshot_rows_have_all_columns(self, rows):
        for row in rows:
            for col in self.EXPECTED_SNAPSHOT_COLUMNS:
                assert col in row, f"missing column {col} in row {row['scraped_at']}"

    def test_catalog_rows_have_all_columns(self):
        with catalog_csv.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                for col in self.EXPECTED_CATALOG_COLUMNS:
                    assert col in row, f"missing column {col} in catalog row"


# ---------------------------------------------------------------------------
# Counts and coverage
# ---------------------------------------------------------------------------

class TestCounts:
    def test_row_count(self, rows):
        expected = len(gh.build_catalog()) * gh.DAYS
        assert len(rows) == expected, f"expected {expected} rows, got {len(rows)}"

    def test_distinct_dates(self, rows, date_range):
        dates = {r["scraped_at"] for r in rows}
        assert len(dates) == gh.DAYS, f"expected {gh.DAYS} distinct dates"
        start, end = date_range
        # 30 consecutive days
        d0 = datetime.strptime(start, "%Y-%m-%d")
        d1 = datetime.strptime(end, "%Y-%m-%d")
        assert (d1 - d0).days == gh.DAYS - 1, "date span does not cover 30 days"

    def test_distinct_products(self, rows):
        urls = {r["url"] for r in rows}
        assert len(urls) == len(gh.build_catalog())

    def test_one_row_per_product_per_day(self, rows):
        seen = set()
        for r in rows:
            key = (r["scraped_at"], r["url"])
            assert key not in seen, f"duplicate snapshot {key}"
            seen.add(key)
        assert len(seen) == len(rows)


# ---------------------------------------------------------------------------
# Planned events (deterministic, checkable)
# ---------------------------------------------------------------------------

class TestPlannedEvents:
    """Assert on the hard-coded events so the eval references can rely on them."""

    @staticmethod
    def _day_to_date(rows, day_index):
        dates = sorted({r["scraped_at"] for r in rows})
        start = datetime.strptime(dates[0], "%Y-%m-%d")
        return (start + timedelta(days=day_index)).strftime("%Y-%m-%d")

    def _row(self, rows, url, day):
        day_str = self._day_to_date(rows, day)
        matches = [r for r in rows if r["url"] == url and r["scraped_at"] == day_str]
        assert matches, f"no row for {url} on day {day}"
        assert len(matches) == 1
        return matches[0]

    def test_price_drop_event(self, rows):
        url = gh.CATALOG[0]["url"]
        day11 = self._day_to_date(rows, 11)
        day12 = self._day_to_date(rows, 12)
        p_before = float(self._row(rows, url, 11)["price"])
        p_after = float(self._row(rows, url, 12)["price"])
        # The planned factor is 0.85, but the walk nudges the price before the
        # event, so we assert on direction and a meaningful drop, not an exact pct.
        assert p_after < p_before, "price should drop on day 12"
        drop_pct = (p_after - p_before) / p_before
        assert drop_pct < -0.05, f"drop too small: {drop_pct:.3%}"

        # After the drop the price stays meaningfully below the day-11 level
        # for the rest of the history (the event shifts the baseline).
        for d in range(13, gh.DAYS):
            later = float(self._row(rows, url, d)["price"])
            assert later < p_before, f"price on day {d} drifted back above pre-drop level"

    def test_stockout_event(self, rows):
        url = gh.CATALOG[1]["url"]
        # in stock before day 20
        for d in range(0, 20):
            assert self._row(rows, url, d)["in_stock"] == "true", f"day {d} unexpectedly OOS"
        # out of stock on days 20..24
        for d in range(20, 25):
            assert self._row(rows, url, d)["in_stock"] == "false", f"day {d} unexpectedly in stock"
        # back in stock from day 25 onward
        for d in range(25, gh.DAYS):
            assert self._row(rows, url, d)["in_stock"] == "true", f"day {d} still OOS"

    def test_promo_event_requiem(self, rows):
        url = gh.CATALOG[5]["url"]
        # no promo before day 8
        for d in range(0, 8):
            assert self._row(rows, url, d)["promo"] == "", f"day {d} unexpectedly has promo"
        # promo present days 8..12
        for d in range(8, 13):
            assert self._row(rows, url, d)["promo"] == "Flash sale", f"day {d} missing Flash sale"
        # no promo after day 12
        for d in range(13, gh.DAYS):
            assert self._row(rows, url, d)["promo"] == "", f"day {d} unexpectedly has promo"

    def test_promo_event_starving_hearts(self, rows):
        url = gh.CATALOG[10]["url"]
        for d in range(0, 18):
            assert self._row(rows, url, d)["promo"] == "", f"day {d} unexpectedly has promo"
        for d in range(18, 23):
            assert self._row(rows, url, d)["promo"] == "Buy 2, save 15%", f"day {d} missing promo"
        for d in range(23, gh.DAYS):
            assert self._row(rows, url, d)["promo"] == "", f"day {d} unexpectedly has promo"


# ---------------------------------------------------------------------------
# Realism / validity constraints
# ---------------------------------------------------------------------------

class TestRealism:
    def test_prices_within_realistic_band(self, rows):
        catalog = {p["url"]: p for p in gh.build_catalog()}
        for r in rows:
            base = catalog[r["url"]]["base_price"]
            price = float(r["price"])
            assert base * 0.75 <= price <= base * 1.15, (
                f"price {price} for {r['url']} outside realistic band around base {base}"
            )

    def test_ratings_in_range(self, rows):
        for r in rows:
            rating = float(r["rating"])
            assert 1.0 <= rating <= 5.0, f"rating {rating} out of range for {r['url']}"

    def test_prices_nonzero_positive(self, rows):
        for r in rows:
            assert float(r["price"]) > 0, f"non-positive price for {r['url']}"

    def test_currency_consistent_per_product(self, rows):
        seen: dict[str, str] = {}
        for r in rows:
            cur = seen.get(r["url"])
            if cur is None:
                seen[r["url"]] = r["currency"]
            else:
                assert r["currency"] == cur, f"currency changed for {r['url']}"

    def test_in_stock_is_boolean_string(self, rows):
        for r in rows:
            assert r["in_stock"] in ("true", "false"), f"bad in_stock value: {r['in_stock']!r}"

    def test_reviews_count_is_non_negative_int(self, rows):
        for r in rows:
            n = int(r["reviews_count"])
            assert n >= 0, f"negative reviews_count for {r['url']}"


# ---------------------------------------------------------------------------
# Description variety (semantic search should have something to work with)
# ---------------------------------------------------------------------------

class TestDescriptions:
    def test_all_products_have_non_empty_description(self, rows):
        for r in rows:
            assert r["description"], f"empty description for {r['url']}"
            assert len(r["description"]) > 40, (
                f"description too short for {r['url']}: {r['description']!r}"
            )

    def test_descriptions_vary_across_products(self, rows):
        by_url = {r["url"]: r["description"] for r in rows}
        descriptions = list(by_url.values())
        unique = set(descriptions)
        assert len(unique) == len(descriptions), "all descriptions are identical"

    def test_catalog_description_matches_latest_snapshot(self, rows):
        """The catalog.csv description for each product equals the description
        in the most recent snapshot for that product."""
        # rebuild latest per url from rows
        by_url: dict[str, dict] = {}
        for r in rows:
            cur = by_url.get(r["url"])
            if cur is None or r["scraped_at"] > cur["scraped_at"]:
                by_url[r["url"]] = r

        with catalog_csv.open(newline="", encoding="utf-8") as f:
            for cat_row in csv.DictReader(f):
                snap = by_url[cat_row["url"]]
                assert snap["description"] == cat_row["description"], (
                    f"catalog description mismatch for {cat_row['url']}"
                )


# ---------------------------------------------------------------------------
# Seed reproducibility
# ---------------------------------------------------------------------------

class TestReproducibility:
    def test_same_seed_same_output(self, rows):
        rows2 = gh.generate_snapshots(seed=gh.SEED)
        assert rows == rows2, "regenerating with the same seed produced different rows"

    def test_different_seed_different_output(self, rows):
        rows2 = gh.generate_snapshots(seed=99999)
        assert rows != rows2, "different seeds produced identical rows (suspicious)"


# ---------------------------------------------------------------------------
# write_csv / write_catalog round-trip
# ---------------------------------------------------------------------------

class TestWriteCsv:
    def test_write_csv_produces_valid_file(self, tmp_path):
        path = tmp_path / "out.csv"
        gh.write_csv([], path=path)
        # empty list should still write a header
        assert path.exists()
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            assert reader.fieldnames == TestSchema.EXPECTED_SNAPSHOT_COLUMNS

    def test_write_csv_round_trip(self, tmp_path, rows):
        path = tmp_path / "roundtrip.csv"
        gh.write_csv(rows, path=path)
        with path.open(newline="", encoding="utf-8") as f:
            read_back = list(csv.DictReader(f))
        # DictReader returns all values as strings; normalize `rows` to the
        # same representation before comparing.
        def normalize(row):
            return {k: str(v) for k, v in row.items()}
        assert [normalize(r) for r in rows] == read_back

    def test_write_catalog_round_trip(self, tmp_path, rows):
        path = tmp_path / "catalog_roundtrip.csv"
        gh.write_catalog(rows=rows, path=path)
        with path.open(newline="", encoding="utf-8") as f:
            read_back = list(csv.DictReader(f))
        # The catalog is the latest snapshot per URL. Derive the expected
        # latest-per-URL from `rows` and compare catalog columns only.
        cols = TestSchema.EXPECTED_CATALOG_COLUMNS

        def latest_per_url(src):
            by: dict[str, dict] = {}
            for r in src:
                cur = by.get(r["url"])
                if cur is None or r["scraped_at"] > cur["scraped_at"]:
                    by[r["url"]] = r
            return sorted(by.values(), key=lambda r: r["url"])

        def subset(row):
            return {k: str(row[k]) for k in cols}

        expected = [subset(r) for r in latest_per_url(rows)]
        actual = [subset(r) for r in sorted(read_back, key=lambda r: r["url"])]
        assert expected == actual
