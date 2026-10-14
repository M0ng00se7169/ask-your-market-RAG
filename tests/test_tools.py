"""
Tests for src/tools.py.

Cover tool_definitions format, run_query_snapshots for every question_type,
run_search_descriptions with a real Chroma collection, run_get_data_summary,
run_tool dispatch, and the format helpers.
"""

from pathlib import Path

import pandas as pd
import pytest

from src.data import COLUMNS, load_snapshots
from src.index import build_index, get_collection
from src.tools import (
    _cap_int,
    _df_to_records,
    _row_to_record,
    run_get_data_summary,
    run_query_snapshots,
    run_search_descriptions,
    run_tool,
    search_descriptions_tool,
    set_collection,
    tool_definitions,
    clear_collection,
)


# Fixture: a real CSV with descriptions so search tests can hit Chroma.
@pytest.fixture(scope="session")
def catalog_csv(tmp_path_factory):
    rows = [
        # Product A: two snapshots, price drops on day 3 (detectable within 7-day window from day 4).
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
            description="A mystery novel set in Victorian London with a detective and foggy streets.",
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
            description="A mystery novel set in Victorian London with a detective and foggy streets.",
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
            description="A bright children's picture book about animals in a colorful forest.",
        ),
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
            description="A practical cookbook with vegetarian recipes for busy weeknights.",
        ),
        dict(
            scraped_at="2026-10-03T00:00:00",
            url="https://books.toscrape.com/d",
            title="D",
            price=12.0,
            currency="GBP",
            in_stock=True,
            promo="",
            rating=4.0,
            reviews_count=15,
            description="A sci-fi short story collection about space travel and distant galaxies.",
        ),
    ]
    df = pd.DataFrame(rows, columns=COLUMNS)
    p = tmp_path_factory.mktemp("data") / "snapshots.csv"
    df.to_csv(p, index=False)
    return p


@pytest.fixture
def df(catalog_csv):
    return load_snapshots(catalog_csv)


@pytest.fixture
def collection(catalog_csv, tmp_path):
    coll = build_index(catalog_csv, tmp_path / "chroma_db")
    set_collection(coll)
    yield coll
    clear_collection()


class TestToolDefinitions:
    def test_count(self):
        defs = tool_definitions()
        assert len(defs) == 3

    def test_names(self):
        names = [d["function"]["name"] for d in tool_definitions()]
        assert set(names) == {"query_snapshots", "search_descriptions", "get_data_summary"}

    def test_openai_format(self):
        for d in tool_definitions():
            assert d["type"] == "function"
            assert "description" in d["function"]
            assert d["function"]["parameters"]["type"] == "object"
            assert "properties" in d["function"]["parameters"]
            assert "required" in d["function"]["parameters"]
            assert d["function"]["parameters"]["additionalProperties"] is False

    def test_query_snapshots_required(self):
        d = next(d for d in tool_definitions() if d["function"]["name"] == "query_snapshots")
        assert "question_type" in d["function"]["parameters"]["required"]
        assert "params" not in d["function"]["parameters"]["required"]

    def test_search_descriptions_required(self):
        d = next(d for d in tool_definitions() if d["function"]["name"] == "search_descriptions")
        assert "query" in d["function"]["parameters"]["required"]
        assert "top_k" not in d["function"]["parameters"]["required"]

    def test_get_data_summary_no_required(self):
        d = next(d for d in tool_definitions() if d["function"]["name"] == "get_data_summary")
        assert d["function"]["parameters"]["required"] == []


class TestCapInt:
    def test_default(self):
        assert _cap_int(None, 10, 20) == 10

    def test_value_within_range(self):
        assert _cap_int(5, 10, 20) == 5

    def test_value_above_max(self):
        assert _cap_int(25, 10, 20) == 20

    def test_zero(self):
        assert _cap_int(0, 10, 20) == 10

    def test_negative(self):
        assert _cap_int(-3, 10, 20) == 10

    def test_bad_type(self):
        assert _cap_int("abc", 10, 20) == 10


class TestFormatHelpers:
    def test_df_to_records_empty(self):
        df = pd.DataFrame(columns=["a", "b"])
        assert _df_to_records(df) == []

    def test_df_to_records_shape(self, df):
        from src.data import most_expensive
        top = most_expensive(df, 2)
        recs = _df_to_records(top)
        assert len(recs) == 2
        assert all("url" in r for r in recs)

    def test_df_to_records_nan_safe(self):
        df = pd.DataFrame({"x": [1.0, float("nan")], "y": ["a", "b"]})
        recs = _df_to_records(df)
        assert recs[0]["x"] == 1.0
        assert recs[1]["x"] is None

    def test_row_to_record(self, df):
        from src.data import find_by_url
        row = find_by_url(df, "https://books.toscrape.com/a")
        rec = _row_to_record(row)
        assert rec["url"] == "https://books.toscrape.com/a"
        assert rec["price"] == 16.0


class TestQuerySnapshots:
    def test_price_changes(self, df):
        result = run_query_snapshots("price_changes", {"days": 7}, df=df)
        assert result["question_type"] == "price_changes"
        assert "result" in result
        assert isinstance(result["result"], list)
        # Product A dropped price within 7 days.
        urls = {r["url"] for r in result["result"]}
        assert "https://books.toscrape.com/a" in urls

    def test_price_changes_caps_days(self, df):
        result = run_query_snapshots("price_changes", {"days": 999}, df=df)
        assert result["days"] == 90

    def test_price_changes_empty_query(self, df):
        result = run_query_snapshots("price_changes", {"days": 1}, df=df)
        assert result["result"] == []

    def test_out_of_stock(self, df):
        result = run_query_snapshots("out_of_stock", {}, df=df)
        assert result["question_type"] == "out_of_stock"
        urls = {r["url"] for r in result["result"]}
        assert "https://books.toscrape.com/b" in urls

    def test_product_history(self, df):
        result = run_query_snapshots("product_history", {"url": "https://books.toscrape.com/a"}, df=df)
        assert result["question_type"] == "product_history"
        assert result["url"] == "https://books.toscrape.com/a"
        assert len(result["result"]) == 2  # A has 2 snapshots in the fixture (price drop)
        prices = [r["price"] for r in result["result"]]
        assert prices == [20.0, 16.0]  # oldest to newest

    def test_product_history_missing_url(self, df):
        result = run_query_snapshots("product_history", {}, df=df)
        assert "error" in result

    def test_product_history_unknown_url(self, df):
        result = run_query_snapshots("product_history", {"url": "https://x.example/nope"}, df=df)
        assert "error" in result

    def test_most_expensive(self, df):
        result = run_query_snapshots("most_expensive", {"n": 2}, df=df)
        assert result["question_type"] == "most_expensive"
        assert len(result["result"]) == 2
        prices = [r["price"] for r in result["result"]]
        assert prices == sorted(prices, reverse=True)

    def test_most_expensive_caps_n(self, df):
        result = run_query_snapshots("most_expensive", {"n": 999}, df=df)
        assert result["n"] == 50

    def test_least_expensive(self, df):
        result = run_query_snapshots("least_expensive", {"n": 2}, df=df)
        assert result["question_type"] == "least_expensive"
        prices = [r["price"] for r in result["result"]]
        assert prices == sorted(prices)
        assert prices[0] == 10.0  # B

    def test_current_product(self, df):
        result = run_query_snapshots("current_product", {"url": "https://books.toscrape.com/c"}, df=df)
        assert result["question_type"] == "current_product"
        assert result["result"][0]["price"] == 30.0

    def test_current_product_missing_url(self, df):
        result = run_query_snapshots("current_product", {}, df=df)
        assert "error" in result

    def test_current_product_unknown_url(self, df):
        result = run_query_snapshots("current_product", {"url": "https://x.example/nope"}, df=df)
        assert "error" in result

    def test_available_data(self, df):
        result = run_query_snapshots("available_data", {}, df=df)
        assert result["question_type"] == "available_data"
        assert result["result"]["columns"] == list(COLUMNS)
        assert result["result"]["product_count"] == 4

    def test_unknown_question_type(self, df):
        result = run_query_snapshots("bogus_type", {}, df=df)
        assert "error" in result


class TestSearchDescriptions:
    def test_search_returns_results(self, collection, catalog_csv):
        result = run_search_descriptions("vegetarian recipes for busy weeknights", 10)
        assert result["query"] == "vegetarian recipes for busy weeknights"
        assert result["count"] >= 1
        # The cookbook should be the top match.
        top = result["result"][0]
        assert top["url"] == "https://books.toscrape.com/c"
        assert top["title"] == "C"

    def test_search_capped_top_k(self, collection):
        result = run_search_descriptions("mystery novel", 1)
        assert len(result["result"]) <= 1

    def test_search_empty_query(self):
        result = run_search_descriptions("", 10)
        assert result["count"] == 0
        assert result["result"] == []

    def test_search_needs_collection(self):
        clear_collection()
        with pytest.raises(RuntimeError, match="No Chroma collection"):
            run_search_descriptions("anything", 5)


class TestGetDataSummary:
    def test_summary_with_df(self, df):
        result = run_get_data_summary(verbose=False, df=df)
        assert result["verbose"] is False
        assert result["result"]["product_count"] == 4

    def test_summary_verbose(self, df):
        result = run_get_data_summary(verbose=True, df=df)
        assert result["verbose"] is True
        assert "note" in result
        assert "sales volume" in result["note"].lower() or "data summary" in result["note"].lower()

    def test_summary_fallback_when_no_df(self):
        # When df is None, falls back to available_data_descriptor (static).
        result = run_get_data_summary(verbose=False, df=None)
        assert result["result"]["product_count"] is None
        assert result["result"]["date_min"] is None


class TestToolDispatch:
    def test_dispatch_query_snapshots(self, df):
        call = {"name": "query_snapshots", "arguments": {"question_type": "out_of_stock", "params": {}}}
        result = run_tool(call, df=df)
        assert result["question_type"] == "out_of_stock"

    def test_dispatch_search_descriptions(self, collection):
        call = {"name": "search_descriptions", "arguments": {"query": "mystery novel", "top_k": 3}}
        result = run_tool(call)
        assert result["count"] >= 1
        assert result["result"][0]["title"] == "A"

    def test_dispatch_get_data_summary(self, df):
        call = {"name": "get_data_summary", "arguments": {"verbose": True}}
        result = run_tool(call, df=df)
        assert result["verbose"] is True
        assert result["result"]["product_count"] == 4

    def test_dispatch_unknown_tool(self, df):
        call = {"name": "nonexistent", "arguments": {}}
        result = run_tool(call, df=df)
        assert "error" in result
        assert "Unknown tool" in result["error"]


class TestSearchDescriptionsToolConvenience:
    def test_search_descriptions_tool_uses_singleton(self, collection):
        result = search_descriptions_tool("vegetarian recipes", 5)
        assert result["count"] >= 1
        assert result["result"][0]["title"] == "C"

    def test_search_descriptions_tool_no_collection_raises(self):
        clear_collection()
        with pytest.raises(RuntimeError, match="No Chroma collection"):
            search_descriptions_tool("anything", 5)
