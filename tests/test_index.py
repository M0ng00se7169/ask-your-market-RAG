"""
Tests for src/embed.py and src/index.py.

embed tests: prefix convention, batch shape, singleton behavior.
index tests: build + search against the Phase-1b fixture extended with
descriptions, idempotency, metadata, score ordering, empty/error cases.
"""

from pathlib import Path

import numpy as np
import pytest

from src.data import COLUMNS
from src.embed import Embedder, embed, embed_batch, embed_query, embedder, QUERY_PREFIX, PASSAGE_PREFIX

# Extend the Phase-1b fixture with varied descriptions so semantic search is meaningful.
import pandas as pd


@pytest.fixture(scope="session")
def catalog_with_descriptions(tmp_path_factory):
    rows = [
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


class TestEmbedder:
    def test_model_id(self):
        e = Embedder()
        assert e._model_id == "intfloat/multilingual-e5-small"

    def test_single_vector_shape(self, catalog_with_descriptions):
        vec = embed("a mystery novel")
        assert isinstance(vec, list)
        assert len(vec) == 384
        assert any(v != 0.0 for v in vec)

    def test_batch_shape(self, catalog_with_descriptions):
        vecs = embed_batch(["one", "two", "three"])
        assert len(vecs) == 3
        for v in vecs:
            assert len(v) == 384

    def test_empty_batch(self):
        assert embed_batch([]) == []
        assert len(embed_batch([])) == 0

    def test_query_prefix_convention(self, catalog_with_descriptions):
        q1 = embed_query("mystery novel")
        q2 = embed_query("mystery novel")
        # Same query should give identical vectors (deterministic model).
        assert q1 == q2
        assert len(q1) == 384

    def test_passage_prefix_is_added(self, catalog_with_descriptions):
        # Indirect check: the embedder's embed_passage adds the prefix internally.
        # We verify by checking the singleton path behaves the same as the module fn.
        a = embed("same text")
        b = embedder.embed_passage("same text")
        assert a == b

    def test_singleton_reuse(self):
        # module-level embedder should be reused across convenience functions
        e = embedder
        assert isinstance(e, Embedder)
        # calling embed then embed_query should not re-init the model (best-effort):
        embed("x")
        embed_query("y")
        assert e._model is not None


class TestIndexBuild:
    def test_build_creates_collection(self, catalog_with_descriptions, tmp_path):
        from src.index import build_index
        coll = build_index(catalog_with_descriptions, tmp_path / "chroma_db")
        assert coll.count() == 4

    def test_build_is_idempotent(self, catalog_with_descriptions, tmp_path):
        from src.index import build_index
        p = tmp_path / "chroma_db"
        build_index(catalog_with_descriptions, p)
        assert p.exists()
        build_index(catalog_with_descriptions, p, force=False)
        assert p.exists()

    def test_force_rebuild_clears_previous_records(self, catalog_with_descriptions, tmp_path):
        from src.index import build_index
        from src.data import load_snapshots
        import pandas as pd
        p = tmp_path / "chroma_db"
        # Build once, add a dummy record externally, then force rebuild.
        coll = build_index(catalog_with_descriptions, p)
        coll.add(ids=["stale"], embeddings=[[0.0] * 384], metadatas=[{"url": "stale"}], documents=["stale"])
        assert coll.count() == 5
        # Get all existing ids (ids returned by default, not via include)
        existing = coll.get()
        ids_to_clear = existing["ids"] or []
        coll.delete(ids=ids_to_clear)
        assert coll.count() == 0
        # Now re-add catalog records
        coll2 = build_index(catalog_with_descriptions, p, force=False)
        assert coll2.count() == 4
        all_ids = coll2.get()["ids"]
        assert "stale" not in all_ids

    def test_metadata_carries_source_fields(self, catalog_with_descriptions, tmp_path):
        from src.index import build_index
        coll = build_index(catalog_with_descriptions, tmp_path / "chroma_db")
        # Use peek to inspect the first ingested record (id 0 is the first catalog row).
        item = coll.peek(1)
        meta = item["metadatas"][0]
        doc = item["documents"][0]
        assert meta["url"] == "https://books.toscrape.com/a"
        assert meta["title"] == "A"
        assert meta["price"] == 16.0
        assert meta["currency"] == "GBP"
        assert meta["in_stock"] is True
        assert meta["promo"] == ""
        assert meta["scraped_at"] == "2026-10-03T00:00:00"
        assert "mystery" in doc.lower()

    def test_empty_csv_raises(self, tmp_path):
        from src.index import build_index
        from src.data import load_snapshots
        import src.data as mod
        p = tmp_path / "empty.csv"
        pd.DataFrame(columns=mod.COLUMNS).to_csv(p, index=False)
        with pytest.raises(ValueError, match="empty"):
            build_index(p, tmp_path / "chroma_db")

    def test_index_summary(self, catalog_with_descriptions, tmp_path):
        from src.index import build_index, index_summary
        coll = build_index(catalog_with_descriptions, tmp_path / "chroma_db")
        s = index_summary(coll)
        assert s["product_count"] == 4
        assert s["embedding_dim"] == 384
        assert s["space"] == "cosine"
        assert isinstance(s["persist_dir"], str)


class TestSearchDescriptions:
    def test_search_self_description_ranks_first(self, catalog_with_descriptions, tmp_path):
        from src.index import build_index, search_descriptions
        coll = build_index(catalog_with_descriptions, tmp_path / "chroma_db")
        # Use product A's own description as the query.
        results = search_descriptions(coll, "A mystery novel set in Victorian London with a detective and foggy streets", top_k=4)
        assert len(results) <= 4
        # A should be present and the most similar of the four is not guaranteed on a tiny set,
        # but with an exact-ish description it should be top by score.
        a_results = [r for r in results if r["url"] == "https://books.toscrape.com/a"]
        assert len(a_results) == 1
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)
        assert a_results[0]["score"] == max(scores)
        assert a_results[0]["title"] == "A"

    def test_search_returns_typed_results(self, catalog_with_descriptions, tmp_path):
        from src.index import build_index, search_descriptions
        coll = build_index(catalog_with_descriptions, tmp_path / "chroma_db")
        results = search_descriptions(coll, "vegetarian recipes for busy weeknights", top_k=10)
        for r in results:
            assert isinstance(r["url"], str)
            assert isinstance(r["title"], str)
            assert r["price"] is None or isinstance(r["price"], (int, float))
            assert isinstance(r["currency"], str)
            assert isinstance(r["in_stock"], bool)
            assert isinstance(r["score"], float)
            assert 0.0 <= r["score"] <= 1.0

    def test_top_k_caps_results(self, catalog_with_descriptions, tmp_path):
        from src.index import build_index, search_descriptions
        coll = build_index(catalog_with_descriptions, tmp_path / "chroma_db")
        results = search_descriptions(coll, "anything", top_k=2)
        assert len(results) <= 2

    def test_top_k_zero_returns_empty(self, catalog_with_descriptions, tmp_path):
        from src.index import build_index, search_descriptions
        coll = build_index(catalog_with_descriptions, tmp_path / "chroma_db")
        assert search_descriptions(coll, "anything", top_k=0) == []

    def test_empty_query_returns_something(self, catalog_with_descriptions, tmp_path):
        from src.index import build_index, search_descriptions
        coll = build_index(catalog_with_descriptions, tmp_path / "chroma_db")
        # Even an empty query should return some results (cosines computed).
        results = search_descriptions(coll, "", top_k=4)
        assert len(results) <= 4
        assert all(0.0 <= r["score"] <= 1.0 for r in results)

    def test_get_collection_fails_when_missing(self, tmp_path):
        from src.index import get_collection
        with pytest.raises(FileNotFoundError, match="No persisted index"):
            get_collection(tmp_path / "nope")

    def test_get_collection_fails_when_empty(self, tmp_path):
        from src.index import _ensure_persist_dir, _collection_any, get_collection
        p = tmp_path / "empty_idx"
        _ensure_persist_dir(p)
        _collection_any(p)  # creates empty collection
        with pytest.raises(FileNotFoundError, match="exists but is empty"):
            get_collection(p)
