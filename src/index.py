"""
Ask Your Market — vector index for product descriptions.

Reads snapshots.csv, dedups by URL (one record per product), embeds each
description, and stores the result in a local persisted Chroma collection.

Public API:
    build_index(csv_path, persist_dir, *, force=False) -> Collection
    get_collection(persist_dir) -> Collection
    search_descriptions(collection, query, top_k) -> list[dict]
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from chromadb import PersistentClient
from chromadb.config import Settings

from src.data import load_snapshots
from src.embed import embedder

# Collection name used inside the persisted Chroma dir.
COLLECTION_NAME = "products"
# Dimensionality of multilingual-e5-small.
EMBED_DIM = 384


def _ensure_persist_dir(persist_dir: Path) -> Path:
    persist_dir.mkdir(parents=True, exist_ok=True)
    return persist_dir


def _collection_any(persist_dir: Path) -> Any:
    """Return the Chroma PersistentClient.collection() object, creating the dir if needed."""
    client = PersistentClient(path=str(_ensure_persist_dir(persist_dir)))
    try:
        return client.get_collection(name=COLLECTION_NAME)
    except Exception:
        return client.create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine", "dimension": EMBED_DIM},
        )


def _catalog_from_csv(csv_path: Path) -> pd.DataFrame:
    """One row per product (latest snapshot), with metadata-ready columns."""
    df = load_snapshots(csv_path)
    if df.empty:
        raise ValueError(f"snapshots.csv at {csv_path} is empty")

    latest = df.sort_values("scraped_at").groupby("url", as_index=False).last()
    latest = latest[["url", "title", "price", "currency", "in_stock", "promo", "scraped_at", "description"]].copy()
    latest["scraped_at"] = latest["scraped_at"].dt.strftime("%Y-%m-%dT%H:%M:%S")
    latest["in_stock"] = latest["in_stock"].astype(bool)
    latest["price"] = latest["price"].astype(float)
    return latest


def _metadata(record: dict) -> dict:
    """Chroma metadata dict for one product record."""
    return {
        "url": record["url"],
        "title": record["title"],
        "price": float(record["price"]) if record["price"] is not None else None,
        "currency": record["currency"],
        "in_stock": bool(record["in_stock"]),
        "promo": str(record["promo"]) if record["promo"] else "",
        "scraped_at": str(record["scraped_at"]),
    }


def build_index(
    csv_path: str | Path,
    persist_dir: str | Path,
    *,
    force: bool = False,
) -> Any:
    """Build (or rebuild) the product description index.

    One Chroma record per product, keyed by URL, with metadata carrying
    title/price/currency/in_stock/promo/scraped_at so source attribution
    does not need a fresh CSV lookup on every search.

    Returns the Chroma collection object.
    """
    csv_path = Path(csv_path)
    persist_dir = Path(persist_dir)

    catalog = _catalog_from_csv(csv_path)
    texts = catalog["description"].fillna("").tolist()
    embeddings = embedder.embed_batch(texts)

    coll = _collection_any(persist_dir)

    if force:
        # Clear ALL existing records via the Chroma API so the live client
        # stays healthy (avoids nuking files that are memory-mapped on Windows).
        try:
            existing = coll.get(include=["ids"])
            ids_to_clear = existing.get("ids", []) or []
            if ids_to_clear:
                coll.delete(ids=ids_to_clear)
        except Exception:
            pass

    ids = [str(i) for i in range(len(catalog))]
    metadatas = [_metadata(row) for _, row in catalog.iterrows()]
    coll.add(
        ids=ids,
        embeddings=embeddings,
        metadatas=metadatas,
        documents=catalog["description"].fillna("").tolist(),
    )
    return coll


def get_collection(persist_dir: str | Path) -> Any:
    """Load an existing persisted collection. Raises if not found/empty."""
    persist_dir = Path(persist_dir)
    if not persist_dir.exists() or not any(persist_dir.iterdir()):
        raise FileNotFoundError(
            f"No persisted index found at {persist_dir}. "
            "Run build_index(csv_path, persist_dir) first."
        )
    coll = _collection_any(persist_dir)
    count = coll.count()
    if count == 0:
        raise FileNotFoundError(
            f"Persisted index at {persist_dir} exists but is empty. "
            "Run build_index(csv_path, persist_dir) first."
        )
    return coll


def search_descriptions(
    collection: Any,
    query: str,
    top_k: int = 10,
) -> list[dict]:
    """Semantic search over product descriptions.

    Returns up to `top_k` products, each as a dict with url/title/price/
    currency/in_stock/promo/scraped_at and a cosine similarity score in [0,1]
    (higher = more similar). The query is embedded with the E5 `query:` prefix.
    """
    if top_k <= 0:
        return []

    results = collection.query(
        query_embeddings=[embedder._model_ref().encode(
            "query: " + query, convert_to_numpy=True
        ).tolist()],
        n_results=min(top_k, collection.count() or top_k),
        include=["metadatas", "documents", "distances"],
    )

    # Distances from cosine space are 1 - cosine similarity.
    out: list[dict] = []
    ids = results.get("ids", [[]])[0] or []
    dists = results.get("distances", [[]])[0] or []
    metas = results.get("metadatas", [[]])[0] or []
    docs = results.get("documents", [[]])[0] or []

    for i, pid in enumerate(ids):
        dist = float(dists[i]) if i < len(dists) else 0.0
        score = max(0.0, min(1.0, 1.0 - dist))
        meta = dict(metas[i]) if i < len(metas) else {}
        doc = str(docs[i]) if i < len(docs) else ""
        out.append({
            "id": pid,
            "url": meta.get("url", ""),
            "title": meta.get("title", ""),
            "price": meta.get("price"),
            "currency": meta.get("currency", ""),
            "in_stock": meta.get("in_stock", True),
            "promo": meta.get("promo", ""),
            "scraped_at": meta.get("scraped_at", ""),
            "description": doc,
            "score": score,
        })
    out.sort(key=lambda r: r["score"], reverse=True)
    return out


def index_summary(collection: Any) -> dict:
    """Small descriptor of the built index for logs / the LLM context."""
    try:
        count = collection.count()
    except Exception:
        count = 0
    return {
        "collection": COLLECTION_NAME,
        "persist_dir": str(collection.path if hasattr(collection, "path") else "chroma_db"),
        "product_count": count,
        "embedding_dim": EMBED_DIM,
        "space": "cosine",
    }
