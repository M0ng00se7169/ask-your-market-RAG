"""
Ask Your Market — local embeddings.

Singleton loader for intfloat/multilingual-e5-small, using the E5
`query:` / `passage:` prefix convention that was verified in Phase 0.

Public API:
    embed(text) -> list[float]       # single passage
    embed_batch(texts) -> list[list[float]]  # multiple passages
    embed_query(text) -> list[float]          # single query (query: prefix)
"""

from __future__ import annotations

from typing import Sequence

from sentence_transformers import SentenceTransformer

# Verified in Phase 0: this model + the e5 prefix convention produces
# ~0.87 cosine for a related pair and ~0.80 for an unrelated pair on the
# synthetic descriptions. No fallback needed for a 20-100 product catalog.
MODEL_ID = "intfloat/multilingual-e5-small"

# E5 convention: passages are prefixed with "passage: ", queries with "query: ".
PASSAGE_PREFIX = "passage: "
QUERY_PREFIX = "query: "


class Embedder:
    """Singleton wrapper around the local E5 model."""

    def __init__(self, model_id: str = MODEL_ID) -> None:
        self._model_id = model_id
        self._model: SentenceTransformer | None = None

    def _model_ref(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self._model_id)
        return self._model

    def embed(self, text: str) -> list[float]:
        """Embed one passage. Caller should pass the raw text; we add the prefix."""
        return self.embed_passage(text)

    def embed_passage(self, text: str) -> list[float]:
        """Embed one passage with the `passage:` prefix."""
        return self._model_ref().encode(
            PASSAGE_PREFIX + text, convert_to_numpy=True
        ).tolist()

    def embed_queries(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed multiple queries with the `query:` prefix.

        Kept separate from embed_batch so callers can opt into query-prefixing
        only where needed (search time).
        """
        if not texts:
            return []
        model = self._model_ref()
        prefixed = [QUERY_PREFIX + t for t in texts]
        out = model.encode(prefixed, convert_to_numpy=True)
        return out.tolist()

    def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed multiple passages with the `passage:` prefix (index build time)."""
        if not texts:
            return []
        model = self._model_ref()
        prefixed = [PASSAGE_PREFIX + t for t in texts]
        out = model.encode(prefixed, convert_to_numpy=True)
        return out.tolist()


# Module-level singleton; the app imports `embedder` directly.
embedder = Embedder()


def embed(text: str) -> list[float]:
    """Convenience: embed one passage via the module singleton."""
    return embedder.embed_passage(text)


def embed_batch(texts: Sequence[str]) -> list[list[float]]:
    """Convenience: embed multiple passages via the module singleton."""
    return embedder.embed_batch(texts)


def embed_query(text: str) -> list[float]:
    """Convenience: embed one query via the module singleton."""
    return embedder._model_ref().encode(
        QUERY_PREFIX + text, convert_to_numpy=True
    ).tolist()
