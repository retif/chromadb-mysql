"""Embedding configuration.

DEFAULT (db mode): embeddings are generated IN-DATABASE by HeatWave
``sys.ML_EMBED_ROW(text, JSON_OBJECT('model_id', <model>))`` — no client model,
no torch/sentence-transformers in the closure. The SQL is emitted by
``_collection``; this module only supplies the model id + dim config.

FALLBACK (MEMPALACE_EMBED_MODE=client): the bundled ``MiniLMEmbedder`` computes
vectors client-side (lazy sentence-transformers; needs the ``embed`` extra).
Kept for parity/testing; not used by the light mysql image.
"""

from __future__ import annotations

import os
from typing import Protocol

# HeatWave embedding model_id (underscored form), 384-dim — closest to Chroma's
# default all-MiniLM-L6-v2 family. NOTE: L12 (HeatWave) ≠ L6 (Chroma); both
# 384-dim so no schema change, but vectors are NOT cross-comparable — a palace
# must be embedded entirely by one model.
DEFAULT_MODEL = "all_minilm_l12_v2"
DEFAULT_DIM = 384

ENV_MODEL = "MEMPALACE_EMBED_MODEL"
ENV_MODE = "MEMPALACE_EMBED_MODE"


def model_from_env() -> str:
    return os.environ.get(ENV_MODEL, DEFAULT_MODEL)


def mode_from_env() -> str:
    return os.environ.get(ENV_MODE, "db").strip().lower()


class Embedder(Protocol):
    dim: int

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class MiniLMEmbedder:
    """Optional client-side fallback (MEMPALACE_EMBED_MODE=client) — bundled
    sentence-transformers MiniLM (384-dim cosine). Lazy import so the default
    db-mode closure carries no torch."""

    dim = DEFAULT_DIM

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self._model_name = model_name
        self._model = None

    def _ensure(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # lazy

            self._model = SentenceTransformer(self._model_name)
        return self._model

    def embed(self, texts: list[str]) -> list[list[float]]:
        model = self._ensure()
        vecs = model.encode(list(texts), normalize_embeddings=True)
        return [list(map(float, v)) for v in vecs]


def vector_to_sql(vec: list[float]) -> str:
    """MySQL 9 takes a vector as a JSON array string via STRING_TO_VECTOR().
    Used by the precomputed-embeddings escape hatch + unit tests."""
    return "[" + ",".join(repr(float(x)) for x in vec) + "]"
