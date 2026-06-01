"""Embedding generation. Chroma embedded its documents implicitly; once we own
the storage we must own the embedding too.

Option A (default, recommended for cutover): bundle the *same* model Chroma used
by default — ``all-MiniLM-L6-v2`` (384-dim) — so vectors live in the same space
and the existing palace migrates 1:1 with no re-embed.

Option B (later): HeatWave in-DB ``ML_EMBED`` — implemented in `_mysql` as a SQL
path; selected by config, behind this same interface.

The model is loaded lazily so importing the package (and the switch shim) never
pulls in torch/onnx.
"""

from __future__ import annotations

from typing import Protocol

DEFAULT_MODEL = "all-MiniLM-L6-v2"
DEFAULT_DIM = 384


class Embedder(Protocol):
    dim: int

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class MiniLMEmbedder:
    """Option A — bundled sentence-transformers MiniLM (384-dim cosine)."""

    dim = DEFAULT_DIM

    def __init__(self, model_name: str = DEFAULT_MODEL):
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
    """MySQL 9 takes a vector as a JSON array string via STRING_TO_VECTOR()."""
    return "[" + ",".join(repr(float(x)) for x in vec) + "]"
