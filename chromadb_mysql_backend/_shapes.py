"""Build ChromaDB-shaped result dicts from plain DB rows.

Fidelity to Chroma's exact return shapes is the whole game — mempalace indexes
into ``results["distances"]``, ``results["metadatas"]``, ``r["ids"]`` etc., and
distinguishes ``get`` (flat lists) from ``query`` (nested per-query lists). These
are pure functions so they can be conformance-tested against real Chroma without
a database.

A "row" is a dict: ``{"id": str, "document": str|None, "metadata": dict|None,
"distance": float|None}``.
"""

from __future__ import annotations

# Chroma's defaults when the caller doesn't pass `include`.
GET_DEFAULT_INCLUDE = ["metadatas", "documents"]
QUERY_DEFAULT_INCLUDE = ["metadatas", "documents", "distances"]


def build_get_result(rows: list[dict], include: list[str] | None) -> dict:
    """Flat result for Collection.get(). ids are always returned."""
    inc = GET_DEFAULT_INCLUDE if include is None else include
    out: dict = {
        "ids": [r["id"] for r in rows],
        "embeddings": None,
        "documents": [r.get("document") for r in rows] if "documents" in inc else None,
        "metadatas": [r.get("metadata") for r in rows] if "metadatas" in inc else None,
        "uris": None,
        "data": None,
        "included": list(inc),
    }
    if "embeddings" in inc:
        out["embeddings"] = [r.get("embedding") for r in rows]
    return out


def build_query_result(per_query: list[list[dict]], include: list[str] | None) -> dict:
    """Nested per-query result for Collection.query(). One inner list per query
    text. ids are always returned."""
    inc = QUERY_DEFAULT_INCLUDE if include is None else include
    out: dict = {
        "ids": [[r["id"] for r in q] for q in per_query],
        "embeddings": None,
        "documents": (
            [[r.get("document") for r in q] for q in per_query]
            if "documents" in inc
            else None
        ),
        "metadatas": (
            [[r.get("metadata") for r in q] for q in per_query]
            if "metadatas" in inc
            else None
        ),
        "distances": (
            [[r.get("distance") for r in q] for q in per_query]
            if "distances" in inc
            else None
        ),
        "uris": None,
        "data": None,
        "included": list(inc),
    }
    if "embeddings" in inc:
        out["embeddings"] = [[r.get("embedding") for r in q] for q in per_query]
    return out
