"""MySQL/HeatWave replacement for mempalace's embedded-Chroma HNSW capacity probe.

mempalace.backends.chroma.hnsw_capacity_status guards the embedded-ChromaDB
#1222 failure mode: the on-disk HNSW index can freeze at a small ``max_elements``
(e.g. 16 384) while ``chroma.sqlite3`` keeps accumulating embeddings (192 997),
so loading the undersized HNSW segment segfaults the process. It detects this by
reading ``chroma.sqlite3`` (embedding count) + the HNSW metadata pickle (element
count) *directly as files* and comparing them; on divergence it disables vector
search and falls back to BM25.

None of that exists on the MySQL/HeatWave backend:

* there is no HNSW segment, no metadata pickle, no ``max_elements`` ceiling;
* ``VECTOR_DISTANCE`` flat-scans the ``embedding`` column of the one table, so
  every stored row is always visible — there is nothing to "diverge" from and
  no undersized segment to segfault on.

So the faithful MySQL equivalent is "never capacity-diverged". Critically, it
must NOT read the vestigial ``chroma.sqlite3`` (left over from the embedded era):
on this deployment that file's data lives on a decommissioned object store, so
the read hangs for ~tens of seconds per call. We report the live row count from
MySQL instead (a fast COUNT) purely for transparency, mirroring the original
dict shape so mempalace's caller (``_refresh_vector_disabled_flag``) is satisfied.
"""

from __future__ import annotations

from typing import Any, Optional


def hnsw_capacity_status(
    palace_path: str, collection_name: str = "mempalace_drawers"
) -> dict:
    """MySQL-backed stand-in: vectors are never capacity-diverged.

    Same return shape as mempalace.backends.chroma.hnsw_capacity_status. Never
    reads chroma.sqlite3; never raises (a probe that throws would defeat the
    point — mempalace's caller swallows exceptions anyway).
    """
    out: dict[str, Any] = {
        "segment_id": None,
        "sqlite_count": None,
        "hnsw_count": None,
        "divergence": 0,
        "diverged": False,  # flat scan — never undersized, never disable vectors
        "status": "ok",
        "message": "",
    }
    n: Optional[int] = None
    try:
        from ._mysql import Pool

        rows = (
            Pool().execute(f"SELECT COUNT(*) AS n FROM `{collection_name}`", fetch=True)
            or []
        )
        n = int(rows[0]["n"]) if rows else 0
    except Exception:  # noqa: BLE001 - never raise from a safeguard probe
        out["message"] = (
            "MySQL/HeatWave VECTOR backend: flat scan, no HNSW capacity limit"
        )
        return out

    # In a flat-scan backend every stored row is "in the index", so sqlite_count
    # and hnsw_count are by definition equal — divergence is structurally zero.
    out["sqlite_count"] = n
    out["hnsw_count"] = n
    out["message"] = (
        f"MySQL/HeatWave VECTOR backend: {n:,} drawers, flat scan "
        "(no HNSW segment / no capacity ceiling)"
    )
    return out
