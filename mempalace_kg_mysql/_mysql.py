"""MySQL connection + DDL for the temporal knowledge-graph backend.

Shares the *same* HeatWave MySQL connection env as the ChromaDB-compatible
vector backend (``MEMPALACE_MYSQL_*``) — the KG lives in the same ``mempalace``
database, in its own ``entities`` / ``triples`` tables. No vectors here: the KG
is plain relational + bitemporal, so this needs neither VECTOR columns nor
ML_EMBED, just small INSERT/SELECT round-trips.

LIVE-DB CODE — not exercised by the pure-logic unit tests; validated against
HeatWave in the conformance phase.
"""

from __future__ import annotations

import os

# Mirror of the SQLite schema in mempalace/knowledge_graph.py, translated to
# MySQL: TEXT PKs -> VARCHAR(191) (fits the utf8mb4 index-length budget),
# properties TEXT -> JSON, CURRENT_TIMESTAMP defaults preserved. Temporal
# columns stay VARCHAR (ISO strings, compared lexicographically) exactly as the
# SQLite KG does — the comparison normalisation lives in the temporal SQL exprs.
_DDL_ENTITIES = """
CREATE TABLE IF NOT EXISTS entities (
  id          VARCHAR(191) PRIMARY KEY,
  name        TEXT NOT NULL,
  type        VARCHAR(64) DEFAULT 'unknown',
  properties  JSON,
  created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
"""

_DDL_TRIPLES = """
CREATE TABLE IF NOT EXISTS triples (
  id              VARCHAR(191) PRIMARY KEY,
  subject         VARCHAR(191) NOT NULL,
  predicate       VARCHAR(191) NOT NULL,
  object          VARCHAR(191) NOT NULL,
  valid_from      VARCHAR(40),
  valid_to        VARCHAR(40),
  confidence      DOUBLE DEFAULT 1.0,
  source_closet   TEXT,
  source_file     TEXT,
  source_drawer_id TEXT,
  adapter_name    VARCHAR(191),
  extracted_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  KEY idx_triples_subject (subject),
  KEY idx_triples_object (object),
  KEY idx_triples_predicate (predicate),
  KEY idx_triples_valid (valid_from, valid_to)
)
"""


def config_from_env() -> dict:
    # Host has no default on purpose — deployment infra, supplied via env (the
    # Helm chart / HelmRelease), keeping the VCN topology out of source. Same
    # vars the vector backend reads, so one secret drives both.
    return {
        "host": os.environ["MEMPALACE_MYSQL_HOST"],
        "port": int(os.environ.get("MEMPALACE_MYSQL_PORT", "3306")),
        "user": os.environ.get("MEMPALACE_MYSQL_USER", "mempalace"),
        "password": os.environ.get("MEMPALACE_MYSQL_PASSWORD", ""),
        "database": os.environ.get("MEMPALACE_MYSQL_DB", "mempalace"),
    }


class KGStore:
    """pymysql helper for the KG (pure-Python driver — no protobuf/gRPC/C deps).

    Unlike the vector backend's per-call autocommit ``Pool``, the KG needs
    multi-statement transactions (``add_triple`` upserts two entities, checks
    for a live duplicate, then inserts the triple atomically). So this opens one
    connection per *high-level operation* via the ``tx``/``ro`` context managers
    and commits at the end — matching the SQLite ``with conn:`` semantics of the
    original.
    """

    def __init__(self, cfg: dict | None = None):
        self._cfg = cfg or config_from_env()

    def _connect(self, autocommit: bool):
        import pymysql  # lazy — keeps the package importable without the driver
        from pymysql.cursors import DictCursor

        return pymysql.connect(
            host=self._cfg["host"],
            port=self._cfg["port"],
            user=self._cfg["user"],
            password=self._cfg["password"],
            database=self._cfg["database"],
            autocommit=autocommit,
            cursorclass=DictCursor,
        )

    # Read-only single statement(s); autocommit, returns rows.
    def query(self, sql: str, params: list | None = None) -> list:
        conn = self._connect(autocommit=True)
        try:
            with conn.cursor() as cur:
                cur.execute(sql, params or [])
                return list(cur.fetchall())
        finally:
            conn.close()

    def execute(self, sql: str, params: list | None = None) -> None:
        conn = self._connect(autocommit=True)
        try:
            with conn.cursor() as cur:
                cur.execute(sql, params or [])
        finally:
            conn.close()

    # Multi-statement transaction. The callable receives a DictCursor; its
    # return value is propagated after commit. Rolls back on exception.
    def tx(self, fn):
        conn = self._connect(autocommit=False)
        try:
            with conn.cursor() as cur:
                result = fn(cur)
            conn.commit()
            return result
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def init_schema(self) -> None:
        self.execute(_DDL_ENTITIES)
        self.execute(_DDL_TRIPLES)
