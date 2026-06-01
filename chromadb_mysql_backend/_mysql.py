"""MySQL connection + DDL for the ChromaDB-compatible backend.

Connection config comes from env (set by the Helm chart / HelmRelease, mirroring
the grafana/gitea HeatWave wiring). The host reaches a VCN-private MySQL with TLS
off (``require_secure_transport=OFF``). No host default — it's supplied per
deployment, never baked into source.

LIVE-DB CODE — not exercised by the unit tests (which cover the pure
where/shape/switch logic). Validated against HeatWave in the conformance phase
(milestone #60, issue #7).
"""

from __future__ import annotations

import os

# One table per Chroma collection. metadata is JSON; wing/room are generated
# columns purely to back an index for the hot equality filters. embedding is a
# fixed-width VECTOR; dim is set from the embedder at create time.
_DDL = """
CREATE TABLE IF NOT EXISTS `{table}` (
  id           VARCHAR(191) PRIMARY KEY,
  document     LONGTEXT,
  metadata     JSON,
  embedding    VECTOR({dim}),
  wing         VARCHAR(255) AS (JSON_UNQUOTE(JSON_EXTRACT(metadata,'$.wing'))) STORED,
  room         VARCHAR(255) AS (JSON_UNQUOTE(JSON_EXTRACT(metadata,'$.room'))) STORED,
  KEY idx_wing_room (wing, room)
)
"""


def config_from_env() -> dict:
    # Host has no default on purpose — it's deployment infra, supplied via env
    # (the Helm chart / HelmRelease). Keeps the VCN topology out of the source.
    return {
        "host": os.environ["MEMPALACE_MYSQL_HOST"],
        "port": int(os.environ.get("MEMPALACE_MYSQL_PORT", "3306")),
        "user": os.environ.get("MEMPALACE_MYSQL_USER", "mempalace"),
        "password": os.environ.get("MEMPALACE_MYSQL_PASSWORD", ""),
        "database": os.environ.get("MEMPALACE_MYSQL_DB", "mempalace"),
    }


class Pool:
    """Thin lazy wrapper over a mysql-connector pool."""

    def __init__(self, cfg: dict | None = None, size: int = 4):
        self._cfg = cfg or config_from_env()
        self._size = size
        self._pool = None

    def _ensure(self):
        if self._pool is None:
            from mysql.connector import pooling  # lazy

            self._pool = pooling.MySQLConnectionPool(
                pool_name="mempalace", pool_size=self._size, **self._cfg
            )
        return self._pool

    def execute(self, sql: str, params: list | None = None, *, fetch: bool = False):
        conn = self._ensure().get_connection()
        try:
            cur = conn.cursor(dictionary=True)
            cur.execute(sql, params or [])
            rows = cur.fetchall() if fetch else None
            if not fetch:
                conn.commit()
            cur.close()
            return rows
        finally:
            conn.close()

    def ddl_create_table(self, table: str, dim: int) -> None:
        self.execute(_DDL.format(table=table, dim=dim))
