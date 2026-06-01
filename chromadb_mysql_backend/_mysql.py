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
    """Connection helper over pymysql (pure-Python — no protobuf/gRPC/C deps,
    unlike mysql-connector). mempalace is a single low-QPS MCP child, so we
    open a fresh autocommit connection per call: trivial overhead, and it
    sidesteps stale-connection handling over the long-lived VCN/ZT link.

    Name kept as ``Pool`` for _client/_collection compatibility.
    """

    def __init__(self, cfg: dict | None = None, **_):
        self._cfg = cfg or config_from_env()

    def _connect(self):
        import pymysql  # lazy — keeps the package importable for unit tests
        from pymysql.cursors import DictCursor

        return pymysql.connect(
            host=self._cfg["host"],
            port=self._cfg["port"],
            user=self._cfg["user"],
            password=self._cfg["password"],
            database=self._cfg["database"],
            autocommit=True,
            cursorclass=DictCursor,
        )

    def execute(self, sql: str, params: list | None = None, *, fetch: bool = False):
        conn = self._connect()
        try:
            with conn.cursor() as cur:
                cur.execute(sql, params or [])
                return cur.fetchall() if fetch else None
        finally:
            conn.close()

    def ddl_create_table(self, table: str, dim: int) -> None:
        self.execute(_DDL.format(table=table, dim=dim))
