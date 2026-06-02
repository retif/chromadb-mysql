"""Test doubles for the KG live-DB seam.

MySQLKnowledgeGraph talks to MySQL only through a ``KGStore`` exposing
``init_schema()``, ``execute(sql, params)``, ``query(sql, params) -> rows`` and
``tx(fn)`` (fn receives a DictCursor-like object). Faking that seam lets the
SQL-building and result-shaping be asserted with no database — the same contract
the conformance phase checks against real HeatWave.

Also installs a minimal fake ``mempalace.knowledge_graph`` / ``mempalace.config``
into sys.modules so the dialect-independent helpers (_temporal_start_key,
_temporal_end_key, sanitize_iso_temporal) resolve without the real package.
"""

from __future__ import annotations

import sys
import types


class FakeCursor:
    """Records execute() calls; serves canned fetchone/fetchall results in order."""

    def __init__(self, fetchone_results=None, fetchall_results=None):
        self.calls = []  # (sql, params)
        self._one = list(fetchone_results or [])
        self._all = list(fetchall_results or [])

    def execute(self, sql, params=None):
        self.calls.append((sql, list(params or [])))

    def fetchone(self):
        return self._one.pop(0) if self._one else None

    def fetchall(self):
        return self._all.pop(0) if self._all else []


class FakeKGStore:
    """Records execute()/query()/tx() and serves canned rows."""

    def __init__(self, query_results=None, cursor=None):
        self.schema_inited = False
        self.execs = []  # (sql, params) from execute()
        self.queries = []  # (sql, params) from query()
        self.tx_cursors = []  # FakeCursor used in each tx()
        self._query_results = list(query_results or [])
        self._cursor = cursor or FakeCursor()

    def init_schema(self):
        self.schema_inited = True

    def execute(self, sql, params=None):
        self.execs.append((sql, list(params or [])))

    def query(self, sql, params=None):
        self.queries.append((sql, list(params or [])))
        return self._query_results.pop(0) if self._query_results else []

    def tx(self, fn):
        self.tx_cursors.append(self._cursor)
        return fn(self._cursor)


def install_fake_mempalace():
    """Register minimal fake mempalace modules so _import_helpers() resolves.

    The helpers are pure functions; we provide faithful tiny implementations
    (passthrough sanitize, ISO date-only normalisation) so temporal logic is
    still exercised. Returns a teardown callable."""
    saved = {k: sys.modules.get(k) for k in ("mempalace", "mempalace.knowledge_graph", "mempalace.config")}

    pkg = types.ModuleType("mempalace")
    pkg.__path__ = []  # mark as package

    kg = types.ModuleType("mempalace.knowledge_graph")

    def _is_date_only(v):
        return isinstance(v, str) and len(v) == 10 and v[4] == "-" and v[7] == "-"

    def _temporal_start_key(v):
        if v is None:
            return None
        return f"{v}T00:00:00Z" if _is_date_only(v) else v

    def _temporal_end_key(v):
        if v is None:
            return None
        return f"{v}T23:59:59Z" if _is_date_only(v) else v

    kg._temporal_start_key = _temporal_start_key
    kg._temporal_end_key = _temporal_end_key

    # A dummy default class so the switch has something to replace.
    class KnowledgeGraph:  # noqa: D401 - placeholder
        def __init__(self, db_path=None):
            self.db_path = db_path

    kg.KnowledgeGraph = KnowledgeGraph
    kg.DEFAULT_KG_PATH = "/tmp/fake-kg.sqlite3"

    cfg = types.ModuleType("mempalace.config")

    def sanitize_iso_temporal(value, field_name="value"):
        return value  # passthrough — validation is covered by mempalace's own tests

    cfg.sanitize_iso_temporal = sanitize_iso_temporal

    sys.modules["mempalace"] = pkg
    sys.modules["mempalace.knowledge_graph"] = kg
    sys.modules["mempalace.config"] = cfg

    def teardown():
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v

    return teardown
