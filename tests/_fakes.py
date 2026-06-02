"""Test doubles for the live-DB seam.

Collection/Client talk to MySQL only through ``pool.execute(sql, params,
fetch=)`` and ``pool.ddl_create_table(table, dim)``. Faking that seam lets the
SQL-building and result-shaping be asserted with no database — the same contract
the conformance phase (#7) checks against real HeatWave.
"""

from __future__ import annotations


class FakePool:
    """Records every execute() and serves canned fetch results in order."""

    def __init__(self, fetch_results=None):
        self.calls = []  # (sql, params, fetch)
        self.ddl = []  # (table, dim)
        self._fetch = list(fetch_results or [])

    def execute(self, sql, params=None, *, fetch=False):
        self.calls.append((sql, list(params or []), fetch))
        if fetch:
            return self._fetch.pop(0) if self._fetch else []
        return None

    def execute_seq(self, statements, *, fetch_last=True):
        # records every statement (so tests can assert the SET @qv + SELECT
        # pair) and serves one canned fetch result for the sequence.
        for sql, params in statements:
            self.calls.append((sql, list(params or []), False))
        if fetch_last:
            return self._fetch.pop(0) if self._fetch else []
        return None

    def ddl_create_table(self, table, dim):
        self.ddl.append((table, dim))

    # convenience accessors -------------------------------------------------
    @property
    def sqls(self):
        return [c[0] for c in self.calls]

    def last(self):
        return self.calls[-1]


class FakeEmbedder:
    """Deterministic 3-dim embedder — vector encodes the input length so the
    embed call is observable in the emitted SQL params."""

    dim = 3

    def __init__(self):
        self.seen = []  # each embed() batch, in order

    def embed(self, texts):
        texts = list(texts)
        self.seen.append(texts)
        return [[float(len(t)), 0.0, 1.0] for t in texts]
