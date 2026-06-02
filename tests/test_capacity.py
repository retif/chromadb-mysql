"""Tests for the MySQL HNSW-capacity stand-in + the switch that reroutes
mempalace's embedded-Chroma probe to it."""

import sys
import types

import chromadb_mysql_backend._capacity as cap
import chromadb_mysql_backend._mysql as mysql
import chromadb_switch


class _FakePool:
    def __init__(self, *a, **k):
        pass

    def execute(self, sql, params=None, *, fetch=False):
        assert fetch and "COUNT(*)" in sql and "mempalace_drawers" in sql
        return [{"n": 243}]


def test_capacity_never_diverged(monkeypatch):
    monkeypatch.setattr(mysql, "Pool", _FakePool)
    out = cap.hnsw_capacity_status("/workspace/palace")
    # flat scan: never diverged, vectors stay enabled, counts equal, no read of
    # chroma.sqlite3 (the FakePool only answers a COUNT).
    assert out["diverged"] is False
    assert out["status"] == "ok"
    assert out["sqlite_count"] == 243 and out["hnsw_count"] == 243
    assert out["divergence"] == 0
    assert "243" in out["message"]


def test_capacity_never_raises_on_db_error(monkeypatch):
    class _Boom:
        def __init__(self, *a, **k):
            pass

        def execute(self, *a, **k):
            raise RuntimeError("db down")

    monkeypatch.setattr(mysql, "Pool", _Boom)
    out = cap.hnsw_capacity_status("/x")
    # a safeguard probe must never raise and must never disable vectors
    assert out["diverged"] is False and out["status"] == "ok"


def test_switch_reroutes_capacity_probe(monkeypatch):
    # fake mempalace.backends.chroma with the original (sqlite-reading) symbol
    saved = {k: sys.modules.get(k) for k in (
        "mempalace", "mempalace.backends", "mempalace.backends.chroma", "mempalace.mcp_server"
    )}
    try:
        pkg = types.ModuleType("mempalace")
        pkg.__path__ = []
        be = types.ModuleType("mempalace.backends")
        be.__path__ = []
        chroma = types.ModuleType("mempalace.backends.chroma")

        def _original(palace_path, collection_name="mempalace_drawers"):
            return {"status": "read-sqlite", "diverged": False}

        chroma.hnsw_capacity_status = _original
        # a consumer that already imported the symbol by value
        consumer = types.ModuleType("mempalace.mcp_server")
        consumer.hnsw_capacity_status = _original
        sys.modules.update({
            "mempalace": pkg, "mempalace.backends": be,
            "mempalace.backends.chroma": chroma, "mempalace.mcp_server": consumer,
        })

        chromadb_switch._patch_capacity_probe()

        assert chroma.hnsw_capacity_status is cap.hnsw_capacity_status
        assert consumer.hnsw_capacity_status is cap.hnsw_capacity_status  # rebound
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
