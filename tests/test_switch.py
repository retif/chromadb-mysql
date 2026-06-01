import sys

import chromadb_mysql_backend
import chromadb_switch


def _reset():
    sys.modules.pop("chromadb", None)


def test_disabled_is_noop(monkeypatch):
    _reset()
    monkeypatch.delenv(chromadb_switch.ENV_VAR, raising=False)
    assert chromadb_switch.activate() is False
    assert "chromadb" not in sys.modules


def test_enabled_aliases_chromadb(monkeypatch):
    _reset()
    monkeypatch.setenv(chromadb_switch.ENV_VAR, "mysql")
    assert chromadb_switch.activate() is True
    import chromadb  # now resolves to the backend

    assert chromadb is chromadb_mysql_backend
    assert hasattr(chromadb, "PersistentClient")
    assert chromadb.__version__.endswith("+mysql")
    assert chromadb_switch.activate() is True  # idempotent
    _reset()


def test_enabled_value_must_match(monkeypatch):
    _reset()
    monkeypatch.setenv(chromadb_switch.ENV_VAR, "chroma")
    assert chromadb_switch.activate() is False
