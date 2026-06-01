"""The redirect must expose chromadb.errors as a real submodule so mempalace's
module-level `from chromadb.errors import NotFoundError` resolves. Regression
guard for the import-time blocker."""

import sys

import chromadb_switch


def _reset():
    sys.modules.pop("chromadb", None)
    sys.modules.pop("chromadb.errors", None)


def test_activate_exposes_errors_submodule(monkeypatch):
    _reset()
    monkeypatch.setenv(chromadb_switch.ENV_VAR, "mysql")
    assert chromadb_switch.activate() is True

    from chromadb.errors import NotFoundError  # must resolve via sys.modules

    assert issubclass(NotFoundError, Exception)

    import chromadb

    assert hasattr(chromadb, "PersistentClient")
    assert chromadb.__version__.endswith("+mysql")
    _reset()
