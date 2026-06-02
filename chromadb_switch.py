"""Runtime import-redirector — the switch between standard ChromaDB and the
MySQL-backed extension.

mempalace does a plain ``import chromadb``. When the deployment opts into the
MySQL backend (Helm value -> env var), this module aliases ``chromadb`` to
``chromadb_mysql_backend`` in ``sys.modules`` *before* mempalace imports it, so
the real ``chromadb`` package is never loaded. When the env var is unset/false,
this is a no-op and the standard ChromaDB path runs unchanged — the extension is
strictly opt-in and the default behaviour is preserved.

Activation is automatic via the installed ``activate-chromadb-mysql.pth`` (a
``.pth`` whose ``import chromadb_switch`` line runs at interpreter startup). It is
also safe to call ``activate()`` explicitly (e.g. from an entrypoint or a test).

Toggle env var: ``MEMPALACE_CHROMA_BACKEND`` — ``mysql`` enables the extension;
anything else (or unset) keeps standard ChromaDB.
"""

from __future__ import annotations

import importlib
import os
import sys

ENV_VAR = "MEMPALACE_CHROMA_BACKEND"
ENABLE_VALUE = "mysql"

# Modules that do `from .backends.chroma import hnsw_capacity_status` at import
# time, so their local binding must be rebound when we swap the source symbol.
_CAPACITY_CONSUMERS = ("mempalace.mcp_server",)


def enabled() -> bool:
    return os.environ.get(ENV_VAR, "").strip().lower() == ENABLE_VALUE


def _patch_capacity_probe() -> None:
    """Reroute mempalace's embedded-Chroma HNSW capacity probe to the MySQL
    stand-in.

    ``mempalace.backends.chroma.hnsw_capacity_status`` reads ``chroma.sqlite3``
    and the HNSW metadata pickle *directly as files* to guard the embedded
    #1222 segfault. Under the MySQL backend there is no HNSW segment, and the
    vestigial chroma.sqlite3 sits on a decommissioned object store, so that read
    hangs ~tens of seconds per search. Swap in the MySQL stand-in (never
    diverged; reports the live row count; never touches chroma.sqlite3) before
    mempalace's mcp_server imports the symbol, and rebind it if already imported.
    Best-effort: never raise — a failed patch must not break the chromadb
    redirect or the standard path.
    """
    repl = importlib.import_module(
        "chromadb_mysql_backend._capacity"
    ).hnsw_capacity_status
    chroma_mod = importlib.import_module("mempalace.backends.chroma")
    if getattr(chroma_mod, "hnsw_capacity_status", None) is repl:
        return
    chroma_mod.hnsw_capacity_status = repl
    for name in _CAPACITY_CONSUMERS:
        m = sys.modules.get(name)
        if m is not None and hasattr(m, "hnsw_capacity_status"):
            m.hnsw_capacity_status = repl


def activate() -> bool:
    """Alias ``chromadb`` -> the MySQL backend if enabled. Returns True if the
    redirect is in place, False otherwise. Idempotent."""
    if not enabled():
        return False
    backend = importlib.import_module("chromadb_mysql_backend")
    if sys.modules.get("chromadb") is not backend:
        sys.modules["chromadb"] = backend
        # Register submodules so `from chromadb.errors import NotFoundError`
        # resolves (the import machinery looks up sys.modules['chromadb.errors'],
        # not getattr on the aliased package). mempalace imports this at load.
        errors_mod = importlib.import_module("chromadb_mysql_backend.errors")
        sys.modules["chromadb.errors"] = errors_mod
        backend.errors = errors_mod
    # Reroute the HNSW capacity safeguard to the MySQL stand-in (idempotent,
    # best-effort). Done after the chromadb alias so importing
    # mempalace.backends.chroma resolves `import chromadb` to the shim.
    try:
        _patch_capacity_probe()
    except Exception:  # noqa: BLE001 - capacity patch must not break the redirect
        pass
    return True


# Auto-activate on import (triggered by the .pth at startup). Never raise — a
# broken switch must not take down the standard path.
try:  # pragma: no cover - exercised via interpreter startup, not unit tests
    activate()
except Exception:  # noqa: BLE001
    pass
