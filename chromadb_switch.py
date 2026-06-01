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


def enabled() -> bool:
    return os.environ.get(ENV_VAR, "").strip().lower() == ENABLE_VALUE


def activate() -> bool:
    """Alias ``chromadb`` -> the MySQL backend if enabled. Returns True if the
    redirect is in place, False otherwise. Idempotent."""
    if not enabled():
        return False
    backend = importlib.import_module("chromadb_mysql_backend")
    if sys.modules.get("chromadb") is backend:
        return True
    sys.modules["chromadb"] = backend
    return True


# Auto-activate on import (triggered by the .pth at startup). Never raise — a
# broken switch must not take down the standard path.
try:  # pragma: no cover - exercised via interpreter startup, not unit tests
    activate()
except Exception:  # noqa: BLE001
    pass
