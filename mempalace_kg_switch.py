"""Runtime class-swapper — the switch between the standard SQLite knowledge
graph and the MySQL-backed one.

Unlike the ChromaDB switch (which aliases a *third-party* ``chromadb`` module in
``sys.modules``), the knowledge graph is *first-party* mempalace code:
``mempalace.knowledge_graph.KnowledgeGraph``. There is no third-party import to
hijack, so this swaps the class symbol directly. When the deployment opts into
the MySQL backend (Helm value -> env var), ``activate()`` replaces
``mempalace.knowledge_graph.KnowledgeGraph`` with ``MySQLKnowledgeGraph``
*before* mempalace's mcp_server / fact_checker import it, so every
``from .knowledge_graph import KnowledgeGraph`` binds the MySQL class. When the
env var is unset/false this is a no-op and the default SQLite path runs
unchanged — strictly opt-in, default behaviour preserved.

Activation is automatic via the installed ``activate-mempalace-kg-mysql.pth``
(its ``import mempalace_kg_switch`` line runs at interpreter startup). It is also
safe to call ``activate()`` explicitly from an entrypoint or a test.

Toggle env var: ``MEMPALACE_KG_BACKEND`` — ``mysql`` enables the extension;
anything else (or unset) keeps the standard SQLite KG.
"""

from __future__ import annotations

import importlib
import os
import sys

ENV_VAR = "MEMPALACE_KG_BACKEND"
ENABLE_VALUE = "mysql"

# Modules that bind ``KnowledgeGraph`` into their own namespace at import time.
# If any are already imported when activate() runs (shouldn't happen at .pth
# startup, but belt-and-suspenders for explicit/late activation), rebind them.
_BINDING_MODULES = ("mempalace.mcp_server", "mempalace.fact_checker")


def enabled() -> bool:
    return os.environ.get(ENV_VAR, "").strip().lower() == ENABLE_VALUE


def activate() -> bool:
    """Swap ``mempalace.knowledge_graph.KnowledgeGraph`` -> the MySQL backend if
    enabled. Returns True if the swap is in place, False otherwise. Idempotent."""
    if not enabled():
        return False

    from mempalace_kg_mysql import MySQLKnowledgeGraph

    kg_mod = importlib.import_module("mempalace.knowledge_graph")
    if getattr(kg_mod, "KnowledgeGraph", None) is MySQLKnowledgeGraph:
        return True

    kg_mod.KnowledgeGraph = MySQLKnowledgeGraph
    # Rebind any consumer that already did `from .knowledge_graph import
    # KnowledgeGraph` (their local name points at the old class object).
    for modname in _BINDING_MODULES:
        m = sys.modules.get(modname)
        if m is not None and hasattr(m, "KnowledgeGraph"):
            m.KnowledgeGraph = MySQLKnowledgeGraph
    return True


# Auto-activate on import (triggered by the .pth at startup). Never raise — a
# broken switch must not take down the standard path. Note: when disabled (the
# default), enabled() short-circuits BEFORE importing mempalace, so unrelated
# tools sharing this interpreter pay nothing.
try:  # pragma: no cover - exercised via interpreter startup, not unit tests
    activate()
except Exception:  # noqa: BLE001
    pass
