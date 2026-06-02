"""Switch tests for the first-party KG class-swap. Unlike the chromadb switch
(sys.modules alias), this replaces mempalace.knowledge_graph.KnowledgeGraph and
rebinds already-imported consumers."""

import sys
import types

import mempalace_kg_switch
from _kg_fakes import install_fake_mempalace


def test_disabled_is_noop(monkeypatch):
    monkeypatch.delenv(mempalace_kg_switch.ENV_VAR, raising=False)
    # When disabled, activate() must short-circuit BEFORE importing mempalace.
    sys.modules.pop("mempalace", None)
    assert mempalace_kg_switch.activate() is False
    assert "mempalace" not in sys.modules  # never imported the app


def test_enabled_value_must_match(monkeypatch):
    monkeypatch.setenv(mempalace_kg_switch.ENV_VAR, "sqlite")
    assert mempalace_kg_switch.activate() is False


def test_enabled_swaps_class_and_rebinds_consumers(monkeypatch):
    teardown = install_fake_mempalace()
    try:
        # A consumer that already did `from .knowledge_graph import KnowledgeGraph`
        consumer = types.ModuleType("mempalace.mcp_server")
        consumer.KnowledgeGraph = sys.modules[
            "mempalace.knowledge_graph"
        ].KnowledgeGraph
        sys.modules["mempalace.mcp_server"] = consumer

        monkeypatch.setenv(mempalace_kg_switch.ENV_VAR, "mysql")
        assert mempalace_kg_switch.activate() is True

        from mempalace_kg_mysql import MySQLKnowledgeGraph

        # the module symbol is swapped
        assert (
            sys.modules["mempalace.knowledge_graph"].KnowledgeGraph
            is MySQLKnowledgeGraph
        )
        # and the already-imported consumer's local binding is rebound too
        assert consumer.KnowledgeGraph is MySQLKnowledgeGraph
        # idempotent
        assert mempalace_kg_switch.activate() is True
    finally:
        sys.modules.pop("mempalace.mcp_server", None)
        teardown()
