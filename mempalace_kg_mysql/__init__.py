"""mempalace-kg-mysql — a switchable MySQL/HeatWave backend for mempalace's
temporal knowledge graph.

Drop-in replacement for ``mempalace.knowledge_graph.KnowledgeGraph`` that stores
entities and bitemporal triples in the shared HeatWave ``mempalace`` database
(same ``MEMPALACE_MYSQL_*`` connection env as the vector backend) instead of a
local ``knowledge_graph.sqlite3``. Activated via ``mempalace_kg_switch`` when
``MEMPALACE_KG_BACKEND=mysql``; otherwise the standard SQLite KG is untouched.

Sibling of the ChromaDB-compatible vector backend in this repo — together they
put the whole palace (vectors + graph) on one HeatWave backend.
"""

from __future__ import annotations

from ._kg import MySQLKnowledgeGraph

__all__ = ["MySQLKnowledgeGraph", "__version__"]
__version__ = "0.1.0"
