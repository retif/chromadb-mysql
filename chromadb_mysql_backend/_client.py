"""Client — the ChromaDB PersistentClient surface mempalace uses.

Methods: get_collection, get_or_create_collection, create_collection,
delete_collection, list_collections. ``path`` is accepted for signature
compatibility and ignored (state lives in MySQL, not on disk).
"""

from __future__ import annotations

from . import _embed, _mysql
from ._collection import Collection


class _CollectionError(Exception):
    """Raised when a collection is missing — mirrors Chroma's behaviour."""


class Client:
    def __init__(self, path: str | None = None, embedder: _embed.Embedder | None = None,
                 pool: "_mysql.Pool | None" = None):
        self._path = path
        self._embed = embedder or _embed.MiniLMEmbedder()
        self._pool = pool or _mysql.Pool()

    def _exists(self, name: str) -> bool:
        rows = self._pool.execute(
            "SELECT 1 FROM information_schema.tables "
            "WHERE table_schema = DATABASE() AND table_name = %s",
            [name],
            fetch=True,
        )
        return bool(rows)

    def create_collection(self, name, metadata=None, **_):
        self._pool.ddl_create_table(name, self._embed.dim)
        return Collection(name, self._pool, self._embed)

    def get_or_create_collection(self, name, metadata=None, **_):
        self._pool.ddl_create_table(name, self._embed.dim)
        return Collection(name, self._pool, self._embed)

    def get_collection(self, name, **_):
        if not self._exists(name):
            raise _CollectionError(f"Collection {name} does not exist.")
        return Collection(name, self._pool, self._embed)

    def delete_collection(self, name, **_):
        self._pool.execute(f"DROP TABLE IF EXISTS `{name}`")

    def list_collections(self):
        rows = self._pool.execute(
            "SELECT table_name AS name FROM information_schema.tables "
            "WHERE table_schema = DATABASE()",
            fetch=True,
        ) or []
        return [Collection(r["name"], self._pool, self._embed) for r in rows]
