"""Client — the ChromaDB PersistentClient surface mempalace uses.

Methods: get_collection, get_or_create_collection, create_collection,
delete_collection, list_collections. ``path`` is accepted for signature
compatibility and ignored (state lives in MySQL). Embeddings default to in-DB
ML_EMBED; the embedder is None unless a client fallback is injected.
"""

from __future__ import annotations

from . import _embed, _mysql
from ._collection import Collection
from .errors import NotFoundError

# Back-compat alias: callers/tests that referenced the old private name still
# catch the chromadb-compatible NotFoundError.
_CollectionError = NotFoundError


class Client:
    def __init__(
        self,
        path: str | None = None,
        embed_model: str | None = None,
        embedder: "_embed.Embedder | None" = None,
        pool: "_mysql.Pool | None" = None,
    ):
        self._path = path
        self._embed_model = embed_model or _embed.model_from_env()
        self._embedder = embedder  # optional client fallback, default None
        self._pool = pool or _mysql.Pool()

    def _exists(self, name: str) -> bool:
        rows = self._pool.execute(
            "SELECT 1 FROM information_schema.tables "
            "WHERE table_schema = DATABASE() AND table_name = %s",
            [name],
            fetch=True,
        )
        return bool(rows)

    def _collection(self, name, metadata=None):
        return Collection(
            name,
            self._pool,
            embed_model=self._embed_model,
            embedder=self._embedder,
            metadata=metadata,
        )

    def create_collection(self, name, metadata=None, **_):
        self._pool.ddl_create_table(name, _embed.DEFAULT_DIM)
        return self._collection(name, metadata)

    def get_or_create_collection(self, name, metadata=None, **_):
        self._pool.ddl_create_table(name, _embed.DEFAULT_DIM)
        return self._collection(name, metadata)

    def get_collection(self, name, **_):
        if not self._exists(name):
            raise NotFoundError(f"Collection {name} does not exist.")
        return self._collection(name)

    def delete_collection(self, name, **_):
        self._pool.execute(f"DROP TABLE IF EXISTS `{name}`")

    def list_collections(self):
        rows = (
            self._pool.execute(
                "SELECT table_name AS name FROM information_schema.tables "
                "WHERE table_schema = DATABASE()",
                fetch=True,
            )
            or []
        )
        return [self._collection(r["name"]) for r in rows]
