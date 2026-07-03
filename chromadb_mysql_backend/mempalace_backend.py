"""Native mempalace 3.5.0 storage backend backed by HeatWave MySQL.

mempalace 3.5.0 restructured storage into ``mempalace.backends``. Its
``ChromaBackend`` reads ``chroma.sqlite3`` *directly* via ``sqlite3.connect``,
bypassing ``import chromadb`` — so the historical "chromadb-mysql shim" (which
only aliases ``import chromadb``) never reaches MySQL on 3.5.0's read path.

This module closes that gap: it implements the real 3.5.0 ABC
(``mempalace.backends.base.BaseBackend`` / ``BaseCollection``) as a *thin
adapter* over this package's existing MySQL SQL layer. The heavy lifting —
``add``/``upsert``/``update``/``delete``/``get``/``query``/``count``/``count_by``
against the live 554k-drawer schema, in-DB ``ML_EMBED`` embedding, and
``VECTOR_DISTANCE COSINE`` search — already lives in
``chromadb_mysql_backend._collection.Collection``. Here we only:

* map the ABC's kwargs-only method surface onto the shim ``Collection``, and
* repackage the shim's chromadb-shaped dicts into the typed ``QueryResult`` /
  ``GetResult`` dataclasses the 3.5.0 core now expects.

The ABC is imported at module top from ``mempalace.backends.base`` — it is
present at runtime inside the mempalace image. Registered as the ``mysql``
backend via the ``mempalace.backends`` entry point in ``pyproject.toml``.
"""

from __future__ import annotations

import os
from typing import Optional

from mempalace.backends.base import (
    BaseBackend,
    BaseCollection,
    CollectionNotInitializedError,
    GetResult,
    HealthStatus,
    PalaceRef,
    QueryResult,
    _IncludeSpec,
)

from ._client import Client
from .errors import NotFoundError


# ---------------------------------------------------------------------------
# Collection adapter
# ---------------------------------------------------------------------------


class MySQLCollection(BaseCollection):
    """Adapts a shim ``chromadb_mysql_backend`` ``Collection`` to the 3.5.0 ABC.

    Writes delegate straight through (the ABC's kwargs-only names map onto the
    shim's positional args). Reads call the shim, which returns chromadb-shaped
    dicts, and are repackaged into typed ``QueryResult`` / ``GetResult``.
    """

    def __init__(self, shim_collection):
        self._c = shim_collection

    # -- identity ----------------------------------------------------------
    @property
    def name(self) -> str:
        return getattr(self._c, "name", "")

    # -- writes ------------------------------------------------------------
    def add(
        self,
        *,
        documents: list[str],
        ids: list[str],
        metadatas: Optional[list[dict]] = None,
        embeddings: Optional[list[list[float]]] = None,
    ) -> None:
        self._c.add(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)

    def upsert(
        self,
        *,
        documents: list[str],
        ids: list[str],
        metadatas: Optional[list[dict]] = None,
        embeddings: Optional[list[list[float]]] = None,
    ) -> None:
        self._c.upsert(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)

    def update(
        self,
        *,
        ids: list[str],
        documents: Optional[list[str]] = None,
        metadatas: Optional[list[dict]] = None,
        embeddings: Optional[list[list[float]]] = None,
    ) -> None:
        # The shim has a native partial per-id UPDATE (only provided columns are
        # touched, re-embedding from new document text in-DB when needed), so
        # override the ABC's non-atomic get+merge+upsert default with it.
        self._c.update(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)

    def delete(
        self,
        *,
        ids: Optional[list[str]] = None,
        where: Optional[dict] = None,
    ) -> None:
        self._c.delete(ids=ids, where=where)

    # -- reads -------------------------------------------------------------
    def get(
        self,
        *,
        ids: Optional[list[str]] = None,
        where: Optional[dict] = None,
        where_document: Optional[dict] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        include: Optional[list[str]] = None,
    ) -> GetResult:
        d = self._c.get(
            ids=ids,
            where=where,
            where_document=where_document,
            include=include,
            limit=limit,
            offset=offset,
        )
        return self._to_get_result(d, include)

    def query(
        self,
        *,
        query_texts: Optional[list[str]] = None,
        query_embeddings: Optional[list[list[float]]] = None,
        n_results: int = 10,
        where: Optional[dict] = None,
        where_document: Optional[dict] = None,
        include: Optional[list[str]] = None,
    ) -> QueryResult:
        num_queries = self._num_queries(query_texts, query_embeddings)
        spec = _IncludeSpec.resolve(include, default_distances=True)
        d = self._c.query(
            query_texts=query_texts,
            query_embeddings=query_embeddings,
            n_results=n_results,
            where=where,
            where_document=where_document,
            include=include,
        )
        return self._to_query_result(d, spec, num_queries)

    def count(self) -> int:
        return int(self._c.count())

    def estimated_count(self) -> int:
        return self.count()

    # -- aggregate / taxonomy helpers -------------------------------------
    def count_by(self, field: str, where: Optional[dict] = None) -> dict:
        """Server-side ``GROUP BY`` over an indexed generated column.

        Passthrough to the shim's ``count_by`` (``wing`` / ``room`` /
        ``source_file``) so the aggregate MCP tools (``status``, ``list_wings``,
        ``get_taxonomy``, ``graph_stats``) count in MySQL rather than streaming
        every drawer's metadata to the client.
        """
        return self._c.count_by(field, where=where)

    def get_all_metadata(self, where: Optional[dict] = None) -> list[dict]:
        """Page every matching record's metadata through the native SQL cursor.

        The shim's ``get(limit=, offset=)`` is a real server-side
        ``LIMIT/OFFSET``, so paging here walks MySQL's cursor rather than
        materializing-and-slicing. (This is the correct, per-record shape the
        callers expect; a pure ``count_by`` reconstruction cannot recover
        cross-field pairs such as (wing, room), so aggregate tools that need
        single-field totals should call :meth:`count_by` directly, which they
        can now do natively.)
        """
        out: list[dict] = []
        offset = 0
        page = 1000
        while True:
            batch = self.get(where=where, include=["metadatas"], limit=page, offset=offset)
            metas = batch.metadatas or []
            if not metas:
                break
            out.extend(metas)
            if len(metas) < page:
                break
            offset += len(metas)
        return out

    def health(self) -> HealthStatus:
        try:
            self._c.count()
        except Exception as exc:  # noqa: BLE001 - health must summarize, not raise
            return HealthStatus.unhealthy(str(exc))
        return HealthStatus.healthy()

    # -- shape conversion --------------------------------------------------
    @staticmethod
    def _num_queries(query_texts, query_embeddings) -> int:
        if query_texts:
            return len(query_texts)
        if query_embeddings:
            return len(query_embeddings)
        return 1

    @staticmethod
    def _to_get_result(d: Optional[dict], include: Optional[list[str]]) -> GetResult:
        if not d:
            return GetResult.empty()
        spec = _IncludeSpec.resolve(include, default_distances=False)
        ids = list(d.get("ids") or [])
        if not ids:
            return GetResult.empty()

        docs_raw = d.get("documents")
        documents = (
            [x if x is not None else "" for x in docs_raw]
            if spec.documents and docs_raw is not None
            else []
        )
        metas_raw = d.get("metadatas")
        metadatas = (
            [m if m is not None else {} for m in metas_raw]
            if spec.metadatas and metas_raw is not None
            else []
        )
        emb_raw = d.get("embeddings")
        embeddings = (
            [list(v) if v is not None else [] for v in emb_raw]
            if spec.embeddings and emb_raw is not None
            else None
        )
        return GetResult(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )

    @staticmethod
    def _to_query_result(d: Optional[dict], spec: _IncludeSpec, num_queries: int) -> QueryResult:
        ids_nested = (d or {}).get("ids") or []
        if not ids_nested:
            return QueryResult.empty(num_queries, embeddings_requested=spec.embeddings)

        n = len(ids_nested)
        ids = [list(q) for q in ids_nested]

        def _nested(raw, requested: bool, fill):
            if not requested or raw is None:
                return [[] for _ in range(n)]
            return [[x if x is not None else fill for x in q] for q in raw]

        documents = _nested(d.get("documents"), spec.documents, "")
        metadatas = _nested(d.get("metadatas"), spec.metadatas, {})
        distances = _nested(d.get("distances"), spec.distances, 0.0)
        if spec.embeddings:
            emb_raw = d.get("embeddings")
            if emb_raw is None:
                embeddings = [[] for _ in range(n)]
            else:
                embeddings = [
                    [list(v) if v is not None else [] for v in q] for q in emb_raw
                ]
        else:
            embeddings = None

        return QueryResult(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            distances=distances,
            embeddings=embeddings,
        )

    def close(self) -> None:
        return None


# ---------------------------------------------------------------------------
# Backend factory
# ---------------------------------------------------------------------------


class MySQLBackend(BaseBackend):
    """Native ``mysql`` backend: a server-mode SQL vector store over HeatWave.

    Lightweight on construction (no I/O); all connection work is deferred to
    :meth:`get_collection`, which builds a shim ``Client`` (a MySQL ``Pool``
    reading ``MEMPALACE_MYSQL_*`` env) and returns a :class:`MySQLCollection`.
    """

    name = "mysql"
    #: Server-mode SQL vector backend. Note the ABSENCE of
    #: ``requires_explicit_embeddings``: the shim embeds query text in-DB via
    #: HeatWave ``ML_EMBED`` (``query_texts`` path), so callers are NOT required
    #: to pre-compute query vectors.
    capabilities = frozenset(
        {
            "supports_embeddings_in",
            "supports_embeddings_passthrough",
            "supports_embeddings_out",
            "supports_metadata_filters",
            "supports_namespace_isolation",
            "supports_server_side_indexes",
            "server_embedder",
            "server_mode",
        }
    )
    distance_metric = "cosine"
    # MySQL/HeatWave maintains its own vector index and statistics; there is no
    # operator-facing analyze/reindex/compact analogue exposed through the shim,
    # so no maintenance kinds are advertised (declaring a no-op kind is a
    # conformance failure per RFC 001).
    maintenance_kinds = frozenset()

    def __init__(self):
        self._clients: dict[str, Client] = {}

    def _client_for(self, palace: PalaceRef) -> Client:
        # State lives in MySQL (env-configured); the shim ignores `path`. Cache
        # one client per palace id for handle reuse / isolation bookkeeping.
        client = self._clients.get(palace.id)
        if client is None:
            client = Client(path=palace.local_path)
            self._clients[palace.id] = client
        return client

    def get_collection(
        self,
        *args,
        **kwargs,
    ) -> MySQLCollection:
        palace, collection_name, create, _options = self._normalize_args(args, kwargs)
        client = self._client_for(palace)
        if create:
            shim = client.get_or_create_collection(collection_name)
        else:
            try:
                shim = client.get_collection(collection_name)
            except NotFoundError as exc:
                # Palace/DB is reachable but the collection was never
                # bootstrapped — mirror pgvector's not-initialized signal.
                raise CollectionNotInitializedError(collection_name) from exc
        return MySQLCollection(shim)

    @staticmethod
    def _normalize_args(args, kwargs):
        """Accept both the keyword-only ABC form and pgvector's positional form.

        ABC/canonical: ``get_collection(palace=PalaceRef, collection_name=...,
        create=False, options=None)``. Also tolerated: a leading positional
        ``palace_path`` (wrapped into a ``PalaceRef``) as pgvector does.
        """
        if "palace" in kwargs:
            palace = kwargs.pop("palace")
            if not isinstance(palace, PalaceRef):
                raise TypeError("palace= must be a PalaceRef instance")
            collection_name = kwargs.pop("collection_name")
            create = bool(kwargs.pop("create", False))
            options = kwargs.pop("options", None)
            if args or kwargs:
                raise TypeError("unexpected arguments to get_collection")
            return palace, collection_name, create, options
        if args:
            palace_path = args[0]
            rest = list(args[1:])
            collection_name = kwargs.pop("collection_name", None) or (rest.pop(0) if rest else None)
            if collection_name is None:
                raise TypeError("collection_name is required")
            create = kwargs.pop("create", False)
            if rest:
                create = rest.pop(0)
            options = kwargs.pop("options", None)
            if rest or kwargs:
                raise TypeError("unexpected arguments to get_collection")
            return (
                PalaceRef(id=palace_path, local_path=palace_path),
                collection_name,
                bool(create),
                options,
            )
        if "palace_path" in kwargs:
            palace_path = kwargs.pop("palace_path")
            collection_name = kwargs.pop("collection_name")
            create = bool(kwargs.pop("create", False))
            options = kwargs.pop("options", None)
            if kwargs:
                raise TypeError("unexpected arguments to get_collection")
            return (
                PalaceRef(id=palace_path, local_path=palace_path),
                collection_name,
                create,
                options,
            )
        raise TypeError("get_collection requires palace= or a positional palace_path")

    def close_palace(self, palace: PalaceRef) -> None:
        self._clients.pop(palace.id, None)

    def close(self) -> None:
        self._clients.clear()

    def health(self, palace: Optional[PalaceRef] = None) -> HealthStatus:
        try:
            client = self._client_for(palace) if palace is not None else Client()
            client.list_collections()
        except Exception as exc:  # noqa: BLE001 - health summarizes
            return HealthStatus.unhealthy(str(exc))
        return HealthStatus.healthy()

    @classmethod
    def detect(cls, path: str) -> bool:
        # Env/config-based, NOT a file-on-disk check: MySQL palaces have no local
        # artifact to sniff. This drives resolve_backend_for_palace's auto-detect
        # and the mismatch-protection path.
        return os.environ.get("MEMPALACE_BACKEND") == "mysql"
