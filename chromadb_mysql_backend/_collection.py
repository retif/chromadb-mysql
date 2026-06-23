"""Collection — the ChromaDB Collection API surface mempalace uses, over MySQL.

Surface: add, upsert, update, get, query, delete, count, plus ``name`` and a
``metadata`` property. ``get`` returns flat lists; ``query`` returns nested
per-query lists (see _shapes). ``where`` supports equality + ``$and`` (see
_where); ``where_document`` is not supported and raises (never emitted by
mempalace, but the signature must accept it).

Embeddings default to IN-DB ML_EMBED: documents/query texts are embedded inline
in SQL via ``sys.ML_EMBED_ROW(text, JSON_OBJECT('model_id', <model>))`` — no
client model. A precomputed-embeddings path (callers passing ``embeddings`` /
``query_embeddings``) is retained as an escape hatch (STRING_TO_VECTOR).
"""

from __future__ import annotations

import json

from . import _embed, _shapes, _where

_NO_WHERE_DOC = "where_document is not supported by the mysql backend"


def _record_skips(table: str, skipped: list) -> None:
    """Append a durable, queryable record of rows that could not be inserted
    (per-row fallback in _write). One JSON object per line in
    ``$MEMPALACE_SKIP_LOG`` (default ``~/.mempalace/embed-skips.ndjson``) with
    the drawer id (encodes wing/room/source_file/chunk), source_file, error, and
    a doc preview — so a skipped chunk can be found and re-mined later. Best
    effort: never raise from the logging path."""
    import logging
    import os
    import time

    path = os.environ.get("MEMPALACE_SKIP_LOG") or os.path.expanduser(
        "~/.mempalace/embed-skips.ndjson"
    )
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        ts = time.time()
        with open(path, "a") as f:
            for rec in skipped:
                f.write(json.dumps({"ts": ts, "table": table, **rec}) + "\n")
    except Exception:  # noqa: BLE001 - logging must not break the write path
        pass
    logging.getLogger(__name__).warning(
        "chromadb_mysql_backend: skipped %d unfilable row(s) in %s -> %s",
        len(skipped),
        table,
        path,
    )


class Collection:
    def __init__(self, name, pool, embed_model=None, embedder=None, metadata=None):
        self.name = name
        self._pool = pool
        self._embed_model = embed_model or _embed.model_from_env()
        self._embedder = embedder  # optional client fallback, default None
        self._metadata = metadata or {}

    @property
    def metadata(self):
        return self._metadata or {}

    # ---- writes -----------------------------------------------------------
    def add(self, ids, documents=None, metadatas=None, embeddings=None):
        self._write(ids, documents, metadatas, embeddings)

    def upsert(self, ids, documents=None, metadatas=None, embeddings=None):
        self._write(ids, documents, metadatas, embeddings)

    # Rows per INSERT statement. A mempalace upsert batch is up to ~1000 chunks;
    # the backend used to issue one round-trip PER ROW, so a 1000-row batch was
    # 1000 tunnel round-trips (~46 drawers/min over an SSH tunnel — the backfill
    # bottleneck). Multi-row INSERT collapses that to ceil(n/_WRITE_BATCH) round
    # trips, each still embedding in-DB via one ML_EMBED_ROW per value tuple.
    # 100 keeps the statement well under max_allowed_packet (64 MB) and bounds
    # the per-statement ML_EMBED fan-out.
    _WRITE_BATCH = 100

    def _write(self, ids, documents, metadatas, embeddings):
        if not ids:
            return
        documents = documents or [None] * len(ids)
        metadatas = metadatas or [{}] * len(ids)
        # Client-side embedding (MEMPALACE_EMBED_MODE=client): if an embedder is
        # configured and the caller didn't pass vectors, embed the documents
        # locally (fast, batched) and take the precomputed STRING_TO_VECTOR
        # path below — bypassing HeatWave ML_EMBED (~0.9s/text). Vectors are in
        # the same space (model parity), so they're interchangeable.
        if embeddings is None and self._embedder is not None:
            embeddings = self._embedder.embed([d or "" for d in documents])
        if embeddings is None:
            # mempalace path — embed the document text in-DB via ML_EMBED.
            tuple_sql = (
                "(%s, %s, %s, sys.ML_EMBED_ROW(%s, JSON_OBJECT('model_id', %s)))"
            )

            def row_params(i):
                return [
                    ids[i],
                    documents[i],
                    json.dumps(metadatas[i] or {}),
                    documents[i] or "",
                    self._embed_model,
                ]
        else:
            # precomputed-embeddings escape hatch.
            tuple_sql = "(%s, %s, %s, STRING_TO_VECTOR(%s))"

            def row_params(i):
                return [
                    ids[i],
                    documents[i],
                    json.dumps(metadatas[i] or {}),
                    _embed.vector_to_sql(embeddings[i]),
                ]

        n = len(ids)
        for start in range(0, n, self._WRITE_BATCH):
            idx = range(start, min(start + self._WRITE_BATCH, n))
            values = ", ".join(tuple_sql for _ in idx)
            params: list = []
            for i in idx:
                params.extend(row_params(i))
            on_dup = (
                " AS new ON DUPLICATE KEY UPDATE document=new.document, "
                "metadata=new.metadata, embedding=new.embedding"
            )
            sql = (
                f"INSERT INTO `{self.name}` (id, document, metadata, embedding) "
                f"VALUES {values}{on_dup}"
            )
            try:
                self._pool.execute(sql, params)
            except Exception:
                # A single bad row (e.g. a vector MySQL's STRING_TO_VECTOR
                # rejects, error 6138) fails the whole multi-row INSERT. Retry
                # the batch row-by-row so the rest still lands; skip + count any
                # individual row that still fails, so a backfill survives rare
                # degenerate embeddings instead of crashing.
                single = (
                    f"INSERT INTO `{self.name}` (id, document, metadata, embedding) "
                    f"VALUES {tuple_sql}{on_dup}"
                )
                skipped = []
                for i in idx:
                    try:
                        self._pool.execute(single, row_params(i))
                    except Exception as exc:  # noqa: BLE001
                        skipped.append(
                            {
                                "id": ids[i],
                                "source_file": (metadatas[i] or {}).get("source_file"),
                                "room": (metadatas[i] or {}).get("room"),
                                "error": str(exc)[:300],
                                "doc_preview": (documents[i] or "")[:200],
                            }
                        )
                if skipped:
                    _record_skips(self.name, skipped)

    def update(self, ids, documents=None, metadatas=None, embeddings=None):
        # Partial per-id UPDATE: only the provided columns are touched.
        for i, _id in enumerate(ids):
            sets, params = [], []
            if documents is not None:
                sets.append("document=%s")
                params.append(documents[i])
            if metadatas is not None:
                sets.append("metadata=%s")
                params.append(json.dumps(metadatas[i] or {}))
            if embeddings is not None:
                sets.append("embedding=STRING_TO_VECTOR(%s)")
                params.append(_embed.vector_to_sql(embeddings[i]))
            elif documents is not None:
                # re-embed from the new document text in-DB.
                sets.append(
                    "embedding=sys.ML_EMBED_ROW(%s, JSON_OBJECT('model_id', %s))"
                )
                params.extend([documents[i] or "", self._embed_model])
            if not sets:
                continue
            params.append(_id)
            self._pool.execute(
                f"UPDATE `{self.name}` SET {', '.join(sets)} WHERE id=%s", params
            )

    def delete(self, ids=None, where=None, where_document=None):
        if where_document is not None:
            raise NotImplementedError(_NO_WHERE_DOC)
        if ids:
            placeholders = ",".join(["%s"] * len(ids))
            self._pool.execute(
                f"DELETE FROM `{self.name}` WHERE id IN ({placeholders})", list(ids)
            )
        elif where:
            clause, params = _where.translate(where)
            self._pool.execute(f"DELETE FROM `{self.name}` WHERE {clause}", params)

    # ---- reads ------------------------------------------------------------
    def count(self) -> int:
        rows = self._pool.execute(
            f"SELECT COUNT(*) AS n FROM `{self.name}`", fetch=True
        )
        return int(rows[0]["n"]) if rows else 0

    # Generated columns that are safe to GROUP BY / filter on by name. Whitelist
    # (not interpolated user input) — these are the STORED/VIRTUAL columns the
    # _DDL declares, with idx_wing_room / idx_source_file backing them.
    _GROUPABLE = ("wing", "room", "source_file")

    def count_by(self, field: str, where=None) -> dict:
        """Server-side ``GROUP BY`` aggregation over a generated column.

        Returns ``{value: count}`` for the given indexed column (``wing`` /
        ``room`` / ``source_file``), counting rows per distinct value in MySQL
        instead of streaming every row's metadata to the client. A NULL value
        (drawer whose metadata lacks the field) is reported under the key
        ``"unknown"`` to match mempalace's ``meta.get(field, "unknown")``
        semantics. ``where`` is an optional chromadb-style filter (e.g.
        ``{"wing": "claude_history"}`` to scope a room count to one wing); it is
        translated through the same ``_where`` layer ``get``/``query`` use.

        idx_wing_room makes ``GROUP BY wing`` (and ``GROUP BY room`` within a
        wing) an index scan, so this is O(distinct values) on the wire instead
        of O(rows) — the fix for list_wings timing out on a 487k-drawer palace.
        """
        if field not in self._GROUPABLE:
            raise ValueError(
                f"count_by: field {field!r} not groupable; expected one of {self._GROUPABLE}"
            )
        params: list = []
        sql = f"SELECT `{field}` AS k, COUNT(*) AS n FROM `{self.name}`"
        if where:
            clause, wparams = _where.translate(where)
            if clause:
                sql += f" WHERE {clause}"
                params.extend(wparams)
        sql += f" GROUP BY `{field}`"
        rows = self._pool.execute(sql, params, fetch=True) or []
        return {(r["k"] if r["k"] is not None else "unknown"): int(r["n"]) for r in rows}

    def get(
        self,
        ids=None,
        where=None,
        where_document=None,
        include=None,
        limit=None,
        offset=None,
    ):
        if where_document is not None:
            raise NotImplementedError(_NO_WHERE_DOC)
        clauses, params = [], []
        if ids:
            clauses.append("id IN (" + ",".join(["%s"] * len(ids)) + ")")
            params.extend(ids)
        if where:
            c, p = _where.translate(where)
            if c:
                clauses.append(c)
                params.extend(p)
        sql = f"SELECT id, document, metadata FROM `{self.name}`"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        if limit is not None:
            sql += " LIMIT %s" % int(limit)
        if offset is not None:
            sql += " OFFSET %s" % int(offset)
        rows = self._pool.execute(sql, params, fetch=True) or []
        return _shapes.build_get_result([self._row(r) for r in rows], include)

    def query(
        self,
        query_texts=None,
        query_embeddings=None,
        n_results=10,
        where=None,
        where_document=None,
        include=None,
    ):
        if where_document is not None:
            raise NotImplementedError(_NO_WHERE_DOC)
        clause, wparams = _where.translate(where) if where else ("", [])
        per_query = []
        if query_embeddings is None:
            # ML_EMBED path — embed the query text in-DB, but ONCE: set @qv to
            # the query embedding, then compare each row against that constant.
            # Inlining sys.ML_EMBED_ROW(text) inside the per-row VECTOR_DISTANCE
            # makes MySQL re-evaluate the (expensive ~seconds) embed once per
            # row — N×embed-cost, minutes for a few hundred drawers, which reads
            # as a hung search. @qv is computed once on the same connection.
            for text in list(query_texts or []):
                set_qv = "SET @qv = sys.ML_EMBED_ROW(%s, JSON_OBJECT('model_id', %s))"
                sql = (
                    "SELECT id, document, metadata, "
                    "VECTOR_DISTANCE(embedding, @qv, 'COSINE') AS distance "
                    f"FROM `{self.name}`"
                )
                params: list = list(wparams)
                if clause:
                    sql += f" WHERE {clause}"
                sql += " ORDER BY distance ASC LIMIT %s"
                params.append(int(n_results))
                rows = (
                    self._pool.execute_seq(
                        [(set_qv, [text, self._embed_model]), (sql, params)]
                    )
                    or []
                )
                per_query.append([self._row(r) for r in rows])
        else:
            for vec in query_embeddings:
                params = [_embed.vector_to_sql(vec)] + wparams
                sql = (
                    "SELECT id, document, metadata, "
                    "VECTOR_DISTANCE(embedding, STRING_TO_VECTOR(%s), 'COSINE') AS distance "
                    f"FROM `{self.name}`"
                )
                if clause:
                    sql += f" WHERE {clause}"
                sql += " ORDER BY distance ASC LIMIT %s"
                params.append(int(n_results))
                rows = self._pool.execute(sql, params, fetch=True) or []
                per_query.append([self._row(r) for r in rows])
        return _shapes.build_query_result(per_query, include)

    @staticmethod
    def _row(r: dict) -> dict:
        meta = r.get("metadata")
        if isinstance(meta, (str, bytes)):
            meta = json.loads(meta)
        out = {"id": r["id"], "document": r.get("document"), "metadata": meta}
        if "distance" in r:
            out["distance"] = float(r["distance"])
        return out
