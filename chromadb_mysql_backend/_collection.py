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

    def _write(self, ids, documents, metadatas, embeddings):
        documents = documents or [None] * len(ids)
        metadatas = metadatas or [{}] * len(ids)
        if embeddings is None:
            # mempalace path — embed the document text in-DB via ML_EMBED.
            sql = (
                f"INSERT INTO `{self.name}` (id, document, metadata, embedding) "
                "VALUES (%s, %s, %s, sys.ML_EMBED_ROW(%s, JSON_OBJECT('model_id', %s))) AS new "
                "ON DUPLICATE KEY UPDATE document=new.document, "
                "metadata=new.metadata, embedding=new.embedding"
            )
            for i, _id in enumerate(ids):
                self._pool.execute(
                    sql,
                    [
                        _id,
                        documents[i],
                        json.dumps(metadatas[i] or {}),
                        documents[i] or "",
                        self._embed_model,
                    ],
                )
        else:
            # precomputed-embeddings escape hatch.
            sql = (
                f"INSERT INTO `{self.name}` (id, document, metadata, embedding) "
                "VALUES (%s, %s, %s, STRING_TO_VECTOR(%s)) AS new "
                "ON DUPLICATE KEY UPDATE document=new.document, "
                "metadata=new.metadata, embedding=new.embedding"
            )
            for i, _id in enumerate(ids):
                self._pool.execute(
                    sql,
                    [
                        _id,
                        documents[i],
                        json.dumps(metadatas[i] or {}),
                        _embed.vector_to_sql(embeddings[i]),
                    ],
                )

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

    def get(self, ids=None, where=None, where_document=None, include=None,
            limit=None, offset=None):
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

    def query(self, query_texts=None, query_embeddings=None, n_results=10,
              where=None, where_document=None, include=None):
        if where_document is not None:
            raise NotImplementedError(_NO_WHERE_DOC)
        clause, wparams = _where.translate(where) if where else ("", [])
        per_query = []
        if query_embeddings is None:
            # ML_EMBED path — embed the query text in-DB.
            for text in list(query_texts or []):
                params = [text, self._embed_model] + wparams
                sql = (
                    "SELECT id, document, metadata, "
                    "VECTOR_DISTANCE(embedding, sys.ML_EMBED_ROW(%s, "
                    "JSON_OBJECT('model_id', %s)), 'COSINE') AS distance "
                    f"FROM `{self.name}`"
                )
                if clause:
                    sql += f" WHERE {clause}"
                sql += " ORDER BY distance ASC LIMIT %s"
                params.append(int(n_results))
                rows = self._pool.execute(sql, params, fetch=True) or []
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
