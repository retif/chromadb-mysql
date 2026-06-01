"""Collection — the ChromaDB Collection API surface mempalace uses, over MySQL.

Surface (the whole of it): add, upsert, get, query, delete, count, plus a
``name`` attribute. ``get`` returns flat lists; ``query`` returns nested
per-query lists (see _shapes). ``where`` supports equality + ``$and`` (see
_where).

The SQL methods need a live DB (conformance phase). The result assembly is
delegated to the pure functions in _shapes so it is unit-tested without one.
"""

from __future__ import annotations

from . import _embed, _shapes, _where


class Collection:
    def __init__(self, name: str, pool, embedder: _embed.Embedder):
        self.name = name
        self._pool = pool
        self._embed = embedder

    # ---- writes -----------------------------------------------------------
    def add(self, ids, documents=None, metadatas=None, embeddings=None):
        self._write(ids, documents, metadatas, embeddings)

    def upsert(self, ids, documents=None, metadatas=None, embeddings=None):
        self._write(ids, documents, metadatas, embeddings)

    def _write(self, ids, documents, metadatas, embeddings):
        import json

        documents = documents or [None] * len(ids)
        metadatas = metadatas or [{}] * len(ids)
        if embeddings is None:
            embeddings = self._embed.embed([d or "" for d in documents])
        rows = []
        for i, _id in enumerate(ids):
            rows.append(
                [
                    _id,
                    documents[i],
                    json.dumps(metadatas[i] or {}),
                    _embed.vector_to_sql(embeddings[i]),
                ]
            )
        # Alias form (INSERT … AS new … = new.col): the VALUES() form in the
        # UPDATE clause is deprecated in MySQL 8.0.20+ and warns on 9.x. Verified
        # against HeatWave 9.7.0.
        sql = (
            f"INSERT INTO `{self.name}` (id, document, metadata, embedding) "
            "VALUES (%s, %s, %s, STRING_TO_VECTOR(%s)) AS new "
            "ON DUPLICATE KEY UPDATE document=new.document, "
            "metadata=new.metadata, embedding=new.embedding"
        )
        for row in rows:
            self._pool.execute(sql, row)

    def delete(self, ids=None, where=None):
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

    def get(self, ids=None, where=None, include=None, limit=None, offset=None):
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
        include=None,
    ):
        if query_embeddings is None:
            query_embeddings = self._embed.embed(list(query_texts or []))
        clause, wparams = _where.translate(where) if where else ("", [])
        per_query = []
        for vec in query_embeddings:
            params = [_embed.vector_to_sql(vec)] + wparams
            sql = (
                f"SELECT id, document, metadata, "
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
        import json

        meta = r.get("metadata")
        if isinstance(meta, (str, bytes)):
            meta = json.loads(meta)
        out = {"id": r["id"], "document": r.get("document"), "metadata": meta}
        if "distance" in r:
            out["distance"] = float(r["distance"])
        return out
