"""Collection SQL + result-shaping, over a fake pool (no live MySQL).

Covers the path the unit suite previously skipped: the actual SQL emitted by
add/upsert/get/query/delete/count and how DB rows are reassembled into Chroma
shapes. Mirrors issue #7 (conformance) at the unit level.
"""

from __future__ import annotations

import json

from chromadb_mysql_backend import _embed
from chromadb_mysql_backend._collection import Collection

from _fakes import FakeEmbedder, FakePool


def _coll(pool=None, emb=None):
    return Collection("mem", pool or FakePool(), emb or FakeEmbedder())


# ---- writes ---------------------------------------------------------------
def test_add_autoembeds_and_emits_upsert_sql():
    pool, emb = FakePool(), FakeEmbedder()
    _coll(pool, emb).add(ids=["a"], documents=["hello"])

    assert emb.seen == [["hello"]]  # documents were embedded
    assert len(pool.calls) == 1
    sql, params, fetch = pool.last()
    assert fetch is False
    assert "INSERT INTO `mem` (id, document, metadata, embedding)" in sql
    assert "STRING_TO_VECTOR(%s)) AS new" in sql
    assert "ON DUPLICATE KEY UPDATE" in sql
    assert "new.document" in sql and "VALUES(" not in sql  # MySQL 9 alias form
    assert params[0] == "a"
    assert params[1] == "hello"
    assert json.loads(params[2]) == {}  # default empty metadata, as JSON
    assert params[3] == _embed.vector_to_sql([5.0, 0.0, 1.0])  # len("hello")=5


def test_add_explicit_embeddings_skips_embedder():
    pool, emb = FakePool(), FakeEmbedder()
    _coll(pool, emb).add(
        ids=["a"], documents=["hello"], embeddings=[[0.1, 0.2, 0.3]]
    )
    assert emb.seen == []  # caller supplied embeddings → no embedding
    assert pool.last()[1][3] == _embed.vector_to_sql([0.1, 0.2, 0.3])


def test_add_serializes_metadata_and_one_execute_per_row():
    pool = FakePool()
    _coll(pool).add(
        ids=["a", "b"],
        documents=["x", "y"],
        metadatas=[{"wing": "infra"}, {"wing": "nix"}],
    )
    assert len(pool.calls) == 2  # one INSERT per row
    assert json.loads(pool.calls[0][1][2]) == {"wing": "infra"}
    assert json.loads(pool.calls[1][1][2]) == {"wing": "nix"}


def test_upsert_uses_same_path_as_add():
    pool = FakePool()
    _coll(pool).upsert(ids=["a"], documents=["hello"])
    assert "ON DUPLICATE KEY UPDATE" in pool.last()[0]


# ---- get ------------------------------------------------------------------
def test_get_builds_where_ids_limit_offset_and_shapes():
    rows = [{"id": "a", "document": "doc-a", "metadata": '{"wing": "infra"}'}]
    pool = FakePool(fetch_results=[rows])
    out = _coll(pool).get(
        ids=["a"], where={"wing": "infra"}, limit=5, offset=2
    )
    sql, params, fetch = pool.last()
    assert fetch is True
    assert "SELECT id, document, metadata FROM `mem`" in sql
    assert "WHERE id IN (%s) AND" in sql
    assert "LIMIT 5" in sql and "OFFSET 2" in sql
    assert params == ["a", "$.wing", "infra"]
    # metadata stored as a JSON string is decoded back to a dict
    assert out["ids"] == ["a"]
    assert out["metadatas"] == [{"wing": "infra"}]
    assert out["documents"] == ["doc-a"]


def test_get_no_filters_omits_where():
    pool = FakePool(fetch_results=[[]])
    out = _coll(pool).get()
    assert "WHERE" not in pool.last()[0]
    assert out["ids"] == []


def test_get_handles_dict_metadata_without_double_decode():
    rows = [{"id": "a", "document": None, "metadata": {"wing": "infra"}}]
    pool = FakePool(fetch_results=[rows])
    out = _coll(pool).get(ids=["a"])
    assert out["metadatas"] == [{"wing": "infra"}]


# ---- query ----------------------------------------------------------------
def test_query_one_sql_per_vector_with_distance_and_shape():
    q1 = [{"id": "a", "document": "d", "metadata": "{}", "distance": 0.1}]
    q2 = [{"id": "b", "document": "e", "metadata": "{}", "distance": 0.2}]
    pool, emb = FakePool(fetch_results=[q1, q2]), FakeEmbedder()
    out = _coll(pool, emb).query(query_texts=["x", "yy"], n_results=3)

    assert emb.seen == [["x", "yy"]]
    assert len(pool.calls) == 2  # one query per embedded text
    sql, params, fetch = pool.calls[0]
    assert fetch is True
    assert "VECTOR_DISTANCE(embedding, STRING_TO_VECTOR(%s), 'COSINE')" in sql
    assert "ORDER BY distance ASC LIMIT %s" in sql
    assert params[0] == _embed.vector_to_sql([1.0, 0.0, 1.0])  # len("x")=1
    assert params[-1] == 3  # n_results
    # nested per-query shape, distances cast to float
    assert out["ids"] == [["a"], ["b"]]
    assert out["distances"] == [[0.1], [0.2]]


def test_query_with_where_merges_clause_and_params_before_limit():
    rows = [{"id": "a", "document": "d", "metadata": "{}", "distance": 0.1}]
    pool = FakePool(fetch_results=[rows])
    _coll(pool).query(
        query_embeddings=[[0.0, 0.0, 1.0]], where={"wing": "infra"}, n_results=2
    )
    sql, params, _ = pool.last()
    assert "WHERE" in sql
    assert sql.index("WHERE") < sql.index("ORDER BY")
    assert params == [_embed.vector_to_sql([0.0, 0.0, 1.0]), "$.wing", "infra", 2]


def test_query_explicit_embeddings_skip_embedder():
    pool, emb = FakePool(fetch_results=[[]]), FakeEmbedder()
    _coll(pool, emb).query(query_embeddings=[[0.0, 1.0, 0.0]])
    assert emb.seen == []


# ---- delete / count -------------------------------------------------------
def test_delete_by_ids():
    pool = FakePool()
    _coll(pool).delete(ids=["a", "b"])
    sql, params, _ = pool.last()
    assert "DELETE FROM `mem` WHERE id IN (%s,%s)" in sql
    assert params == ["a", "b"]


def test_delete_by_where():
    pool = FakePool()
    _coll(pool).delete(where={"wing": "infra"})
    sql, params, _ = pool.last()
    assert sql.startswith("DELETE FROM `mem` WHERE ")
    assert params == ["$.wing", "infra"]


def test_count_reads_scalar():
    pool = FakePool(fetch_results=[[{"n": 7}]])
    assert _coll(pool).count() == 7
    assert "SELECT COUNT(*) AS n FROM `mem`" in pool.last()[0]


def test_count_empty_is_zero():
    pool = FakePool(fetch_results=[[]])
    assert _coll(pool).count() == 0
