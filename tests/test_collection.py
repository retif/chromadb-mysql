from chromadb_mysql_backend import _embed
from chromadb_mysql_backend._collection import Collection

import pytest

from tests._fakes import FakePool


def _coll(pool=None, **kw):
    return Collection("mem", pool or FakePool(), **kw)


# ---- add / upsert: in-DB ML_EMBED default --------------------------------
def test_add_uses_ml_embed_inline():
    p = FakePool()
    _coll(p).add(["a"], documents=["hello"])
    sql, params, _ = p.last()
    assert "sys.ML_EMBED_ROW(%s, JSON_OBJECT('model_id', %s))) AS new" in sql
    assert "ON DUPLICATE KEY UPDATE" in sql
    assert params == ["a", "hello", "{}", "hello", "all_minilm_l12_v2"]


def test_add_precomputed_embeddings_uses_string_to_vector():
    p = FakePool()
    _coll(p).add(["a"], documents=["hello"], embeddings=[[0.1, 0.2, 0.3]])
    sql, params, _ = p.last()
    assert "STRING_TO_VECTOR(%s)" in sql
    assert "ML_EMBED_ROW" not in sql
    assert params[-1] == _embed.vector_to_sql([0.1, 0.2, 0.3])


def test_add_model_override():
    p = FakePool()
    _coll(p, embed_model="multilingual-e5-small").add(["a"], documents=["x"])
    assert p.last()[1][4] == "multilingual-e5-small"


def test_update_reembeds_from_document():
    p = FakePool()
    _coll(p).update(["a"], documents=["hi"])
    sql, params, _ = p.last()
    assert sql.startswith("UPDATE `mem` SET")
    assert "document=%s" in sql
    assert "sys.ML_EMBED_ROW(%s, JSON_OBJECT('model_id', %s))" in sql
    assert sql.endswith("WHERE id=%s")
    # document=%s ('hi'), then ML_EMBED text ('hi') + model, then id.
    assert params == ["hi", "hi", "all_minilm_l12_v2", "a"]


# ---- query ----------------------------------------------------------------
def test_query_ml_embed_two_texts():
    p = FakePool(fetch_results=[[], []])
    _coll(p).query(query_texts=["x", "yy"], n_results=3)
    # Each text embeds ONCE into @qv (SET) then scans against it (SELECT):
    # 2 texts × 2 statements = 4 calls. The embed must NOT be inlined in the
    # per-row VECTOR_DISTANCE (that re-embeds once per row — minutes/hang).
    assert len(p.calls) == 4
    set_sql, set_params, _ = p.calls[0]
    sel_sql, sel_params, _ = p.calls[1]
    assert "SET @qv = sys.ML_EMBED_ROW(%s, JSON_OBJECT('model_id', %s))" in set_sql
    assert set_params == ["x", "all_minilm_l12_v2"]
    assert "VECTOR_DISTANCE(embedding, @qv, 'COSINE')" in sel_sql
    assert "ML_EMBED_ROW" not in sel_sql  # embed not re-evaluated per row
    assert "ORDER BY distance ASC LIMIT %s" in sel_sql
    assert sel_params[-1] == 3


def test_query_with_where():
    p = FakePool(fetch_results=[[]])
    _coll(p).query(query_texts=["q"], where={"wing": "infra"}, n_results=2)
    # SET @qv carries text+model; the SELECT carries where-params + limit.
    assert p.calls[0][1] == ["q", "all_minilm_l12_v2"]
    assert p.calls[1][1] == ["$.wing", "infra", 2]


def test_query_precomputed():
    p = FakePool(fetch_results=[[]])
    _coll(p).query(query_embeddings=[[0, 0, 1]], n_results=1)
    sql = p.last()[0]
    assert "STRING_TO_VECTOR(%s)" in sql and "ML_EMBED_ROW" not in sql


# ---- where_document unsupported ------------------------------------------
def test_where_document_raises():
    for fn in ("query", "get", "delete"):
        with pytest.raises(NotImplementedError):
            getattr(_coll(), fn)(where_document="x")


# ---- metadata property ----------------------------------------------------
def test_metadata_property():
    c = _coll(metadata={"hnsw:space": "cosine"})
    assert c.metadata == {"hnsw:space": "cosine"}
    assert _coll().metadata == {}


# ---- get / delete / count -------------------------------------------------
def test_get_flat_shape():
    p = FakePool(fetch_results=[[{"id": "a", "document": "d", "metadata": "{}"}]])
    out = _coll(p).get(ids=["a"], include=["documents", "metadatas"])
    assert out["ids"] == ["a"] and out["documents"] == ["d"]


def test_delete_by_id():
    p = FakePool()
    _coll(p).delete(ids=["a", "b"])
    assert "DELETE FROM `mem` WHERE id IN (%s,%s)" in p.last()[0]


def test_count():
    p = FakePool(fetch_results=[[{"n": 5}]])
    assert _coll(p).count() == 5
