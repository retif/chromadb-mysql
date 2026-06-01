"""Client collection-lifecycle mapping to MySQL DDL/lookup/drop (issue #5),
over a fake pool. The embedder is faked so importing the suite never pulls in
sentence-transformers.
"""

from __future__ import annotations

import pytest

from chromadb_mysql_backend._client import Client, _CollectionError
from chromadb_mysql_backend._collection import Collection

from _fakes import FakeEmbedder, FakePool


def _client(pool):
    return Client(embedder=FakeEmbedder(), pool=pool)


def test_create_collection_issues_ddl_with_embedder_dim():
    pool = FakePool()
    coll = _client(pool).create_collection("mem")
    assert isinstance(coll, Collection)
    assert coll.name == "mem"
    assert pool.ddl == [("mem", 3)]  # FakeEmbedder.dim


def test_get_or_create_is_ddl_idempotent():
    pool = FakePool()
    _client(pool).get_or_create_collection("mem")
    assert pool.ddl == [("mem", 3)]


def test_get_collection_missing_raises():
    pool = FakePool(fetch_results=[[]])  # information_schema → not found
    with pytest.raises(_CollectionError):
        _client(pool).get_collection("nope")


def test_get_collection_existing_returns_collection():
    pool = FakePool(fetch_results=[[{"1": 1}]])  # exists
    coll = _client(pool).get_collection("mem")
    assert isinstance(coll, Collection)
    assert coll.name == "mem"


def test_delete_collection_drops_table():
    pool = FakePool()
    _client(pool).delete_collection("mem")
    assert "DROP TABLE IF EXISTS `mem`" in pool.last()[0]


def test_list_collections_maps_rows_to_collections():
    pool = FakePool(fetch_results=[[{"name": "mem"}, {"name": "kg"}]])
    colls = _client(pool).list_collections()
    assert [c.name for c in colls] == ["mem", "kg"]
    assert all(isinstance(c, Collection) for c in colls)
