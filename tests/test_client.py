"""Client collection-lifecycle mapping to MySQL DDL/lookup/drop, over a fake
pool. Embeddings default to in-DB ML_EMBED so no client embedder is needed."""

import pytest

from chromadb_mysql_backend._client import Client
from chromadb_mysql_backend.errors import NotFoundError

from tests._fakes import FakePool


def test_create_collection_ddl_fixed_dim():
    p = FakePool()
    Client(pool=p).create_collection("mem")
    assert p.ddl == [("mem", 384)]


def test_get_or_create_ddl():
    p = FakePool()
    Client(pool=p).get_or_create_collection("mem")
    assert p.ddl == [("mem", 384)]


def test_get_collection_missing_raises_notfound():
    p = FakePool(fetch_results=[[]])  # _exists -> no rows
    with pytest.raises(NotFoundError):
        Client(pool=p).get_collection("absent")


def test_get_collection_present():
    p = FakePool(fetch_results=[[{"1": 1}]])  # _exists -> a row
    c = Client(pool=p).get_collection("mem")
    assert c.name == "mem"


def test_delete_collection_drops():
    p = FakePool()
    Client(pool=p).delete_collection("mem")
    assert "DROP TABLE IF EXISTS `mem`" in p.last()[0]


def test_list_collections():
    p = FakePool(fetch_results=[[{"name": "a"}, {"name": "b"}]])
    cols = Client(pool=p).list_collections()
    assert [c.name for c in cols] == ["a", "b"]
