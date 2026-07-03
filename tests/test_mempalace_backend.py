"""Adapter tests for the native mempalace 3.5.0 MySQL backend — no live DB.

We put the 3.5.0 source tree on ``sys.path`` so ``mempalace.backends.base``
imports, then drive :class:`MySQLCollection` / :class:`MySQLBackend` against a
FAKE shim collection that returns canned chromadb-shaped dicts. This validates
the pure adapter/repackaging logic (typed result shapes, ``include`` handling,
delegation, ABC conformance) without any MySQL connection.
"""

from __future__ import annotations

import os
import sys

import pytest

# Make mempalace.backends.base importable from the 3.5.0 reference tree.
_SRC350 = "/home/oleks/.claude/jobs/330b19e2/tmp/src350"
if os.path.isdir(_SRC350) and _SRC350 not in sys.path:
    sys.path.insert(0, _SRC350)

import mempalace.backends.base as b  # noqa: E402

from chromadb_mysql_backend.mempalace_backend import (  # noqa: E402
    MySQLBackend,
    MySQLCollection,
)


class FakeShimCollection:
    """Canned stand-in for chromadb_mysql_backend._collection.Collection."""

    def __init__(self):
        self.name = "mempalace"
        self.calls = []

    # writes just record the call
    def add(self, **kw):
        self.calls.append(("add", kw))

    def upsert(self, **kw):
        self.calls.append(("upsert", kw))

    def update(self, **kw):
        self.calls.append(("update", kw))

    def delete(self, **kw):
        self.calls.append(("delete", kw))

    def count(self) -> int:
        self.calls.append(("count", {}))
        return 42

    def count_by(self, field, where=None):
        self.calls.append(("count_by", {"field": field, "where": where}))
        return {"alpha": 3, "beta": 5}

    def get(self, ids=None, where=None, where_document=None, include=None, limit=None, offset=None):
        self.calls.append(("get", {"include": include, "limit": limit, "offset": offset}))
        # chromadb flat get shape (build_get_result); metadatas/documents default on.
        return {
            "ids": ["d1", "d2"],
            "embeddings": None,
            "documents": ["doc one", "doc two"],
            "metadatas": [{"wing": "w1"}, {"wing": "w2"}],
            "uris": None,
            "data": None,
            "included": ["metadatas", "documents"],
        }

    def query(
        self,
        query_texts=None,
        query_embeddings=None,
        n_results=10,
        where=None,
        where_document=None,
        include=None,
    ):
        self.calls.append(("query", {"include": include, "n_results": n_results}))
        n = len(query_texts or query_embeddings or [None])
        # chromadb nested query shape (build_query_result), one inner list / query.
        docs = None if (include is not None and "documents" not in include) else None
        dists = None if (include is not None and "distances" not in include) else None
        return {
            "ids": [["a", "b"] for _ in range(n)],
            "embeddings": None,
            "documents": (
                [["doc a", "doc b"] for _ in range(n)]
                if include is None or "documents" in include
                else None
            ),
            "metadatas": (
                [[{"k": 1}, {"k": 2}] for _ in range(n)]
                if include is None or "metadatas" in include
                else None
            ),
            "distances": (
                [[0.1, 0.2] for _ in range(n)]
                if include is None or "distances" in include
                else None
            ),
            "uris": None,
            "data": None,
            "included": include or ["metadatas", "documents", "distances"],
        }


def _coll():
    return MySQLCollection(FakeShimCollection())


# --------------------------------------------------------------------------
# ABC conformance
# --------------------------------------------------------------------------


def test_subclasses_and_instantiable():
    assert issubclass(MySQLBackend, b.BaseBackend)
    assert issubclass(MySQLCollection, b.BaseCollection)
    # No TypeError => every abstractmethod is implemented.
    c = _coll()
    assert isinstance(c, b.BaseCollection)


# --------------------------------------------------------------------------
# query -> QueryResult
# --------------------------------------------------------------------------


def test_query_default_include_shape():
    c = _coll()
    r = c.query(query_texts=["hello", "world"])
    assert isinstance(r, b.QueryResult)
    assert r.ids == [["a", "b"], ["a", "b"]]
    assert r.documents == [["doc a", "doc b"], ["doc a", "doc b"]]
    assert r.metadatas == [[{"k": 1}, {"k": 2}], [{"k": 1}, {"k": 2}]]
    assert r.distances == [[0.1, 0.2], [0.1, 0.2]]
    assert r.embeddings is None  # not requested


def test_query_include_filtering():
    c = _coll()
    r = c.query(query_texts=["hello"], include=["metadatas"])
    # documents/distances not requested => empty lists of correct outer shape.
    assert r.ids == [["a", "b"]]
    assert r.metadatas == [[{"k": 1}, {"k": 2}]]
    assert r.documents == [[]]
    assert r.distances == [[]]
    assert r.embeddings is None


def test_query_embeddings_requested_shape():
    c = _coll()
    # shim returns embeddings=None even when requested; adapter must still
    # produce the correct outer shape (not None) when embeddings are in include.
    r = c.query(query_texts=["a", "b"], include=["embeddings"])
    assert r.embeddings == [[], []]


def test_query_empty_uses_empty_constructor():
    class Empty(FakeShimCollection):
        def query(self, **kw):
            return {"ids": [], "documents": None, "metadatas": None, "distances": None}

    c = MySQLCollection(Empty())
    r = c.query(query_texts=["x", "y"], include=["embeddings"])
    assert r.ids == [[], []]
    assert r.distances == [[], []]
    assert r.embeddings == [[], []]  # embeddings_requested preserves outer shape


# --------------------------------------------------------------------------
# get -> GetResult
# --------------------------------------------------------------------------


def test_get_shape():
    c = _coll()
    r = c.get(ids=["d1", "d2"])
    assert isinstance(r, b.GetResult)
    assert r.ids == ["d1", "d2"]
    assert r.documents == ["doc one", "doc two"]
    assert r.metadatas == [{"wing": "w1"}, {"wing": "w2"}]
    assert r.embeddings is None


def test_get_empty():
    class Empty(FakeShimCollection):
        def get(self, **kw):
            return {"ids": [], "documents": None, "metadatas": None}

    r = MySQLCollection(Empty()).get(ids=["nope"])
    assert r.ids == [] and r.documents == [] and r.metadatas == []
    assert r.embeddings is None


# --------------------------------------------------------------------------
# delegation
# --------------------------------------------------------------------------


def test_count_delegates():
    fake = FakeShimCollection()
    assert MySQLCollection(fake).count() == 42
    assert ("count", {}) in fake.calls


def test_count_by_delegates():
    fake = FakeShimCollection()
    out = MySQLCollection(fake).count_by("wing", where={"room": "r1"})
    assert out == {"alpha": 3, "beta": 5}
    assert ("count_by", {"field": "wing", "where": {"room": "r1"}}) in fake.calls


def test_write_delegation_uses_shim_kwargs():
    fake = FakeShimCollection()
    c = MySQLCollection(fake)
    c.add(documents=["d"], ids=["1"], metadatas=[{"w": 1}])
    c.upsert(documents=["d"], ids=["1"])
    c.delete(ids=["1"])
    kinds = [name for name, _ in fake.calls]
    assert kinds == ["add", "upsert", "delete"]
    assert fake.calls[0][1]["ids"] == ["1"]


def test_get_all_metadata_pages_via_get():
    fake = FakeShimCollection()  # get() always returns 2 rows (< page) -> one page
    metas = MySQLCollection(fake).get_all_metadata()
    assert metas == [{"wing": "w1"}, {"wing": "w2"}]


# --------------------------------------------------------------------------
# backend
# --------------------------------------------------------------------------


def test_backend_name_and_detect(monkeypatch):
    assert MySQLBackend.name == "mysql"
    monkeypatch.setenv("MEMPALACE_BACKEND", "mysql")
    assert MySQLBackend.detect("/anything") is True
    monkeypatch.setenv("MEMPALACE_BACKEND", "chroma")
    assert MySQLBackend.detect("/anything") is False
    monkeypatch.delenv("MEMPALACE_BACKEND", raising=False)
    assert MySQLBackend.detect("/anything") is False


def test_backend_capabilities_no_explicit_embeddings():
    # query_texts must be allowed (in-DB ML_EMBED), so the backend must NOT
    # advertise requires_explicit_embeddings.
    assert "requires_explicit_embeddings" not in MySQLBackend.capabilities
    assert "supports_metadata_filters" in MySQLBackend.capabilities
