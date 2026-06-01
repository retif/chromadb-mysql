from chromadb_mysql_backend import _shapes


ROWS = [
    {"id": "a", "document": "doc-a", "metadata": {"wing": "infra"}},
    {"id": "b", "document": "doc-b", "metadata": {"wing": "nix"}},
]


def test_get_default_include_is_flat():
    out = _shapes.build_get_result(ROWS, None)
    assert out["ids"] == ["a", "b"]
    assert out["documents"] == ["doc-a", "doc-b"]
    assert out["metadatas"] == [{"wing": "infra"}, {"wing": "nix"}]
    assert out["included"] == ["metadatas", "documents"]


def test_get_include_subset_nulls_others():
    out = _shapes.build_get_result(ROWS, ["documents"])
    assert out["documents"] == ["doc-a", "doc-b"]
    assert out["metadatas"] is None


def test_get_empty():
    out = _shapes.build_get_result([], ["metadatas"])
    assert out["ids"] == []
    assert out["metadatas"] == []


def test_query_is_nested_per_query():
    q = [
        [
            {"id": "a", "document": "doc-a", "metadata": {"w": 1}, "distance": 0.1},
            {"id": "b", "document": "doc-b", "metadata": {"w": 2}, "distance": 0.4},
        ]
    ]
    out = _shapes.build_query_result(q, ["metadatas", "documents", "distances"])
    assert out["ids"] == [["a", "b"]]
    assert out["distances"] == [[0.1, 0.4]]
    assert out["documents"] == [["doc-a", "doc-b"]]
    assert out["metadatas"] == [[{"w": 1}, {"w": 2}]]


def test_query_distances_only():
    q = [[{"id": "a", "document": "d", "metadata": {}, "distance": 0.2}]]
    out = _shapes.build_query_result(q, ["distances"])
    assert out["distances"] == [[0.2]]
    assert out["documents"] is None
    assert out["metadatas"] is None
