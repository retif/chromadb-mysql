from chromadb_mysql_backend import _where


def test_empty():
    assert _where.translate(None) == ("", [])
    assert _where.translate({}) == ("", [])


def test_single_equality():
    clause, params = _where.translate({"source_file": "/a/b.md"})
    assert clause == "JSON_UNQUOTE(JSON_EXTRACT(metadata, %s)) = %s"
    assert params == ["$.source_file", "/a/b.md"]


def test_implicit_and_multi_key():
    clause, params = _where.translate({"wing": "infra", "room": "heatwave"})
    assert clause == (
        "JSON_UNQUOTE(JSON_EXTRACT(metadata, %s)) = %s AND "
        "JSON_UNQUOTE(JSON_EXTRACT(metadata, %s)) = %s"
    )
    assert params == ["$.wing", "infra", "$.room", "heatwave"]


def test_explicit_and():
    clause, params = _where.translate({"$and": [{"wing": "infra"}, {"room": "hw"}]})
    assert clause == (
        "(JSON_UNQUOTE(JSON_EXTRACT(metadata, %s)) = %s) AND "
        "(JSON_UNQUOTE(JSON_EXTRACT(metadata, %s)) = %s)"
    )
    assert params == ["$.wing", "infra", "$.room", "hw"]


def test_eq_operator_unwrap():
    clause, params = _where.translate({"wing": {"$eq": "infra"}})
    assert params == ["$.wing", "infra"]
