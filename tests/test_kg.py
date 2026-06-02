"""SQL-shape + result-shaping tests for MySQLKnowledgeGraph, driven through the
FakeKGStore seam (no database). Asserts MySQL dialect (%s placeholders, INSERT
IGNORE, ON DUPLICATE KEY, the temporal CASE exprs, NULLS-LAST rewrite) and that
return shapes match the SQLite KG's dict contract.
"""

from __future__ import annotations

import pytest

from _kg_fakes import FakeCursor, FakeKGStore, install_fake_mempalace


@pytest.fixture()
def fake_mempalace():
    teardown = install_fake_mempalace()
    yield
    teardown()


def _make(monkeypatch, store):
    """Build a MySQLKnowledgeGraph whose _store is the fake (bypass env)."""
    import mempalace_kg_mysql._kg as kgmod

    monkeypatch.setattr(kgmod, "KGStore", lambda *a, **k: store)
    return kgmod.MySQLKnowledgeGraph()


def test_init_creates_schema(monkeypatch, fake_mempalace):
    store = FakeKGStore()
    _make(monkeypatch, store)
    assert store.schema_inited is True


def test_add_entity_upsert_sql(monkeypatch, fake_mempalace):
    store = FakeKGStore()
    kg = _make(monkeypatch, store)
    eid = kg.add_entity("Max Power", "person", {"gender": "m"})
    assert eid == "max_power"  # slug
    sql, params = store.execs[-1]
    assert "ON DUPLICATE KEY UPDATE" in sql
    assert "%s" in sql and "?" not in sql
    assert params[0] == "max_power" and params[1] == "Max Power"


def test_add_triple_inserts_when_new(monkeypatch, fake_mempalace):
    # cursor: dedup SELECT returns nothing -> a fresh INSERT happens
    cur = FakeCursor(fetchone_results=[None])
    store = FakeKGStore(cursor=cur)
    kg = _make(monkeypatch, store)
    tid = kg.add_triple("Max", "loves", "chess", valid_from="2025-10-01")
    assert tid.startswith("t_max_loves_chess_")
    sqls = [c[0] for c in cur.calls]
    # two INSERT IGNORE entity upserts, one dedup SELECT, one triple INSERT
    assert sum("INSERT IGNORE INTO entities" in s for s in sqls) == 2
    assert any(s.startswith("SELECT id FROM triples") for s in sqls)
    assert any("INSERT INTO triples" in s for s in sqls)
    assert all("?" not in s for s in sqls)


def test_add_triple_dedup_returns_existing(monkeypatch, fake_mempalace):
    cur = FakeCursor(fetchone_results=[{"id": "t_existing"}])
    store = FakeKGStore(cursor=cur)
    kg = _make(monkeypatch, store)
    tid = kg.add_triple("Max", "loves", "chess")
    assert tid == "t_existing"
    # no triple INSERT when a live duplicate exists
    assert not any("INSERT INTO triples" in c[0] for c in cur.calls)


def test_add_triple_rejects_inverted_interval(monkeypatch, fake_mempalace):
    store = FakeKGStore()
    kg = _make(monkeypatch, store)
    with pytest.raises(ValueError):
        kg.add_triple("A", "x", "B", valid_from="2026-05-10", valid_to="2026-05-01")


def test_query_entity_outgoing_shape(monkeypatch, fake_mempalace):
    row = {
        "predicate": "loves",
        "obj_name": "Chess",
        "valid_from": "2025-10-01",
        "valid_to": None,
        "confidence": 1.0,
        "source_closet": None,
    }
    store = FakeKGStore(query_results=[[row]])
    kg = _make(monkeypatch, store)
    out = kg.query_entity("Max", direction="outgoing")
    assert out == [
        {
            "direction": "outgoing",
            "subject": "Max",
            "predicate": "loves",
            "object": "Chess",
            "valid_from": "2025-10-01",
            "valid_to": None,
            "confidence": 1.0,
            "source_closet": None,
            "current": True,
        }
    ]
    sql, params = store.queries[-1]
    assert "JOIN entities e ON t.object = e.id" in sql
    assert params == ["max"]


def test_query_entity_as_of_adds_temporal_filter(monkeypatch, fake_mempalace):
    store = FakeKGStore(query_results=[[]])
    kg = _make(monkeypatch, store)
    kg.query_entity("Max", as_of="2026-01-15", direction="outgoing")
    sql, params = store.queries[-1]
    assert "CONCAT(t.valid_from, 'T00:00:00Z')" in sql
    assert "CONCAT(t.valid_to, 'T23:59:59Z')" in sql
    # entity id + two as-of keys (normalised to midnight)
    assert params == ["max", "2026-01-15T00:00:00Z", "2026-01-15T00:00:00Z"]


def test_timeline_uses_nulls_last_rewrite(monkeypatch, fake_mempalace):
    store = FakeKGStore(query_results=[[]])
    kg = _make(monkeypatch, store)
    kg.timeline()
    sql, _ = store.queries[-1]
    assert "ORDER BY (t.valid_from IS NULL), t.valid_from ASC" in sql
    assert "NULLS LAST" not in sql


def test_stats_shape(monkeypatch, fake_mempalace):
    store = FakeKGStore(
        query_results=[
            [{"cnt": 5}],  # entities
            [{"cnt": 7}],  # triples
            [{"cnt": 4}],  # current
            [{"predicate": "loves"}, {"predicate": "works_on"}],  # predicates
        ]
    )
    kg = _make(monkeypatch, store)
    s = kg.stats()
    assert s == {
        "entities": 5,
        "triples": 7,
        "current_facts": 4,
        "expired_facts": 3,
        "relationship_types": ["loves", "works_on"],
    }


def test_invalidate_updates_valid_to(monkeypatch, fake_mempalace):
    cur = FakeCursor(fetchall_results=[[{"id": "t1", "valid_from": "2025-01-01"}]])
    store = FakeKGStore(cursor=cur)
    kg = _make(monkeypatch, store)
    kg.invalidate("Max", "has_issue", "injury", ended="2026-02-15")
    sqls = [c[0] for c in cur.calls]
    assert any(s.startswith("UPDATE triples SET valid_to=%s") for s in sqls)
