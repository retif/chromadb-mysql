"""MySQLKnowledgeGraph — a drop-in replacement for
``mempalace.knowledge_graph.KnowledgeGraph`` backed by HeatWave MySQL.

Same public surface (``add_entity``, ``add_triple``, ``invalidate``,
``query_entity``, ``query_relationship``, ``timeline``, ``stats``,
``seed_from_entity_facts``, ``close``, context-manager) and the same return
shapes, so mempalace's mcp_server / fact_checker can't tell the difference. The
``mempalace_kg_switch`` patches the class symbol in when ``MEMPALACE_KG_BACKEND
=mysql``; the default SQLite path is untouched.

The bitemporal semantics are preserved exactly. The date-only normalisation
(``valid_from='2026-05-06'`` compares as midnight, ``valid_to`` as end-of-day)
that SQLite did via ``CASE WHEN length()=10 ... THEN col||'T..'`` is reproduced
with MySQL ``LENGTH``/``SUBSTRING``/``CONCAT``. ``ORDER BY ... NULLS LAST`` (a
SQLite extension) becomes ``ORDER BY (col IS NULL), col``.

The pure-Python helpers (entity-id slug, temporal comparison keys, the ISO
validator) are imported from the real mempalace modules at call time so the two
backends share one source of truth and can never drift.
"""

from __future__ import annotations

import hashlib
import json
import threading
from datetime import date, datetime
from typing import Optional

from ._mysql import KGStore


# ── temporal SQL (MySQL dialect; mirrors knowledge_graph._sql_temporal_*) ──
def _sql_temporal_start_expr(column: str) -> str:
    return (
        f"CASE WHEN LENGTH({column}) = 10 "
        f"AND SUBSTRING({column}, 5, 1) = '-' "
        f"AND SUBSTRING({column}, 8, 1) = '-' "
        f"THEN CONCAT({column}, 'T00:00:00Z') ELSE {column} END"
    )


def _sql_temporal_end_expr(column: str) -> str:
    return (
        f"CASE WHEN LENGTH({column}) = 10 "
        f"AND SUBSTRING({column}, 5, 1) = '-' "
        f"AND SUBSTRING({column}, 8, 1) = '-' "
        f"THEN CONCAT({column}, 'T23:59:59Z') ELSE {column} END"
    )


def _temporal_filter_sql(as_of_key: str) -> tuple[str, list]:
    vf = _sql_temporal_start_expr("t.valid_from")
    vt = _sql_temporal_end_expr("t.valid_to")
    return (
        f" AND (t.valid_from IS NULL OR {vf} <= %s) "
        f"AND (t.valid_to IS NULL OR {vt} >= %s)",
        [as_of_key, as_of_key],
    )


def _import_helpers():
    """Late import of the dialect-independent helpers from real mempalace, so
    the package stays importable (for the switch / tests) without mempalace."""
    from mempalace.knowledge_graph import _temporal_start_key, _temporal_end_key
    from mempalace.config import sanitize_iso_temporal

    return _temporal_start_key, _temporal_end_key, sanitize_iso_temporal


def _entity_id(name: str) -> str:
    # Identical to KnowledgeGraph._entity_id (pure string slug).
    return name.lower().replace(" ", "_").replace("'", "")


class MySQLKnowledgeGraph:
    def __init__(self, db_path: Optional[str] = None):
        # db_path is accepted for signature parity but ignored — storage is the
        # shared HeatWave MySQL (entities/triples in the `mempalace` database).
        self.db_path = db_path
        self._store = KGStore()
        self._lock = threading.Lock()
        self._store.init_schema()

    # ── lifecycle (no persistent connection — connections are per-op) ──────
    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    def _entity_id(self, name: str) -> str:
        return _entity_id(name)

    # ── Write operations ──────────────────────────────────────────────────
    def add_entity(
        self, name: str, entity_type: str = "unknown", properties: Optional[dict] = None
    ):
        eid = _entity_id(name)
        props = json.dumps(properties or {})
        with self._lock:
            # Upsert keeps created_at (REPLACE would drop it).
            self._store.execute(
                "INSERT INTO entities (id, name, type, properties) VALUES (%s, %s, %s, %s) "
                "ON DUPLICATE KEY UPDATE name=VALUES(name), type=VALUES(type), "
                "properties=VALUES(properties)",
                [eid, name, entity_type, props],
            )
        return eid

    def add_triple(
        self,
        subject: str,
        predicate: str,
        obj: str,
        valid_from: Optional[str] = None,
        valid_to: Optional[str] = None,
        confidence: float = 1.0,
        source_closet: Optional[str] = None,
        source_file: Optional[str] = None,
        source_drawer_id: Optional[str] = None,
        adapter_name: Optional[str] = None,
    ):
        _temporal_start_key, _temporal_end_key, sanitize_iso_temporal = (
            _import_helpers()
        )

        valid_from = sanitize_iso_temporal(valid_from, "valid_from")
        valid_to = sanitize_iso_temporal(valid_to, "valid_to")

        if (
            valid_from is not None
            and valid_to is not None
            and _temporal_end_key(valid_to) < _temporal_start_key(valid_from)
        ):
            raise ValueError(
                f"valid_to={valid_to!r} is before valid_from={valid_from!r}; "
                "an inverted interval would be invisible to every KG query"
            )

        sub_id = _entity_id(subject)
        obj_id = _entity_id(obj)
        pred = predicate.lower().replace(" ", "_")

        def _do(cur):
            cur.execute(
                "INSERT IGNORE INTO entities (id, name) VALUES (%s, %s)",
                (sub_id, subject),
            )
            cur.execute(
                "INSERT IGNORE INTO entities (id, name) VALUES (%s, %s)", (obj_id, obj)
            )
            cur.execute(
                "SELECT id FROM triples WHERE subject=%s AND predicate=%s "
                "AND object=%s AND valid_to IS NULL",
                (sub_id, pred, obj_id),
            )
            existing = cur.fetchone()
            if existing:
                return existing["id"]  # already exists and still valid

            triple_id = (
                f"t_{sub_id}_{pred}_{obj_id}_"
                f"{hashlib.sha256(f'{valid_from}{datetime.now().isoformat()}'.encode()).hexdigest()[:12]}"
            )
            cur.execute(
                "INSERT INTO triples ("
                "id, subject, predicate, object, valid_from, valid_to, "
                "confidence, source_closet, source_file, source_drawer_id, adapter_name"
                ") VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                (
                    triple_id,
                    sub_id,
                    pred,
                    obj_id,
                    valid_from,
                    valid_to,
                    confidence,
                    source_closet,
                    source_file,
                    source_drawer_id,
                    adapter_name,
                ),
            )
            return triple_id

        with self._lock:
            return self._store.tx(_do)

    def invalidate(
        self, subject: str, predicate: str, obj: str, ended: Optional[str] = None
    ):
        _temporal_start_key, _temporal_end_key, sanitize_iso_temporal = (
            _import_helpers()
        )
        sub_id = _entity_id(subject)
        obj_id = _entity_id(obj)
        pred = predicate.lower().replace(" ", "_")
        ended = sanitize_iso_temporal(ended or date.today().isoformat(), "ended")

        def _do(cur):
            cur.execute(
                "SELECT id, valid_from FROM triples "
                "WHERE subject=%s AND predicate=%s AND object=%s AND valid_to IS NULL",
                (sub_id, pred, obj_id),
            )
            for row in cur.fetchall():
                valid_from = row["valid_from"]
                if valid_from is not None and _temporal_end_key(
                    ended
                ) < _temporal_start_key(valid_from):
                    raise ValueError(
                        f"valid_to={ended!r} is before valid_from={valid_from!r}; "
                        "an inverted interval would be invisible to every KG query"
                    )
            cur.execute(
                "UPDATE triples SET valid_to=%s "
                "WHERE subject=%s AND predicate=%s AND object=%s AND valid_to IS NULL",
                (ended, sub_id, pred, obj_id),
            )

        with self._lock:
            self._store.tx(_do)

    # ── Query operations ──────────────────────────────────────────────────
    def query_entity(
        self, name: str, as_of: Optional[str] = None, direction: str = "outgoing"
    ):
        _temporal_start_key, _temporal_end_key, sanitize_iso_temporal = (
            _import_helpers()
        )
        as_of = sanitize_iso_temporal(as_of, "as_of")
        eid = _entity_id(name)
        results = []

        temporal_sql = ""
        temporal_params: list = []
        if as_of:
            temporal_sql, temporal_params = _temporal_filter_sql(
                _temporal_start_key(as_of)
            )

        with self._lock:
            if direction in ("outgoing", "both"):
                rows = self._store.query(
                    "SELECT t.*, e.name as obj_name FROM triples t "
                    "JOIN entities e ON t.object = e.id WHERE t.subject = %s"
                    + temporal_sql,
                    [eid] + temporal_params,
                )
                for row in rows:
                    results.append(
                        {
                            "direction": "outgoing",
                            "subject": name,
                            "predicate": row["predicate"],
                            "object": row["obj_name"],
                            "valid_from": row["valid_from"],
                            "valid_to": row["valid_to"],
                            "confidence": row["confidence"],
                            "source_closet": row["source_closet"],
                            "current": row["valid_to"] is None,
                        }
                    )

            if direction in ("incoming", "both"):
                rows = self._store.query(
                    "SELECT t.*, e.name as sub_name FROM triples t "
                    "JOIN entities e ON t.subject = e.id WHERE t.object = %s"
                    + temporal_sql,
                    [eid] + temporal_params,
                )
                for row in rows:
                    results.append(
                        {
                            "direction": "incoming",
                            "subject": row["sub_name"],
                            "predicate": row["predicate"],
                            "object": name,
                            "valid_from": row["valid_from"],
                            "valid_to": row["valid_to"],
                            "confidence": row["confidence"],
                            "source_closet": row["source_closet"],
                            "current": row["valid_to"] is None,
                        }
                    )

        return results

    def query_relationship(self, predicate: str, as_of: Optional[str] = None):
        _temporal_start_key, _temporal_end_key, sanitize_iso_temporal = (
            _import_helpers()
        )
        as_of = sanitize_iso_temporal(as_of, "as_of")
        pred = predicate.lower().replace(" ", "_")

        query = (
            "SELECT t.*, s.name as sub_name, o.name as obj_name "
            "FROM triples t "
            "JOIN entities s ON t.subject = s.id "
            "JOIN entities o ON t.object = o.id "
            "WHERE t.predicate = %s"
        )
        params: list = [pred]
        if as_of:
            temporal_sql, temporal_params = _temporal_filter_sql(
                _temporal_start_key(as_of)
            )
            query += temporal_sql
            params.extend(temporal_params)

        results = []
        with self._lock:
            for row in self._store.query(query, params):
                results.append(
                    {
                        "subject": row["sub_name"],
                        "predicate": pred,
                        "object": row["obj_name"],
                        "valid_from": row["valid_from"],
                        "valid_to": row["valid_to"],
                        "current": row["valid_to"] is None,
                    }
                )
        return results

    def timeline(self, entity_name: Optional[str] = None):
        with self._lock:
            if entity_name:
                eid = _entity_id(entity_name)
                rows = self._store.query(
                    "SELECT t.*, s.name as sub_name, o.name as obj_name "
                    "FROM triples t "
                    "JOIN entities s ON t.subject = s.id "
                    "JOIN entities o ON t.object = o.id "
                    "WHERE (t.subject = %s OR t.object = %s) "
                    "ORDER BY (t.valid_from IS NULL), t.valid_from ASC "
                    "LIMIT 100",
                    [eid, eid],
                )
            else:
                rows = self._store.query(
                    "SELECT t.*, s.name as sub_name, o.name as obj_name "
                    "FROM triples t "
                    "JOIN entities s ON t.subject = s.id "
                    "JOIN entities o ON t.object = o.id "
                    "ORDER BY (t.valid_from IS NULL), t.valid_from ASC "
                    "LIMIT 100",
                )

        return [
            {
                "subject": r["sub_name"],
                "predicate": r["predicate"],
                "object": r["obj_name"],
                "valid_from": r["valid_from"],
                "valid_to": r["valid_to"],
                "current": r["valid_to"] is None,
            }
            for r in rows
        ]

    # ── Stats ─────────────────────────────────────────────────────────────
    def stats(self):
        with self._lock:
            entities = self._store.query("SELECT COUNT(*) as cnt FROM entities")[0][
                "cnt"
            ]
            triples = self._store.query("SELECT COUNT(*) as cnt FROM triples")[0]["cnt"]
            current = self._store.query(
                "SELECT COUNT(*) as cnt FROM triples WHERE valid_to IS NULL"
            )[0]["cnt"]
            expired = triples - current
            predicates = [
                r["predicate"]
                for r in self._store.query(
                    "SELECT DISTINCT predicate FROM triples ORDER BY predicate"
                )
            ]
        return {
            "entities": entities,
            "triples": triples,
            "current_facts": current,
            "expired_facts": expired,
            "relationship_types": predicates,
        }

    # ── Seed from known facts ─────────────────────────────────────────────
    def seed_from_entity_facts(self, entity_facts: dict):
        # Identical control flow to the SQLite KG — it only calls add_entity /
        # add_triple, which are reimplemented above, so reuse it verbatim.
        for key, facts in entity_facts.items():
            name = facts.get("full_name", key.capitalize())
            etype = facts.get("type", "person")
            self.add_entity(
                name,
                etype,
                {
                    "gender": facts.get("gender", ""),
                    "birthday": facts.get("birthday", ""),
                },
            )
            parent = facts.get("parent")
            if parent:
                self.add_triple(
                    name,
                    "child_of",
                    parent.capitalize(),
                    valid_from=facts.get("birthday"),
                )
            partner = facts.get("partner")
            if partner:
                self.add_triple(name, "married_to", partner.capitalize())
            relationship = facts.get("relationship", "")
            if relationship == "daughter":
                self.add_triple(
                    name,
                    "is_child_of",
                    facts.get("parent", "").capitalize() or name,
                    valid_from=facts.get("birthday"),
                )
            elif relationship == "husband":
                self.add_triple(
                    name, "is_partner_of", facts.get("partner", name).capitalize()
                )
            elif relationship == "brother":
                self.add_triple(
                    name, "is_sibling_of", facts.get("sibling", name).capitalize()
                )
            elif relationship == "dog":
                self.add_triple(
                    name, "is_pet_of", facts.get("owner", name).capitalize()
                )
                self.add_entity(name, "animal")
            for interest in facts.get("interests", []):
                self.add_triple(
                    name, "loves", interest.capitalize(), valid_from="2025-01-01"
                )
