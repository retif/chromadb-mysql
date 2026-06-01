"""Translate the tiny subset of ChromaDB `where` filters mempalace uses into a
SQL WHERE fragment.

mempalace only ever issues:
  * equality maps         {"source_file": "..."}            (implicit AND of keys)
  * a single ``$and``     {"$and": [{"wing": w}, {"room": r}]}

We also accept ``$or`` and nesting for robustness, but nothing else (no
``$contains`` / ``$in`` / ``$nin`` / ``where_document`` — mempalace never emits
them). Metadata is stored in a JSON column, so every key compares against a JSON
path; the generated columns (wing, room) are an index optimisation, not a
correctness requirement.

``translate(where)`` returns ``(clause, params)`` where ``clause`` has no leading
``WHERE`` and uses ``%s`` placeholders. An empty/None filter yields ``("", [])``.
"""

from __future__ import annotations

_META = "metadata"


def _leaf(key: str, value) -> tuple[str, list]:
    # JSON_UNQUOTE so string comparisons match the stored scalar, not a quoted
    # JSON token. mempalace's filtered keys (wing/room/source_file) are strings.
    frag = f"JSON_UNQUOTE(JSON_EXTRACT({_META}, %s)) = %s"
    return frag, [f"$.{key}", _scalar(value)]


def _scalar(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    return value if isinstance(value, str) else str(value)


def _combine(op: str, parts: list[dict]) -> tuple[str, list]:
    frags: list[str] = []
    params: list = []
    for part in parts:
        f, p = translate(part)
        if f:
            frags.append(f"({f})")
            params.extend(p)
    if not frags:
        return "", []
    return f" {op} ".join(frags), params


def translate(where: dict | None) -> tuple[str, list]:
    if not where:
        return "", []

    if "$and" in where:
        return _combine("AND", where["$and"])
    if "$or" in where:
        return _combine("OR", where["$or"])

    # Implicit AND over each key=value pair (Chroma semantics).
    frags: list[str] = []
    params: list = []
    for key, value in where.items():
        if isinstance(value, dict):
            # e.g. {"k": {"$eq": v}} — unwrap the single supported operator.
            if "$eq" in value:
                value = value["$eq"]
            else:  # pragma: no cover - mempalace never emits these
                raise ValueError(f"unsupported where operator on {key!r}: {value!r}")
        frag, p = _leaf(key, value)
        frags.append(frag)
        params.extend(p)
    return " AND ".join(frags), params
