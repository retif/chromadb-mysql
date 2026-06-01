# chromadb-mysql-backend

A drop-in, **ChromaDB-API-compatible** backend stored in **MySQL 9 `VECTOR`**
(and, later, HeatWave `ML_EMBED`). Built as a **switchable extension for
[mempalace](https://github.com/milla-jovovich/mempalace)** — mempalace's source
is never modified; standard ChromaDB remains the default and the MySQL backend is
opt-in per deployment.

## How it works

mempalace does a plain `import chromadb`. This package provides:

- **`chromadb_mysql_backend`** — a module exposing exactly the surface mempalace
  uses (`PersistentClient`, `__version__`; client `get/create/get_or_create/
  delete/list_collection(s)`; collection `add/upsert/get/query/delete/count`).
- **`chromadb_switch`** — an import-redirector. When `MEMPALACE_CHROMA_BACKEND=mysql`,
  it aliases `chromadb` → `chromadb_mysql_backend` in `sys.modules` at interpreter
  startup (via the installed `activate-chromadb-mysql.pth`). Unset/other value =
  no-op, standard ChromaDB runs unchanged.

So the switch is a single env var, wired from a Helm value — no fork, no code
edits in mempalace, and standard behaviour is preserved by default.

## Switch + connection env

| Env var | Meaning | Default |
|---------|---------|---------|
| `MEMPALACE_CHROMA_BACKEND` | `mysql` enables the extension; else standard ChromaDB | unset (standard) |
| `MEMPALACE_MYSQL_HOST` | MySQL host (required, no default) | – |
| `MEMPALACE_MYSQL_PORT` | port | `3306` |
| `MEMPALACE_MYSQL_USER` / `_PASSWORD` / `_DB` | credentials / schema | `mempalace` / – / `mempalace` |

## Status

- ✅ Implemented + unit-tested (no DB needed): `where` translator (equality +
  `$and`), Chroma result shapes (flat `get`, nested `query`), the on/off switch.
- 🚧 Live-DB code written, pending HeatWave conformance: `Collection` SQL
  (`VECTOR_DISTANCE … 'COSINE'`, `STRING_TO_VECTOR`, `INSERT … ON DUPLICATE KEY`),
  `Client` DDL, MiniLM-L6-v2 embeddings (Option A, 384-dim parity).

Tracked under milestone #60 on `oleks/mempalace`. Plan:
`servers/cluster/plans/mempalace-mysql-backend.md`.

## Test

```sh
PYTHONPATH=. pytest -q
```
