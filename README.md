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

## Laptop CLI — `mempalace-remote`

The stock `mempalace` CLI can't reach the cluster palace from a laptop for two
reasons: (1) it only ever opens a **local on-disk ChromaDB**
(`PersistentClient(path=…)`) — a bare `mempalace mine` writes a *separate* local
palace, not the cluster's; (2) the cluster's MySQL (`10.0.2.15`, the OCI HeatWave
instance) is **VCN-private**, reachable only via node `armer`.

`bin/mempalace-remote` bridges both: it opens an idempotent SSH local-forward
`127.0.0.1:13306 → 10.0.2.15:3306` through `armer`, exports the backend +
connection env (so the `.pth` shim redirects `chromadb` → this package), and
`exec`s the stock CLI. So `mempalace-remote mine …` writes to the **real**
cluster palace.

### Install (uv tool, with this backend injected)

```sh
# public PyPI; clear the home's Gitea/Nexus index env so resolution doesn't 401
env -u UV_INDEX_URL -u UV_EXTRA_INDEX_URL -u PIP_INDEX_URL -u PIP_EXTRA_INDEX_URL \
  uv tool install mempalace --no-config \
  --default-index https://pypi.org/simple \
  --with "chromadb-mysql-backend[mysql] @ file://$PWD"
```

This lands both `.pth` files in the tool env's `site-packages`, so the MySQL
redirect auto-activates whenever `MEMPALACE_CHROMA_BACKEND=mysql` is set.

### Use

```sh
bin/mempalace-remote mine ~/.claude/projects/ --mode convos --dry-run   # preview
bin/mempalace-remote mine ~/.claude/projects/ --mode convos             # backfill
```

Password comes from `pass infra/mempalace/mysql-password` by default. Overrides:
`MEMPALACE_REMOTE_SSH` (jump host, default `armer.oracle.cloud`),
`MEMPALACE_REMOTE_LPORT` (default `13306`), `MEMPALACE_MYSQL_PASSWORD`.

### Resumable backfill — in-place mine

A full `~/.claude/projects` backfill is a multi-hour, single-process, CPU-bound
job (measured: ~96% miner-CPU — parsing/chunking/room-detection — so neither
faster embedding, running on `armer`, nor parallel workers speed it; one worker
already saturates the CPU). **It is also already resumable** — the miner writes
a per-file sentinel keyed on `source_file` and **skips files it has already
mined** on re-run. So the canonical backfill is just an in-place mine:

```sh
MEMPALACE_EMBED_MODE=client MEMPALACE_EMBED_OPENVINO_DEVICE=CPU \
  bin/mempalace-remote mine ~/.claude/projects --mode convos --wing claude_history
```

Kill it anytime; re-run to resume (already-mined files are skipped). Because the
drawer id is `(wing, room, source_file, …)`, mining the **real** paths is
idempotent and consistent with the Stop/PreCompact hooks (which mine those same
paths). For all-time history, point it at the archive once the live window has
been pruned (`~/Documents/Claude_JSONL_Backup`).

> **`bin/mempalace-batch-ingest` is DEPRECATED.** It staged each batch into a
> throwaway `/tmp/mp-batch-XXXX` dir, which defeats the path-keyed dedup (the
> miner never recognised the temp path as already-mined → no skip, duplicate
> drawers on re-run, dead `/tmp` provenance). Use the in-place mine above.

### Read-side guard shim

Some **read-side** CLI commands (`status`, `wake-up`, `sync`) hard-gate on a
local `chroma.sqlite3` file (cli.py — a leftover on-disk-ChromaDB assumption)
and would print "No palace found" in MySQL mode even though the data is fully
reachable. The wrapper works around this without patching mempalace: it creates
a stub `$MEMPALACE_PALACE_PATH/chroma.sqlite3` (empty) so the existence guard
passes, after which the command reads through the MySQL-aliased chromadb client.
Verified: `mempalace-remote status` reports the live cluster drawer count and
full wing/room tree; `mempalace-remote wake-up` renders L0/L1 from MySQL.

Note: query embedding for `search` is computed server-side in HeatWave
(`ML_EMBED`), so no client embedder is needed. mempalace still imports
`ONNXMiniLM_L6_V2` from `chromadb.utils.embedding_functions` to build a
(decorative, in db mode) client EF; the backend exposes an inert stub for that
import (registered by `chromadb_switch`) so it resolves quietly instead of
raising a `No module named 'chromadb.utils'` traceback.

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
