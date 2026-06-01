"""chromadb_mysql_backend — a ChromaDB-API-compatible module backed by MySQL 9
VECTOR (and, later, HeatWave ML_EMBED).

The import-redirector (`chromadb_switch`) aliases this module to ``chromadb`` in
``sys.modules`` when the MySQL backend is enabled, so mempalace's unmodified
``import chromadb`` resolves here. The public surface is exactly what mempalace
touches: ``PersistentClient`` and ``__version__``.
"""

from __future__ import annotations

from ._client import Client

# Reported back to callers as chromadb.__version__ (mempalace only prints it).
# Advertise the API generation we emulate (0.6.x — the pre-1.0 surface mempalace
# was written against), suffixed to make the backend identifiable.
__version__ = "0.6.3+mysql"


def PersistentClient(path: str | None = None, **kwargs):  # noqa: N802 (Chroma name)
    return Client(path=path)


__all__ = ["PersistentClient", "Client", "__version__"]
