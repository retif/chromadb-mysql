"""chromadb_mysql_backend — a ChromaDB-API-compatible module backed by MySQL 9
VECTOR with in-DB HeatWave ML_EMBED embeddings (the active default).

The import-redirector (`chromadb_switch`) aliases this module to ``chromadb`` in
``sys.modules`` when the MySQL backend is enabled, and registers the submodules
(``chromadb.errors``) so mempalace's unmodified ``import chromadb`` /
``from chromadb.errors import NotFoundError`` resolve here. Public surface:
``PersistentClient``, ``Client``, ``__version__``, and the ``errors`` submodule.
"""

from __future__ import annotations

from . import errors
from ._client import Client

# Reported back as chromadb.__version__ (mempalace prints it). Advertise the
# pre-1.0 API generation we emulate, suffixed to identify the backend.
__version__ = "0.6.3+mysql"


def PersistentClient(path: str | None = None, **kwargs):  # noqa: N802 (Chroma name)
    return Client(path=path)


__all__ = ["PersistentClient", "Client", "errors", "__version__"]
