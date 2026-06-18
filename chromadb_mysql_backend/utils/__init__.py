"""Stand-in for ``chromadb.utils`` under the MySQL backend alias.

mempalace does ``from chromadb.utils.embedding_functions import
ONNXMiniLM_L6_V2`` to build a client-side embedding function. Under the MySQL
backend the real ``chromadb`` package is never loaded (it's aliased to
``chromadb_mysql_backend``), so that import would raise ``ModuleNotFoundError:
No module named 'chromadb.utils'`` — which mempalace catches and reports as a
noisy "Failed to build embedding function; using chromadb default" traceback.

In db mode embeddings are computed *server-side* via HeatWave ``ML_EMBED`` and
the collection ignores any client embedding function, so the client EF is
decorative. This package exposes a constructible stub so the import succeeds
quietly — no behavioural change, just no traceback. See
``chromadb_switch.activate`` for the sys.modules registration.
"""

from __future__ import annotations
