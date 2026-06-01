"""chromadb.errors shim — only the symbols mempalace imports.

mempalace does a module-level ``from chromadb.errors import NotFoundError`` (in
backends/chroma.py, mcp_server.py, repair.py) and catches it on the
create-on-first-use path. The redirect (chromadb_switch) registers this module
as ``sys.modules['chromadb.errors']`` so that import resolves to here.
"""

from __future__ import annotations


class ChromaError(Exception):
    pass


class NotFoundError(ChromaError):
    """Mirror of chromadb.errors.NotFoundError (raised on a missing collection)."""

    pass


__all__ = ["ChromaError", "NotFoundError"]
