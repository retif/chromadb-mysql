"""Stub ``chromadb.utils.embedding_functions`` for the MySQL backend alias.

Only the symbol mempalace imports is provided: ``ONNXMiniLM_L6_V2``. mempalace
subclasses it (renaming ``name()`` to ``"default"``) and constructs it with a
``preferred_providers=`` kwarg, then hands the instance to the collection as
``embedding_function=…`` — which the MySQL backend ignores (embeddings are
computed in-DB via ``ML_EMBED``). So the stub only needs to be a constructible
class; it is never invoked to embed in db mode.
"""

from __future__ import annotations


class ONNXMiniLM_L6_V2:  # noqa: N801 - mirror ChromaDB's public class name
    """Inert stand-in for ChromaDB's ONNX MiniLM embedding function.

    Constructible (accepts and ignores ``preferred_providers`` and any other
    kwargs) so mempalace can subclass and instantiate it. It is never called to
    embed under the MySQL backend — embedding happens server-side via
    ``ML_EMBED``. If something *does* call it (i.e. a non-db embed path leaked
    in), fail loudly rather than silently returning wrong vectors.
    """

    def __init__(self, *args, **kwargs) -> None:
        self._args = args
        self._kwargs = kwargs

    @staticmethod
    def name() -> str:
        return "onnx_mini_lm_l6_v2"

    def __call__(self, *args, **kwargs):
        raise RuntimeError(
            "chromadb_mysql_backend stub ONNXMiniLM_L6_V2 was called to embed, "
            "but db mode embeds server-side via ML_EMBED. Set "
            "MEMPALACE_EMBED_MODE=client (with the [embed] extra) for a real "
            "client-side embedder."
        )
