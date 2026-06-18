"""Embedding configuration.

DEFAULT (db mode): embeddings are generated IN-DATABASE by HeatWave
``sys.ML_EMBED_ROW(text, JSON_OBJECT('model_id', <model>))`` — no client model,
no torch/sentence-transformers in the closure. The SQL is emitted by
``_collection``; this module only supplies the model id + dim config.

FALLBACK (MEMPALACE_EMBED_MODE=client): the bundled ``MiniLMEmbedder`` computes
vectors client-side (lazy sentence-transformers; needs the ``embed`` extra).
Kept for parity/testing; not used by the light mysql image.
"""

from __future__ import annotations

import os
from typing import Protocol

# HeatWave embedding model_id (underscored form), 384-dim — closest to Chroma's
# default all-MiniLM-L6-v2 family. NOTE: L12 (HeatWave) ≠ L6 (Chroma); both
# 384-dim so no schema change, but vectors are NOT cross-comparable — a palace
# must be embedded entirely by one model.
DEFAULT_MODEL = "all_minilm_l12_v2"
DEFAULT_DIM = 384

ENV_MODEL = "MEMPALACE_EMBED_MODEL"
ENV_MODE = "MEMPALACE_EMBED_MODE"


def model_from_env() -> str:
    return os.environ.get(ENV_MODEL, DEFAULT_MODEL)


def mode_from_env() -> str:
    return os.environ.get(ENV_MODE, "db").strip().lower()


# Map a HeatWave ML_EMBED model_id to its sentence-transformers equivalent, so
# client-side embedding (MEMPALACE_EMBED_MODE=client) produces vectors in the
# SAME space as the in-DB ML_EMBED path. Verified parity: HeatWave
# all_minilm_l12_v2 vs ST all-MiniLM-L12-v2 give cosine ~= 1.0 for the same
# text, so a client-embedded palace is interchangeable with an ML_EMBED one.
_ST_MODEL_BY_ID = {
    "all_minilm_l12_v2": "sentence-transformers/all-MiniLM-L12-v2",
    "all_minilm_l6_v2": "sentence-transformers/all-MiniLM-L6-v2",
}


def sentence_transformers_name(model_id: str) -> str:
    """ST model name matching a HeatWave model_id (passthrough if unknown)."""
    return _ST_MODEL_BY_ID.get(model_id, model_id)


ENV_OPENVINO_DEVICE = "MEMPALACE_EMBED_OPENVINO_DEVICE"


def make_client_embedder(embed_model: str | None = None):
    """Return the client-side embedder when MEMPALACE_EMBED_MODE=client, else
    None (in-DB ML_EMBED stays the default). The model matches the HeatWave
    model_id so client- and ML_EMBED-produced vectors are cross-compatible.

    If MEMPALACE_EMBED_OPENVINO_DEVICE is set (e.g. "GPU" / "CPU") the OpenVINO
    embedder is used (iGPU-capable); otherwise the sentence-transformers
    MiniLMEmbedder."""
    if mode_from_env() != "client":
        return None
    ov_device = os.environ.get(ENV_OPENVINO_DEVICE, "").strip()
    if ov_device:
        return OpenVINOEmbedder(device=ov_device)
    model_id = embed_model or model_from_env()
    return MiniLMEmbedder(sentence_transformers_name(model_id))


class Embedder(Protocol):
    dim: int

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class MiniLMEmbedder:
    """Optional client-side fallback (MEMPALACE_EMBED_MODE=client) — bundled
    sentence-transformers MiniLM (384-dim cosine). Lazy import so the default
    db-mode closure carries no torch."""

    dim = DEFAULT_DIM

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L12-v2"):
        # Default matches the palace's HeatWave model (all_minilm_l12_v2). Using
        # the L6 default here would produce an incompatible vector space.
        self._model_name = model_name
        self._model = None

    def _ensure(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # lazy

            self._model = SentenceTransformer(self._model_name)
        return self._model

    def embed(self, texts: list[str]) -> list[list[float]]:
        model = self._ensure()
        vecs = model.encode(list(texts), normalize_embeddings=True)
        return [list(map(float, v)) for v in vecs]


class OpenVINOEmbedder:
    """Client-side embedder via OpenVINO — runs all-MiniLM-L12-v2 on the Intel
    iGPU (device="GPU") or CPU. Loads the prebuilt openvino_model.xml from the
    HF repo and tokenizes with the `tokenizers` lib + manual mean-pool, bypassing
    transformers/optimum (whose version pins break). fp32 → cosine parity 1.0
    with HeatWave's all_minilm_l12_v2 (verified). Needs the openvino + tokenizers
    + huggingface_hub extras and, for device=GPU, the Level Zero / OpenCL runtime
    (see xpu-bench/flake.nix)."""

    dim = DEFAULT_DIM

    def __init__(self, device: str = "GPU", model: str = "sentence-transformers/all-MiniLM-L12-v2"):
        self._device = device
        self._model = model
        self._compiled = None
        self._tok = None
        self._inputs = None

    def _ensure(self):
        if self._compiled is None:
            import openvino as ov  # lazy
            from huggingface_hub import hf_hub_download
            from tokenizers import Tokenizer

            xml = hf_hub_download(self._model, "openvino/openvino_model.xml")
            hf_hub_download(self._model, "openvino/openvino_model.bin")
            tok = Tokenizer.from_file(hf_hub_download(self._model, "tokenizer.json"))
            tok.enable_padding(length=None)
            tok.enable_truncation(max_length=256)
            self._tok = tok
            core = ov.Core()
            m = core.read_model(xml)
            self._inputs = [i.get_any_name() for i in m.inputs]
            self._compiled = core.compile_model(m, self._device)
        return self._compiled

    def embed(self, texts: list[str]) -> list[list[float]]:
        import numpy as np

        compiled = self._ensure()
        encs = self._tok.encode_batch(list(texts))
        ids = np.array([e.ids for e in encs], dtype=np.int64)
        mask = np.array([e.attention_mask for e in encs], dtype=np.int64)
        feed = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in self._inputs:
            feed["token_type_ids"] = np.zeros_like(ids)
        tok_emb = compiled(feed)[compiled.output(0)]
        m = mask[:, :, None].astype(np.float32)
        mean = (tok_emb * m).sum(axis=1) / np.clip(m.sum(axis=1), 1e-9, None)
        norm = mean / np.linalg.norm(mean, axis=1, keepdims=True)
        return norm.tolist()


def vector_to_sql(vec: list[float]) -> str:
    """MySQL 9 takes a vector as a JSON array string via STRING_TO_VECTOR().
    Used by the precomputed-embeddings escape hatch + unit tests."""
    return "[" + ",".join(repr(float(x)) for x in vec) + "]"
