"""Bench ONNX Runtime + OpenVINO embedding backends for all-MiniLM-L12-v2,
vs the PyTorch CPU baseline, with cosine parity vs HeatWave ML_EMBED.

The OpenVINO-GPU case is the one that matters: OpenVINO supports Iris Xe (Gen12)
where torch-XPU does not. Each backend is isolated in try/except so one failure
doesn't abort the rest.
"""

import json
import math
import time

from sentence_transformers import SentenceTransformer

HW = json.load(open("/tmp/claude-1000/hw_l12.json"))
MODEL = "sentence-transformers/all-MiniLM-L12-v2"
TEXTS = [
    f"benchmark sentence number {i} with a handful of representative words here"
    for i in range(512)
]


def cos(a, b):
    d = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return d / (na * nb)


def bench(label, **kw):
    try:
        m = SentenceTransformer(MODEL, **kw)
    except Exception as e:  # noqa: BLE001
        print(f"{label}: LOAD FAILED: {type(e).__name__}: {str(e)[:180]}")
        return
    try:
        m.encode(TEXTS[:16])  # warmup
        t = time.time()
        m.encode(TEXTS, batch_size=64)
        dt = time.time() - t
        pv = m.encode([HW["text"]], normalize_embeddings=True)[0]
        c = cos([float(x) for x in pv], HW["vec"])
        print(f"{label}: {len(TEXTS) / dt:6.1f}/s | cosine vs HeatWave={c:.5f}")
    except Exception as e:  # noqa: BLE001
        print(f"{label}: RUN FAILED: {type(e).__name__}: {str(e)[:180]}")


bench("PyTorch-CPU fp32", device="cpu")
bench("ONNX-CPU fp32", backend="onnx", model_kwargs={"file_name": "onnx/model.onnx"})
bench(
    "ONNX-CPU int8",
    backend="onnx",
    model_kwargs={"file_name": "onnx/model_qint8_avx512_vnni.onnx"},
)
bench(
    "OpenVINO-CPU fp32",
    backend="openvino",
    model_kwargs={"file_name": "openvino/openvino_model.xml"},
)
bench(
    "OpenVINO-GPU(iGPU) fp32",
    backend="openvino",
    model_kwargs={"file_name": "openvino/openvino_model.xml", "device": "GPU"},
)
bench(
    "OpenVINO-GPU(iGPU) int8",
    backend="openvino",
    model_kwargs={
        "file_name": "openvino/openvino_model_qint8_quantized.xml",
        "device": "GPU",
    },
)
