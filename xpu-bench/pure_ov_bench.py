"""Pure-OpenVINO embedding bench — no transformers/optimum (those kept breaking
on dep/version/config issues). Loads the prebuilt openvino_model.xml from the HF
repo, tokenizes with the `tokenizers` lib, mean-pools + L2-normalizes manually,
and times CPU vs the Iris Xe iGPU (device="GPU") with cosine parity vs HeatWave.
"""

import json
import math
import time

import numpy as np
import openvino as ov
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

MODEL = "sentence-transformers/all-MiniLM-L12-v2"
HW = json.load(open("/tmp/claude-1000/hw_l12.json"))
TEXTS = [
    f"benchmark sentence number {i} with a handful of representative words here"
    for i in range(512)
]

xml = hf_hub_download(MODEL, "openvino/openvino_model.xml")
hf_hub_download(MODEL, "openvino/openvino_model.bin")  # sibling, fetched next to xml
tok = Tokenizer.from_file(hf_hub_download(MODEL, "tokenizer.json"))
tok.enable_padding(length=None)
tok.enable_truncation(max_length=256)

core = ov.Core()
print("OpenVINO devices:", core.available_devices)
model = core.read_model(xml)
in_names = [i.get_any_name() for i in model.inputs]
print("model inputs:", in_names)


def encode(compiled, texts):
    encs = tok.encode_batch(texts)
    ids = np.array([e.ids for e in encs], dtype=np.int64)
    mask = np.array([e.attention_mask for e in encs], dtype=np.int64)
    feed = {"input_ids": ids, "attention_mask": mask}
    if "token_type_ids" in in_names:
        feed["token_type_ids"] = np.zeros_like(ids)
    out = compiled(feed)
    tok_emb = out[compiled.output(0)]  # [B, T, H] last_hidden_state
    m = mask[:, :, None].astype(np.float32)
    summed = (tok_emb * m).sum(axis=1)
    counts = np.clip(m.sum(axis=1), 1e-9, None)
    mean = summed / counts
    norm = mean / np.linalg.norm(mean, axis=1, keepdims=True)
    return norm


def cos(a, b):
    d = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return d / (na * nb)


def bench(device):
    try:
        compiled = core.compile_model(model, device)
    except Exception as e:  # noqa: BLE001
        print(f"{device}: COMPILE FAILED: {type(e).__name__}: {str(e)[:160]}")
        return
    try:
        encode(compiled, TEXTS[:16])  # warmup
        t = time.time()
        for i in range(0, len(TEXTS), 64):
            encode(compiled, TEXTS[i : i + 64])
        dt = time.time() - t
        pv = encode(compiled, [HW["text"]])[0]
        c = cos([float(x) for x in pv], HW["vec"])
        print(f"{device}: {len(TEXTS) / dt:6.1f}/s | cosine vs HeatWave={c:.5f}")
    except Exception as e:  # noqa: BLE001
        print(f"{device}: RUN FAILED: {type(e).__name__}: {str(e)[:160]}")


bench("CPU")
if "GPU" in core.available_devices:
    bench("GPU")
