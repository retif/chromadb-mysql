import time

import torch
from sentence_transformers import SentenceTransformer

print("torch:", torch.__version__)
ok = hasattr(torch, "xpu") and torch.xpu.is_available()
print("XPU available:", ok)
if ok:
    print("XPU device:", torch.xpu.get_device_name(0))
MODEL = "sentence-transformers/all-MiniLM-L12-v2"
texts = [
    f"benchmark sentence number {i} with a handful of representative words here"
    for i in range(512)
]


def bench(dev):
    m = SentenceTransformer(MODEL, device=dev)
    m.encode(texts[:16], batch_size=16)  # warmup
    if dev == "xpu":
        torch.xpu.synchronize()
    t = time.time()
    m.encode(texts, batch_size=64)
    if dev == "xpu":
        torch.xpu.synchronize()
    dt = time.time() - t
    print(f"{dev.upper()}: {len(texts)} texts in {dt:.2f}s = {len(texts) / dt:.1f}/s")


bench("cpu")
if ok:
    bench("xpu")
