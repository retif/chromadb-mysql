# xpu-bench — Intel iGPU embedding acceleration: research artifact

Investigation (2026-06-18) into whether the **Intel Iris Xe** iGPU on `emmett`
can accelerate sentence-embedding for mempalace backfill, vs CPU / HeatWave
in-DB `ML_EMBED`. **Conclusion: not via PyTorch-XPU/IPEX (hardware below the
support floor); OpenVINO is the only viable iGPU route and is untested. And it
doesn't matter for the backfill anyway — see below.**

## Hardware

- GPU: **Intel Iris Xe Graphics (Xe-LP, Gen12, Tiger Lake)** — PCI `8086:9A49`,
  driver `i915`, render node `renderD128`.
- No NVIDIA/CUDA GPU. ROCm is AMD-only → N/A.

## What was tested: native PyTorch XPU (NOT IPEX)

`torch==2.12.1+xpu` from `https://download.pytorch.org/whl/xpu` +
`sentence-transformers`, in the devShell here (which supplies the Level Zero
loader + Intel L0 GPU driver — see `flake.nix`).

Result:

- `torch.xpu.is_available()` → **True**; device enumerated as
  `Intel(R) Iris(R) Xe Graphics`. So the Level Zero plumbing is correct.
- PyTorch warned: *"The detected GPU (Intel Iris Xe Graphics) is not officially
  supported by PyTorch XPU … use Intel Arc (Alchemist) series or newer."*
- First real compute op (`torch.embedding`) **crashed**:
  `RuntimeError: The program was built for 1 devices` +
  `module got recompiled from IR because provided native binary is incompatible
  with underlying device and/or driver`.

**Why:** PyTorch's XPU wheel ships AOT-compiled kernels for certified arches
(**Arc / Xe-HPG / Data Center Max**). Iris Xe is **Xe-LP** — two generations
older, uncertified. The prebuilt binary is incompatible with the Xe-LP ISA, the
JIT-from-IR fallback fails to build a valid program for the device, and some
kernels additionally hit `Double type is not supported on this platform`
(Iris Xe has no fp64). No env var / oneAPI version fixes this — it's the silicon
class.

## IPEX (intel-extension-for-pytorch): not a live alternative

The IPEX repo was **archived 2026-03-30**; XPU support is now folded into
mainline PyTorch (what was tested above). Community issues confirm integrated
Xe-LP GPUs are unresolved on both IPEX and mainline:

- IPEX [#292](https://github.com/intel/intel-extension-for-pytorch/issues/292)
  (TigerLake-LP GT2, this exact GPU) — "Number of dpcpp devices should be
  greater than zero", unresolved.
- IPEX [#321](https://github.com/intel/intel-extension-for-pytorch/issues/321)
  — the "built for 1 devices" error; root cause "Double type is not supported".
- PyTorch [#164070](https://github.com/pytorch/pytorch/issues/164070) — Intel
  13500H iGPU (Xe-LP class), `UR error`, triaged/unresolved.

Officially-supported Intel iGPUs for `torch.xpu` are the recent **Core Ultra
(Meteor / Lunar / Arrow Lake)** Arc/Xe2 iGPUs — not Tiger Lake.

## The iGPU route that WORKS: OpenVINO (benchmarked)

OpenVINO runs on Intel iGPUs back through **Gen9** (Iris Xe Gen12 = supported)
via the OpenCL/Level-Zero **NEO** runtime. With the OpenCL ICD wired in the
devShell (`OCL_ICD_VENDORS` → intel-compute-runtime's `intel-neo.icd`),
OpenVINO enumerates the iGPU as a `GPU` device and **runs the embedding model on
it** — where torch-XPU cannot. Measured (`pure_ov_bench.py`, 512 texts,
all-MiniLM-L12-v2, prebuilt `openvino_model.xml`):

| backend | rate | cosine vs HeatWave |
|---|---|---|
| sentence-transformers PyTorch CPU | ~176/s | 1.00000 |
| sentence-transformers ONNX CPU | ~123/s | 1.00000 |
| sentence-transformers OpenVINO CPU | ~179/s | 1.00000 |
| **pure OpenVINO CPU** (no torch/ST overhead) | **427/s** | 1.00000 |
| **pure OpenVINO GPU — Iris Xe iGPU** | **741/s** | **1.00000** |

So the iGPU does embedding at ~741/s with **perfect fp32 parity** (drop-in with
the existing L12 palace). Key gotchas: (1) drive OpenVINO **directly** +
`tokenizers` + manual mean-pool — the `transformers`/`optimum-intel` path kept
breaking on a transformers downgrade (`optimum-intel` pins old transformers that
can't even `AutoConfig` a BERT model). (2) Use **fp32**; int8 quant would
perturb vectors and break cosine parity with the palace. (3) The OpenCL ICD
(`OCL_ICD_VENDORS`) must point at the Intel driver or the GPU plugin fails with
`m_device_map.empty()`.

`pure_ov_bench.py` is the working bench; `onnx_bench.py` is the
sentence-transformers-backend variant (CPU works; its OpenVINO-GPU path hit the
transformers/optimum dep issue). `bench.py` is the torch-XPU attempt (crashes).

## The result that makes all of the above moot for backfill

Embedding is **not** the backfill bottleneck:

| | rate |
|---|---|
| HeatWave `ML_EMBED_ROW` (in-DB) | ~0.9 s/text, serial (~1/s) |
| sentence-transformers CPU (warm, batched) | **317 texts/s** |
| Iris Xe via torch-XPU | crashes (unsupported silicon) |
| Iris Xe via **OpenVINO** | **741 texts/s** (works, parity 1.0) |
| **full client-mode mine (steady)** | **521 drawers/min (~8.7/s)** |

The full `~/.claude/projects` backfill (511,856 drawers) is **~16h**, bound by
the **miner pipeline over the SSH tunnel** (JSONL parse, chunk, room-detection,
per-batch dedup query, insert) — embedding is ~3% of that. No embedding backend
(CPU/ONNX/iGPU) changes the ETA. The only real lever is running the miner
**on `armer`** (HeatWave-local, no tunnel RTT).

## The devShell works (reusable for a future Arc GPU)

`flake.nix` provides a working Level Zero stack: `level-zero` loader +
`intel-compute-runtime.drivers` output (`libze_intel_gpu.so.1`, which lives in
the package's `drivers` output, not `out`) + a C/C++ runtime for the manylinux
torch wheels (nix-ld resolves the rest). On a supported GPU (Arc+) this devShell
with `bench.py` would just work.

```sh
nix develop          # enters shell; sets LD_LIBRARY_PATH + ZE_ENABLE_ALT_DRIVERS
uv venv .venv-xpu && . .venv-xpu/bin/activate
# clear the host's gitea/nexus index env or pip 401s:
env -u UV_INDEX_URL -u UV_EXTRA_INDEX_URL -u PIP_INDEX_URL -u PIP_EXTRA_INDEX_URL \
  uv pip install --no-config --index-url https://download.pytorch.org/whl/xpu torch pytorch-triton-xpu
env -u UV_INDEX_URL -u UV_EXTRA_INDEX_URL -u PIP_INDEX_URL -u PIP_EXTRA_INDEX_URL \
  uv pip install --no-config --default-index https://pypi.org/simple sentence-transformers
python bench.py
```

`.venv-xpu/` is regenerable and gitignored.
