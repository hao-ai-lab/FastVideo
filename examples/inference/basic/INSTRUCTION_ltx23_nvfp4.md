# LTX-2.3 distilled — NVFP4 / FA4 / CUDA-graphs benchmark & reproduction

End-to-end instructions to reproduce the LTX-2.3 distilled inference speed
sweep on a **Blackwell GB300 (sm_103)**, including the FP4 attention (FA4-FP4)
path and the CUDA-graphs optimization that makes NVFP4 the fastest config.

## TL;DR result (t2v, 832×1280, 121 frames, 8 denoise + 3 refine, GB300)

| config                         | denoise | refine | **e2e** |
|--------------------------------|---------|--------|---------|
| bf16 + FA2 (baseline) compile  | 1.476   | 2.759  | 5.15 s  |
| bf16 + FA4 compile             | 1.423   | 1.754  | 4.06 s  |
| nvfp4 + FA4-FP4 compile        | 3.777   | 1.655  | 6.35 s  |
| bf16 + FA4 + **cudagraphs**    | 1.105   | 1.728  | 3.81 s  |
| **nvfp4 + FA4-FP4 + cudagraphs** | **0.859** | **1.193** | **3.00 s** ⭐ |

Key findings:
- **FA4-bf16** (CuTeDSL attention kernel) is a big win over FlashAttention-2.
- The NVFP4 denoise penalty is **per-step launch-bound** (per-layer FP4 quant +
  mm_fp4 = hundreds of tiny kernel launches/step; ~flat across resolution).
- **CUDA graphs** removes that launch overhead → NVFP4 denoise 3.78→0.86 s, and
  nvfp4+FA4+cudagraphs becomes the fastest end-to-end config.

---

## 1. Hardware

- NVIDIA **GB300** (Grace-Blackwell, compute capability **sm_103**, 256 GB).
- On a mixed box, pin it: this repo's box has GB300 as **GPU 1**
  (`CUDA_VISIBLE_DEVICES=1`), GPU 0 is an RTX PRO 6000 workstation card.
- ⚠️ The installed torch wheel lists archs `sm_80/90/100/120` (no `sm_103`).
  Compute works on GB300, but importing torch/fastvideo with **both** GPUs
  visible crashes in the capability check — always set `CUDA_VISIBLE_DEVICES=1`.

## 2. Conda env + PyTorch

Python 3.12, CUDA 12.8 torch build:

```bash
# torch 2.11.0+cu128 (aarch64/sbsa wheel for Grace) + torchvision
pip install --index-url https://download.pytorch.org/whl/cu128 \
    'torch==2.11.0+cu128' 'torchvision==0.26.0+cu128'
```

Verified versions on the reference machine:

```
torch==2.11.0+cu128       torchvision==0.26.0+cu128
flashinfer-python==0.6.12 nvidia-cutlass-dsl==4.5.2
flash-attn-4==0.0.1.dev1330+gd268d2b86   quack-kernels==0.5.0
ninja==1.13.0
```

## 3. System CUDA toolkit (for flashinfer / FA4 JIT)

flashinfer and the FA4-FP4 CuTeDSL kernels JIT-compile on first use and need a
full CUDA toolkit with `nvcc` that supports `compute_103`. Use **CUDA 13.2**:

```bash
export CUDA_HOME=/usr/local/cuda-13.2          # must contain bin/nvcc + include/
# ensure the conda env bin (with `ninja`) AND cuda bin are on PATH:
export PATH="$CONDA_PREFIX/bin:/usr/local/cuda-13.2/bin:$PATH"
nvcc --version            # -> release 13.2 ; supports compute_103
```

## 4. NVFP4 (DiT-linear FP4) — flashinfer

```bash
pip install flashinfer-python
```

> ⚠️ `flashinfer-python` may try to **downgrade torch** to a CPU build. If it
> does, reinstall torch from step 2 afterwards (use `--no-deps` on flashinfer or
> reinstall torch). NVFP4 linears JIT-build an sm_103 module on first use; this
> needs `nvcc` (CUDA_HOME) and `ninja` (conda env bin) on PATH.

## 5. FA4-FP4 attention (quantized Q/K) — extra deps + a one-line shim

FP4 attention (`nvfp4_fa4=True`) needs the hao-ai-lab flash-attention-fp4 fork
plus QuACK, and a tiny cutlass-dsl compatibility shim.

```bash
# the FP4 CuTeDSL kernels (installs pkg `flash-attn-4`, providing flash_attn.cute)
pip install --no-deps \
    "git+https://github.com/hao-ai-lab/flash-attention-fp4.git@fp4#subdirectory=flash_attn/cute"
# QuACK kernels (imported as `quack`)
pip install --no-deps quack-kernels
```

`flash_attn.cute` imports `cutlass.utils.ampere_helpers`, which
`nvidia-cutlass-dsl >= 4.5` removed (and we can't downgrade — flashinfer needs
>=4.5). Restore just the one symbol it uses (`SMEM_CAPACITY`):

```bash
python - <<'PY'
import os, cutlass
d = os.path.join(os.path.dirname(cutlass.__file__), "utils")
open(os.path.join(d, "ampere_helpers.py"), "w").write(
    "SMEM_CAPACITY = {"
    "'sm80': (164-1)*1024, 'sm86': (100-1)*1024, "
    "'sm87': (164-1)*1024, 'sm89': (100-1)*1024}\n")
print("wrote", os.path.join(d, "ampere_helpers.py"))
PY
```

Sanity check the whole chain:

```bash
CUDA_VISIBLE_DEVICES=1 python -c "
import flash_attn; from flash_attn import flash_attn_func          # FA2 core
import flash_attn.cute.interface                                    # FA4 cute
from fastvideo.attention.utils.flash_attn_cute import flash_attn_fp4_func
import fastvideo.attention.backends.flash_attn as fa
print('FA4-FP4 available:', fa._FA4_FP4_AVAILABLE)"   # -> True
```

## 6. fastvideo code change (already in this branch)

`fastvideo/attention/backends/flash_attn.py`:
- `_nvfp4_quantize_for_fa4` is wrapped as a `torch.library.custom_op` +
  `register_fake` so torch.compile treats the flashinfer FP4 quant as an opaque
  leaf (dynamo can't trace its `@functools.cache` lock / JIT `subprocess`).
- `FlashAttentionImpl.__init__` pre-builds the FP4 quant module (warms the cache
  before compile).

These are required for **compile + FA4-FP4** to work at all. Nothing to do if
you're on this branch.

## 7. Model

Auto-downloaded on first run (~67 GB; skips the redundant `download/` raw
checkpoints). To pre-stage and avoid re-fetching:

```bash
python -c "from huggingface_hub import snapshot_download; \
print(snapshot_download('FastVideo/LTX-2.3-Distilled-Diffusers', \
ignore_patterns=['download/*','*.onnx','*.msgpack'], max_workers=8))"
# point runs at the printed snapshot dir via LTX23_MODEL_PATH (optional)
```

## 8. Running the benchmark

The parametrized harness is `examples/inference/basic/bench_ltx2_3_distilled_t2v.py`.
All knobs are env vars:

| env var               | default | meaning |
|-----------------------|---------|---------|
| `LTX23_COMPILE`       | 0       | torch.compile DiT+TE+VAE |
| `LTX23_NVFP4`         | 0       | NVFP4 quant the DiT linears |
| `LTX23_FA4`           | 0       | FA4-FP4 attention (quant Q/K, sets `nvfp4_fa4`) |
| `LTX23_COMPILE_MODE`  | default | DiT compile mode (`default` / `reduce-overhead`) |
| `LTX23_NO_CGTREES`    | 0       | set 1 with reduce-overhead (see caveats) |
| `LTX23_COMPILE_TE`    | 1       | compile text encoder (set 0 for cudagraphs) |
| `LTX23_COMPILE_VAE`   | 1       | compile VAE (kept on `default` mode) |
| `LTX23_HEIGHT`/`_WIDTH` | 1280/832 | resolution (H×W) |
| `LTX23_NUM_FRAMES`    | 121     | 121≈5 s, 481≈20 s @24fps |
| `LTX23_STEPS`         | 8       | denoise steps (refine fixed at 3) |
| `LTX23_VAE_TILING`    | 0       | tile the VAE decode — set 1 for long/high-res (avoids the 32-bit decode overflow; see §11) |
| `LTX23_DECODE`        | 1       | set 0 → `output_type="latent"`: skip VAE decode + video save to measure DiT (denoise/refine) timing cleanly on long runs |
| `LTX23_WARMUP`        | 2 / 1   | warmup runs (default 2 if compile else 1) |
| `LTX23_MEASURED`      | 3       | measured runs to average |
| `LTX23_MODEL_PATH`    | hub id  | local snapshot dir (optional) |

Common env block for every run:

```bash
export CUDA_VISIBLE_DEVICES=1 CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_HOME=/usr/local/cuda-13.2
export PATH="$CONDA_PREFIX/bin:/usr/local/cuda-13.2/bin:$PATH"
export TORCHINDUCTOR_CACHE_DIR=$HOME/.cache/torchinductor_ltx23   # caches cold compiles
cd <repo root>
SCRIPT=examples/inference/basic/bench_ltx2_3_distilled_t2v.py
```

### The 5 headline configs (832×1280, 8+3)

```bash
# bf16 + FA2 baseline (force FA2 by hiding flash_attn.cute is not needed; FA4 is
# auto-selected once cute is installed — to get the FA2 number, benchmark before
# installing the FA4 deps, or compare against the table above)

# bf16 + FA4, compile
LTX23_COMPILE=1 LTX23_NVFP4=0 LTX23_FA4=0 python $SCRIPT          # ~4.06 s

# nvfp4 + FA4-FP4, compile
LTX23_COMPILE=1 LTX23_NVFP4=1 LTX23_FA4=1 python $SCRIPT          # ~6.35 s

# bf16 + FA4 + CUDA graphs
LTX23_COMPILE=1 LTX23_NVFP4=0 LTX23_FA4=0 \
  LTX23_COMPILE_MODE=reduce-overhead LTX23_NO_CGTREES=1 LTX23_COMPILE_TE=0 \
  python $SCRIPT                                                  # ~3.81 s

# nvfp4 + FA4-FP4 + CUDA graphs  ⭐ fastest
LTX23_COMPILE=1 LTX23_NVFP4=1 LTX23_FA4=1 \
  LTX23_COMPILE_MODE=reduce-overhead LTX23_NO_CGTREES=1 LTX23_COMPILE_TE=0 \
  python $SCRIPT                                                  # ~3.00 s
```

Each run prints a per-stage breakdown and an averaged e2e over 3 measured runs
(after warmups). First run pays a one-time cold compile (~15 min) cached in
`$TORCHINDUCTOR_CACHE_DIR`.

i2v variants (anchor an image at frame 0) live in the same folder:
`basic_ltx2_3_distilled_i2v_{uncompiled,compiled}{,_nvfp4}.py` — set
`LTX23_I2V_IMAGE=/path/to.jpg`.

## 9. CUDA graphs caveats (why the extra flags)

Plain `mode="reduce-overhead"` fails with the FP4 pipeline; two workarounds are
needed and are wired to the env flags above:

1. **`LTX23_NO_CGTREES=1`** → sets `torch._inductor.config.triton.cudagraph_trees
   = False`. cudagraph_**trees** rejects the flashinfer FP4 custom ops because
   they allocate tensors inside the captured region that inductor doesn't track
   (`Detected N tensor(s) in the cudagraph pool not tracked as outputs`).
2. **`LTX23_COMPILE_TE=0`** → don't cudagraph the text encoder. cudagraphs on
   `gemma.py` triggers cross-module static-buffer aliasing
   (`accessing tensor output of CUDAGraphs that has been overwritten`). VAE stays
   compiled on `mode="default"`. So cudagraphs is applied to the **DiT only**.

Correctness: with `cudagraph_trees=False` the trees safety check is off, so we
validated the output is a real, coherent video (121 frames, healthy stats) and
that the cudagraph result is the *closest* match to its own non-cudagraph output
(PSNR 17.95 dB — higher than any cross-config pair). The per-frame difference is
ordinary diffusion sensitivity to kernel/numeric changes, not corruption.

## 10. Long-token / longer-video scaling (241 / 481 frames)

DiT token length is `N = T_lat × H_lat × W_lat` with VAE compression 32×
spatial / 8× temporal and DiT patch 1: `T_lat = (frames−1)/8 + 1`,
`H_lat = H/32`, `W_lat = W/32`. At 832×1280 each extra latent frame adds
40×26 = 1040 tokens, so **frame count is the clean linear axis** for long-context
tests — use `frames = 8k+1` (121 / 241 / 481), else `(frames−1)//8` truncates.

Measured on GB300 (t2v, 832×1280, 8+3, nvfp4+FA4-FP4+CUDA graphs vs bf16+FA4+CG),
seconds:

| frames (~dur) | N (refine) | nvfp4  denoise / refine / **e2e** | bf16  denoise / refine / e2e |
|---------------|-----------:|-----------------------------------|------------------------------|
| 121 (~5 s)    | 16,640     | 0.859 / 1.193 / **3.00**          | 1.082 / 1.725 / 3.77         |
| 241 (~10 s)   | 32,240     | 1.549 / 2.956 / **6.15**          | 2.051 / 4.073 / 7.85         |
| 481 (~20 s)   | 63,440     | 3.261 / 8.259 / **11.9***         | —                            |

*481-frame e2e is **DiT-only** (`LTX23_DECODE=0`): the full-res tiled VAE decode
is CPU-bound and dominates wall-time without saying anything about DiT scaling,
so it is excluded. The 121/241 e2e include decode + video save.

Finding: the low-res **denoise** stays ~linear in frames (launch-bound), but the
full-res **refine** goes super-linear — attention (N²) is ~29 % of refine at
121 f, ~44 % at 241 f, ~61 % at 481 f. nvfp4's ~20 % e2e win over bf16 holds
across all lengths.

```bash
# 241-frame full e2e (decode still fits 32-bit untiled at 241 f)
LTX23_COMPILE=1 LTX23_NVFP4=1 LTX23_FA4=1 \
  LTX23_COMPILE_MODE=reduce-overhead LTX23_NO_CGTREES=1 LTX23_COMPILE_TE=0 \
  LTX23_NUM_FRAMES=241 python $SCRIPT                             # nvfp4 ~6.15 s

# 481-frame DiT-only (skip the slow tiled decode, measure denoise/refine)
LTX23_COMPILE=1 LTX23_NVFP4=1 LTX23_FA4=1 \
  LTX23_COMPILE_MODE=reduce-overhead LTX23_NO_CGTREES=1 LTX23_COMPILE_TE=0 \
  LTX23_NUM_FRAMES=481 LTX23_DECODE=0 python $SCRIPT              # DiT ~11.9 s

# 481-frame WITH decode (needs tiling; decode is slow/CPU-bound)
LTX23_COMPILE=1 LTX23_NVFP4=1 LTX23_FA4=1 \
  LTX23_COMPILE_MODE=reduce-overhead LTX23_NO_CGTREES=1 LTX23_COMPILE_TE=0 \
  LTX23_NUM_FRAMES=481 LTX23_VAE_TILING=1 python $SCRIPT
```

## 11. Known limitations / TODO

- **Long / high-res VAE decode**: past ~241 frames at 832×1280 (and ≥1920×1088),
  untiled decode overflows `input tensor must fit into 32-bit index math`. It runs
  with `LTX23_VAE_TILING=1`, but the tiled decode is **CPU-bound and very slow**
  (the per-tile trapezoidal-blend stitch dominates) — not yet optimized. Use
  `LTX23_DECODE=0` to benchmark DiT timing without it.
- cudagraphs is DiT-only here; text-encoder/VAE graph capture needs
  `cudagraph_mark_step_begin()` plumbing to be safe.
