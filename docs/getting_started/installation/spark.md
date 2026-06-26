# NVIDIA DGX Spark (GB10)

Instructions to install FastVideo on an **NVIDIA DGX Spark** — the GB10
Grace-Blackwell platform.

The Spark is **ARM64 (`aarch64`) with CUDA 13**, which the standard
[NVIDIA GPU guide](gpu.md) does not cover: there is no prebuilt `aarch64` wheel
for `fastvideo-kernel`, so it is **compiled from source** as part of the
install, and the system Python typically lacks the development headers that
build needs. The steps below handle both, **without requiring `sudo`**.

## Requirements

- **OS: Linux (`aarch64`)**
- **GPU: NVIDIA GB10** (compute capability `sm_121`), at least 1
- **CUDA Toolkit: 13.x** at `/usr/local/cuda` (`nvcc` on `PATH`)
- **Python: 3.10–3.12**
- **Compilers:** gcc/g++ 10+ (the Spark ships gcc 13)

!!! note "Different compute capability?"
    These instructions target the GB10's `sm_121`. The kernel build auto-detects
    it from your GPU, so there is normally nothing to set. To check your device:
    ```bash
    python -c "import torch; print(torch.cuda.get_device_capability(0))"  # GB10 -> (12, 1)
    ```

## Install

Run from the repository root:

```bash
# 0. (once) install uv and put it on PATH
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
git clone https://github.com/hao-ai-lab/FastVideo.git && cd FastVideo

# 1. Create a venv on a uv-managed CPython 3.12 (bundles the dev headers the
#    kernel build needs; the Spark's system python3.12 lacks them).
uv venv .venv --python 3.12 --python-preference only-managed --seed
source .venv/bin/activate

# 2. Initialise the kernel submodules (cutlass + ThunderKittens headers).
git submodule update --init --recursive \
  fastvideo-kernel/include/cutlass fastvideo-kernel/include/tk

# 3. Install FastVideo (editable; compiles the in-tree kernel for the GB10).
#    UV_TORCH_BACKEND=auto selects the CUDA-13 (cu130) torch for this box.
UV_TORCH_BACKEND=auto uv pip install -e .
```

Contributors who want the lint/test tooling:
`UV_TORCH_BACKEND=auto uv pip install -e ".[dev]"`.

Then jump to [Verify the install](#verify-the-install).

## Building without a visible GPU (CI / Docker)

With no GPU visible the kernel build can't probe the arch and `auto` can't detect
the driver — name both explicitly:

```bash
UV_TORCH_BACKEND=cu130 TORCH_CUDA_ARCH_LIST=12.1 uv pip install -e .
```

## Verify the install

```bash
python - <<'PY'
import torch, fastvideo
print("fastvideo:", fastvideo.__version__)
print("torch    :", torch.__version__, "| cuda:", torch.version.cuda, "| avail:", torch.cuda.is_available())
print("device   :", torch.cuda.get_device_name(0))
PY

fastvideo --help
```

Expected: a `+cu130` torch, `device: NVIDIA GB10`, and the CLI listing
`generate / serve / router-serve / bench / eval`.

To confirm the **compiled** CUDA kernel actually runs on the GB10 (importing
alone doesn't execute the `.so`):

```bash
python - <<'PY'
import torch
from fastvideo_kernel import Int8Linear
lin  = torch.nn.Linear(512, 256, bias=False).cuda().to(torch.bfloat16)
qlin = Int8Linear.from_linear(lin, quantize=True)        # compiled quant_cuda
x    = torch.randn(128, 512, device="cuda", dtype=torch.bfloat16)
y, ref = qlin(x), lin(x)                                  # compiled gemm_cuda
rel = (y.float() - ref.float()).norm() / ref.float().norm()
print(f"int8 GEMM rel err vs fp32: {rel.item():.4f}  (~0.01 is correct; int8 is lossy)")
PY
```

## Manual build (advanced / fallback)

The one-liner above is the supported path. Build the kernel yourself only if you
are iterating on the CUDA source or the auto-build fails:

```bash
# install torch + the kernel's build deps into the active venv first
UV_TORCH_BACKEND=auto uv pip install torch torchvision torchaudio scikit-build-core cmake ninja setuptools wheel
# build just the kernel against that torch (auto-detects sm_121)
uv pip install ./fastvideo-kernel --no-build-isolation
# then the rest of FastVideo
uv pip install -e .
```

Or `cd fastvideo-kernel && ./build.sh`, which auto-detects the arch and does the
same compile (it runs a global `git submodule update --init --recursive`, which
also clones `vbench`).

## Optional: flash-attn

Not installed by default — there is no prebuilt `aarch64` wheel, so it is a heavy
from-source build, and FastVideo falls back to other attention backends without
it. If you want it:

```bash
export CUDA_HOME=/usr/local/cuda
export CUDACXX=/usr/local/cuda/bin/nvcc
uv pip install flash-attn==2.8.1 --no-cache-dir --no-build-isolation -v
```

## Troubleshooting

| Symptom | Cause / Fix |
|---|---|
| `Could NOT find Python (missing: ... Development.Module)` | venv built from system Python without headers. Recreate with `--python-preference only-managed` (add `--clear` to reuse the path), or `sudo apt install python3.12-dev`. |
| kernel build can't find cutlass headers | Submodules not initialised — run the `git submodule update` step. |
| `fastvideo-kernel: could not determine the target CUDA architecture` | The build couldn't see a GPU and no arch was given. Build on the Spark itself, or pass `TORCH_CUDA_ARCH_LIST=12.1` (see [Building without a visible GPU](#building-without-a-visible-gpu-ci--docker)). |
| `nvcc fatal: Unsupported gpu architecture 'compute_121'` | `nvcc` older than CUDA 12.9/13. Confirm `nvcc --version` is 13.x and `CUDACXX=/usr/local/cuda/bin/nvcc`. |
| `ninja: command not found` (manual build only) | `uv pip install scikit-build-core cmake ninja setuptools wheel`. |

If you hit other issues, please open an issue on our
[GitHub repository](https://github.com/hao-ai-lab/FastVideo). You can also join
our [Slack community](https://join.slack.com/t/fastvideo/shared_invite/zt-38u6p1jqe-yDI1QJOCEnbtkLoaI5bjZQ)
for additional support.

## Development Environment Setup

If you're planning to contribute to FastVideo please see the
[Contributor Guide](../../contributing/overview.md).
