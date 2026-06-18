# NVIDIA DGX Spark (GB10)

Instructions to install FastVideo on an **NVIDIA DGX Spark** — the GB10
Grace-Blackwell platform.

The Spark is **ARM64 (`aarch64`) with CUDA 13**, which the standard
[NVIDIA GPU guide](gpu.md) does not cover: there is no prebuilt `aarch64` wheel
for `fastvideo-kernel`, so it must be compiled from source, and the system
Python typically lacks the development headers that build needs. The steps below
handle both, **without requiring `sudo`**.

## Requirements

- **OS: Linux (`aarch64`)**
- **GPU: NVIDIA GB10** (compute capability `sm_121`), at least 1
- **CUDA Toolkit: 13.x** at `/usr/local/cuda` (`nvcc` on `PATH`)
- **Python: 3.10–3.12**
- **Compilers:** gcc/g++ 10+ (the Spark ships gcc 13)

!!! note "Different compute capability?"
    These instructions target the GB10's `sm_121`. If your device reports
    something else, substitute it wherever you see `121` / `12.1` below. Detect it
    after installing torch with:
    ```bash
    python -c "import torch; print(torch.cuda.get_device_capability(0))"  # GB10 -> (12, 1)
    ```

## Quick install (copy-paste)

Run from the repository root:

```bash
# 0. Install uv (skip if present) and ensure it is on PATH
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# 1. Clone FastVideo
git clone https://github.com/hao-ai-lab/FastVideo.git && cd FastVideo

# 2. Create a venv on a uv-MANAGED CPython 3.12 (bundles the dev headers the
#    kernel build needs; avoids `sudo apt install python3.12-dev`)
uv venv .venv --python 3.12 --python-preference only-managed --seed
source .venv/bin/activate

# 3. Initialise the kernel's git submodules (cutlass + ThunderKittens headers)
git submodule update --init --recursive \
  fastvideo-kernel/include/cutlass fastvideo-kernel/include/tk

# 4. Install PyTorch FIRST. On aarch64 this resolves to torch 2.11.0+cu130
#    (CUDA 13), which matches the system nvcc.
uv pip install "torch==2.11.0" torchvision torchaudio --prerelease=allow

# 5. Compile fastvideo-kernel from source for the GB10 (sm_121)
uv pip install scikit-build-core cmake ninja setuptools wheel
export CUDA_HOME=/usr/local/cuda
export CUDACXX=/usr/local/cuda/bin/nvcc
export TORCH_CUDA_ARCH_LIST=12.1
export CMAKE_ARGS="-DCMAKE_CUDA_ARCHITECTURES=121 -DFASTVIDEO_KERNEL_BUILD_TK=OFF -DGPU_BACKEND=CUDA"
uv pip install ./fastvideo-kernel -v --no-build-isolation

# 6. Install the rest of FastVideo (editable)
uv pip install -e . --no-sources --prerelease=allow
```

Then jump to [Verify the install](#verify-the-install).

## What each step does (and why)

These notes explain the non-obvious choices in the **Quick install** block above
(same step numbers); they don't repeat the commands.

**Step 2 — use a managed Python.** The kernel is a C++/CUDA extension whose CMake
build calls `find_package(Python ... Development.Module)`, which needs
`Python.h`. The system `python3.12` on the Spark has **no dev headers**, so a venv
seeded from it fails with
`Could NOT find Python (missing: Interpreter Development.Module)`. A **uv-managed**
CPython (`--python-preference only-managed`) ships the headers, so the build works
with no `sudo`. Re-running on an existing venv? Add `--clear`. The system
alternative is `sudo apt install python3.12-dev`.

**Step 3 — the kernel submodules.** `fastvideo-kernel` vendors **cutlass** and
**ThunderKittens** as submodules under `fastvideo-kernel/include/`. A fresh clone
leaves them empty and the GEMM kernel won't compile without the cutlass headers.

**Step 4 — PyTorch first, and CUDA 13 not 12.8.** On `aarch64`, PyPI
`torch==2.11.0` resolves to **`2.11.0+cu130`** (a CUDA 13 build). This is
intentionally different from FastVideo's `pyproject.toml`, which pins the
**cu128** (CUDA 12.8) index: the Spark only has the CUDA 13 toolkit, so pairing
**cu130 torch with `nvcc` 13** gives a matched toolchain for the kernel compile.
PyTorch must be installed **before** the kernel, which links against it. Confirm
it sees the GPU before continuing:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# -> 2.11.0+cu130 True NVIDIA GB10
```

**Step 5 — compiling the kernel for the GB10.** The env vars are what make it
work:

- **`--no-build-isolation`** makes the build link against the cu130 torch you
  just installed and detect the real GPU arch. With isolation, uv would pull a
  throwaway CPU-only torch into the build env, detect the wrong GPU arch, and link the
  wrong `libtorch`.
- **`CMAKE_CUDA_ARCHITECTURES=121` / `TORCH_CUDA_ARCH_LIST=12.1`** target the
  GB10's `sm_121` natively.
- **`FASTVIDEO_KERNEL_BUILD_TK=OFF`**: ThunderKittens kernels are Hopper-only
  (`sm_90a`); on Blackwell they're disabled and FastVideo uses Triton fallbacks
  at runtime.

A successful build ends with `Built fastvideo_kernel-0.2.6-cp312-cp312-linux_aarch64.whl`.

> **Alternative:** `cd fastvideo-kernel && ./build.sh` auto-detects the arch and
> does the same thing, but it runs a global `git submodule update --init
> --recursive` (also clones `vbench`). The explicit commands above avoid that.

**Step 6 — `--no-sources` is required.** FastVideo's `pyproject.toml` pins torch
to the **cu128** index via `[tool.uv.sources]`. Without `--no-sources`, this step
would reinstall torch as `2.11.0+cu128`, replacing your cu130 torch and
**breaking the kernel's ABI** (it was compiled against cu130). `--no-sources`
makes uv ignore that table, keep the installed cu130 torch, and only add the
remaining pure-Python dependencies.

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

Expected: `torch 2.11.0+cu130`, `device: NVIDIA GB10`, and the CLI listing
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

## Optional: flash-attn

Not installed by default — there is no prebuilt `aarch64` wheel, so it is a heavy
from-source build, and FastVideo falls back to other attention backends without
it. If you want it (uses the same `CUDA_HOME` / `CUDACXX` env as the kernel step):

```bash
uv pip install flash-attn==2.8.1 --no-cache-dir --no-build-isolation -v
```

## Troubleshooting

| Symptom | Cause / Fix |
|---|---|
| `Could NOT find Python (missing: ... Development.Module)` | venv built from system Python without headers. Recreate with `--python-preference only-managed`, or `sudo apt install python3.12-dev`. |
| kernel build can't find cutlass headers | Submodules not initialised — run Step 3. |
| `ninja: command not found` / no build backend | `uv pip install scikit-build-core cmake ninja setuptools wheel`. |
| `import fastvideo_kernel` → `ModuleNotFoundError: einops` | Pure-Python deps not installed — run Step 6. |
| torch got reinstalled as `+cu128`, kernel now fails to import | You ran `uv pip install -e .` **without** `--no-sources`. Reinstall cu130 torch, rebuild the kernel, then redo Step 6 **with** `--no-sources`. |
| `nvcc fatal: Unsupported gpu architecture 'compute_121'` | `nvcc` older than CUDA 12.9/13. Confirm `nvcc --version` is 13.x and `CUDACXX=/usr/local/cuda/bin/nvcc`. |

If you hit other issues, please open an issue on our
[GitHub repository](https://github.com/hao-ai-lab/FastVideo). You can also join
our [Slack community](https://join.slack.com/t/fastvideo/shared_invite/zt-38u6p1jqe-yDI1QJOCEnbtkLoaI5bjZQ)
for additional support.

## Development Environment Setup

If you're planning to contribute to FastVideo please see the
[Contributor Guide](../../contributing/overview.md).
