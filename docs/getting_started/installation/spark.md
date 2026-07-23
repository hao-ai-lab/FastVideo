# NVIDIA DGX Spark (GB10)

Instructions to install FastVideo on an **NVIDIA DGX Spark**: the GB10
Grace-Blackwell platform.

Use this guide instead of the standard [NVIDIA GPU guide](gpu.md) when your
machine is a Spark / GB10 system. Spark combines Linux **ARM64 (`aarch64`)**,
CUDA 13, and a GB10 GPU (`sm_121`), so the regular install path needs a few
changes:

- PyPI does not provide a matching `aarch64` `fastvideo-kernel` wheel for GB10,
  so FastVideo builds the in-tree kernel from source during installation.
- The build must target CUDA 13 / PyTorch `cu130`.
- The kernel build needs Python development headers; the system Python on Spark
  may not include them.
- FlashAttention, if needed, should use a matching Linux ARM64 wheel instead of
  the usual x86_64 install path.

The steps below handle those differences without requiring `sudo`.

!!! note "Container alternative"
    If you prefer Docker, use the explicit CUDA 13 Spark image tag:
    `ghcr.io/hao-ai-lab/fastvideo/fastvideo-dev:py3.12-cuda13.0.0-latest`.
    Do not use `py3.12-latest` on DGX Spark; that tag currently points to the
    CUDA 12.6 ARM64 image, which targets GH200-class hardware rather than
    GB10.

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

## Install from source

Run from the repository root:

```bash
# 0. (once) install uv and put it on PATH
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
git clone https://github.com/hao-ai-lab/FastVideo.git && cd FastVideo

# 1. Create a venv on a uv-managed CPython 3.12. This bundles the development
#    headers needed by the kernel build; the Spark's system Python may not.
uv venv .venv --python 3.12 --python-preference only-managed --seed
source .venv/bin/activate

# 2. Initialise the kernel submodules (cutlass + ThunderKittens headers).
git submodule update --init --recursive \
  fastvideo-kernel/include/cutlass fastvideo-kernel/include/tk

# 3. Install FastVideo in editable mode. On aarch64, this compiles the in-tree
#    fastvideo-kernel source for GB10 instead of using a PyPI kernel wheel.
UV_TORCH_BACKEND=cu130 uv pip install -e .
```

Contributors who want the lint/test tooling:
`UV_TORCH_BACKEND=cu130 uv pip install -e ".[dev]"`.

Then jump to [Verify the install](#verify-the-install).

## Docker image

Use the CUDA 13 image on DGX Spark:

```bash
docker run --gpus all -it \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  ghcr.io/hao-ai-lab/fastvideo/fastvideo-dev:py3.12-cuda13.0.0-latest
```

The published FastVideo Docker tags are multi-platform, but the default
`latest` / `py3.12-latest` tags select the CUDA 12.6 image. On ARM64, that
CUDA 12.6 image is built for GH200-class hardware, not GB10. Spark users should
therefore select the explicit CUDA 13 tag above.

If you build the image locally on a Spark host, target `sm_121` explicitly:

```bash
docker build --platform linux/arm64 -f docker/Dockerfile \
  --build-arg TORCH_CUDA_ARCH_LIST=12.1 \
  -t fastvideo-dev:spark .
```

## Building without a visible GPU (CI / Docker)

With no GPU visible, the kernel build cannot probe the architecture and `uv`
cannot infer the CUDA backend from the host driver. Name both explicitly:

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

Expected:

- `torch` reports a `+cu130` build.
- `torch.cuda.is_available()` is `True`.
- `device` is `NVIDIA GB10`.
- `fastvideo --help` lists commands such as `generate`, `serve`,
  `router-serve`, `bench`, and `eval`.

To confirm one of the **compiled** CUDA kernels actually runs on the GB10
(importing alone doesn't execute the `.so`):

```bash
python - <<'PY'
import torch
from fastvideo_kernel import turbodiffusion_ops

x = torch.randn((2, 16, 128), dtype=torch.bfloat16, device="cuda")
w = torch.randn(128, dtype=torch.bfloat16, device="cuda")
y = turbodiffusion_ops.rmsnorm(x, w, 1e-5)
if isinstance(y, tuple):
    y = y[0]
torch.cuda.synchronize()
print("rmsnorm ok", y.shape, y.dtype)
PY
```

!!! note "INT8 TurboDiffusion kernels"
    The Spark build includes the TurboDiffusion extension symbols, including
    `quant_cuda` and `gemm_cuda`, but the install smoke test above intentionally
    uses `rms_norm_cuda`. Treat INT8 quantization or GEMM failures on GB10 as a
    kernel compatibility issue to debug separately, not as a failed FastVideo
    Python installation.

## Optional: generate a smoke-test video

After installation is verified, you can run a tiny text-to-video generation to
exercise model download, model loading, denoising, and video export. This is a
functional smoke test, not a quality benchmark:

```bash
export FASTVIDEO_ATTENTION_BACKEND=TORCH_SDPA
export HF_HOME=$HOME/.cache/huggingface

fastvideo generate --config scripts/inference/inference_wan.yaml \
  --generator.engine.num_gpus 1 \
  --generator.engine.parallelism.tp_size 1 \
  --generator.engine.parallelism.sp_size 1 \
  --generator.engine.offload.dit true \
  --generator.engine.offload.vae true \
  --generator.engine.offload.text_encoder true \
  --request.sampling.num_frames 17 \
  --request.sampling.height 256 \
  --request.sampling.width 256 \
  --request.sampling.num_inference_steps 4 \
  --request.sampling.guidance_scale 6.0 \
  --request.output.output_path outputs_video/spark_smoke/
```

Check the generated files with:

```bash
find outputs_video/spark_smoke -type f
```

For better visual quality, increase resolution, frames, and denoising steps
from the smoke-test values above. The default `inference_wan.yaml` settings are
closer to a real quality run but take much longer.

If Hugging Face returns `401` or `404` while downloading the model, authenticate
with a token that has access to the referenced model:

```bash
huggingface-cli login
```

Avoid setting both `request.prompt` and `request.inputs.prompt_path` in the same
run; the CLI rejects ambiguous prompt sources.

## Manual build (advanced / fallback)

The one-liner above is the supported path. Build the kernel yourself only if you
are iterating on the CUDA source or the auto-build fails:

```bash
# install torch + the kernel's build deps into the active venv first
UV_TORCH_BACKEND=cu130 uv pip install torch torchvision torchaudio scikit-build-core cmake ninja setuptools wheel
# build just the kernel against that torch (auto-detects sm_121)
uv pip install ./fastvideo-kernel --no-build-isolation
# then the rest of FastVideo
uv pip install -e .
```

Or `cd fastvideo-kernel && ./build.sh`, which auto-detects the arch and does the
same compile (it initializes only the kernel's `cutlass` and `tk` submodules).

## Optional: flash-attn

Not installed by default. FastVideo falls back to other attention backends
without it. For the guide's Python 3.12, PyTorch 2.12, and CUDA 13 environment,
install the matching prebuilt Linux ARM64 wheel from the
[FlashAttention prebuilt wheel index](https://mjunya.com/flash-attention-prebuild-wheels/?python=3.12&torch=2.12&cuda=13.0&platform=Linux+arm64):

```bash
uv pip install "https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.9.22/flash_attn-2.8.3%2Bcu130torch2.12-cp312-cp312-linux_aarch64.whl"
```

## Troubleshooting

| Symptom | Cause / Fix |
|---|---|
| `Could NOT find Python (missing: ... Development.Module)` | venv built from system Python without headers. Recreate with `--python-preference only-managed` (add `--clear` to reuse the path), or `sudo apt install python3.12-dev`. |
| kernel build can't find cutlass headers | Submodules not initialised — run the `git submodule update` step. |
| `fastvideo-kernel: could not determine the target CUDA architecture` | The build couldn't see a GPU and no arch was given. Build on the Spark itself, or pass `TORCH_CUDA_ARCH_LIST=12.1` (see [Building without a visible GPU](#building-without-a-visible-gpu-ci--docker)). |
| `nvcc fatal: Unsupported gpu architecture 'compute_121'` | `nvcc` older than CUDA 12.9/13. Confirm `nvcc --version` is 13.x and `CUDACXX=/usr/local/cuda/bin/nvcc`. |
| `Segmentation fault` when running `Int8Linear` or `int8_quant` | The Python package and compiled extension may be installed correctly, but the TurboDiffusion INT8 quantization/GEMM path is not validated by this guide on GB10. Use the `rmsnorm ok` smoke test above for install verification and debug INT8 kernel support separately. |
| Hugging Face returns `401` or `404` during generation | The model is private/gated, the repo id is wrong, or the Spark machine is not authenticated. Run `huggingface-cli login` and verify the model id in the config. |
| `ninja: command not found` (manual build only) | `uv pip install scikit-build-core cmake ninja setuptools wheel`. |

## What this guide does not verify

The commands above are intended for a real DGX Spark / GB10 machine. On another
machine you can review the docs and build logic, but you cannot fully validate
the GB10 kernel compile, kernel execution, or FlashAttention ARM64 wheel without
Spark hardware.

The generation smoke test verifies a small inference path only. Training,
distillation, and model-specific optimized kernels should be validated per
workload before treating them as supported on Spark.

If you hit other issues, please open an issue on our
[GitHub repository](https://github.com/hao-ai-lab/FastVideo). You can also join
our [Slack community](https://join.slack.com/t/fastvideo/shared_invite/zt-38u6p1jqe-yDI1QJOCEnbtkLoaI5bjZQ)
for additional support.

## Development Environment Setup

If you're planning to contribute to FastVideo please see the
[Contributor Guide](../../contributing/overview.md).
