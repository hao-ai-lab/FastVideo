
# Optimizations

This page describes the various options for speeding up generation times in FastVideo.

## Table of Contents

- Optimized Attention Backends

  - [Flash Attention](#flash-attention)
  - [Sliding Tile Attention (Archived)](#sliding-tile-attention-archived)
  - [Sage Attention](#sage-attention)
  - [Sage Attention 3](#sage-attention-3)
- [Adaptive Guidance (CFG gating)](#adaptive-guidance-cfg-gating)

- [torch.compile](#torch-compile)

## Attention Backends

### Available Backends

- Torch SDPA: `FASTVIDEO_ATTENTION_BACKEND=TORCH_SDPA`
- Flash Attention 2 and 3: `FASTVIDEO_ATTENTION_BACKEND=FLASH_ATTN`
- Video Sparse Attention: `FASTVIDEO_ATTENTION_BACKEND=VIDEO_SPARSE_ATTN`
- Sage Attention: `FASTVIDEO_ATTENTION_BACKEND=SAGE_ATTN`
- Sage Attention 3: `FASTVIDEO_ATTENTION_BACKEND=SAGE_ATTN_THREE`
- Attn-QAT inference (modified SageAttention3 FP4, sm_120/RTX 5090): `FASTVIDEO_ATTENTION_BACKEND=ATTN_QAT_INFER`
- Video MoBA Attention: `FASTVIDEO_ATTENTION_BACKEND=VMOBA_ATTN`
- Sparse Linear Attention: `FASTVIDEO_ATTENTION_BACKEND=SLA_ATTN`
- SageSLA Attention: `FASTVIDEO_ATTENTION_BACKEND=SAGE_SLA_ATTN`
- Sliding Tile Attention (archived branch only):
  `FASTVIDEO_ATTENTION_BACKEND=SLIDING_TILE_ATTN`

### Configuring Backends

There are two ways to configure the attention backend in FastVideo.

#### 1. In Python

In python, set the `FASTVIDEO_ATTENTION_BACKEND` environment variable before instantiating `VideoGenerator` like this:

```python
os.environ["FASTVIDEO_ATTENTION_BACKEND"] = "VIDEO_SPARSE_ATTN"
```

#### 2. In CLI

You can also set the environment variable on the command line:

```bash
FASTVIDEO_ATTENTION_BACKEND=SAGE_ATTN python example.py
```

### Flash Attention

**`FLASH_ATTN`**

We recommend always installing [Flash Attention 2](https://github.com/Dao-AILab/flash-attention):

```bash
uv pip install flash-attn==2.7.4.post1 --no-build-isolation
```

And if using a Hopper+ GPU (ie H100), installing [Flash Attention 3](https://github.com/Dao-AILab/flash-attention?tab=readme-ov-file#flashattention-3-beta-release) by compiling it from source (takes about 10 minutes for me):

```bash
git clone https://github.com/Dao-AILab/flash-attention.git && cd flash-attention

cd hopper
uv pip install ninja
python setup.py install
```

### FP4 Flash Attention 4 (Blackwell only)

**`FLASH_ATTN`** with **`--nvfp4_fa4`**

On Blackwell GPUs (B200/B300), you can enable FP4 quantized Q/K attention for up to **1.31x kernel speedup** over BF16 FA4, peaking at **2018 TFLOPS**. This quantizes Q and K to NVFP4 E2M1 with per-block E4M3 scale factors while keeping V in BF16 or FP8.

See the [Attn-QAT paper](https://arxiv.org/abs/2603.00040) and [flash-attention-fp4 benchmark results](https://github.com/hao-ai-lab/flash-attention-fp4/blob/fp4/flash_attn/cute/README.md) for details.

#### Requirements

- **GPU**: NVIDIA Blackwell (sm100a or sm103a) — B200, B300, GB200, GB300
- **CUDA**: 12.8+
- **Python**: 3.10 or 3.11

#### Installation

Install the FP4 flash attention kernel (without upgrading your existing torch):

```bash
pip install --no-deps "git+ssh://git@github.com/hao-ai-lab/flash-attention-fp4.git@fp4#subdirectory=flash_attn/cute"
pip install "nvidia-cutlass-dsl>=4.4.2" apache-tvm-ffi flashinfer-python
```

The `--no-deps` flag prevents upgrading torch/torchvision. The kernel requires torch >= 2.4 with CUDA 12.8+ support (already present in FastVideo's environment).

#### Usage

Enable FP4 attention via the `--nvfp4_fa4` flag:

```bash
python examples/inference/optimizations/fp4_attn_wan2_1_1_3b.py --nvfp4_fa4
```

Or in Python via the `nvfp4_fa4` kwarg (sets env vars automatically):

```python
from fastvideo import VideoGenerator
gen = VideoGenerator.from_pretrained(
    "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    nvfp4_fa4=True,
    num_gpus=1,
    use_fsdp_inference=False,  # FSDP is incompatible with FP4 pointer path
)
gen.generate_video(prompt="A raccoon in sunflowers", save_video=True)
```

#### Known Limitations

- `use_fsdp_inference=True` is incompatible with the FP4 path (FSDP shards invalidate tensor pointers)
- Per-call cosine similarity vs BF16: ~0.99 (slight quantization error accumulates over denoising steps)
- Only supports `headdim >= 128`

### NVFP4 + Attn-QAT (modified SageAttention3, Blackwell sm_120)

**`ATTN_QAT_INFER`** with **`transformer_quant="NVFP4"`**

Runs the DiT fully in 4-bit: NVFP4 linear layers (activations quantized on the
fly) plus the modified SageAttention3 FP4 attention backend. This is the
inference half of the Quantization-Aware Distillation (QAD) recipe and the path
used for the RTX 5090 release.

The `attn_qat_infer` kernel hard-gates on **sm_120 (consumer Blackwell / RTX
5090)**; on other GPUs the backend logs a notice and falls back to Flash
Attention. See the [Attn-QAT paper](https://arxiv.org/abs/2603.00040).

Enable both halves — attention via the env var, linear via `transformer_quant`:

```python
import os
os.environ["FASTVIDEO_ATTENTION_BACKEND"] = "ATTN_QAT_INFER"

from fastvideo import VideoGenerator
gen = VideoGenerator.from_pretrained(
    "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    num_gpus=1,
    transformer_quant="NVFP4",   # or "nvfp4_qat" to match a QAT-distilled checkpoint
    use_fsdp_inference=False,     # FSDP shards invalidate the FP4 tensor pointers
)
gen.generate(request={"prompt": "A raccoon in sunflowers", "output": {"save_video": True}})
```

Or run the example script:

```bash
python examples/inference/optimizations/nvfp4_qat_wan2_1_1_3b.py
python examples/inference/optimizations/nvfp4_qat_wan2_1_1_3b.py --bf16  # baseline
```

### Sliding Tile Attention (Archived)

**`SLIDING_TILE_ATTN`**

The full STA integration in `fastvideo/` is archived from `main` and preserved
at:

- https://github.com/hao-ai-lab/FastVideo/tree/sta_do_not_delete

We keep STA off `main` because we believe VSA is strictly better than STA for
the actively maintained FastVideo path.

Kernel code in `fastvideo-kernel` is still retained. For mask search and STA
inference workflow, see [STA docs](../attention/sta/index.md).

### Video Sparse Attention

**`VIDEO_SPARSE_ATTN`**

Video Sparse Attention is provided by `fastvideo-kernel`.
See [VSA docs](../attention/vsa/index.md) for installation details.

### Sage Attention

**`SAGE_ATTN`**

To use [SageAttention](https://github.com/thu-ml/SageAttention) 2.1.1, please compile from source:

```bash
git clone https://github.com/thu-ml/SageAttention.git
cd sageattention
python setup.py install  # or uv pip install -e .
```

### Sage Attention 3

**`SAGE_ATTN_THREE`**

[SageAttention 3](https://github.com/thu-ml/SageAttention/tree/main/sageattention3_blackwell) is an advanced attention mechanism that leverages FP4 quantization and Blackwell GPU Tensor Cores for significant performance improvements.

#### Hardware Requirements

- RTX5090

#### Installation

Note that Sage Attention 3 requires `python>=3.13`, `torch>=2.8.0`, `CUDA >=12.8`. If you are using `uv` and using `torch==2.8.0` make sure that `sentencepiece==0.2.1` in the pyproject.toml file.

To use Sage Attention 3 in FastVideo, follow the `README.md` in the linked repository to install the package from source.

### V-MoBA / SLA / SageSLA

These backends are model-specific and require the corresponding kernels and
dependencies. Use the support matrix and model examples to confirm compatibility
before enabling them.

<a id="torch-compile"></a>

## torch.compile

FastVideo can `torch.compile` the DiT (transformer) for a substantial
end-to-end speedup. It is **off by default** and enabled per-run.

### Enabling

```python
generator = VideoGenerator.from_pretrained(
    "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    enable_torch_compile=True,
)
```

A complete A/B example (eager vs compiled, warmup excluded) is in
[`examples/inference/optimizations/torch_compile_example.py`](https://github.com/hao-ai-lab/FastVideo/blob/main/examples/inference/optimizations/torch_compile_example.py).

`fastvideo generate` is config-file driven; to enable `torch.compile`
from the CLI, set the relevant field in your run config and pass it via
`fastvideo generate --config run.yaml`. There is no top-level
`--enable-torch-compile` flag on the subcommand.

Only DiT submodules that declare `_compile_conditions` are compiled
(most shipped models). The text encoder and VAE are not compiled by this
flag.

### What to expect

| Config | Effect |
|---|---|
| Wan2.1-T2V-1.3B, A100-80GB, 480×832×81f, 50 steps | end-to-end **259.7s → 198.1s (−23.7%)**; per-step **4.91 → 3.78 s/it** |

The speedup is **configuration-dependent** — it varies with model,
resolution, step count, and GPU. Treat the number above as one measured
data point, not a guarantee; benchmark your own config (recipe below).

There is a **one-time graph-build cost** on the first generation (tens of
seconds to minutes, model-dependent). It amortizes over subsequent
generations with the same input shapes. Always exclude the first
(warmup) generation when measuring steady-state latency — measuring the
warmup is the most common way to wrongly conclude "compile is slower".

**Numerics.** Inductor's lowering is designed to preserve eager
semantics within floating-point tolerance, but per-model equivalence is
not asserted by any standing SSIM regression here — the SSIM tests in
[`fastvideo/tests/ssim/`](https://github.com/hao-ai-lab/FastVideo/tree/main/fastvideo/tests/ssim)
run with `enable_torch_compile` disabled. If you depend on compile
output staying close to eager (or your previous compiled run), run an
MS-SSIM gate on *your* config, especially when combining
`enable_torch_compile=True` with other numerics-affecting flags
(quantized attention backends, FP4, layerwise offload edge cases).

### Known interactions

- **Layerwise CPU offload** (`dit_layerwise_offload=True`, the default):
  the offload hook previously caused an implicit graph break once per
  transformer layer, fragmenting the compiled region. Addressed in
  hao-ai-lab/FastVideo#1365 — keep that fix to get a clean compiled
  region under the default offload path.
- **`mode="reduce-overhead"` / CUDA graphs**: not yet supported
  end-to-end. The attention dispatch is an untraceable custom op and
  still breaks the graph, which CUDA-graph trees cannot span. Use the
  default inductor mode (shown above) until that is resolved.

Extra `torch.compile` options are passed through `torch_compile_kwargs`
(a dict), accepted by `VideoGenerator.from_pretrained(...)` and by the
CLI as a JSON string via `--torch-compile-kwargs`. Example (currently
**not** recommended — see the CUDA-graphs caveat above):

```python
VideoGenerator.from_pretrained(
    "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    enable_torch_compile=True,
    torch_compile_kwargs={"mode": "reduce-overhead"},  # may error today
)
```

### Benchmarking torch.compile

Same discipline as attention backends — same prompt, same seed, same
config; **discard the first generation** (graph build):

```python
import time
from fastvideo import VideoGenerator

gen = VideoGenerator.from_pretrained("your-model-id", enable_torch_compile=True)
req = {"prompt": "Your prompt", "sampling": {"seed": 1024},
       "output": {"save_video": False}}
gen.generate(req)                                            # warmup: graph build, discard
t0 = time.perf_counter()
gen.generate(req)                                            # measured: shapes reused
print(f"compiled steady-state: {time.perf_counter() - t0:.2f}s")
```

See `examples/inference/optimizations/torch_compile_example.py` for a
baseline-vs-compile A/B with the warmup correctly excluded.

## Benchmarking different optimizations

To benchmark backend performance, generate the same prompt with the same seed and compare end-to-end generation times:

```python
import os
import time

for backend in ["TORCH_SDPA", "FLASH_ATTN", "SAGE_ATTN"]:
    os.environ["FASTVIDEO_ATTENTION_BACKEND"] = backend
    generator = VideoGenerator.from_pretrained("your-model-id")
    start_time = time.perf_counter()
    generator.generate_video(
        prompt="Your prompt",
        seed=1024,
    )
    elapsed = time.perf_counter() - start_time
    print(f"{backend}: {elapsed:.2f}s")
```

Note: reinstantiate `VideoGenerator` after changing `FASTVIDEO_ATTENTION_BACKEND`.

## Adaptive Guidance (CFG gating)

CFG gating accelerates classifier-free guidance by reusing the cached
`noise_pred_cond - noise_pred_uncond` delta after a configurable fraction of
the denoising schedule, skipping the unconditional model forward for the
remaining steps. The technique is the LinearAG variant of Adaptive Guidance
(Castillo et al. 2023, [arXiv:2312.12487](https://arxiv.org/abs/2312.12487)).

### Enabling

Set the `FASTVIDEO_CFG_GATE_STEP` environment variable to a float in `[0, 1]`:

| Value | Behavior |
|-------|----------|
| `1.0` (default) | Disabled — legacy two-pass CFG every step. |
| `0.5` | Cache the delta after `len(timesteps) * 0.5` steps; reuse for the rest. |
| `0.0` | Cache from the very first step (most aggressive). |

```bash
export FASTVIDEO_CFG_GATE_STEP=0.5
```

### Trade-offs

- **Memory**: one extra model-output-sized tensor per rank held during the
  gating window.
- **Quality**: VBench-measured quality is preserved within noise on 4 of 5
  dimensions at `FASTVIDEO_CFG_GATE_STEP=0.5` for Wan T2V 1.3B per the PR's
  reported numbers (see [#1372](https://github.com/hao-ai-lab/FastVideo/pull/1372)).
- **Speed**: ~22% e2e on 4xL40S and ~24% on 1xH100 at the same settings.

Default behavior is byte-for-byte equivalent to the legacy two-pass CFG path;
the feature is fully opt-in.
