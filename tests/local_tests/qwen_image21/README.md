# Qwen-Image-2.1 validation

Use the package tests for regression coverage and the opt-in local tests to
compare FastVideo with pinned upstream Diffusers. For generation commands and
memory measurements, see the [cookbook](../../../docs/cookbook/qwen-image21.md).

## Confirmed coverage

Validation on NVIDIA GB200 at `c3d59459` completed on 2026-10-23: **36 cases
passed, with no failures or skips**.

| Check | Cases | Result |
| --- | ---: | --- |
| Package attention and stage tests | 9 | Backend selection, masks, caches and pipeline stages pass |
| FA4 transformer-block golden | 1 | Zero drift against the reviewed device/runtime reference |
| FA4 T2I and editing image regressions | 2 | SSIM 1.0; unchanged threshold 0.98 |
| Encoder, DiT and VAE component parity | 12 | All component cases pass |
| SDPA upstream pipeline parity | 12 | All modes below pass at 40 steps |

The pipeline matrix covers T2I, plain I2I, editing, reference-to-image,
transparent RGBA generation and editing, annotations, mask images, true CFG,
uncached execution and ten references at 1024 x 1024. A separate T2I case covers
2048 x 2048. Conditioning, initial/final latents, all 40 noise predictions and
latent updates, floating-point RGBA output and quantized PNG pixels were exact.
Raw FP32 scheduler arrays have maximum differences of `1.1920928955078125e-7`
for sigmas and `0.0001220703125` for timesteps; they are not bitwise equal.

Upstream comparisons explicitly use SDPA with matching cuDNN SDPA settings.
FA4 is validated separately against its golden and reviewed image references;
FA4 and SDPA outputs are not expected to be bitwise equal. The automatic
attention selector was also observed selecting FA4 for all 64 blocks across the
two image regressions with `FASTVIDEO_FA4=1`.

These results establish the tested GB200 configurations. They do not establish
other GPU/backend combinations, 2K with ten references, or hard preservation of
pixels outside a semantic mask.

## Package regression tests

Install FastVideo using the [CUDA installation guide](../../../docs/getting_started/installation/gpu.md)
and run commands from the repository root on an allocated CUDA GPU:

```bash
python -m pytest fastvideo/tests/transformers/test_qwen_image21_attention.py \
  fastvideo/tests/stages/test_qwen_image21_stages.py -xq

FASTVIDEO_ATTENTION_BACKEND=FLASH_ATTN FASTVIDEO_FA4=1 python -m pytest \
  fastvideo/tests/golden_gate/test_qwen_image21.py -xq

FASTVIDEO_ATTENTION_BACKEND=FLASH_ATTN FASTVIDEO_FA4=1 python -m pytest \
  fastvideo/tests/ssim/test_qwen_image21_similarity.py -xq
```

These files cover attention, pipeline stages, transformer blocks and generated
images through the existing package CI lanes. The golden loads one released
BF16 transformer block. The image tests generate T2I and editing outputs at
512 x 512, four steps and seed 21, comparing all four RGBA channels.

The shared harnesses download missing references from
[`FastVideo/ssim-reference-videos`](https://huggingface.co/datasets/FastVideo/ssim-reference-videos).
Qwen references are available for the GB200 FA4 golden and default image tier.
The optional `--ssim-full-quality` tier uses 1024 x 1024 and 40 steps and needs
its own reviewed references; those are not covered by the published default
tier. See the [SSIM guide](../../../fastvideo/tests/ssim/README.md) for reference
management.

## Prepare upstream parity

Local parity tests under `tests/local_tests/` are opt-in and run outside CI.
Use BF16 on a CUDA GPU, a complete Qwen checkpoint and a clean checkout of the
pinned upstream revision. The pipeline harness uses CPU offload, VAE tiling
and CPU prefix KV storage; its pytest duration is not a generation benchmark.
Use the cookbook's measured memory profiles to plan resources.

```bash
git clone https://github.com/huggingface/diffusers.git ../diffusers-qwen-image21
git -C ../diffusers-qwen-image21 checkout 578c9b2c6636ab2424a0e56186268b83623656b2
hf download Qwen/Qwen-Image-2.1 \
  --revision d26bb61231c349cf6b7896fa83353113880e1ba3 \
  --local-dir ../Qwen-Image-2.1

export QWEN_IMAGE21_DIFFUSERS_DIR="$PWD/../diffusers-qwen-image21"
export QWEN_IMAGE21_MODEL_ROOT="$PWD/../Qwen-Image-2.1"
export PYTHONPATH="$QWEN_IMAGE21_DIFFUSERS_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export FASTVIDEO_ATTENTION_BACKEND=TORCH_SDPA
```

An existing checkpoint can be used by changing `QWEN_IMAGE21_MODEL_ROOT`. It
must include `model_index.json`, `processor`, `text_encoder`, `transformer`,
`vae`, `scheduler` and all weight shards. No checkpoint conversion is needed.
Explicitly enabled real-weight tests fail if prerequisites are missing.

### Component parity

```bash
QWEN_IMAGE21_RUN_ENCODER_PARITY=1 python -m pytest \
  tests/local_tests/encoders/test_qwen_image21_conditioner_parity.py -xq

QWEN_IMAGE21_RUN_DIT_PARITY=1 python -m pytest \
  tests/local_tests/transformers/test_qwen_image21_transformer_parity.py -xq

QWEN_IMAGE21_RUN_VAE_PARITY=1 python -m pytest \
  tests/local_tests/vaes/test_qwen_image21_vae_parity.py -xq
```

These checks cover text and 1/2/10-image conditioning, every DiT block and
cached decoding, and VAE posterior/decode behavior including tiling. Optional
`QWEN_IMAGE21_ENCODER_PARITY_REPORT` and `QWEN_IMAGE21_DIT_PARITY_REPORT` paths
save component diagnostics as JSON.

### Pipeline parity

Run all eleven 1024-pixel modes and the separate 2K T2I case at 40 steps:

```bash
QWEN_IMAGE21_RUN_PIPELINE_PARITY=1 QWEN_IMAGE21_RUN_2K_PARITY=1 \
  QWEN_IMAGE21_PARITY_STEPS=40 \
  python -m pytest tests/local_tests/qwen_image21/test_pipeline_parity.py -xq \
  --basetemp outputs/qwen_image21/parity-40step
```

Choose a fresh `--basetemp` path for each run: pytest replaces an existing
directory at that path. For one mode, select its node, for example
`tests/local_tests/qwen_image21/test_pipeline_parity.py::test_qwen_image21_modes_1024_pipeline_parity[edit]`.
Without `QWEN_IMAGE21_PARITY_STEPS`, the harness defaults to four steps.

Each case runs native and upstream pipelines sequentially in separate
processes, with identical BF16 initial latents, sigmas and cache settings.
Synthetic reference images are generated by default.
`QWEN_IMAGE21_PARITY_CASES_JSON` can supply per-case prompts and ordered image
paths; relative paths resolve beside that JSON file. For parallel invocations,
assign distinct `CUDA_VISIBLE_DEVICES`, `MASTER_PORT` and artifact paths.

The tests assert RGBA output parity within the existing BF16 tolerance and
verify PNG alpha export. They also write final-output error metrics. The stronger
exact-trajectory results reported above came from separate diagnostic runs;
a passing pixel assertion alone does not establish bitwise equality at every
stage.
