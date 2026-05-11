# FLUX.1-dev Local Tests

## Prerequisites

- CUDA GPU
- `official_weights/FLUX.1-dev` (Diffusers layout) or set `FLUX_DEV_ROOT`
- `FASTVIDEO_ATTENTION_BACKEND=TORCH_SDPA` for parity tests

Download weights:
```bash
huggingface-cli download black-forest-labs/FLUX.1-dev --local-dir official_weights/FLUX.1-dev
```

## Component Loader Smoke Test

Verifies CLIP/T5 encoders, tokenizers, VAE, and FlowMatch scheduler all load
from the Diffusers checkpoint layout.

```bash
pytest tests/local_tests/flux/test_flux_dev_component_loaders.py -vs
```

Status: requires weights — not run in CI.

## Pipeline Smoke Test

Short denoise + decode (2 steps, 256×256) end-to-end.

```bash
FASTVIDEO_ATTENTION_BACKEND=TORCH_SDPA \
pytest tests/local_tests/pipelines/test_flux_dev_pipeline_smoke.py -vs
```

Status: requires weights — not run in CI.

## Pipeline Parity Test

Compares FastVideo FluxPipeline decoded image output against Diffusers
FluxPipeline under identical prompt, seed, and inference parameters (4
steps, 256×256, seed=42). Tolerance: atol=5e-2.

```bash
FASTVIDEO_ATTENTION_BACKEND=TORCH_SDPA \
pytest tests/local_tests/pipelines/test_flux_dev_pipeline_parity.py -vs
```

Status: requires weights — pending local run (see PORT_STATUS.md).

## DiT Parity Test

Single-forward comparison of FastVideo `FluxTransformer2DModel` against
Diffusers `FluxTransformer2DModel` under identical random inputs.

```bash
FASTVIDEO_ATTENTION_BACKEND=TORCH_SDPA \
pytest fastvideo/tests/transformers/test_flux.py -vs
```

Status: requires weights — not run in CI.
Pass evidence: pending (see PORT_STATUS.md).

## SSIM Regression Test

Full-quality image generation compared against per-device reference images.
Reference images must be seeded first via `reference_videos_cli.py`.

```bash
FASTVIDEO_ATTENTION_BACKEND=TORCH_SDPA \
FLUX_T2I_MODEL_DIR=official_weights/FLUX.1-dev \
pytest fastvideo/tests/ssim/test_flux_t2i_similarity.py -vs
```

Status: reference images not yet committed — see PORT_STATUS.md.
