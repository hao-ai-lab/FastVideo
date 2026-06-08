# Hunyuan GameCraft Local Tests

Local-only parity tests for the `gamecraft` FastVideo port. They compare
FastVideo against the official `Hunyuan-GameCraft-1.0` implementation and are
skipped in CI; run locally on a single GPU (~40 GB+ memory).

## Reference Assets

| Field | Value |
|---|---|
| Model family | `gamecraft` |
| Workload types | `T2V`, `I2V` |
| Official reference | `Hunyuan-GameCraft-1.0` (local clone, see test files) |
| Local reference dir | `Hunyuan-GameCraft-1.0/` (env: `GAMECRAFT_OFFICIAL_PATH`) |
| Official commit/version | `<TODO>` |
| HF weights | `<TODO>` |
| HF revision | `<TODO>` |
| Local weights dir | `Hunyuan-GameCraft-1.0/weights/` (env: `GAMECRAFT_WEIGHTS_PATH`) |
| Source layout | `<TODO>` |
| Needs conversion | `<TODO>` |

> Use only the env-var **name** for tokens (e.g., `HF_TOKEN`). Never paste a token value.

## Shared Environment Setup

Run from the FastVideo repo root in the same env used for FastVideo. See the
`add-model-prep` skill for canonical clone + install commands.

```bash
# Clone the official reference under the FastVideo repo root.
git clone <official_repo_url> Hunyuan-GameCraft-1.0
uv pip install --no-deps -e ./Hunyuan-GameCraft-1.0
```

Do not change core dependency versions (`torch`, `diffusers`, `transformers`,
`flash-attn`, `triton`, CUDA packages) without explicit approval.

## Official Environment Status

```text
dependency_changes: <TODO>
official_env_status: <TODO: imports_ok | private_deps_need_stubs | blocked>
private_dep_stubs: <TODO>
blocked_on: <TODO>
```

## Tests in this directory

Run the whole family:

```bash
DISABLE_SP=1 pytest tests/local_tests/gamecraft/ -v -s
```

| Component | Test | Concerns | Status |
|---|---|---|---|
| `text encoders` (LLaVA-LLaMA-3-8B + CLIP ViT-L/14) | [`test_gamecraft_encoders_parity.py`](./test_gamecraft_encoders_parity.py) | `<TODO>` | `<TODO>` |
| `transformer/DiT` | [`test_gamecraft_parity.py`](./test_gamecraft_parity.py) | `<TODO>` | `<TODO>` |
| `vae` (`AutoencoderKLCausal3D`) | [`test_gamecraft_vae_parity.py`](./test_gamecraft_vae_parity.py) | `<TODO>` | `<TODO>` |
| `pipeline` | [`test_gamecraft_pipeline_parity.py`](./test_gamecraft_pipeline_parity.py) | text encoding -> denoising -> VAE decode | `<TODO>` |

## Review Notes

- Required before handoff: non-skip PASS for each component parity test,
  including reused components that own weights or numerical behavior.
- Pipeline parity may start as a scaffold; final handoff requires non-skip
  PASS or an explicit blocker accepted via the escape-hatch process.
