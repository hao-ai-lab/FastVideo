# FLUX.1-dev Port Status

Model family: flux
Official ref: black-forest-labs/FLUX.1-dev (Diffusers layout)
Workload: T2I
Last updated: 2026-05-11

## Component Status

| Component | Type | Parity test | Status | Notes |
|---|---|---|---|---|
| FluxTransformer2DModel | DiT (ported) | fastvideo/tests/transformers/test_flux.py | pending local run | Requires weights + CUDA |
| AutoencoderKL | VAE (reused) | tests/local_tests/flux/test_flux_dev_component_loaders.py | loader smoke only | Shared component, no dedicated parity |
| CLIPTextModel | encoder (reused) | tests/local_tests/flux/test_flux_dev_component_loaders.py | loader smoke only | Shared component |
| T5EncoderModel | encoder (reused) | tests/local_tests/flux/test_flux_dev_component_loaders.py | loader smoke only | Shared component |
| FlowMatchEulerDiscreteScheduler | scheduler (reused) | tests/local_tests/flux/test_flux_dev_component_loaders.py | loader smoke only | Shared component |

## Conversion

No conversion script needed — FLUX.1-dev uses native Diffusers checkpoint layout.
Components load via production loaders with no key remapping.
Strict-load evidence: pending (loader smoke test verifies class resolution only).

## Pipeline

| Test | Status | Notes |
|---|---|---|
| Pipeline smoke (2-step, 256×256) | pending local run | tests/local_tests/pipelines/test_flux_dev_pipeline_smoke.py |
| Pipeline parity vs Diffusers FluxPipeline | pending local run | tests/local_tests/pipelines/test_flux_dev_pipeline_parity.py |

## Quality

| Item | Status |
|---|---|
| SSIM test | written (fastvideo/tests/ssim/test_flux_t2i_similarity.py) |
| Reference images committed | not yet — blocker: needs full-res inference run on target hardware |

## Known Blockers

1. SSIM reference images not committed — test will FileNotFoundError before comparison.
   Resolution: run seed-ssim-references skill on A40/L40S after weights are available.

2. Pipeline parity test written but not yet run — smoke test only verifies non-null finite output.
   Resolution: run tests/local_tests/pipelines/test_flux_dev_pipeline_parity.py on GPU
   with weights and record PASS output here.

3. DiT parity PASS not recorded — test exists but no local output committed.
   Resolution: run on GPU with weights, record pytest output here.

4. Registry early-return guard (registry.py) — if any test pre-populates
   _CONFIG_REGISTRY before _register_configs() runs, FLUX registration will be
   silently skipped. Known pre-existing pattern; accepted risk for now.
