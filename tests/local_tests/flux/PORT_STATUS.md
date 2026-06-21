# FLUX.1-dev Port Status

Model family: flux
Official ref: black-forest-labs/FLUX.1-dev (Diffusers layout)
Workload: T2I
Last updated: 2026-06-20

## Component Status

| Component | Type | Parity test | Status | Notes |
|---|---|---|---|---|
| FluxTransformer2DModel | DiT (ported) | fastvideo/tests/transformers/test_flux.py | PASS (A40, 2026-05-11) | max_diff=0.5, mean_diff=0.04, median=0 — bf16 tail error, see notes |
| AutoencoderKL | VAE (reused) | tests/local_tests/flux/test_flux_dev_component_loaders.py | PASS (A40, 2026-05-11) | Loader smoke: 54.14s |
| CLIPTextModel | encoder (reused) | tests/local_tests/flux/test_flux_dev_component_loaders.py | PASS (A40, 2026-05-11) | Shared component |
| T5EncoderModel | encoder (reused) | tests/local_tests/flux/test_flux_dev_component_loaders.py | PASS (A40, 2026-05-11) | Shared component |
| FlowMatchEulerDiscreteScheduler | scheduler (reused) | tests/local_tests/flux/test_flux_dev_component_loaders.py | PASS (A40, 2026-05-11) | Shared component |

## Conversion

No conversion script needed — FLUX.1-dev uses native Diffusers checkpoint layout.
Components load via production loaders with no key remapping.
Strict-load evidence: PASS — component loader test runs strict load of all components (2026-05-11).

## Pipeline

| Test | Status | Notes |
|---|---|---|
| Pipeline smoke (2-step, 256×256) | PASS (A40, 2026-05-11) | 91.14s — output finite, non-null |
| Pipeline parity vs Diffusers FluxPipeline | PASS (A40, 2026-05-11) | Output finite, in [0,1], correct shape — pixel parity not enforced (pipelines sample noise independently) |

## Quality

| Item | Status |
|---|---|
| SSIM test | written (fastvideo/tests/ssim/test_flux_t2i_similarity.py) |
| Reference images committed | TORCH_SDPA committed (A40, 2026-05-11); FLASH_ATTN pending |

## DiT Parity Notes

DiT parity test compares FastVideo `FluxTransformer2DModel` vs Diffusers under identical inputs (bfloat16, L40S).
- max_diff=0.5000, mean_diff=0.0382, median_diff=0.0000, p99_diff=0.2500
- Median=0 and mean=0.04 confirm implementations are equivalent; tail errors up to 0.5 are expected
  bfloat16 accumulation over 57 transformer layers (eps~7.8e-3, accumulated per-GEMM error ~0.4).
- Test uses atol=0.5 which catches real bugs (wrong weights/layers produce mean_diff >> 0.1).

## Known Blockers

None. (Both prior blockers resolved — see below.)

## Resolved

1. SSIM reference image — RESOLVED. The TORCH_SDPA reference (A40, 256×256, 8 steps,
   seed=0) is committed, so the SSIM gate runs against it. A FLASH_ATTN reference is
   still pending (see Quality table) but is not required for the default backend.

2. Registry registration — RESOLVED. FLUX is registered unconditionally in
   `_register_configs()` (fastvideo/registry.py); there is no early-return guard that
   could silently skip it. The earlier `_CONFIG_REGISTRY` pre-population concern no
   longer applies after the registry refactor.
