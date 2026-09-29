# Wan2.1-VACE Port Status

Model family: wan (VACE variant)
Official ref: Wan-AI/Wan2.1-VACE-1.3B-diffusers, Wan-AI/Wan2.1-VACE-14B-diffusers (Diffusers layout)
Workload: controllable T2V (reference images, control video, video + mask)
Last updated: 2026-09-29

## Component Status

| Component | Type | Parity test | Status | Notes |
|---|---|---|---|---|
| WanVACETransformer3DModel | DiT (ported) | tests/local_tests/pipelines/test_wan_vace_pipeline_parity.py | PASS (B200) | Single-step forward vs Diffusers, FP32 and BF16, 1.3B |
| AutoencoderKLWan | VAE (reused) | tests/local_tests/pipelines/test_wan_vace_end_to_end_parity.py | PASS (B200) | Control/reference latents bf16 bitwise equal to Diffusers |
| UMT5 text encoder | encoder (reused) | tests/local_tests/pipelines/test_wan_vace_end_to_end_parity.py | PASS (B200) | Prompt embeddings bf16 bitwise equal; needs `T5PaddedConfig` |
| FlowUniPCMultistepScheduler | scheduler (reused) | tests/local_tests/pipelines/test_wan_vace_end_to_end_parity.py | PASS (B200) | Timesteps bitwise equal |

## Conversion

No conversion script is needed. Both checkpoints use the Diffusers layout and load
through `WanVACEConfig.param_names_mapping`.

## Pipeline

| Test | Status | Notes |
|---|---|---|
| Pipeline smoke | PASS | `test_wan_vace_pipeline_smoke.py`; preflight runs without weights |
| Pipeline parity vs Diffusers | PASS (B200) | 1.3B t2v / reference / video+mask and 14B reference; gates in [README](README.md#parity-gates) |
| Basic example | PASS | `examples/inference/basic/basic_wan_vace.py` |
| SP2 forward (weight-free FP32) | PASS | `fastvideo/tests/distributed/test_sp_wan_vace.py` |
| SP2 weighted BF16 short pipeline | PASS (1.3B local; 14B Slurm job 951756: drift 3.73%, SSIM 0.938) | Drift < 5%, frame SSIM ≥ 0.93; not bitwise because BF16 GEMM depends on sharded row count; see [docs](../../../docs/inference/wan_vace.md#sequence-parallel-status) |

## Quality

| Item | Status |
|---|---|
| SSIM test | written (fastvideo/tests/ssim/test_wan_vace_similarity.py; 1.3B reference-image and video+mask cases, 1 GPU) |
| Reference videos | pending: a maintainer runs the SSIM job with `FASTVIDEO_SSIM_BOOTSTRAP_MODE=1`, reviews the drafts, then promotes them |

## Known Blockers

None. SSIM reference videos still need a maintainer bootstrap run (see Quality).
