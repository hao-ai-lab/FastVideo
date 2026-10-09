# Wan-VACE Local Tests

Local parity and smoke coverage for Wan2.1-VACE controllable video generation.

## Official Reference

- Repository: [Wan-Video/Wan2.1](https://github.com/Wan-Video/Wan2.1)
- Weights:
  - `Wan-AI/Wan2.1-VACE-1.3B-diffusers`
  - `Wan-AI/Wan2.1-VACE-14B-diffusers`

## Setup

```bash
export WAN_VACE_MODEL_DIR=/path/to/Wan2.1-VACE-1.3B-diffusers
export WAN_VACE_14B_MODEL_DIR=/path/to/Wan2.1-VACE-14B-diffusers
export FASTVIDEO_ATTENTION_BACKEND=TORCH_SDPA
```

## Test Inventory

| Scope | File | GPU | Notes |
|---|---|---|---|
| Registry / context shape | `fastvideo/tests/api/test_wan_vace_definitions.py` | no | Buildkite unit lane |
| Input and backend contracts | `fastvideo/tests/api/test_wan_vace_{inputs,backend}.py` | no | Buildkite unit lane |
| Pipeline smoke | `tests/local_tests/pipelines/test_wan_vace_pipeline_smoke.py` | optional | Preflight always runs |
| Transformer parity | `tests/local_tests/pipelines/test_wan_vace_pipeline_parity.py` | yes | Single-step forward vs Diffusers |
| Pipeline parity | `tests/local_tests/pipelines/test_wan_vace_end_to_end_parity.py` | yes | VACE e2e gates (bf16 inputs + every-step teacher-forced DiT + final drift) |
| SP2 forward | `fastvideo/tests/distributed/test_sp_wan_vace.py` | 2 | Weight-free FP32 odd-token/short-control parity |
| SP2 pipeline | `tests/local_tests/pipelines/test_wan_vace_sp_pipeline.py` | 2 | Weighted BF16 single vs SP2 (`WAN_VACE_SP_E2E=1`): drift < 5%, frame SSIM ≥ 0.93 |
| Shared helpers | `tests/local_tests/wan/vace_parity_helpers.py` | — | Official fp32 VAE/T5 reload, FV transformer load |
| Repro matrix | `scripts/run_vace_matrix.py` | yes | Official reference-image case |
| SSIM regression | `fastvideo/tests/ssim/test_wan_vace_similarity.py` | 1 | Reference-image and video+mask cases; CI SSIM lane |
| Port status | `tests/local_tests/wan/PORT_STATUS.md` | — | add-model handoff state |

## Parity Gates

End-to-end tests in `test_wan_vace_end_to_end_parity.py` use VACE-specific gates:

| Gate | What it checks |
|---|---|
| Conditioning inputs | bf16 bitwise equality (prompt, control, video/mask/refs, timesteps, initial latents) |
| Every step, teacher-forced | FastVideo prediction vs the Diffusers DiT on FastVideo's exact step inputs (`atol=rtol=0.05`, drift < 1%) |
| Final latent | free-running abs-mean drift < 4% |

Cases: 1.3B t2v / reference / video+mask, plus 14B reference. The component test
(`test_wan_vace_pipeline_parity.py`) covers an integer and a fractional timestep in
FP32 (`1e-4`) and BF16 (`0.05`, drift < 2%).

The free-running gate is looser because the BF16 DiT itself turns 1-ulp input changes
into about 2% output drift. See
[parity evidence](../../../docs/inference/wan_vace.md#parity-evidence).

## Sequence Parallel

Weight-free FP32 SP2 forward passes at `atol=rtol=1e-5`. The weighted BF16
5-frame short pipeline (`WAN_VACE_SP_E2E=1`) gates drift < 5% and frame SSIM
≥ 0.93, the repo's SP2 norm. BF16 GEMM results depend on the sharded token row
count, so exact equality is not expected; conditioning inputs and both ranks still
match. See
[sequence parallel status](../../../docs/inference/wan_vace.md#sequence-parallel-status).
Set `WAN_VACE_SP_TRACE=1` for read-only per-layer divergence prints.

### 14B block trace

Real-shape 14B reference step0 with aligned transformer inputs:

- Overall step0 noise prediction passes `assert_dit_parity`.
- Per-main-block hook comparison shows accumulating bf16 drift (worst `blocks.39`,
  max abs ~1.6) that stays within the step0 DiT gate; no VACE wiring bug or shared
  Wan DiT regression was found.

## Commands

```bash
pytest fastvideo/tests/api/test_wan_vace_definitions.py -q
pytest fastvideo/tests/api/test_wan_vace_inputs.py fastvideo/tests/api/test_wan_vace_backend.py -q
pytest tests/local_tests/pipelines/test_wan_vace_pipeline_smoke.py -q -k preflight
WAN_VACE_MODEL_DIR=$WAN_VACE_MODEL_DIR pytest tests/local_tests/pipelines/test_wan_vace_pipeline_parity.py -v -s
WAN_VACE_MODEL_DIR=$WAN_VACE_MODEL_DIR WAN_VACE_14B_MODEL_DIR=$WAN_VACE_14B_MODEL_DIR \
  pytest tests/local_tests/pipelines/test_wan_vace_end_to_end_parity.py -v -s
pytest fastvideo/tests/distributed/test_sp_wan_vace.py -v -s
WAN_VACE_SP_E2E=1 WAN_VACE_MODEL_DIR=$WAN_VACE_MODEL_DIR WAN_VACE_14B_MODEL_DIR=$WAN_VACE_14B_MODEL_DIR \
  pytest tests/local_tests/pipelines/test_wan_vace_sp_pipeline.py -v -s
python scripts/run_vace_matrix.py
```

See [`docs/inference/wan_vace.md`](../../../docs/inference/wan_vace.md).
