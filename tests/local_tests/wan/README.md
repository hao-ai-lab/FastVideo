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
export DISABLE_SP=1
export FASTVIDEO_ATTENTION_BACKEND=TORCH_SDPA
```

## Test Inventory

| Scope | File | GPU | Notes |
|---|---|---|---|
| Registry / context shape | `fastvideo/tests/api/test_wan_vace_definitions.py` | no | Buildkite unit lane |
| Input and backend contracts | `fastvideo/tests/api/test_wan_vace_{inputs,backend}.py` | no | Buildkite unit lane |
| Pipeline smoke | `tests/local_tests/pipelines/test_wan_vace_pipeline_smoke.py` | optional | Preflight always runs |
| Transformer parity | `tests/local_tests/pipelines/test_wan_vace_pipeline_parity.py` | yes | Single-step forward vs Diffusers |
| Pipeline parity | `tests/local_tests/pipelines/test_wan_vace_end_to_end_parity.py` | yes | VACE e2e gates (bf16 inputs + step0 DiT + final latent drift) |
| Shared helpers | `tests/local_tests/wan/vace_parity_helpers.py` | — | Official fp32 VAE/T5 reload, FV transformer load |
| Repro matrix | `scripts/run_vace_matrix.py` | yes | Official reference-image case |

## Parity Gates

End-to-end tests in `test_wan_vace_end_to_end_parity.py` use VACE-specific gates:

| Gate | What it checks |
|---|---|
| Conditioning inputs | bf16 bitwise equality (prompt, control, video/mask/refs, timesteps, initial latents) |
| Step0 noise prediction | `assert_dit_parity` against Diffusers (atol/rtol 0.1) |
| Final latent | abs-mean drift below 5% after multi-step denoising |

Cases: 1.3B t2v / reference / video+mask, plus 14B reference.

### 14B block trace (2026-09-24)

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
python scripts/run_vace_matrix.py
```

See [`docs/inference/wan_vace.md`](../../../docs/inference/wan_vace.md).
