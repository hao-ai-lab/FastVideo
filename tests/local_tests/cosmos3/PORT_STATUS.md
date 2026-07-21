# Cosmos3 Port Status

## Current Scope

- Model family: `cosmos3`
- Registered checkpoint: `nvidia/Cosmos3-Nano` only
- Official parity oracle: <https://github.com/NVIDIA/cosmos-framework>
- Checkpoint layout: public diffusers-layout HF repository; no conversion or
  FastVideo-owned weights repository is required
- Public registry workloads: T2V and I2V
- Additional implemented paths: T2I preset/example and opt-in T2VS generation
- Component-only paths: action conditioning and text/image reasoning have native
  DiT/helper parity, but are not first-class public pipeline workloads
- Status: aggregate branch ready for review
- Last updated: 2026-07-21

The branch must not be described as full-omni end-to-end support. The native
model contains the sound, action, and reasoning surfaces needed to strict-load
and exercise the checkpoint, but public action/image-reasoning wiring and
generated-media/audio regression remain follow-ups.

## Implementation Matrix

| Surface | Current support | Evidence / limitation |
|---|---|---|
| Video DiT | Native MoT implementation with production T2V/I2V pipeline | Framework parity and real-weight B200 runs recorded below |
| T2I | Pipeline dispatch, preset, and example | Not registered as a `WorkloadType` |
| Sound | Native DiT sound path plus lazy AVAE decode | Opt-in through `COSMOS3_T2VS`; no registered AV workload or audio-quality baseline |
| Action | Native domain-aware input/output path | Component parity only; pipeline does not build action specs |
| Reasoning | Native text/deepstack reasoner paths | Component parity only; public image-conditioned prefill wiring remains |
| VAE | Native Wan2.2 `AutoencoderKLWan` configuration | Framework parity recorded below |
| Scheduler | Native flow-configured UniPC | Framework timetable/sigma/trajectory parity recorded below |
| Tokenizer | Qwen2 tokenizer passthrough | Loaded from the checkpoint |
| Vision encoder | Qwen3-VL vision component parity | Not a required public pipeline module |
| Checkpoint loading | Direct diffusers-layout load | `cosmos3_convert.py` is a strict-load verifier, not a converter |

## Verification

| Scope | Result | Notes |
|---|---|---|
| Local framework component suite | `150 passed, 0 skipped` | Recorded on 2026-06-07 before the final aggregate rebase; this is not a fresh run against the fingerprint's newer official pin |
| Real-weight B200 inference | T2V, I2V, T2I, T2VS, text reasoning | Recorded on 2026-06-07; outputs were inspected and described as coherent/prompt-matching |
| Transformer fingerprint seed | `2 passed` | Modal L40S, 2026-07-21; T2V/T2VS/action decoder-layer outputs matched the pinned official framework layer bit-exact; config contract passed |
| Transformer fingerprint normal mode | `2 passed` | Modal L40S, 2026-07-21; T2V/T2VS/action/reasoning committed hashes passed without loading the official framework |
| Modal checkout unit tests | `12 passed` | Exact immutable Buildkite commit checkout and bounded retries |

Run the normal transformer fingerprint with:

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 FASTVIDEO_FA4=0 \
  pytest fastvideo/tests/transformers/test_cosmos3_block_fingerprint.py -v -s
```

The test requires the pinned Modal L40S execution envelope and a mounted cache
for `nvidia/Cosmos3-Nano`.

## Transformer Fingerprint Baseline

- Checkpoint revision: `411f42a8fdfb8c5b2583cb8786e0938f49796eaa`
- Official framework revision: `ed8287fd7477113f8ac4f6b84290514d55cf0cdc`
- Container image digest:
  `ghcr.io/hao-ai-lab/fastvideo/fastvideo-dev:py3.12-latest@sha256:5d0e99c518bb6ee06f5062f1e04e94846a3f327a67b3d78bb78e5e949b9f45f2`
- Hardware/software: NVIDIA L40S, PyTorch `2.12.0+cu126`, CUDA `12.6`,
  BF16, MATH SDPA, TF32 disabled, deterministic algorithms enabled
- State loaded: 44 checkpoint tensors or slices, 408,486,912 parameters,
  approximately 779.13 MiB in BF16
- Environment SHA-256:
  `ad3d22373224999f0a0520c96eb406335ad5b58bdf4fd2f3e4ec29de2f6e1dda`
- Config/structural contract SHA-256:
  `5043ad8a281ddfd2595317c45943c2c06d71cf8012525db7efed1db7af7e31e3`
- Input SHA-256:
  `4c2f1879c1847f65372fbb08df981bbb00e2e4b7ad6176c22db5515e69b2846f`
- Selected weights SHA-256:
  `480b5a497bb0c9ef71edabeee62c057b2c54fce753362e8938d073d6bf7cfa62`
- T2V outputs SHA-256:
  `2b15c03d936bace07acea7aba55fe850dd3cadc344b361dce8da68acd94f9cb7`
- T2VS outputs SHA-256:
  `1bffb714b7f316665e0176ddfe2b930df70d6dae1f75b3f761e92b5b9d8c26dd`
- Action2world outputs SHA-256:
  `4cf042b91e1a0c4131efbb454e23d679b73542abda9b36af18cd81bb694b1b34`
- Reasoning outputs SHA-256:
  `112b9ecd765b0e1afc7d21d169b4f3e941901d975c27f59df6062dedef60bc35`

Normal CI runs separate depth-one T2V, T2VS, action2world, and deepstack-reason
forwards twice and checks exact output identity plus the committed hashes.
Seed/reseed mode additionally imports the pinned NVIDIA framework and compares
the captured generation decoder layer bit-for-bit. No extracted fixture or
separate HF upload is needed.

## Recorded Component Parity

The following results were recorded from tiny CPU/FP32 models with framework
weights copied into the native implementations. They establish component
behavior, not current generated-media quality.

| Component / pipeline | Test | Recorded result |
|---|---|---|
| Scheduler (UniPC flow) | `test_cosmos3_scheduler_parity` | timesteps exact; sigmas about `1e-8`; trajectory below about `1e-6` |
| DiT video path | `test_cosmos3_dit_parity_mrope` | exact |
| Sequence packing | `test_cosmos3_packing_parity` | exact |
| VAE (Wan2.2) | `test_cosmos3_vae_parity` | exact |
| Denoise / CFG velocity | `test_cosmos3_denoise_cfg_parity` | max below `1e-6` |
| Resolution flow shift | `test_cosmos3_flow_shift_parity` | exact |
| I2V conditioning | `test_cosmos3_i2v_conditioning_parity` | exact |
| AVAE decoder | `test_cosmos3_avae_parity` | exact |
| Sound DiT / packing / CFG | `test_cosmos3_sound_parity` | exact |
| Action DiT / packing / CFG | `test_cosmos3_action_parity` | exact |
| Text/deepstack reasoning | `test_cosmos3_reasoning_parity` | exact logits/forward and token-exact greedy generation |
| Vision encoder | `test_cosmos3_vision_encoder_parity` | exact |

## Remaining Gaps

- Add Cosmos3 generated-media SSIM references once the desired public
  T2V/I2V/T2I regression cases are selected.
- Add an audio-quality regression metric for T2VS; SSIM cannot cover audio.
- Decide whether to register T2I and an AV workload as public workload types.
- Wire action conditioning and image-conditioned reasoning into first-class
  public pipeline requests if those are in product scope.
- Only Cosmos3-Nano is registered and tested; no other Cosmos3 size is claimed.

## CI Decision

Routine changes limited to `fastvideo/models/dits/cosmos3.py` and its DiT config
use the exact one-layer fingerprint instead of scheduling the full SSIM lane.
Changes to the fingerprint baseline itself, pipelines, sequence packing,
schedulers, encoders, VAEs, shared layers, dependencies, or the container still
schedule SSIM. The full end-to-end SSIM test is retained; it simply does not
need to run on every covered Cosmos3 DiT-only PR.
