# MiniMax H3 Local Tests

Local component and contract tests for `minimax_h3` in FastVideo.
Stage 1 covers model components, FL2VA packing, both schedulers, direct Diffusers component loading, and the minimal
`last_image`/`references`/`audio_latents` request bridge. Stage 2 adds the internal T2VA/FL2VA composed pipeline.
Stage 3 adds ordered Ref2VA media, condition encoding, packing, and its separate pipeline. Stage 4 adds public
registration and real-weight distributed T2VA, FL2VA, and Ref2VA acceptance.

Progress and blockers live in `tests/local_tests/minimax_h3/PORT_STATUS.md`.

## Reference boundary

| Field | Value |
|---|---|
| Model family | `minimax_h3` |
| Workloads | T2VA, FL2VA, and Ref2VA joint video/stereo-audio generation |
| Implementation reference | `https://github.com/huggingface/diffusers/pull/14355` |
| Local reference source | `DiffusersMiniMaxH3/src` |
| Checkpoint | `MiniMaxAI/MiniMax-H3@9bfb6693f2cf6de171db46d1aa586f67d773a1da` |
| Loading boundary | Diffusers component folders |

FastVideo selects `transformer/` for T2VA/FL2VA and `transformer_ref/` for Ref2VA. Shared components load directly
from their published subfolders. The reference source is imported from the local checkout; no core dependency is
changed.

## Stage 1 tests

| Component | Reference | Test | Evidence |
|---|---|---|---|
| Transformer | `transformer_minimax_h3.py` | `tests/local_tests/transformers/test_minimax_h3_transformer_parity.py` | tiny-weight activation and dual-head parity |
| Video VAE | `autoencoder_kl_minimax_h3.py` | `tests/local_tests/vaes/test_minimax_h3_video_vae_parity.py` | activation, encode, decode, and tiling parity |
| Audio VAE | `autoencoder_kl_minimax_h3_audio.py` | `tests/local_tests/vaes/test_minimax_h3_audio_vae_parity.py` | activation, posterior, and decode parity |
| Scheduler | `scheduling_minimax_h3.py` | `tests/local_tests/minimax_h3/test_minimax_h3_scheduler_parity.py` | independent video/audio schedule parity |
| FL2VA packing | `modular_pipelines/minimax_h3/packing.py` | `tests/local_tests/minimax_h3/test_minimax_h3_packing.py` | independent row, position, tag, and RNG contracts |
| Component loading | FastVideo loaders | `tests/local_tests/minimax_h3/test_minimax_h3_loader_contracts.py` | strict meta-device loading, mixed dtypes, both VAEs, and both schedulers |
| Request bridge | FastVideo typed request path | `tests/local_tests/minimax_h3/test_minimax_h3_api_contract.py` | `last_image`, `references`, and `audio_latents` survive to `ForwardBatch` |

Run Stage 1 from the repository root:

```bash
PYTHONPATH=DiffusersMiniMaxH3/src pytest \
  tests/local_tests/transformers/test_minimax_h3_transformer_parity.py \
  tests/local_tests/vaes/test_minimax_h3_video_vae_parity.py \
  tests/local_tests/vaes/test_minimax_h3_audio_vae_parity.py \
  tests/local_tests/minimax_h3 -v -s
```

These tests establish synthetic CPU parity only. They do not establish real-checkpoint compatibility, generated
media quality, CUDA behavior, memory use, or performance.

## Stage 2 tests

| Scope | Test | Evidence |
|---|---|---|
| Qwen3-VL encoder and conditioning | `tests/local_tests/minimax_h3/test_minimax_h3_conditioner.py` | standard `BaseEncoderOutput`; stage-owned picture presentation, tags, multimodal IDs, and layer-50 selection |
| T2VA/FL2VA pipeline | `tests/local_tests/minimax_h3/test_minimax_h3_pipeline.py` | private factory, offload lifecycle, four paths, dual schedules, executor-safe result, and stereo MP4 mux |

Run Stage 2 from the repository root:

```bash
pytest \
  tests/local_tests/minimax_h3/test_minimax_h3_conditioner.py \
  tests/local_tests/minimax_h3/test_minimax_h3_pipeline.py -q
```

Stage 2 evidence uses synthetic tiny components; Stage 4 below records real-checkpoint CUDA and media-quality
acceptance.

## Stage 3 tests

| Scope | Test | Evidence |
|---|---|---|
| Ref2VA media | `tests/local_tests/minimax_h3/test_minimax_h3_ref2va_media.py` | image/video/audio normalization, rate handling, and local decode |
| Ref2VA packing | `tests/local_tests/minimax_h3/test_minimax_h3_ref2va_packing.py` | handwritten oracle plus independent exact reference parity |
| Transformer partition | `tests/local_tests/minimax_h3/test_minimax_h3_ref_loader.py` | logical `transformer` loads only from `transformer_ref/` |
| Ref2VA pipeline | `tests/local_tests/minimax_h3/test_minimax_h3_ref2va_pipeline.py` | ordered presentation, VAE conditions, RNG, validation, and joint AV output |

Run Stage 3 from the repository root:

```bash
PYTHONPATH=DiffusersMiniMaxH3/src pytest \
  tests/local_tests/minimax_h3/test_minimax_h3_ref2va_media.py \
  tests/local_tests/minimax_h3/test_minimax_h3_ref2va_packing.py \
  tests/local_tests/minimax_h3/test_minimax_h3_ref_loader.py \
  tests/local_tests/minimax_h3/test_minimax_h3_ref2va_pipeline.py -q
```

Stage 3 evidence is synthetic CPU evidence; Stage 4 below records the corresponding real-weight Ref2VA run.

## Stage 4 distributed acceptance

The T2VA acceptance runner exercises the released FL2VA partition in its text-only mode. On the GB200 cluster,
the checkpoint is stored at `/mnt/lustre/vlm-k1kong/models/MiniMax-H3` and can be run from the repository root with:

```bash
PYTHONPATH="$PWD:$PWD/DiffusersMiniMaxH3/src" \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun --nproc_per_node=4 tests/local_tests/minimax_h3/run_stage4_t2va.py \
  --model-path /mnt/lustre/vlm-k1kong/models/MiniMax-H3 \
  --output /mnt/lustre/vlm-k1kong/outputs/minimax-h3/stage4_t2va_seed0.mp4
```

Accepted seed-0 evidence:

| Field | Result |
|---|---|
| Distributed path | 4x GB200, FSDP shard size 4, SP=4, FlashAttention-2 |
| Pipeline time | 85.241 seconds after module loading |
| Video | H.264, 960x544, 24 fps, 124 frames, 5.167 seconds |
| Audio | AAC, 32 kHz stereo, 5.216 seconds; finite and non-silent |
| Output SHA-256 | `ea1dbfeb37fd9036f2f6d8ce1d9ffb20c39dbd54d5b5381fa4c3b33bdfeaff6a` |

This proves the real distributed T2VA path; component-level Diffusers numerical parity and SP=1 versus SP=4 parity
remain separate follow-up gates.

FL2VA and Ref2VA use the public manifest-compatible classes through the conditioned runner:

```bash
PYTHONPATH="$PWD" CUDA_VISIBLE_DEVICES=0,1,2,3 \
python -m torch.distributed.run --standalone --nproc_per_node=4 \
  tests/local_tests/minimax_h3/run_stage4_conditioned.py fl2va \
  --model-path /mnt/lustre/vlm-k1kong/models/MiniMax-H3 \
  --prompt-file /mnt/lustre/vlm-k1kong/models/MiniMax-H3/scripts/readme/reproducible-768p-fl2va-request.sh \
  --image /mnt/lustre/vlm-k1kong/inputs/minimax-h3/fl2va_keyframe.png \
  --output /mnt/lustre/vlm-k1kong/outputs/minimax-h3/stage4_fl2va_official_seed0.mp4

PYTHONPATH="$PWD" CUDA_VISIBLE_DEVICES=0,1,2,3 \
python -m torch.distributed.run --standalone --nproc_per_node=4 \
  tests/local_tests/minimax_h3/run_stage4_conditioned.py ref2va \
  --model-path /mnt/lustre/vlm-k1kong/models/MiniMax-H3 \
  --prompt-file /mnt/lustre/vlm-k1kong/models/MiniMax-H3/scripts/readme/reproducible-768p-ref2va-request.sh \
  --reference-video /mnt/lustre/vlm-k1kong/inputs/minimax-h3/ref2va_source.mp4 \
  --reference-audio /mnt/lustre/vlm-k1kong/inputs/minimax-h3/ref2va_voice.mp3 \
  --output /mnt/lustre/vlm-k1kong/outputs/minimax-h3/stage4_ref2va_official_seed0.mp4
```

Accepted official-request evidence:

| Workload | Partition | Output | Audio | SHA-256 |
|---|---|---|---|---|
| FL2VA | `transformer/` | 1344x768, 192 frames, 8.000 s | 32 kHz stereo, 8.032 s | `fa7bd3940b4316804314e9cf98d91187a0d3755d2945cea93ebad7da9de0d3c7` |
| Ref2VA | `transformer_ref/` | 1344x768, 124 frames, 5.167 s | 32 kHz stereo, 5.216 s | `2c7c25ea7385b86a11e2a934df3f698f590839156c1c37a31266cbd041ccdcf6` |

Both outputs decode without errors and have non-silent audio (mean -14.8 dB, peak -3.1 dB). FL2VA completes in
191.954 seconds after module loading, including 58.192 seconds of conditioning and 114.703 seconds of denoising.
Ref2VA completes in 276.582 seconds after module loading; its measured stages include 55.592 seconds of multimodal
conditioning, 11.467 seconds of reference encoding, and 193.615 seconds of denoising.

## Review notes

- Keep the FastVideo and reference packers independent.
- Keep Transformer FP32 islands and both FP32 VAEs explicit.
- Keep `last_image`, `references`, and `audio_latents` in the typed request bridge.
- Keep the isolated Transformers Qwen3-VL base-model passthrough adapter on the standard FastVideo encoder contract;
  load the released full checkpoint strictly before retaining its base model, while H3 presentation and layer
  selection remain in the conditioning stage.
- Keep media I/O in reference preparation, not in the immutable request object.
- Keep the public class names aligned with the official modular manifest; Ref2VA explicitly selects its alternate
  class so only `transformer_ref/` is loaded.
