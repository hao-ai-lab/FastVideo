# MiniMax H3 Local Tests

Local component and contract tests for `minimax_h3` in FastVideo.
Stage 1 covers model components, FL2VA packing, both schedulers, direct Diffusers component loading, and the minimal
`last_image`/`references`/`audio_latents` request bridge. Stage 2 adds the internal T2VA/FL2VA composed pipeline.
Stage 3 adds ordered Ref2VA media, condition encoding, packing, and its separate private pipeline.

Progress and blockers live in `tests/local_tests/minimax_h3/PORT_STATUS.md`.

## Reference boundary

| Field | Value |
|---|---|
| Model family | `minimax_h3` |
| Workloads | T2VA, FL2VA, and Ref2VA joint video/stereo-audio generation |
| Implementation reference | `https://github.com/huggingface/diffusers/pull/14355` |
| Local reference source | `DiffusersMiniMaxH3/src` |
| Checkpoint | `MiniMaxAI/MiniMax-H3`; not available in this workspace |
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

The pipeline remains internal until real-checkpoint and distributed acceptance. Stage 2 evidence uses synthetic tiny
components; real-checkpoint, CUDA, model-output, and media-quality behavior are unverified.

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

Stage 3 evidence remains synthetic CPU evidence. Real weights, CUDA, distributed execution, output quality, memory,
and performance remain unverified.

## Review notes

- Keep the FastVideo and reference packers independent.
- Keep Transformer FP32 islands and both FP32 VAEs explicit.
- Keep `last_image`, `references`, and `audio_latents` in the typed request bridge.
- Keep the isolated Transformers Qwen3-VL base-model passthrough adapter on the standard FastVideo encoder contract;
  H3 presentation and layer selection belong to the conditioning stage.
- Keep media I/O in reference preparation, not in the immutable request object.
- Neither private pipeline adds a public preset or registry detector.
