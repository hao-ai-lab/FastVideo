# MiniMax H3 Local Tests

Local component and contract tests for the FastVideo-native `minimax_h3` port.
Stage 1 covers native components, FL2VA packing, both schedulers, direct Diffusers component loading, and the minimal
`last_image`/`references`/`audio_latents` request bridge. Pipeline composition starts in Stage 2.

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

## Review notes

- Keep the FastVideo and reference packers independent.
- Keep Transformer FP32 islands and both FP32 VAEs explicit.
- Keep `last_image`, `references`, and `audio_latents` in the typed request bridge.
- Decode, normalize, and validate reference contents in the Ref2VA stages.
- Stage 1 does not activate a pipeline, preset, conditioner, denoising loop, or public registry entry.
