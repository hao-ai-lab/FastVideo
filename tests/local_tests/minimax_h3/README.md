# MiniMax H3 Local Tests

Local-only component and contract tests for the `minimax_h3` FastVideo port.
The current branch stops after engineering Stage 1 from
`docs/design/minimax_h3_frontier_and_integration_plan.md`: native components,
packing/scheduler/conversion contracts, minimal request plumbing, and synthetic
parity. Pipeline composition starts in Stage 2.

Progress, open questions, and blockers live in
`tests/local_tests/minimax_h3/PORT_STATUS.md`.

## Reference assets

| Field | Value |
|---|---|
| Model family | `minimax_h3` |
| Workload types | T2V/I2V-compatible requests producing joint video and stereo audio; pipeline deferred to Stage 2 |
| Implementation reference | `https://github.com/huggingface/diffusers/pull/14355` |
| Local reference dir | `DiffusersMiniMaxH3/` |
| Reference commit | `abc5e9bf71fd38f53cd471bc3acaa84bc5ecbfdc` |
| HF weights | `MiniMaxAI/MiniMax-H3` described by the draft; no usable local weights in Stage 1 |
| HF revision | unknown |
| Local weights dir | unavailable |
| Source layout | `raw_official` as described by the pinned conversion script |
| Needs conversion | yes |

No token value belongs in this file. Future gated access should use an exported
`HF_TOKEN`, `HUGGINGFACE_HUB_TOKEN`, or `HF_API_KEY`.

## Shared environment setup

Run from the FastVideo repository root in the same environment as FastVideo:

```bash
python .agents/skills/add-model-01-prep/scripts/clone_reference_repo.py \
  https://github.com/huggingface/diffusers.git \
  DiffusersMiniMaxH3 \
  --commit abc5e9bf71fd38f53cd471bc3acaa84bc5ecbfdc
```

The reference imports directly from `DiffusersMiniMaxH3/src`; it is not
installed and no core dependency version was changed.

```text
dependency_changes: none
official_env_status: imports_ok
private_dep_stubs: none
blocked_on: real checkpoint parity is deferred until accessible weights exist
```

## Prototype and conversion artifacts

Synthetic fixtures exercise the converter without materializing real weights.
Optional key/shape dumps are written under the ignored directory
`converted_weights/minimax_h3/_mapping/`.

```text
official_key_dumps:
  transformer: converted_weights/minimax_h3/_mapping/transformer_official_keys.json
  video_vae: converted_weights/minimax_h3/_mapping/video_vae_official_keys.json
  audio_vae: converted_weights/minimax_h3/_mapping/audio_vae_official_keys.json
fastvideo_key_dumps:
  transformer: converted_weights/minimax_h3/_mapping/transformer_fastvideo_keys.json
  video_vae: converted_weights/minimax_h3/_mapping/video_vae_fastvideo_keys.json
  audio_vae: converted_weights/minimax_h3/_mapping/audio_vae_fastvideo_keys.json
conversion_script: scripts/checkpoint_conversion/convert_minimax_h3_to_diffusers.py
conversion_source_layout: raw_official
converted_weights_dir: converted_weights/minimax_h3
strict_load_status: all three native components pass synthetic strict load; real-weight strict load deferred
```

## Stage 1 tests

| Component | Reference | Test | Stage 1 evidence |
|---|---|---|---|
| Transformer | `transformer_minimax_h3.py`; tiny config from its model test | `tests/local_tests/transformers/test_minimax_h3_transformer_parity.py` | exact state surface plus activation and both-head output parity |
| Video VAE | `autoencoder_kl_minimax_h3.py`; tiny model test config | `tests/local_tests/vaes/test_minimax_h3_video_vae_parity.py` | exact state surface plus activation/encode/decode/tiling parity |
| Audio VAE | `autoencoder_kl_minimax_h3_audio.py`; tiny model test config | `tests/local_tests/vaes/test_minimax_h3_audio_vae_parity.py` | exact state surface plus activation/posterior/decode parity |
| Scheduler | `scheduling_minimax_h3.py` | `tests/local_tests/minimax_h3/test_minimax_h3_scheduler_parity.py` | video/audio sigma, timestep, scale-noise, and full-step parity |
| FL2VA packing | `modular_pipelines/minimax_h3/packing.py` | `tests/local_tests/minimax_h3/test_minimax_h3_packing.py` | independent row/position/tag/RNG contracts |
| Conversion | `scripts/convert_minimax_h3_to_diffusers.py` | `tests/local_tests/minimax_h3/test_minimax_h3_conversion.py` | CLI safetensors round trip: QKV de-interleave, FFN ordering, dropped keys, keys/shapes/dtypes, strict load, and scheduler configs |
| Component loading | current FastVideo loaders | `tests/local_tests/minimax_h3/test_minimax_h3_loader_contracts.py` | actual meta-device mixed-dtype load, full audio VAE, strict video VAE, registry, and independent scheduler shifts |
| Request bridge | current FastVideo typed request path | `tests/local_tests/minimax_h3/test_minimax_h3_api_contract.py` | H3 inputs survive `GenerationRequest` to `ForwardBatch` |

Run Stage 1 locally with:

```bash
pytest \
  tests/local_tests/transformers/test_minimax_h3_transformer_parity.py \
  tests/local_tests/vaes/test_minimax_h3_video_vae_parity.py \
  tests/local_tests/vaes/test_minimax_h3_audio_vae_parity.py \
  tests/local_tests/minimax_h3 -v -s
```

Current Stage 1 result: `33 passed`, all non-skip on CPU. The existing API
translation regression subset also passes `43 passed`.

These are non-skip synthetic parity tests. They are not evidence of real-weight
compatibility, generated-media quality, memory use, or performance.

## Review notes

- The FastVideo and reference packers must remain independent; do not import one
  from the other or generate both expected and actual rows with the same helper.
- Keep mixed-precision islands explicit: Transformer input/time/output modules
  are FP32, the main block stack is BF16 in the described checkpoint, and both
  VAEs are FP32.
- Stage 1 does not add a pipeline, preset, public registry activation, Qwen3-VL
  conditioner, joint denoising loop, or generated-media claim.
- Real component and pipeline parity remains a Stage 4 blocker until accessible
  weights and their license/layout are confirmed.
