# MiniMax H3 validation

Local tests keep checks that need the pinned Diffusers source, published weights, or the public registry surface.
FastVideo-owned unit contracts belong under `fastvideo/tests/`.

## Reference

- Diffusers implementation: `https://github.com/huggingface/diffusers/pull/14355`
- Source checkout: `${MINIMAX_H3_OFFICIAL_REF_DIR:-$PWD/DiffusersMiniMaxH3}`
- Checkpoint: `MiniMaxAI/MiniMax-H3`

The reference helper verifies the pinned source and import origin. A missing checkout may skip a source-parity module;
that skip is not parity evidence.

## Registry smoke

```bash
pytest tests/local_tests/pipelines/test_minimax_h3_pipeline_smoke.py -q
```

## Pinned implementation parity

```bash
PYTHONPATH="${MINIMAX_H3_OFFICIAL_REF_DIR:-$PWD/DiffusersMiniMaxH3}/src:$PWD" pytest \
  tests/local_tests/minimax_h3/test_minimax_h3_scheduler_parity.py \
  tests/local_tests/minimax_h3/test_minimax_h3_packing.py \
  tests/local_tests/minimax_h3/test_minimax_h3_ref2va_packing.py \
  tests/local_tests/minimax_h3/test_minimax_h3_ref2va_media.py -v -s
```

## Checkpoint component parity

```bash
export MINIMAX_H3_MODEL_ROOT=/path/to/MiniMax-H3
export MINIMAX_H3_OFFICIAL_REF_DIR=/path/to/DiffusersMiniMaxH3

PYTHONPATH="$MINIMAX_H3_OFFICIAL_REF_DIR/src:$PWD" \
MINIMAX_H3_RUN_ENCODER_PARITY=1 \
pytest tests/local_tests/encoders/test_minimax_h3_qwen3_vl_parity.py -v -s

PYTHONPATH="$MINIMAX_H3_OFFICIAL_REF_DIR/src:$PWD" \
MINIMAX_H3_RUN_DIT_PARITY=1 \
MINIMAX_H3_RUN_VIDEO_VAE_PARITY=1 \
MINIMAX_H3_RUN_AUDIO_VAE_PARITY=1 \
pytest \
  tests/local_tests/transformers/test_minimax_h3_transformer_parity.py \
  tests/local_tests/vaes/test_minimax_h3_video_vae_parity.py \
  tests/local_tests/vaes/test_minimax_h3_audio_vae_parity.py -v -s
```

With a gate enabled, missing CUDA, source, or weights is a failure. Recorded component evidence is exact for both DiT
partitions, the video VAE, and all Qwen3-VL hidden states; audio decode has maximum absolute drift `2.4e-7`.

FastVideo joint audio/video generation and SP=1/SP=4 latent consistency have been validated. T2VA, FL2VA, and
Ref2VA video/audio latents match the pinned Diffusers pipeline exactly.
