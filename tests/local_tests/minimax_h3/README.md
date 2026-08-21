# MiniMax H3 validation

Local tests keep checks that need the pinned Diffusers source, published weights, or the public registry surface.
FastVideo-owned unit contracts belong under `fastvideo/tests/`.

## Reference

- Diffusers implementation: `https://github.com/huggingface/diffusers/pull/14355`
- Source checkout: `${MINIMAX_H3_OFFICIAL_REF_DIR:-$PWD/DiffusersMiniMaxH3}`
- Checkpoint: `MiniMaxAI/MiniMax-H3`

The reference helper verifies the pinned source and import origin. A missing checkout may skip a source-parity module;
that skip is not parity evidence.

## FastVideo unit contracts

```bash
pytest \
  fastvideo/tests/vaes/test_minimax_h3_video_vae_streaming.py \
  fastvideo/tests/stages/test_minimax_h3_vae_streaming.py -q
```

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
partitions and the video VAE; audio decode has maximum absolute drift `2.4e-7`. The encoder gate compares the slim
forward's selected layer-50 hidden state bit-exactly against the same state from the official full stack across text,
image, and video inputs.

The video VAE test verifies the reference checkout at commit
`abc5e9bf71fd38f53cd471bc3acaa84bc5ecbfdc` and compares the production CPU `uint8` `encode_pixels()` path against
the official posterior element by element.

## Video VAE memory benchmark

The benchmark uses one warmup and three measured runs with `vae_cpu_offload=True`. It reports absolute and
stage-incremental allocated/reserved CUDA peaks for every rank. For SP runs, the reported aggregate is explicitly the
sum of rank-local maxima, not a simultaneous node peak.

```bash
python tests/local_tests/vaes/benchmark_minimax_h3_video_vae_memory.py \
  --source-root "$PWD" --model-root "$MINIMAX_H3_MODEL_ROOT" \
  --revision-label candidate --operation encode

python -m torch.distributed.run --nproc_per_node=4 \
  tests/local_tests/vaes/benchmark_minimax_h3_video_vae_memory.py \
  --source-root "$PWD" --model-root "$MINIMAX_H3_MODEL_ROOT" \
  --revision-label candidate-sp4 --operation decode
```

Run the same script with `--source-root` pointed at the base checkout for a comparable baseline. The default workload
is deterministic `124 x 768 x 1344` video geometry with seed `20260803`; the JSON record includes source/model
revisions, software/allocator metadata, exact measurement boundaries, per-repetition values, and output shapes.

FastVideo joint audio/video generation and SP=1/SP=4 latent consistency have been validated. T2VA, FL2VA, and
Ref2VA video/audio latents match the pinned Diffusers pipeline exactly.
