# Cosmos Predict2.5 distilled validation

This port targets NVIDIA's released 2B distilled **Text2World** checkpoint. It
does not claim distilled Video2World, rolling generation, or real-time DreamVerse
support.

## Reference

- Source: `NVIDIA/Cosmos-Predict2.5` at commit
  `a2c298b0a3df3778b973fe65e9e58877b292d8a7`
- Checkpoint: `nvidia/Cosmos-Predict2.5-2B`, `base/distilled`
- Override the source checkout with `COSMOS25_OFFICIAL_REF_DIR`.

Clone the reference next to FastVideo, or point the environment variable at an
existing checkout:

```bash
git clone https://github.com/NVIDIA/Cosmos-Predict2.5.git cosmos-predict2.5
export COSMOS25_OFFICIAL_REF_DIR="$PWD/cosmos-predict2.5"
```

## CPU sampler tests

```bash
pytest fastvideo/tests/schedulers/test_cosmos25_distilled_scheduler.py -q

COSMOS25_OFFICIAL_REF_DIR=/path/to/Cosmos-Predict2.5 \
pytest tests/local_tests/cosmos25/test_cosmos25_distilled_scheduler_parity.py -v -s
```

The parity test pins NVIDIA's scaling source and compares the full four-step
preconditioning/x0/fixed-noise rollout. It does not load model weights.

## Conversion

The converter keeps only the official student's native `net.*` tensors and
reuses non-transformer components from an existing FastVideo-loadable Cosmos
Predict2.5 package. It writes its own distilled scheduler metadata instead of
inheriting the base package's UniPC scheduler.

```bash
python scripts/checkpoint_conversion/cosmos25_distilled_to_diffusers.py \
  --src-checkpoint /path/to/base/distilled/575edf0f-d973-4c74-b52c-69929a08d0a5_ema_bf16.pt \
  --base-model /path/to/Cosmos-Predict2.5-2B-Diffusers \
  --dst converted_weights/cosmos25-distilled
```

Local conversion contracts:

```bash
pytest tests/local_tests/cosmos25/test_cosmos25_distilled_conversion.py -q
```

The released checkpoint conversion and production FastVideo strict load passed
on the Spark validation host: 685 student tensors and no training counters.

## Validated GPU gates

Conversion, strict load, the real-weight DiT comparison, end-to-end T2W
generation, and decoded-frame return all pass on the Spark validation host.
Distilled V2W/rolling remains explicitly outside the initial support claim.

Run the cheap pipeline contracts, then a small wiring smoke before the full
four-step quality gate:

```bash
pytest tests/local_tests/cosmos25/test_cosmos25_distilled_pipeline.py -q

FASTVIDEO_ATTENTION_BACKEND=TORCH_SDPA \
python examples/inference/basic/basic_cosmos2_5_distilled_t2w.py \
  --model /path/to/converted-model \
  --steps 1 --frames 9 --height 256 --width 448 \
  --output outputs_video/cosmos25_distilled_smoke.mp4

FASTVIDEO_ATTENTION_BACKEND=TORCH_SDPA \
python examples/inference/basic/basic_cosmos2_5_distilled_t2w.py \
  --model /path/to/converted-model
```

For the DreamVerse frame-return contract, rerun the small smoke without MP4
output and require a nonempty decoded frame list:

```bash
FASTVIDEO_ATTENTION_BACKEND=TORCH_SDPA \
python examples/inference/basic/basic_cosmos2_5_distilled_t2w.py \
  --model /path/to/converted-model \
  --steps 1 --frames 9 --height 256 --width 448 --return-frames
```

The Spark frame-return gate produced 9 RGB frames with shape `(256, 448, 3)`.
The full four-step `704x1280x77` run completed in 143.53 seconds after model
load and passed visual inspection.

## Real student DiT parity

This gate loads the raw NVIDIA student into the official and FastVideo DiTs,
runs the same small deterministic BF16 forward through each implementation, and
compares the raw network outputs. It loads the models sequentially to limit GPU
memory use.

```bash
export COSMOS25_OFFICIAL_REF_DIR=/path/to/Cosmos-Predict2.5
export COSMOS25_DISTILLED_CHECKPOINT=/path/to/575edf0f-d973-4c74-b52c-69929a08d0a5_ema_bf16.pt

FASTVIDEO_ATTENTION_BACKEND=TORCH_SDPA \
pytest tests/local_tests/cosmos25/test_cosmos25_distilled_transformer_parity.py -v -s
```

The Spark gate passed with first-block relative mean error `0.000655` and final
relative mean error `0.038397`, with smooth BF16 drift and no discontinuity.
