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

## Conversion scaffold

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

This scaffold is locally tested, but conversion of the released 4 GB checkpoint
and production-loader strictness have not yet been validated.

## Remaining GPU gates

Before wiring a public pipeline or running DreamVerse:

1. Run the converter on the released distilled transformer and verify its
   reported tensor count and production-loader missing/unexpected keys.
2. Compare one real student DiT forward against the official implementation.
3. Compare deterministic T2W latents end to end for the official four-step
   schedule.
4. Only after T2W parity, evaluate experimental V2W/rolling conditioning.

Those checks require the released checkpoint and a CUDA machine. A skipped
local parity test is not pass evidence.
