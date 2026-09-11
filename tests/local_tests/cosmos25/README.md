# Cosmos Predict2.5 distilled validation

This port targets two complementary Cosmos Predict2.5 2B students:

- NVIDIA's released distilled **Text2World** checkpoint for the first segment.
- The Data-Forcing Distillation (DFD) **Video2World** checkpoint for one-frame
  continuation segments.

The DreamVerse consumer uses the T2W student once, then passes each decoded
terminal frame to the DFD student. DFD is not treated as a causal or native
rolling model; each continuation is a fresh one-frame-conditioned sample.

## Reference

- Source: `NVIDIA/Cosmos-Predict2.5` at commit
  `a2c298b0a3df3778b973fe65e9e58877b292d8a7`
- Checkpoint: `nvidia/Cosmos-Predict2.5-2B`, `base/distilled`
- Override the source checkout with `COSMOS25_OFFICIAL_REF_DIR`.

DFD reference:

- Source: `csy2077/data-forcing-distillation` at commit
  `de6416cac1e06562d29aaf96a13bb0ab99099cdf`
- Checkpoint: `csusupergear/cosmos_i2v_checkpoints`,
  `cosmos_dfd_checkpoints/0000040.net_model.zip`
- Override the checkout with `COSMOS25_DFD_REF_DIR` and the extracted DCP
  directory with `COSMOS25_DFD_CHECKPOINT_DIR`.

Clone the reference next to FastVideo, or point the environment variable at an
existing checkout:

```bash
git clone https://github.com/NVIDIA/Cosmos-Predict2.5.git cosmos-predict2.5
export COSMOS25_OFFICIAL_REF_DIR="$PWD/cosmos-predict2.5"
```

The local DFD checkout is staged through the repository add-model workflow:

```bash
git clone https://github.com/csy2077/data-forcing-distillation.git DFDReference
export COSMOS25_DFD_REF_DIR="$PWD/DFDReference"
```

No DFD weights are kept in the repository. The Hugging Face layout is a custom
PyTorch distributed checkpoint with no root `model_index.json`, so it requires
conversion before FastVideo can load it.

## DFD reference contract

The published I2V configuration uses BF16, one conditioning latent frame,
1280x704 output, 81 decoded frames, 24 FPS temporal encoding, and four ODE
student steps at normalized rectified-flow times:

```text
[0.999, 0.937, 0.833, 0.624, 0.0]
```

The inference-only path must load the student transformer directly. The
teacher, fake-score network, discriminator, optimizers, and training counters
are not FastVideo runtime components.

## CPU sampler tests

```bash
pytest fastvideo/tests/schedulers/test_cosmos25_distilled_scheduler.py -q

COSMOS25_OFFICIAL_REF_DIR=/path/to/Cosmos-Predict2.5 \
pytest tests/local_tests/cosmos25/test_cosmos25_distilled_scheduler_parity.py -v -s

pytest fastvideo/tests/schedulers/test_cosmos25_dfd_scheduler.py -q

COSMOS25_DFD_REF_DIR=/path/to/data-forcing-distillation \
pytest tests/local_tests/cosmos25/test_cosmos25_dfd_scheduler_parity.py -v -s
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

Convert the extracted DFD distributed checkpoint without constructing the
teacher, fake-score network, or discriminator:

```bash
python scripts/checkpoint_conversion/cosmos25_dfd_to_diffusers.py \
  --src-dcp /path/to/0000040.net_model \
  --base-model /path/to/Cosmos-Predict2.5-2B-Diffusers \
  --dst converted_weights/cosmos25-dfd-v2w

pytest tests/local_tests/cosmos25/test_cosmos25_dfd_conversion.py -q
```

The DFD converter materializes the PyTorch DCP state, keeps only inference
transformer tensors, enables FPS-modulated RoPE, and writes the fixed DFD
scheduler metadata. A strict load of the real converted package is still a
required validation gate.

Run the isolated DFD pipeline contracts and full real-weight four-step rollout
parity after conversion/component tests:

```bash
pytest tests/local_tests/cosmos25/test_cosmos25_dfd_pipeline.py -q

COSMOS25_DFD_REF_DIR=/path/to/data-forcing-distillation \
COSMOS25_DFD_CHECKPOINT_DIR=/path/to/0000040.net_model \
FASTVIDEO_ATTENTION_BACKEND=TORCH_SDPA \
pytest tests/local_tests/cosmos25/test_cosmos25_dfd_pipeline_parity.py -v -s
```

Then run the converted package at its production contract:

```bash
FASTVIDEO_ATTENTION_BACKEND=TORCH_SDPA \
python examples/inference/basic/basic_cosmos2_5_dfd_i2w.py \
  --model /path/to/converted_weights/cosmos25-dfd-v2w \
  --image /path/to/conditioning.png \
  --prompt "The camera pivots smoothly to the right while the scene continues." \
  --output outputs_video/cosmos25_dfd_i2w.mp4
```

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

The corresponding DFD gate exercises the released V2W wrapper semantics: it
replaces the first noisy latent with the clean image latent, sets that frame's
timestep to zero, includes the condition-mask channel, and enables 24 FPS RoPE
modulation in both implementations.

```bash
export COSMOS25_DFD_REF_DIR=/path/to/data-forcing-distillation
export COSMOS25_DFD_CHECKPOINT_DIR=/path/to/0000040.net_model

FASTVIDEO_ATTENTION_BACKEND=TORCH_SDPA \
pytest tests/local_tests/cosmos25/test_cosmos25_dfd_transformer_parity.py -v -s
```

The non-skip DFD DiT gate passed with maximum absolute error `0.15625`,
mean absolute error `0.01315392`, and relative mean error `0.01986194`.
