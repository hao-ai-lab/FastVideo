# Pika H3 Flex

Pika H3 Flex is a post-trained MiniMax-H3 model that keeps H3's packed
video/audio transformer, Qwen3-VL conditioner, and video/audio VAEs. Pika ships
two rank-128 PEFT adapters instead of a second base checkpoint:

| Adapter | Trunk | Tasks |
|---|---|---|
| `fl2va/adapter.safetensors` | `transformer` | T2VA, I2VA, L2VA, FL2VA |
| `ref2va/adapter.safetensors` | `transformer_ref` | Ref2VA |

FastVideo detects `Pika-Labs/Pika-H3-Flex`, downloads only the selected task's
adapter, and applies it as an ordinary LoRA. The released adapter is selected
automatically; pass `lora_path` only when overriding it with a compatible
Pika adapter.

## Sampling contract

Pika's sampler is fixed, not a tunable step count:

* 6 transformer forwards over clean-time rungs
  `1000, 750, 500, 250, 100, 40, 0`;
* video shift `12.0`, audio shift `3.0`;
* no CFG, `guidance_scale=1.0`, empty negative prompt;
* each step re-noises the predicted clean sample with fresh Gaussian noise from
  a generator seeded `seed + 1`.

`num_inference_steps` must therefore be `6`; other values are rejected. This is
not a PDD checkpoint and it has no trained VSA gates, so PDD fused blocks and
VSA sparsity are not used.

## Tasks

```bash
# text to video + audio
python examples/inference/basic/basic_pika_h3_flex.py --task t2va \
    --prompt "a paper boat drifting down a rainy street"

# first frame, last frame, or both
python examples/inference/basic/basic_pika_h3_flex.py --task i2va \
    --image first.png --prompt "the camera slowly pushes in"
python examples/inference/basic/basic_pika_h3_flex.py --task fl2va \
    --image first.png --last-image last.png --prompt "the flower opens"

# ordered references, including reference video and audio
python examples/inference/basic/basic_pika_h3_flex.py --task ref2va \
    --reference image:person.png --reference video:dance.mp4 \
    --reference audio:voice.wav --prompt "the person dances to the music"
```

Ref2VA accepts up to 12 references: 9 images, 3 videos, and 3 audio. These
counts are enforced locally, as is the rule that audio-only requests are
invalid. A video reference needs at least 22 decoded frames at 24 fps. At least
one image or video is required, and the Ref2VA pipeline must be selected with
`PikaH3FlexRef2VAModularPipeline` (the example does this).

Pika's hosted API also documents a 2–15 s limit per video/audio reference and a
15 s total per modality. **The local path does not currently enforce those
duration caps**; it trims each reference to the requested output duration. Treat
the per-reference and per-modality limits as recommendations until the checks
are added.

## Canvas and frames

Pika uses an exact 32-pixel bucket lattice. A request that matches a bucket is
kept exactly; otherwise FastVideo picks the tier from the requested short edge
(`<= 480` → p480, else p768) and snaps to the nearest aspect bucket in that tier.

| Tier | Short edge | Max pixels | Reference-image short edge |
|---|---|---|---|
| p480 | 480 | 832×480 | 1024 |
| p768 | 768 | 1344×768 | 2048 |

Frame counts follow Pika's `17n + 5` grid at 24 fps (5–15 s, 120–362 aligned
frames). The default 16:9 p768 canvas is 1344×768 and 124 frames.

Pika center-crops both keyframes to the bucket. MiniMax-H3 stretches its first
frame instead; do not reuse that fit for Pika.

## Training-free acceleration

The 6 forwards already use FastVideo's normal inference path. Available
lossless switches:

* `FASTVIDEO_MINIMAX_H3_EXACT_KERNELS=all` — exact fused H3 kernels;
* `FASTVIDEO_H3_REF2VA_MEMO_ENTRIES=16` — cache reference conditioning
  (Ref2VA);
* `FASTVIDEO_H3_VAE_TILE_PARALLEL=1` — parallel VAE tiles;
* `FASTVIDEO_ULYSSES_A2A=auto` — Ulysses all-to-all for multi-GPU sequence
  parallelism;
* `OffloadConfig(lazy_module_load=True)` and `dit_layerwise=True` for smaller
  GPUs;
* `CompileConfig(enabled=True)` for `torch.compile`;
* the standard FlashAttention/SDPA backend selection.

`--lossless-accel` in the example enables the first four before the generator
starts. Kernel work beyond the existing exact kernels is intentionally not part
of this integration.

## Hardware

The selected H3 trunk plus Qwen3-VL does not fit comfortably on a 24 GB card
without offload. A single 80 GB GPU can run with text-encoder offload; use
layerwise DiT offload or multiple GPUs for smaller devices. Ref2VA encodes
reference images at up to a 2048 short edge, so its conditioning memory is
higher than T2VA's.
