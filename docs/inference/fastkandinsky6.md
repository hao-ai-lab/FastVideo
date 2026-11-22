# FastKandinsky 6.0 NVFP4 inference

FastKandinsky 6.0 Pro generates a five-second video with synchronized audio
from a text prompt. The NVFP4 example runs eight pi-Flow steps with guidance 1.0.

## Requirements

Use one RTX 5090 or RTX PRO 6000 Blackwell (SM120) on Linux x86-64. In your
active `uv` environment, use a CUDA 13 FastVideo installation with FlashInfer
and the prebuilt CUDA 13 `fastvideo-kernel` wheel that includes
`ATTN_QAT_INFER`. See [NVIDIA GPU setup](../getting_started/installation/gpu.md).

## Export the BF16 checkpoint

Run from the FastVideo checkout with the `uv` environment active. Download the
BF16 checkpoint and convert it once on the Blackwell GPU.

```bash
hf download kandinskylab/FastKandinsky-6.0-Pro-NVPF4-QAD \
  --local-dir ./FastKandinsky-6.0-Pro-NVPF4-QAD

python scripts/checkpoint_conversion/convert_kandinsky6_nvfp4.py \
  --src ./FastKandinsky-6.0-Pro-NVPF4-QAD
```

The converter writes `nvfp4/transformer/` inside the local model directory.
It contains `nvfp4_weights.safetensors`, the remaining dense shards, their
index, and the transformer configuration. If the BF16 checkpoint is already
on disk, use its directory in both commands that take a model path.

## Generate a clip

Pass the same local model directory. FastVideo loads the shared components
from its root and the packed transformer from `nvfp4/transformer/`.

```bash
python examples/inference/optimizations/nvfp4_qat_kandinsky6.py \
  --model-path ./FastKandinsky-6.0-Pro-NVPF4-QAD \
  --prompt "cinematic shot: a giant stone samurai on a stormy cliff above a neon city opens glowing golden eyes and raises a katana. Blue lightning strikes the blade, creating a massive shockwave through the clouds. The camera rapidly pulls back from a low angle. Photorealistic, epic scale, dark blue and gold lighting, rain, sparks, volumetric lightning, blockbuster quality. Audio: heavy rain, deep thunder, metallic sword hum, rising brass and choir, electrical crackle, perfectly synchronized lightning impact, sub-bass shockwave. No dialogue, text, or logos."
```

The script saves a 768 × 512 MP4 with audio at 24 fps under
`outputs_video/kandinsky6_nvfp4/` by default. It uses FlashInfer for NVFP4
linears and `ATTN_QAT_INFER` for native FP4 attention, with DiT compilation
and layerwise offload enabled.
