# GEN3C: 3D-Informed Camera-Controlled Video Generation

[GEN3C](https://arxiv.org/abs/2503.03751) is NVIDIA's Cosmos-7B-based video model for camera-controlled generation from a single image. The FastVideo integration supports the GEN3C I2V workflow, including 3D cache conditioning and tokenizer-based conditioning latents.

## Key Features

- **Camera trajectory control**: `left/right/up/down/zoom_in/zoom_out/clockwise/counterclockwise`
- **3D cache conditioning**: depth prediction -> point cloud cache -> forward warping -> latent conditioning
- **Single-image to video generation**: 121-frame generation with camera motion
- **Official raw checkpoint conversion**: `model.pt` -> Diffusers/FastVideo layout

## Model Sources

- Official raw checkpoint (not Diffusers): `nvidia/GEN3C-Cosmos-7B`
- Diffusers-format checkpoint: `vbharath/GEN3C-Cosmos-7B-Diffusers`

## Prerequisites

- Install MoGe:

```bash
pip install git+https://github.com/microsoft/MoGe.git
```

- If you hit `ImportError: libGL.so.1` (common on Ubuntu/headless nodes), you can try installing OpenCV runtime libs:

```bash
sudo apt-get update
sudo apt-get install -y libgl1 libglib2.0-0 libsm6 libxext6 libxrender1
```

## Quick Start

### Option A: Use Diffusers-format weights directly

```bash
python examples/inference/basic/basic_gen3c.py \
  --model_path vbharath/GEN3C-Cosmos-7B-Diffusers \
  --image_path /path/to/input.png \
  --prompt "" \
  --trajectory left \
  --movement_distance 0.3 \
  --camera_rotation center_facing \
  --num_inference_steps 35 \
  --guidance_scale 1.0 \
  --output_path outputs_video/gen3c_output.mp4
```

### Option B: Convert official raw checkpoint locally

1. Download:

```bash
huggingface-cli download nvidia/GEN3C-Cosmos-7B --local-dir official_weights/GEN3C-Cosmos-7B
```

1. Convert:

```bash
python scripts/checkpoint_conversion/convert_gen3c_to_fastvideo.py \
  --source official_weights/GEN3C-Cosmos-7B/model.pt \
  --output converted_weights/GEN3C-Cosmos-7B
```

1. Run:

```bash
python examples/inference/basic/basic_gen3c.py \
  --model_path converted_weights/GEN3C-Cosmos-7B \
  --image_path /path/to/input.png \
  --prompt "" \
  --trajectory left \
  --movement_distance 0.3 \
  --camera_rotation center_facing \
  --num_inference_steps 35 \
  --guidance_scale 1.0 \
  --output_path outputs_video/gen3c_output.mp4
```

## FastVideo Defaults

GEN3C defaults in FastVideo:

- `height=704`, `width=1280`
- `num_frames=121`
- `num_inference_steps=35`
- `guidance_scale=1.0`
- `fps=24`

These values are defined in:

- `fastvideo/configs/sample/gen3c.py`
- `fastvideo/configs/pipelines/gen3c.py`

and align with the official GEN3C inference defaults in:

- `tmp/GEN3C/cosmos_predict1/diffusion/inference/inference_utils.py`

## Scheduler Note

The converted GEN3C Diffusers layout may include a FlowMatch scheduler config, but GEN3C denoising uses EDM preconditioning behavior. FastVideo's GEN3C pipeline enforces an EDM scheduler at runtime for parity with official inference behavior.

Implementation path:

- `fastvideo/pipelines/basic/gen3c/gen3c_pipeline.py`

## 3D Cache Conditioning Path

FastVideo GEN3C conditioning stage performs:

1. MoGe depth estimation from input image
2. 3D cache initialization
3. Camera trajectory generation
4. Forward rendering of warped frames + masks
5. VAE/tokenizer encoding of conditioning buffers
6. Denoising with condition mask + condition pose channels

Main implementation:

- `fastvideo/pipelines/basic/gen3c/gen3c_pipeline.py`
- `fastvideo/pipelines/basic/gen3c/cache_3d.py`
- `fastvideo/pipelines/basic/gen3c/depth_estimation.py`
- `fastvideo/models/vaes/gen3c_tokenizer_vae.py`

## References

- [GEN3C Paper](https://arxiv.org/abs/2503.03751)
- [Official Repository](https://github.com/nv-tlabs/GEN3C)
- [Official Checkpoint (raw)](https://huggingface.co/nvidia/GEN3C-Cosmos-7B)
