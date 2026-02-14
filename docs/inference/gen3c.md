# GEN3C: 3D-Informed Camera-Controlled Video Generation

[GEN3C](https://arxiv.org/abs/2503.03751) is NVIDIA's 7B-parameter video diffusion model that generates camera-controlled videos from a single image using a 3D scene cache. It builds on the Cosmos architecture with additional conditioning for camera trajectories and 3D scene understanding.

## Key Features

- **Camera Control**: Generate videos with explicit camera trajectories (orbit, zoom, dolly, etc.)
- **3D Cache**: Uses depth-estimated point clouds for geometrically consistent motion
- **Single Image Input**: Generates multi-view consistent video from one reference image
- **Autoregressive Generation**: Supports extending videos beyond the initial 121 frames
- **7B Parameters**: Based on Cosmos-Predict1 architecture with 28 transformer blocks

## Requirements

- **GPU**: NVIDIA GPU with at least 24 GB VRAM (A40, A100, H100, etc.)
- **Disk**: ~30 GB for model weights
- **Dependencies**: FastVideo with standard installation

## Quick Start

### 1. Convert Official Weights

GEN3C uses an official `model.pt` checkpoint that must be converted to FastVideo's safetensors format:

```bash
# Download and convert (requires ~30 GB disk + ~15 GB GPU memory)
python scripts/checkpoint_conversion/convert_gen3c_to_fastvideo.py \
    --download nvidia/GEN3C-Cosmos-7B \
    --output ./gen3c_fastvideo
```

This will:

1. Download the official checkpoint from HuggingFace (`nvidia/GEN3C-Cosmos-7B`)
2. Apply key remapping to match FastVideo's naming conventions
3. Save as `model.safetensors` along with a `model_index.json`

### 2. Generate Video

```python
from fastvideo import VideoGenerator

generator = VideoGenerator.from_pretrained(
    "gen3c_fastvideo",
    num_gpus=1,
)

prompt = "A camera slowly orbits around a vase of flowers on a table."

result = generator.generate_video(
    prompt,
    height=720,
    width=1280,
    num_frames=121,
    num_inference_steps=50,
    guidance_scale=6.0,
    fps=24,
    seed=42,
    output_path="outputs/gen3c/",
)
```

## Architecture Overview

### Model Components

| Component | Implementation | Notes |
|-----------|---------------|-------|
| **DiT** | `Gen3CTransformer3DModel` | 7B params, 28 blocks, AdaLN-LoRA |
| **VAE** | Cosmos/Wan VAE | 16 latent channels, 8x spatial / 4x temporal compression |
| **Text Encoder** | T5-Large | 1024-dim embeddings, max 512 tokens |
| **Scheduler** | Flow Matching Euler | sigma_max=80, sigma_min=0.002 |

### 3D Cache Conditioning

GEN3C conditions generation on 3D scene information through a cache mechanism:

1. **Depth Estimation**: MoGe estimates monocular depth from the input image
2. **Point Cloud**: Depth + RGB creates an unprojected 3D point cloud
3. **Camera Trajectory**: New camera poses are generated along a trajectory
4. **Forward Warping**: The 3D cache is rendered from new viewpoints
5. **VAE Encoding**: Rendered images and masks are encoded to latent space
6. **Conditioning**: Encoded buffers are concatenated with the noise latent

The DiT receives these additional inputs at each denoising step:

- `condition_video_input_mask` (1 channel): Binary mask indicating conditioning frames
- `condition_video_pose` (64 channels): VAE-encoded 3D cache buffers (2 buffers x 32 channels)
- `condition_video_augment_sigma`: Noise augmentation level for conditioning frames

### DiT Modifications from Cosmos

GEN3C extends the Cosmos 2.5 DiT with:

- **Extended Patch Embedding**: Accepts 82 input channels (16 latent + 1 mask + 64 buffers + 1 padding)
- **AdaLN-LoRA**: Separate AdaLN modulation for self-attention, cross-attention, and MLP
- **3D RoPE**: Rotary positional embeddings with per-axis learnable additions
- **QK Normalization**: RMSNorm applied to Q and K projections

## Configuration

### Pipeline Configuration

Key parameters in `Gen3CConfig`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `frame_buffer_max` | 2 | Number of 3D cache buffers |
| `noise_aug_strength` | 0.0 | Noise augmentation on conditioning frames |
| `fps` | 24 | Generation frame rate |
| `num_frames` | 121 | Frames to generate (pixel space) |
| `video_resolution` | (720, 1280) | Output resolution (H, W) |
| `flow_shift` | 1.0 | Flow matching shift parameter |
| `guidance_scale` | 6.0 | Classifier-free guidance scale |
| `num_inference_steps` | 50 | Denoising steps |

### Sampling Parameters

Default sampling parameters are defined in `Gen3C_Cosmos_7B_SamplingParam`:

```python
height: int = 720
width: int = 1280
num_frames: int = 121
fps: int = 24
guidance_scale: float = 6.0
num_inference_steps: int = 50
```

## Weight Conversion Details

The conversion script (`scripts/checkpoint_conversion/convert_gen3c_to_fastvideo.py`) transforms the official checkpoint's key naming:

| Official Pattern | FastVideo Pattern |
|-----------------|-------------------|
| `net.x_embedder.proj.1.*` | `patch_embed.proj.*` |
| `net.t_embedder.1.linear_*.*` | `time_embed.t_embedder.linear_*.*` |
| `net.blocks.blockN.blocks.0.block.attn.to_q.0.*` | `transformer_blocks.N.attn1.to_q.*` |
| `net.blocks.blockN.blocks.0.block.attn.to_q.1.*` | `transformer_blocks.N.attn1.norm_q.*` |
| `net.blocks.blockN.blocks.1.block.attn.to_q.0.*` | `transformer_blocks.N.attn2.to_q.*` |
| `net.blocks.blockN.blocks.2.block.layer1.*` | `transformer_blocks.N.mlp.fc_in.*` |
| `net.final_layer.linear.*` | `final_layer.proj_out.*` |

Keys not mapped (safely ignored): `net.pos_embedder.*`, `net.accum_*`, `logvar.*`.

## Numerical Parity

The FastVideo implementation achieves close numerical parity with the official GEN3C model:

- **Max absolute difference**: ~0.015 (in bf16)
- **Mean absolute difference**: ~0.002

The small remaining difference is due to different attention backends (official uses Transformer Engine's `DotProductAttention`; FastVideo uses `torch.nn.functional.scaled_dot_product_attention`) accumulated over 28 transformer blocks in bf16 precision.

## Limitations

- **3D Cache Pipeline**: The full 3D cache pipeline (depth estimation, point cloud management, forward warping) requires the official GEN3C repository utilities. The FastVideo integration currently focuses on the DiT and denoising pipeline.
- **Single GPU**: Currently tested on single-GPU configurations. Multi-GPU support follows FastVideo's standard distribution patterns.

## References

- [GEN3C Paper](https://arxiv.org/abs/2503.03751)
- [Official Repository](https://github.com/nv-tlabs/GEN3C)
- [HuggingFace Weights](https://huggingface.co/nvidia/GEN3C-Cosmos-7B)
