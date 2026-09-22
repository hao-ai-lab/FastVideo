# Wan-VACE: Controllable Video Generation

[Wan2.1-VACE](https://github.com/Wan-Video/Wan2.1) is Alibaba's controllable video
generation model. FastVideo supports both 1.3B (480P) and 14B (480P/720P) variants
for reference-image conditioning and video+mask control.

## Model Sources

| Variant | HuggingFace ID | Default Resolution |
|---------|----------------|-------------------|
| 1.3B | `Wan-AI/Wan2.1-VACE-1.3B-diffusers` | 480×832 |
| 14B | `Wan-AI/Wan2.1-VACE-14B-diffusers` | 720×1280 |

## Input Modes

1. **Reference-only** (official `generate.py` example): pass `references` with no
   `video_path`. The pipeline synthesizes zero-pixel video and an all-ones mask.
2. **Video + mask**: pass `video_path` and optional `mask_path` for controllable
   editing.
3. **Video + references**: combine source video with reference images for
   subject/style transfer.

## Official Parameters

The official [Wan2.1 `generate.py`](https://github.com/Wan-Video/Wan2.1) CLI uses
these defaults for VACE (all resolutions):

| Parameter | Official | FastVideo location |
|-----------|----------|-------------------|
| `sample_shift` | 16.0 | `WanVACE*_Config.flow_shift` |
| `sampling_steps` | 50 | preset `num_inference_steps` |
| `guide_scale` | 5.0 | preset `guidance_scale` |
| `sample_neg_prompt` | Chinese | preset `negative_prompt` |
| `conditioning_scale` | 1.0 | `SamplingParam.conditioning_scale` |

These differ from Wan T2V defaults (`flow_shift` 3/5, English negative prompt).

## Quick Start

```bash
python examples/inference/basic/basic_wan_vace.py \
  --model_path Wan-AI/Wan2.1-VACE-1.3B-diffusers \
  --output_path outputs/wan_vace/official_1_3b_480.mp4
```

Reference images (`girl.png`, `snake.png`) are downloaded automatically from the
official Wan2.1 repository on first run.

### 14B at 720P

```bash
python examples/inference/basic/basic_wan_vace.py \
  --model_path Wan-AI/Wan2.1-VACE-14B-diffusers \
  --height 720 --width 1280 \
  --output_path outputs/wan_vace/official_14b_720.mp4
```

## FastVideo Defaults

Defined in:

- `fastvideo/models/wan/pipeline_config.py` (`WanVACE1_3B_Config`, `WanVACE14B_Config`)
- `fastvideo/pipelines/basic/wan/presets.py` (`WAN_VACE_1_3B`, `WAN_VACE_14B`)

## Pipeline Architecture

```
InputValidation → TextEncoding → Conditioning → VACEInput → TimestepPrep
  → VACELatentPrep → VACEContext → Denoising → VACEDecoding
```

- **VACEInput**: loads mask video, preprocesses reference images, synthesizes
  zero-pixel video for reference-only mode.
- **VACEContext**: VAE-encodes video/mask/reference into 96-channel
  `control_hidden_states`.
- **VACEDenoising**: passes `control_hidden_states` to the DiT (no channel concat).
- **VACEDecoding**: strips reference-frame latents before VAE decode.

## Reproducibility Matrix

Run all three resolution × parameter combinations:

```bash
python scripts/run_vace_matrix.py
```

Each run produces an 81-frame MP4 with `seed=0`. Results are written to
`outputs/vace_matrix/results.json`.

## Local Tests

```bash
# Weight-free registry/preset contracts (CI-excluded)
pytest fastvideo/tests/api/test_wan_vace_definitions.py -q

# Import/registry smoke (no GPU)
pytest tests/local_tests/pipelines/test_wan_vace_pipeline_smoke.py -q -k preflight

# Transformer forward parity (GPU + weights)
WAN_VACE_MODEL_DIR=/path/to/Wan2.1-VACE-1.3B-diffusers \
  pytest tests/local_tests/pipelines/test_wan_vace_pipeline_parity.py -v -s

# End-to-end load/generate smoke (GPU + weights)
WAN_VACE_MODEL_DIR=/path/to/Wan2.1-VACE-1.3B-diffusers \
  pytest tests/local_tests/pipelines/test_wan_vace_pipeline_smoke.py -v -s -k load_generate
```

## Known Gaps

- No SSIM regression baseline yet (quality verified via official-case matrix runs).
- Sequence parallelism (`sp_size > 1`) not validated for VACE.
- Video+mask control path not covered by the official matrix script.

## References

- [Wan2.1 Repository](https://github.com/Wan-Video/Wan2.1)
- [Wan2.1-VACE-1.3B on HuggingFace](https://huggingface.co/Wan-AI/Wan2.1-VACE-1.3B-diffusers)
- [Wan2.1-VACE-14B on HuggingFace](https://huggingface.co/Wan-AI/Wan2.1-VACE-14B-diffusers)
