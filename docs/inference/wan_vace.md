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

1. **Pure T2V**: omit source video, mask, and references. The pipeline supplies
   zero-pixel video and an all-ones mask.
2. **Reference-only** (official `generate.py` example): pass `references` with no
   `video_path`. The pipeline synthesizes zero-pixel video and an all-ones mask.
3. **Video + mask**: pass `video_path` and optional `mask_path` for controllable
   editing.
4. **Video + references**: combine source video with reference images for
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
  → VACEContext → VACELatentPrep → Denoising → VACEDecoding
```

- **VACEInput**: aligns mask frames, preserves reference-image aspect ratio, and
  synthesizes zero-pixel video when no source video is supplied.
- **VACEContext**: VAE-encodes video/mask/reference into 96-channel
  `control_hidden_states`.
- **VACEDenoising**: passes `control_hidden_states` to the DiT (no channel concat).
- **VACEDecoding**: strips reference-frame latents before VAE decode.

VACE uses dense attention. A sparse attention request (including VSA) is
rejected during transformer construction, before checkpoint weights load.

## Reproducibility Matrix

Run all three resolution × parameter combinations:

```bash
python scripts/run_vace_matrix.py
```

Each run produces an 81-frame MP4 with `seed=0`. Results are written to
`outputs/vace_matrix/results.json`.

## Local Tests

```bash
# Weight-free registry, input, and backend contracts (Buildkite unit lane)
pytest fastvideo/tests/api/test_wan_vace_definitions.py \
  fastvideo/tests/api/test_wan_vace_inputs.py \
  fastvideo/tests/api/test_wan_vace_backend.py -q

# Import/registry smoke (no GPU)
pytest tests/local_tests/pipelines/test_wan_vace_pipeline_smoke.py -q -k preflight

# Transformer forward parity (GPU + weights)
WAN_VACE_MODEL_DIR=/path/to/Wan2.1-VACE-1.3B-diffusers \
  pytest tests/local_tests/pipelines/test_wan_vace_pipeline_parity.py -v -s

# Small pipeline-level parity against Diffusers (GPU + local 1.3B/14B weights)
WAN_VACE_MODEL_DIR=/path/to/Wan2.1-VACE-1.3B-diffusers \
WAN_VACE_14B_MODEL_DIR=/path/to/Wan2.1-VACE-14B-diffusers \
  pytest tests/local_tests/pipelines/test_wan_vace_end_to_end_parity.py -v -s

# End-to-end load/generate smoke (GPU + weights)
WAN_VACE_MODEL_DIR=/path/to/Wan2.1-VACE-1.3B-diffusers \
  pytest tests/local_tests/pipelines/test_wan_vace_pipeline_smoke.py -v -s -k load_generate
```

## Parity Evidence

End-to-end gates (`test_wan_vace_end_to_end_parity.py`):

| Gate | Check |
|------|-------|
| Conditioning inputs | bf16 bitwise vs Diffusers |
| Step0 noise prediction | `assert_dit_parity` (atol/rtol 0.1) |
| Final latent | abs-mean drift &lt; 5% after 2 denoise steps |

Coverage: 1.3B t2v, reference, video+mask; 14B reference.

14B reference step0 block hooks (aligned inputs, real reference image) show
accumulating per-block bf16 drift while the overall noise prediction still passes
the DiT gate. No VACE-only wiring bug or shared Wan DiT regression was observed.

## Known Gaps

- No SSIM regression baseline yet. The matrix verifies runnable outputs and frame
  counts, not pixel-level quality against an HF reference video.
- End-to-end numerics use the VACE gates above. Tight `1e-2` latent parity across
  all steps is not expected because bf16 attention drift compounds over the
  denoising loop.
- `WanVACETransformer3DModel.forward` still duplicates much of
  `WanTransformer3DModel.forward`; consolidating would require shared forward
  hooks and is deferred.
- Sequence parallelism (`sp_size > 1`) not validated for VACE.
- Video+mask control path not covered by the official matrix script.

## References

- [Wan2.1 Repository](https://github.com/Wan-Video/Wan2.1)
- [Wan2.1-VACE-1.3B on HuggingFace](https://huggingface.co/Wan-AI/Wan2.1-VACE-1.3B-diffusers)
- [Wan2.1-VACE-14B on HuggingFace](https://huggingface.co/Wan-AI/Wan2.1-VACE-14B-diffusers)
