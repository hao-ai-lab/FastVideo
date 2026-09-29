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

Checked-in presets use `fps=16`.

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

# SP2 forward parity (two GPUs, no weights)
pytest fastvideo/tests/distributed/test_sp_wan_vace.py -v -s

# Local SP2 pipeline comparison (two GPUs + cached weights; cases xfail)
WAN_VACE_SP_E2E=1 \
WAN_VACE_MODEL_DIR=/path/to/Wan2.1-VACE-1.3B-diffusers \
WAN_VACE_14B_MODEL_DIR=/path/to/Wan2.1-VACE-14B-diffusers \
  pytest tests/local_tests/pipelines/test_wan_vace_sp_pipeline.py -v -s

# End-to-end load/generate smoke (GPU + weights)
WAN_VACE_MODEL_DIR=/path/to/Wan2.1-VACE-1.3B-diffusers \
  pytest tests/local_tests/pipelines/test_wan_vace_pipeline_smoke.py -v -s -k load_generate
```

## Parity Evidence

End-to-end gates (`test_wan_vace_end_to_end_parity.py`; B200, `TORCH_SDPA`, 5 frames at 64×64,
2 denoising steps, fixed initial latents, no CFG):

| Gate | Check |
|------|-------|
| Conditioning inputs | bf16 bitwise vs Diffusers (prompt, control, video/mask/references, timesteps, initial latents) |
| Every step, teacher-forced | FastVideo prediction vs the Diffusers DiT run on FastVideo's exact inputs for that step: `atol=rtol=0.05`, abs-mean drift &lt; 1% |
| Final latent (free-running) | abs-mean drift &lt; 4% |

| Case | Max abs, step 0 / step 1 (teacher-forced) | Drift, step 0 / step 1 | Final drift |
|------|------:|------:|------:|
| 1.3B t2v | 0.0156 / 0.0156 | 0.20% / 0.18% | 2.53% |
| 1.3B reference | 0.0156 / 0.0156 | 0.20% / 0.18% | 2.57% |
| 1.3B video+mask | 0.0156 / 0.0313 | 0.18% / 0.19% | 2.34% |
| 14B reference | 0.0313 / 0.0313 | 0.11% / 0.09% | 1.79% |

The component test (`test_wan_vace_pipeline_parity.py`) runs the DiT alone at an integer and a
fractional timestep. FP32 matches to about 2e-6; BF16 matches to within 1 bf16 ulp
(max 0.008, drift about 0.5%).

**Why free-running latents drift about 2–3% although every step matches.** The BF16 DiT itself
turns a 1-ulp change in its input into about 2% output drift. Feeding FastVideo's step-1 input
into the *Diffusers* model reproduces the same 2.2% difference, while FastVideo and Diffusers on
identical inputs differ by only 0.18%. The free-running gate therefore bounds accumulation;
the teacher-forced gate is the implementation check.

**Timestep embedding.** FastVideo's shared `timestep_embedding` computes the sinusoidal
frequencies on the CPU, while Diffusers computes them on the GPU. The two `exp` results differ
by an ulp. At fractional timesteps that flips BF16 rounding in the modulation and adds about
2.5% drift (14B step 0 was 0.156 max / 1.1% drift before the fix). Wan-VACE computes the
frequencies on the input device (`WanTimeTextImageEmbedding(timestep_freqs_on_input_device=True)`).
The default is unchanged for other models so their bitwise goldens stay valid.

## Sequence Parallel Status

**FP32 weight-free SP2** (`fastvideo/tests/distributed/test_sp_wan_vace.py`) passes
single-GPU versus SP2 at `atol=rtol=1e-5`, including odd-token padding, shorter
control sequences, gather/unpad, and both ranks.

**Weighted BF16 short pipeline** (`test_wan_vace_sp_pipeline.py`, `WAN_VACE_SP_E2E=1`, 5 frames
at 64×64, two steps, fixed latents). Conditioning inputs match exactly between single-GPU and
SP2, and both SP ranks agree. BF16 `F.linear` results depend on the GEMM row count, and SP
halves the rows per rank; the first traced difference is VACE block 0's FFN output projection,
after bitwise-equal Q/K/V and attention outputs. The BF16 DiT then amplifies these 1-ulp
differences as described above. The gates therefore follow repo norms rather than elementwise
equality: per-step and final drift &lt; 5%, decoded-frame SSIM ≥ 0.93 (the Wan T2V SP2 SSIM gate
is 0.93).

| Case | Final latent drift | Frame SSIM |
|------|-------------------:|-----------:|
| 1.3B reference | 3.51% | 0.957 |
| 1.3B video+mask | 3.16% | 0.986 |
| 14B reference | 2.00%¹ | 0.981¹ |

¹ Measured before the timestep-embedding fix. A rerun was blocked by GPU memory on the
local host; both sides of this comparison are FastVideo, so the fix affects them equally.

**Production-size video**: 1.3B/14B 480p and 14B 720p SP2 videos matched corresponding
single-GPU MP4 hashes (81 frames). This is repeatability evidence, not a substitute
for the short numeric gates above.

Set `WAN_VACE_SP_TRACE=1` to print per-layer SP divergence stats (read-only hooks).

## Known Gaps

- The SSIM regression test (`fastvideo/tests/ssim/test_wan_vace_similarity.py`: reference-image
  and video + mask cases) has no committed reference videos yet. A maintainer needs to run the
  SSIM job with `FASTVIDEO_SSIM_BOOTSTRAP_MODE=1` to create draft references for review.
- BF16 single-GPU vs SP2 outputs are not bitwise equal (see
  [Sequence Parallel](#sequence-parallel-status)); FP32 SP2 is.

## References

- [Wan2.1 Repository](https://github.com/Wan-Video/Wan2.1)
- [Wan2.1-VACE-1.3B on HuggingFace](https://huggingface.co/Wan-AI/Wan2.1-VACE-1.3B-diffusers)
- [Wan2.1-VACE-14B on HuggingFace](https://huggingface.co/Wan-AI/Wan2.1-VACE-14B-diffusers)
