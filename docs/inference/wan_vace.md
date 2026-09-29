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

End-to-end gates (`test_wan_vace_end_to_end_parity.py`):

| Gate | Check |
|------|-------|
| Conditioning inputs | bf16 bitwise vs Diffusers |
| Step0 noise prediction | `assert_dit_parity` (atol/rtol 0.1 and abs-mean drift &lt; 5%) |
| Final latent | abs-mean drift &lt; 5% after 2 denoise steps |

Coverage: 1.3B t2v, reference, video+mask; 14B reference.

Offline B200 rerun: 5 frames at 64×64, 2 denoising steps, fixed initial latents,
`TORCH_SDPA`. All four cases passed the gates above. Strict final-latent
`rtol=atol=1e-2` was diagnostic only and failed in every case.

| Case | Step0 max abs | Final max abs | Final abs-mean drift | Strict `1e-2` mismatches |
|------|--------------:|--------------:|---------------------:|------------------------:|
| 1.3B t2v | 0.015625 | 0.155267 | 2.30% | 888/2048 |
| 1.3B reference | 0.015625 | 0.101291 | 2.65% | 974/2048 |
| 1.3B video+mask | 0.015625 | 0.637095 | 3.06% | 814/2048 |
| 14B reference | 0.15625 | 0.259831 | 3.52% | 1217/2048 |

"Hierarchical gates pass" does **not** mean strict `1e-2` bitwise parity.

## Sequence Parallel Status

**FP32 weight-free SP2** (`fastvideo/tests/distributed/test_sp_wan_vace.py`) passes
single-GPU versus SP2 at `atol=rtol=1e-5`, including odd-token padding, shorter
control sequences, gather/unpad, and both ranks.

**Weighted BF16 short pipeline** (`test_wan_vace_sp_pipeline.py`, 5 frames at 64×64,
two steps, fixed latents) is marked `xfail(strict=True)`. Gates per step:
`atol=rtol=0.02`, final latent drift &lt; 1%, decoded-frame SSIM ≥ 0.99. All three
cases fail on the unmodified path:

| Case | Step0 max abs | Step1 max abs | Final latent drift | Frame SSIM |
|------|--------------:|--------------:|-------------------:|-----------:|
| 1.3B reference | 0.255859 | 0.1875 | 3.78% | 0.94534 |
| 1.3B video+mask | 0.148438 | 0.726562 | 3.48% | 0.97844 |
| 14B reference | 0.125 | 0.15625 | 2.00% | 0.98123 |

Conditioning inputs match exactly between single-GPU and SP2; both SP ranks agree.
The first traced difference is VACE block 0's FFN output projection after
bitwise-equal Q/K/V and attention outputs. Root cause: BF16 `F.linear` results
depend on GEMM row count (full M versus two M/2 shards on the same GPU).

**Production-size video**: 1.3B/14B 480p and 14B 720p SP2 videos matched corresponding
single-GPU MP4 hashes (81 frames). This is repeatability evidence, not a substitute
for the short numeric gates above.

Set `WAN_VACE_SP_TRACE=1` to print per-layer SP divergence stats (read-only hooks).

## Known Gaps

- The SSIM regression test (`fastvideo/tests/ssim/test_wan_vace_similarity.py`) has
  no committed reference videos yet; CI bootstraps draft references for review.
- With BF16 weights, the short-pipeline SP2 parity test is still marked `xfail`
  (see [Sequence Parallel](#sequence-parallel-status) above).
- The video + mask control path has no end-to-end reproduction case; only the
  reference-image case is covered.

## References

- [Wan2.1 Repository](https://github.com/Wan-Video/Wan2.1)
- [Wan2.1-VACE-1.3B on HuggingFace](https://huggingface.co/Wan-AI/Wan2.1-VACE-1.3B-diffusers)
- [Wan2.1-VACE-14B on HuggingFace](https://huggingface.co/Wan-AI/Wan2.1-VACE-14B-diffusers)
