# MagiHuman — Codebase Map Entry

**Family:** daVinci-MagiHuman (joint audio-visual generative model)
**Reference:** [GAIR-NLP/daVinci-MagiHuman](https://github.com/GAIR-NLP/daVinci-MagiHuman)
**Architecture:** 15B-param single-stream DiT, 40 layers, hidden=5120,
head_dim=128, GQA num_query_groups=8. Joint AV denoising in a unified token
sequence; **no cross-attention**.

## Variant Matrix

| Variant | T2V | TI2V | DiT | Steps | CFG | Resolution |
|---|---|---|---|---|---|---|
| `base` | yes | yes | base | 32 | 2 | 480x256 |
| `distill` | yes | yes | distill (DMD-2) | 8 | 1 (no CFG) | 480x256 |
| `sr_540p` | yes | yes | base + sr_540p | 32 + 5 | 2 + cfg-trick | 896x512 |
| `sr_1080p` | yes | yes | base + sr_1080p | 32 + 5 | 2 + cfg-trick | 1920x1056 |

SR-1080p uses block-sparse video→video local-window attention on 32 of 40
SR DiT layers (`frame_receptive_field=11`), implemented as a 3-block SDPA
accumulator that mirrors upstream `flex_flash_attn_func`.

## File Locations

| Role | Path |
|---|---|
| Pipeline class | `fastvideo/pipelines/basic/magi_human/magi_human_pipeline.py` |
| Pipeline package AGENTS.md | `fastvideo/pipelines/basic/magi_human/AGENTS.md` |
| Stages | `fastvideo/pipelines/basic/magi_human/stages/*.py` |
| DiT | `fastvideo/models/dits/magi_human.py` |
| Text encoder (T5-Gemma) | `fastvideo/models/encoders/t5gemma.py` |
| Audio VAE wrapper | `fastvideo/models/vaes/sa_audio.py` (shared with `stable_audio` pipeline) |
| Conversion script | `scripts/checkpoint_conversion/convert_magi_human_to_diffusers.py` |
| Examples | `examples/inference/basic/basic_magi_human*.py` (8 files, one per variant × mode) |
| SSIM regression | `fastvideo/tests/ssim/test_magi_human_similarity.py` |
| Local parity battery | `tests/local_tests/magi_human/` (14 tests, GPU-gated) |
| Port journal | `fastvideo/pipelines/basic/magi_human/JOURNAL.md` |

## Canonical HF Repo

[FastVideo/MagiHuman-Diffusers](https://huggingface.co/FastVideo/MagiHuman-Diffusers)
— umbrella repo with sibling subfolders per variant.

```python
from fastvideo import VideoGenerator
gen = VideoGenerator.from_pretrained("FastVideo/MagiHuman-Diffusers/base")
gen.generate_video(prompt="...", output_path="out.mp4", save_video=True)
```

Four shared upstream components are lazy-loaded by `MagiHumanPipeline.load_modules`:

| Component | Upstream repo |
|---|---|
| Wan 2.2 VAE | `Wan-AI/Wan2.2-TI2V-5B` |
| T5-Gemma 9B UL2 | `google/t5gemma-9b-9b-ul2` |
| Stable Audio VAE | `stabilityai/stable-audio-open-1.0` |
| MagiHuman DiT weights | `GAIR/daVinci-MagiHuman` (gated) |

## Parity Invariants

Three load-bearing invariants. See `fastvideo/pipelines/basic/magi_human/AGENTS.md`
for the full discussion.

1. **Channel-major video token packing** (`stages/latent_preparation.py`)
2. **DiT dtype boundary**: residual stream stays fp32 across blocks
3. **Conversion `_FP32_KEEP_SUFFIXES`** (`scripts/checkpoint_conversion/convert_magi_human_to_diffusers.py`)

## Lessons

- `.agents/lessons/2026-05-07_silent-channel-major-packing-bugs.md`
- `.agents/lessons/2026-05-07_dit-dtype-boundary-with-flash-attn.md`
- `.agents/lessons/2026-05-07_conversion-cast-bf16-suffix-allowlist.md`

## Provenance

Decomposed from PR [#1280](https://github.com/hao-ai-lab/FastVideo/pull/1280)
(`will/magi` @ `4e1603634d27c8e1b5c4cc5d9387f046547f5c49`). See the package
AGENTS.md for the full PR-stack table.
