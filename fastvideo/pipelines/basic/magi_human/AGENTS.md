# `fastvideo/pipelines/basic/magi_human/` — daVinci-MagiHuman

**Generated:** 2026-05-07

Single-stream joint audio-visual generative pipeline. 4 variants × 2 input modes
(T2V / TI2V) = 8 entrypoints. The DiT denoises video and audio latents in one
unified token sequence — no cross-attention, channel-major token packing.

If you are touching this pipeline, **read the parity invariants and cross-refs
sections below before editing any file in the manifest**.

## Manifest

| File | Role |
|------|------|
| `magi_human_pipeline.py` | Composed pipeline class. `load_modules` lazy-loads four shared upstream components (Wan 2.2 VAE, T5-Gemma, Stable Audio VAE, scheduler). |
| `pipeline_configs.py` | Per-variant `PipelineConfig` dataclasses (`base`, `distill`, `sr_540p`, `sr_1080p`). |
| `presets.py` | Preset registry per variant — entry point that `fastvideo/registry.py` imports. |
| `__init__.py` | SPDX header only; no public exports beyond what `presets.py` registers. |
| `stages/audio_decoding.py` | Decodes audio latents through the lazy Stable Audio VAE wrapper. |
| `stages/denoising.py` | Joint AV denoising loop (32-step FlowUniPC with CFG=2 base, 8-step CFG=1 distill). |
| `stages/latent_preparation.py` | Channel-major video token packing + audio interleave + reference-image masking. **Carries the channel-major packing invariant — see Parity Invariants.** |
| `stages/reference_image.py` | TI2V reference-image conditioning. |
| `stages/sr_denoising.py` | SR DiT denoising loop with cfg-trick guidance tensor. SR-1080p uses block-sparse video→video local-window attention on 32 of 40 SR DiT layers via a 3-block SDPA accumulator. |
| `stages/sr_latent_preparation.py` | Trilinear-up of base latent + ZeroSNR noise + audio mix for the SR pass. |
| `stages/__init__.py` | Re-exports stage classes. |
| `JOURNAL.md` | 14-wave port-state journal; root cause writeups for the parity invariants below. |
| `AGENTS.md` | This file. |

External coordinates of related files (read these too if you change wiring):

| Path | Role |
|------|------|
| `fastvideo/models/dits/magi_human.py` | DiT architecture port. |
| `fastvideo/configs/models/dits/magi_human.py` | `MagiHumanVideoConfig` arch dataclass. |
| `fastvideo/models/encoders/t5gemma.py` | T5-Gemma 9B UL2 text encoder port. |
| `fastvideo/configs/models/encoders/t5gemma.py` | `T5GemmaEncoderConfig`. |
| `fastvideo/models/vaes/sa_audio.py` | Lazy `OobleckVAE` wrapper (shared with `pipelines/basic/stable_audio/`). **Not new — pre-existed on main.** |
| `fastvideo/models/vaes/oobleck.py` | `OobleckVAE` itself (pre-existing). |
| `fastvideo/models/loader/component_loader.py` | `sr_transformer` module type alias added here. |
| `scripts/checkpoint_conversion/convert_magi_human_to_diffusers.py` | Reference → FastVideo state-dict converter. **Carries the `_FP32_KEEP_SUFFIXES` invariant — see Parity Invariants.** |
| `scripts/checkpoint_conversion/push_magi_human_to_hf.py` | Push converted weights to umbrella HF repo. |
| `tests/local_tests/magi_human/` | 14-test parity battery (skipped in CI; GPU-gated). |
| `tests/local_tests/helpers/magi_human_upstream.py` | Upstream daVinci-MagiHuman reference loader. |
| `examples/inference/basic/basic_magi_human*.py` | 8 user-facing example scripts (one per variant × mode). |
| `fastvideo/tests/ssim/test_magi_human_similarity.py` | CI-eligible SSIM regression. |

## Parity Invariants

These are **load-bearing**. Each one was a multi-wave bug hunt in the original
port. Breaking any of them re-introduces a known production-only regression.
See the matching lesson under `.agents/lessons/`.

### 1. Channel-major video token packing

Video tokens must be packed **`(C pT pH pW)`** — channel-major.

```python
# CORRECT
einops.rearrange(x, "b c (T pT) (H pH) (W pW) -> b (T H W) (C pT pH pW)", ...)
# WRONG — produced pure noise in production E2E despite passing pipeline parity
einops.rearrange(x, "b c (T pT) (H pH) (W pW) -> b (T H W) (pT pH pW C)", ...)
```

Upstream `UnfoldNd` packs channel-major. Mismatching this passes the FastVideo
self-parity test (both sides use FastVideo's packer) but breaks against the
official reference, which only surfaces in production E2E. Lives in
`stages/latent_preparation.py` — search for `_img2tokens`.

→ `.agents/lessons/2026-05-07_silent-channel-major-packing-bugs.md`

### 2. DiT dtype boundary discipline

Across DiT block boundaries, the residual stream **stays fp32**. SDPA inputs
are cast to bf16 inside the attention call. Post-attention output is upcast to
fp32 before the per-head gating multiply. **There is no block-boundary cast.**

These four rules are cumulative — relaxing any one re-introduces measurable
parity drift (worst case `diff_max=0.5`, best case `diff_max≈1e-3` which is
still not bit-exact).

→ `.agents/lessons/2026-05-07_dit-dtype-boundary-with-flash-attn.md`

### 3. Conversion `_FP32_KEEP_SUFFIXES` allowlist

`scripts/checkpoint_conversion/convert_magi_human_to_diffusers.py` runs with
`--cast-bf16` by default. The allowlist of fp32-keep suffixes prevents a
specific set of 8 tensors from being downcast. The base checkpoint and the
FastVideo `final_linear`/adapter modules require these in fp32. The distill
DiT was the canary: its parity went from `diff_mean=0.114` (silently wrong)
to bit-exact when the allowlist was fixed.

If you add or rename DiT modules, check that any fp32-required tensors are
covered by `_FP32_KEEP_SUFFIXES` and re-run `test_magi_human_distill_parity`.

→ `.agents/lessons/2026-05-07_conversion-cast-bf16-suffix-allowlist.md` (lands in 7/8)

### 4. Umbrella HF repo layout

User code is a single string per variant:

```python
VideoGenerator.from_pretrained("FastVideo/MagiHuman-Diffusers/base")
```

`fastvideo/utils.py:maybe_download_model` recognises the 3-segment
`org/repo/subfolder` form and downloads only that subtree (not the full
~75 GB repo). The umbrella repo:

```
FastVideo/MagiHuman-Diffusers/
├── base/{model_index.json, transformer/, scheduler/}
├── distill/{model_index.json, transformer/, scheduler/}
├── sr_540p/{model_index.json, transformer/, sr_transformer/, scheduler/}
└── sr_1080p/{model_index.json, transformer/, sr_transformer/, scheduler/}
```

Note: `vae/`, `text_encoder/`, `audio_vae/` subfolders are **deliberately
absent**. Those four shared components are lazy-loaded by
`MagiHumanPipeline.load_modules` from their canonical upstream repos
(`Wan-AI/Wan2.2-TI2V-5B`, `google/t5gemma-9b-9b-ul2`,
`stabilityai/stable-audio-open-1.0`). This relies on
`fastvideo/utils.py:verify_model_config_and_directory` honoring
`model_index.json` declarations.

## Cross-Refs (If you change X, re-run Y)

| If you touch... | Re-run at minimum |
|---|---|
| `stages/latent_preparation.py` (any token packing) | `test_magi_human_pipeline_parity` (base T2V) **and** one of the SR-540p/1080p tests **and** `examples/inference/basic/basic_magi_human.py` (E2E mp4 hash) |
| `stages/denoising.py` or `stages/sr_denoising.py` | All four pipeline-parity tests (`{base, ti2v, sr540p, sr1080p}_pipeline_parity`) |
| `fastvideo/models/dits/magi_human.py` (any layer) | `test_magi_human_parity` and `test_magi_human_distill_parity` (DiT-level) **before** the pipeline tests |
| Anything dtype-related in the DiT | All DiT parity tests + verify the residual-stream dtype invariant manually with a layer-by-layer hook trace (see `fastvideo/hooks/activation_trace.py`) |
| `scripts/checkpoint_conversion/convert_magi_human_to_diffusers.py` | Convert the distill checkpoint and run `test_magi_human_distill_parity` (it's the canary for `_FP32_KEEP_SUFFIXES`) |
| `fastvideo/utils.py:maybe_download_model` (3-segment detector) | Smoke-load every existing 2-segment HF id in `registry.py` plus one MagiHuman variant |
| `magi_human_pipeline.py:load_modules` | `test_magi_human_pipeline_smoke` + one full E2E `examples/inference/basic/basic_magi_human.py` |
| `presets.py` or `pipeline_configs.py` | Both DiT parity tests (preset wiring leaks into module construction) |

## Run Book

```bash
# Setup once
export HF_TOKEN=hf_...    # any of HF_TOKEN / HUGGINGFACE_HUB_TOKEN / HF_API_KEY works
# Accept terms at:
#   - https://huggingface.co/GAIR/daVinci-MagiHuman
#   - https://huggingface.co/google/t5gemma-9b-9b-ul2
#   - https://huggingface.co/stabilityai/stable-audio-open-1.0
#   - https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B

# Full parity battery (~233s on a single H100, GPU-gated)
pytest tests/local_tests/magi_human/ -v -s

# E2E mp4 generation (any variant)
python examples/inference/basic/basic_magi_human.py

# CI-eligible SSIM regression
pytest fastvideo/tests/ssim/test_magi_human_similarity.py -v -s
```

The base T2V mp4 hash should be `dcf5f2bf6534c7c0d91e7353e42b23db` — stable
across all 43 commits of the original port.

## Open Questions

- **OQ-7** — Wan VAE decode shows a max diff of ~8e-4 against the diffusers
  reference (not bit-exact). Root cause is a known fp32 op-order drift in the
  Wan 2.2 VAE itself, not MagiHuman wiring. Tracked in `JOURNAL.md`. Documented
  here so the test author knows the tolerance is intentional, not a regression.

## Provenance

This pipeline was decomposed from a single 9,812-line PR
([#1280](https://github.com/hao-ai-lab/FastVideo/pull/1280),
`will/magi` @ `4e1603634d27c8e1b5c4cc5d9387f046547f5c49`) into a stack of
focused PRs:

| Step | PR | Branch | Scope |
|---|---|---|---|
| Prereq A | [#1293](https://github.com/hao-ai-lab/FastVideo/pull/1293) | `will/activation-trace` | Generic activation-tracing infra |
| Prereq B | [#1294](https://github.com/hao-ai-lab/FastVideo/pull/1294) | `will/loader-infra` | Loader umbrella-repo + optional component dirs |
| 1/8 | [#1295](https://github.com/hao-ai-lab/FastVideo/pull/1295) | `will/magi-01-housekeeping` | gitignore, codespell, skills index |
| 2/8 | [#1296](https://github.com/hao-ai-lab/FastVideo/pull/1296) | `will/magi-02-t5gemma` | T5-Gemma encoder + parity test |
| 3/8 | [#1297](https://github.com/hao-ai-lab/FastVideo/pull/1297) | `will/magi-03-dit` | DiT + parity tests |
| 4/8 | [#1298](https://github.com/hao-ai-lab/FastVideo/pull/1298) | `will/magi-04a-stages` | Pipeline stages + sr_transformer alias |
| 5/8 | [#1299](https://github.com/hao-ai-lab/FastVideo/pull/1299) | `will/magi-04b-orchestrator` | Pipeline orchestrator + parity battery |
| 6/8 | this PR | `will/magi-04c-provenance` | This AGENTS.md, JOURNAL.md, lessons, parent AGENTS.md hook |
| 7/8 | (next) | `will/magi-05-conversion` | Checkpoint conversion + 3rd lesson |
| 8/8 | (last) | `will/magi-06-activate` | Registry + examples + SSIM + codebase-map |

Provenance section will be finalized in 8/8 with the actual PR numbers for 7/8 and 8/8.
