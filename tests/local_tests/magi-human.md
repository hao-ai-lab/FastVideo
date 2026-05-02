# Local daVinci-MagiHuman Tests

End-to-end parity tests for the daVinci-MagiHuman joint text-to-audio-video
pipeline. MagiHuman is a 15B-parameter DiT that denoises video and audio
latents in a single loop, producing synchronized video and audio from a text
prompt. The video path uses the Wan 2.2 TI2V-5B VAE (decoder only), the audio
path uses the Stable Audio Open 1.0 `OobleckVAE` (shared with the standalone
Stable Audio pipeline), and text conditioning comes from a T5-Gemma 9B UL2
encoder. The base variant runs 32-step FlowUniPC with CFG=2; the distill
variant runs 8 steps with CFG=1. Reference implementation:
[GAIR-NLP/daVinci-MagiHuman](https://github.com/GAIR-NLP/daVinci-MagiHuman).
These tests compare FastVideo against the published weights and the upstream
reference, so they're skipped in CI and run locally on a single GPU.

## Setup

### 1. Hugging Face access

MagiHuman depends on four gated repos. Accept the terms at each URL once, then
export your token:

| Repo | Terms URL |
|---|---|
| `GAIR/daVinci-MagiHuman` | https://huggingface.co/GAIR/daVinci-MagiHuman |
| `google/t5gemma-9b-9b-ul2` | https://huggingface.co/google/t5gemma-9b-9b-ul2 |
| `stabilityai/stable-audio-open-1.0` | https://huggingface.co/stabilityai/stable-audio-open-1.0 |
| `Wan-AI/Wan2.2-TI2V-5B` | https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B |

```bash
export HF_TOKEN=hf_...
# any of HF_TOKEN / HUGGINGFACE_HUB_TOKEN / HF_API_KEY works
```

The pipeline's `_ensure_hf_token_env` helper (in
`fastvideo/pipelines/basic/magi_human/magi_human_pipeline.py`) aliases all
three names to `HF_TOKEN` and `HUGGINGFACE_HUB_TOKEN` at load time, so
whichever variable you set will be picked up. Tests skip cleanly with a
helpful message if no token is found.

### 2. Optional inference dependencies

The pipeline uses the default FastVideo attention backend. No extra packages
are required for basic inference. If you want the T5-Gemma wrapper to use
PyTorch SDPA instead of Flash Attention, set:

```bash
export FASTVIDEO_ATTENTION_BACKEND=TORCH_SDPA
```

The `T5GemmaEncoderModel` wrapper in
`fastvideo/models/encoders/t5gemma.py` reads this variable and patches
`model.config.attn_implementation` accordingly before the first forward pass.

### 3. Clone the upstream reference repo

The DiT parity test (`test_magi_human_parity.py`) and the pipeline parity test
(`test_magi_human_pipeline_parity.py`) import directly from the upstream
`daVinci-MagiHuman` package. Clone it under the repo root and add it to your
personal ignore list:

```bash
cd <FastVideo repo root>
git clone --depth 1 https://github.com/GAIR-NLP/daVinci-MagiHuman.git
echo "/daVinci-MagiHuman/" >> .git/info/exclude   # personal ignore
```

Tests that need the clone skip cleanly if the directory is absent. The VAE
parity tests and the smoke test do not need the upstream clone.

### 4. Convert weights

Run the conversion script once to produce a Diffusers-layout checkpoint. The
`--bundle-vae`, `--bundle-audio-vae`, and `--bundle-text-encoder` flags copy
the Wan VAE, Oobleck audio VAE, and T5-Gemma encoder into the output directory
so the pipeline can load everything from a single path:

```bash
python scripts/checkpoint_conversion/convert_magi_human_to_diffusers.py \
    --source GAIR/daVinci-MagiHuman \
    --output converted_weights/magi_human_base \
    --bundle-vae \
    --bundle-audio-vae \
    --bundle-text-encoder
```

Disk budget: roughly 30 GB for the base checkpoint. The distill variant is a
similar size; add `--cast-bf16` to halve the transformer shards if storage is
tight.

The tests look for the converted path in `MAGI_HUMAN_DIFFUSERS_PATH` (see
§8 Troubleshooting). If that variable is unset, they fall back to
`converted_weights/magi_human_base` relative to the repo root.

### 5. (Optional) Pre-warm the model cache

The first parity-test run downloads the T5-Gemma encoder (~18 GB), the Wan VAE
(~2 GB), and the Stable Audio Open VAE (~1 GB) if they aren't already cached.
To avoid the download blocking your first test run, fetch them ahead of time:

```bash
python -c "
from huggingface_hub import snapshot_download
snapshot_download('google/t5gemma-9b-9b-ul2')
snapshot_download('Wan-AI/Wan2.2-TI2V-5B')
snapshot_download('stabilityai/stable-audio-open-1.0')
"
```

## Running the tests

All MagiHuman local tests in one shot:

```bash
pytest tests/local_tests/magi_human/test_magi_human_parity.py \
       tests/local_tests/magi_human/test_magi_human_t5gemma_parity.py \
       tests/local_tests/magi_human/test_magi_human_sa_audio_parity.py \
       tests/local_tests/magi_human/test_magi_human_sa_audio_official_parity.py \
       tests/local_tests/magi_human/test_magi_human_vae_parity.py \
       tests/local_tests/magi_human/test_magi_human_pipeline_smoke.py \
       tests/local_tests/magi_human/test_magi_human_pipeline_parity.py \
       fastvideo/tests/ssim/test_magi_human_similarity.py \
       -v -s
```

Add `-s` to print per-test diff numbers (shape / abs_mean / max diff / drift).

### What each test covers

**`test_magi_human_parity.py`** — DiT component parity. Loads the
`MagiHumanTransformer3DModel` from the converted checkpoint and the upstream
reference DiT from the `daVinci-MagiHuman` clone, feeds identical latent
inputs, and checks that output tensors match within tolerance. Requires both
the upstream clone and the converted weights.

**`test_magi_human_t5gemma_parity.py`** — T5-Gemma encoder wrapper parity.
Compares `fastvideo.models.encoders.t5gemma.T5GemmaEncoderModel` against a
direct HuggingFace `T5GemmaEncoderModel.from_pretrained` call on the same
checkpoint. Verifies that the FastVideo wrapper's lazy-load path and
`named_parameters` exclusion don't alter the encoder's output embeddings.

**`test_magi_human_sa_audio_parity.py`** — Stable Audio Open VAE wrapper
parity. Compares the FastVideo `OobleckVAE` (shared with the standalone Stable
Audio pipeline) against HuggingFace Diffusers' `AutoencoderOobleck` on the
`stabilityai/stable-audio-open-1.0` weights. Encode + decode + round-trip;
expected to be bit-identical in fp32.

**`test_magi_human_sa_audio_official_parity.py`** — Stable Audio Open VAE
parity vs the official daVinci-MagiHuman integration layer. Compares FastVideo's
`SAAudioVAEModel` against the upstream `SAAudioFeatureExtractor.decode()` path
from the `daVinci-MagiHuman` clone. Catches drift between FastVideo's full SA
wrapper and the official repo's custom Stable-Audio module. Requires the upstream
clone and the `stabilityai/stable-audio-open-1.0` gated repo. Expected to be
bit-exact (diff=0) in fp32.

**`test_magi_human_vae_parity.py`** — Wan video VAE parity. Compares the
FastVideo Wan VAE decoder against the upstream `Wan2_2_VAE` on
`Wan-AI/Wan2.2-TI2V-5B` weights. Decoder-only path (MagiHuman never encodes
video at inference time).

**`test_magi_human_pipeline_smoke.py`** — Preflight and smoke. Imports the
pipeline, resolves the registry entries (`magi_human_base`,
`magi_human_distill`), checks preset wiring, and verifies the pipeline can
instantiate without a GPU. CPU-only; no model weights required beyond the
converted path.

**`test_magi_human_pipeline_parity.py`** — End-to-end joint AV latent parity.
Runs a short denoising loop through the full pipeline and compares the final
video and audio latents against the upstream reference pipeline. Requires the
upstream clone, the converted weights, and a GPU.

**`test_magi_human_similarity.py`** — Video SSIM regression (CI-runnable).
Generates a short clip from a fixed prompt and seed, then compares frame-level
SSIM against reference videos stored in the `FastVideo/ssim-reference-videos`
HF dataset. The test skips cleanly until reference videos are seeded (see §7
Open questions).

### Reproducing a single test

Each test file is independent. Run one:

```bash
pytest tests/local_tests/magi_human/test_magi_human_pipeline_parity.py -v -s
```

## Phase 11 status

Branch tip `eeef855b` (rebased onto `origin/main` `c77a76c6`), Wave 1+4 changes applied (uncommitted working tree), NVIDIA B200. Wave 2-3 numerical-alignment investigation completed 2026-05-01; see §Numerical-alignment investigation below.

| Test | Status | Diff numbers | Notes |
|---|---|---|---|
| `tests/local_tests/magi_human/test_magi_human_t5gemma_parity.py::test_magi_human_t5gemma_wrapper_parity` | PASS | exact (`assert_close(atol=1e-3, rtol=1e-3)`) | gated repo, requires HF token |
| `tests/local_tests/magi_human/test_magi_human_parity.py::test_magi_human_dit_parity` | FAIL | video diff_max=0.057, diff_mean=0.008; audio diff_max=0.034, diff_mean=0.008; text exact (diff_max=0) | Tightened to `atol=0.03, rtol=0.01` (Wave 1). Bf16-noise-floor; per-layer drift ~1e-3 accumulates over 40 layers. Root cause of OQ-6 compounding. See §Numerical-alignment investigation. |
| `tests/local_tests/magi_human/test_magi_human_vae_parity.py::test_magi_human_vae_decode_parity` | PASS | diff_max=8e-4, diff_mean=4.9e-5 | Wan VAE. Deferred to `atol=1e-3, rtol=1e-3` per OQ-7 (Wave 4). Tighten to `atol=1e-4` once Wan VAE op-order fix lands. |
| `tests/local_tests/magi_human/test_magi_human_sa_audio_parity.py::test_magi_human_sa_audio_vae_decode_parity` | PASS | exact (`assert_close(atol=1e-5, rtol=1e-5)`, machine epsilon) | gated repo, requires HF token; uses main's shared `OobleckVAE` + `SAAudioVAEModel` wrapper |
| `tests/local_tests/magi_human/test_magi_human_sa_audio_official_parity.py::test_magi_human_sa_audio_official_decode_parity` | PASS | `atol=1e-5, rtol=1e-5`, diff_max=0, diff_mean=0 (bit-exact) | Wave 7. Compares FV `SAAudioVAEModel` vs upstream `SAAudioFeatureExtractor.decode()`. Confirms OQ-6 is NOT in audio VAE. Requires upstream clone + gated SA repo. |
| `tests/local_tests/magi_human/test_magi_human_pipeline_smoke.py::test_magi_human_typed_surface_preflight` | PASS | CPU-only key/preset checks, exact key set equality, 331 keys | no skip conditions met locally |
| `tests/local_tests/magi_human/test_magi_human_pipeline_smoke.py::test_magi_human_pipeline_smoke` | PASS | shape-only; 2 inference steps, output shape `[B,C,T,H,W]` validated | wallclock ~50s |
| `tests/local_tests/magi_human/test_magi_human_pipeline_parity.py::test_magi_human_pipeline_latent_parity` | FAIL | video: diff_max=6.69, diff_mean=0.47; audio: diff_max=3.45, diff_mean=1.01 | Wave 7.5: now uses real preset prompts via T5-Gemma. Wave 8 production fixes don't move parity numbers (both sides use same encoder). Residual drift is bf16+CFG amplification floor; tracked as OQ-6 RESOLVED-PRODUCTION. |
| `fastvideo/tests/ssim/test_magi_human_similarity.py::test_magi_human_base_inference_similarity` | DEFERRED | n/a | Reference videos not yet seeded to `FastVideo/ssim-reference-videos` HF repo; tracked as OQ-2. Requires Modal L40S seeding via `seed-ssim-references` skill. |
| _(debug)_ | INFO | Per-side layer logs: `/tmp/opencode/magi_dit_up_layers.log`, `/tmp/opencode/magi_dit_fv_layers.log` | Added in Wave 1 to `_debug_magi_human_block_parity.py`. See `add-model-trace` skill at `~/.config/opencode/skill/add-model-trace/`. |
| `fastvideo/tests/hooks/test_activation_trace.py::*` | PASS | 6 tests covering off/on/filter/stats/step-filter/cleanup | Wave 9 activation trace infrastructure |

_Last verified: 2026-05-01 (Wave 10 dtype refactor on rebased branch @ 3caeaad1; tests now under `tests/local_tests/magi_human/`)_

## Design notes

### T5-Gemma lazy-load exception

`fastvideo/models/encoders/gemma.py:10` establishes the FastVideo precedent for
gated foundation-model encoders: the HF model class
(`Gemma3ForConditionalGeneration`) is imported at module top-level, and the
actual weights are loaded lazily via `from_pretrained` inside a property or
method.

`fastvideo/models/encoders/t5gemma.py:60` follows the same pattern but is
strictly more conservative: the HF class (`T5GemmaEncoderModel`) is imported
inside `_build_t5gemma_model` rather than at module top-level. This avoids an
import-time failure if `transformers.models.t5gemma` isn't available in the
environment. The `named_parameters` override on the same class hides the
upstream encoder from FastVideo's weight loader so the converted repo directory
isn't scanned for T5-Gemma shards.

This is the established FastVideo pattern for gated foundation-model encoders
not yet ported natively. It is not a workaround; it's the documented approach.

**Native T5-Gemma port — TRACKED FOLLOW-UP.** A future native port is
desirable for full Phase 11 hard-rule compliance (no HF model-class imports in
production runtime code). Scope estimate is multi-week: Gemma decoder blocks,
T5 encoder cross-attention, RMS norm, RoPE, and tokenizer wiring all need
native FastVideo implementations. This is tracked here until claimed by a
follow-up PR.

### Audio quality regression deferral

`tests/local_tests/stable-audio.md` sets the precedent: the Stable Audio Open
1.0 port ships local parity tests, a smoke test, and self-consistency checks
for inpainting and audio-to-audio variation, with no `fastvideo/tests/audio/`
quality regression test.

MagiHuman's audio path is covered by `test_magi_human_pipeline_parity.py`
(joint AV latent comparison against the upstream reference) and the basic
example mp4 spot-check (`examples/inference/basic/basic_magi_human.py`). A
mel-spectrogram L1 or multi-resolution STFT regression test is listed as a
follow-up if audio drift becomes a concern in practice.

### Pipeline parity tolerance budget (1-step / CFG=2)

Drift is dominated by CFG amplification of single-DiT bf16 mismatch. The
single-DiT diff_mean is ~0.008 (per `test_magi_human_dit_parity`); CFG mixes
`v = v_uncond + 5*(v_cond - v_uncond)`, so independent bf16 errors in
cond/uncond paths compound by ~5x, giving an expected pipeline diff_mean of
~0.04. Observed is 0.069. `diff_max` is the noisiest statistic for bf16+CFG
(a single fma quantization can blow it up); `atol=0.40` accommodates that.

Two ratio guards catch real structural bugs:

- **`abs_mean` drift < 1%** (gross-bug catcher: scheduler state leak, dropped
  modality, CFG sign flip)
- **`diff_mean / ref_abs` < 4%** (systematic per-element bias guard)

All three guards currently pass with margin: video abs_mean rel=0.36%, audio
abs_mean rel=0.33%; video diff_mean/ref=3.07%, audio diff_mean/ref=2.66%.

The test uses `num_inference_steps=1, cfg_number=2, guidance=5.0`. Per Oracle
analysis in this PR's review notes, this is expected bf16+CFG behavior, not a
structural bug.

## Numerical-alignment investigation (2026-05-01)

Wave 2-3 investigation into why the 4-step pipeline parity fails and whether the
DiT parity failure at `atol=0.03` indicates a real bug.

### Methodology

TDD-style: tighten tolerances to surface real drift, run, drill into the
largest contributor, bisect to confirm pre-existence, then rule out hypotheses
one by one.

1. **Wave 1 (bug-surfacing changes):** Tightened DiT parity from `atol=0.1` to
   `atol=0.03, rtol=0.01`. Tightened Wan VAE parity from `atol=5e-2` to
   `atol=1e-4` (later deferred to `atol=1e-3` per OQ-7). Bumped pipeline parity
   `num_inference_steps` from 1 to 4. Fixed `_find_base_shard_dir` with
   `snapshot_download` fallback (resolves OQ-4). Added per-side layer log files
   to `_debug_magi_human_block_parity.py`. Created new `add-model-trace` skill
   in user dotfiles.

2. **Wave 2 (run and measure):** DiT parity fails at new `atol=0.03`
   (diff_max=0.057, diff_mean=0.008). Wan VAE parity fails at `atol=1e-4`
   (diff_max=8e-4). 4-step pipeline parity fails with video diff_mean=1.30 vs
   1-step 0.069, a ratio of 18.85x (expected ~4x linear). Per-block drift never
   exceeds 0.5% threshold; cumulative peaks at MM layers (blocks 0-3 and 36-39,
   matching `mm_layers=[0,1,2,3,36,37,38,39]`).

3. **Wave 3 (drill and bisect):** Tested PackedExpertLinear hypothesis via A/B
   patch. Bisected compounding bug to original commit. Drilled into Block[02]
   MM-layer MLP `down_proj` amplification. Verified expert chunk ordering
   bit-exact.

### Key findings

| Finding | Result | Evidence |
|---|---|---|
| PackedExpertLinear routing bug | **REJECTED** | A/B with `MAGI_DEBUG_PATCH_LINEAR=1` (mirrors upstream `_BF16ComputeLinear`) showed zero change in drift |
| Wave 1 commits caused compounding | **REJECTED** | `git revert` bisect: 4-step diff_mean=1.20 with reverts vs 1.30 with Wave 1; bug pre-exists in commit 620aaf41 |
| Expert chunk ordering mismatch | **REJECTED** | Direct FV `PackedExpertLinear` vs upstream `NativeMoELinear` test: diff=0 (bit-exact) |
| Wan VAE op-order drift | **CONFIRMED** | FV uses `z * std + mean`; upstream uses `z / (1/std) + mean`. Bitwise non-equivalent. Shared Wan-family bug (OQ-7). |
| MM-layer MLP `down_proj` amplification | **NORMAL** | Block[02] input drift 0.0005 → output drift 0.022 = 44x amplification. Normal sensitivity for a 15360x20480 matrix; not a routing bug. |
| Per-forward DiT drift | **BF16 NOISE FLOOR** | diff_max=0.057 from cumulative ~1e-3 per-layer over 40 layers. Consistent with random-walk bf16 accumulation. |

### Root-cause hypothesis

Per-forward DiT drift is bf16 noise, not a structural bug. Diffusion sampling
amplifies per-step bf16 perturbations geometrically over the denoise loop (a
known ill-conditioned-ODE phenomenon). The 18.85x compounding ratio at 4 steps
vs the expected 4x linear ratio confirms geometric amplification. The "blurry
abstract" output at 32 steps (OQ-5) is the downstream symptom.

Wave 3 ruled out all discrete implementation bugs: PackedExpertLinear routing,
expert chunk ordering, and the conversion script are all bit-exact. The
remaining candidates are dtype boundary mismatches around sensitive MM-layer ops
(pre-norm, attention, MLP activation) where upstream may cast to fp32 and FV
stays in bf16.

### Wave 7 (2026-05-01): CFG + negative prompt investigation

Findings:
- **CFG math identical**: FV `v = uncond + g * (cond - uncond)` matches upstream at `denoising.py:178-181` ↔ `video_generate.py:426,456-457`. Video has `t > 500` cutoff (`5.0 → 2.0`); audio has none. Both sides apply the same formula.
- **Scheduler args identical for T2AV base path**: `step(model_output, t, sample, return_dict=False)[0]`. Audio-skip modes (`is_a2v`/SR) are not exercised in base.
- **Audio decode path bit-exact vs official**: New parity test [`test_magi_human_sa_audio_official_parity.py`] passes at machine-eps (diff=0). FV's `SAAudioVAEModel` is identical to upstream `SAAudioFeatureExtractor.decode()`. Confirms OQ-6 is NOT in audio VAE.
- **Production root cause identified**: FV's preset `_MAGI_HUMAN_NEGATIVE_PROMPT` was missing the audio-quality + speech-delivery blocks present in upstream `video_generate.py:222-224`. Fix applied at `presets.py`. Audio CFG amplifies the missing-block delta 5x → consistent with observed step-1 audio amplification of ~3x.
- **Hardening**: Replaced silent zero-fallback in `denoising.py:127-135` with `ValueError`. Missing negative embeds at CFG=2 is a real bug, not silent-success.

Caveat — parity test path bypasses preset prompts: `test_magi_human_pipeline_parity.py:291-298` uses random `txt_feat` and `neg_txt_feat` (identical on both sides), so the negative-prompt fix does NOT change parity numbers. Production inference (basic example) DOES use the preset and benefits from the fix.

Reframed OQ-6 root cause:
- Production-facing "blurry abstract" output: caused by incomplete negative prompt (audio CFG didn't have the right negatives). FIXED in this commit.
- Parity-test 4-step compounding (1.196 mean): separate phenomenon — inherent FlowUniPC multistep scheduler amplification of per-call bf16 noise (~2x per DiT call expansively, 8 calls = ~256x). NOT a code bug; would require fp32 sensitive ops or a different scheduler to materially change.

### Wave 8 (2026-05-01): broader CFG/preset/fallback audit + targeted fixes

Audit found 4 more HARMFUL FV-vs-upstream divergences in addition to the negative-prompt incompleteness fixed in Wave 7:

| # | Item | Severity | Status |
|---|---|---|---|
| 1 | T5-Gemma tokenizer pre-pads to 640 BEFORE encoding (pad-token hidden states pollute DiT input; magi_original_text_lens lies about real length) | HARMFUL | FIXED — `t5gemma.py:57-64` no longer passes `truncation`/`padding`/`max_length`; pad/trim handled post-encode by `MagiHumanLatentPreparationStage._pad_or_trim_dim1` |
| 3 | Default resolution 448x256 vs upstream's 480x272 (snapped to 256). Production users got different aspect ratio than upstream | HARMFUL | FIXED — `presets.py:50-84` and `latent_preparation.py:130-133` now use `480x256` |
| 10 | Audio decoding silently returned no audio if `batch.audio_latents` missing (joint AV makes this a real bug) | AMBIGUOUS→HARMFUL | FIXED — `audio_decoding.py:90-96` now raises `ValueError` |
| 7 (stale) | Parity test FV scheduler helper claimed "double-shift" | (false alarm) | Already fixed in Wave 1A; audit was reading stale state |

Other items from audit (BENIGN or out-of-scope for base T2AV): distill DDIM shortcut (cfg_number=1 path), Turbo VAE default (out-of-scope), A2V branch (out-of-scope), text_offset propagation (BENIGN for default v2 coords), frame_receptive_field (BENIGN for base local_attn_layers=[]), seed fallback (AMBIGUOUS edge case).

**Parity-test test edit (Wave 7.5)**: pipeline parity test now encodes real preset prompts via T5-Gemma (`test_magi_human_pipeline_parity.py:59-153, 388-395`) instead of random tensors. Validates that production-facing preset values flow through the test path.

**Critical caveat — parity numbers DON'T move with these fixes**: The parity test uses the SAME encoder/decoder/tokenizer on both FV and upstream sides. So fixing tokenizer-side pre-padding doesn't change FV-vs-upstream parity (both sides got the same wrong → now both get the same right). Wave 8 fixes are real PRODUCTION improvements (actual user inference now matches upstream's tokenization, resolution, and joint-AV invariants) but the residual ~0.47 (video) / ~1.0 (audio) drift in 4-step pipeline parity is the inherent bf16+CFG amplification floor through the multistep FlowUniPC scheduler.

Per-test parity numbers post-Wave-8:
| Test | Status | diff_max | diff_mean |
|---|---|---:|---:|
| DiT parity (single forward) | FAIL @ atol=0.03 | 0.057 | 0.0053 |
| T5-Gemma parity | PASS | 0.0 | 0.0 |
| Wan VAE parity (loose per OQ-7) | PASS @ atol=1e-3 | 8e-4 | 5e-5 |
| SA Audio VAE parity | PASS | 0.0 | 0.0 |
| SA official parity (NEW Wave 7) | PASS | 0.0 | 0.0 |
| Pipeline parity (real prompts, 4-step) | FAIL @ atol=0.40 | video 6.69 / audio 3.45 | video 0.47 / audio 1.01 |

OQ-6 status update:
- **Production-facing root causes**: ALL identified and FIXED — incomplete neg prompt (Wave 7), tokenizer pre-padding (Wave 8 #1), resolution defaults (Wave 8 #3), silent fallbacks (Wave 7 + Wave 8 #10).
- **Parity-test compounding**: bf16+CFG inherent amplification floor. Cannot be improved without fp32 sensitive ops or a less-amplifying scheduler. Tracked as `RESOLVED-PRODUCTION` for OQ-6 with a separate `OPEN-IF-NEEDED` follow-up for fp32 path investigation.

### Wave 9 (2026-05-01): activation trace infrastructure

Built Extension 0 of FastVideo's activation trace mode at `fastvideo/hooks/activation_trace.py` (env-gated zero-overhead module forward hooks). Designed for parity-debug across model ports — enable on both FastVideo's and upstream's path, diff resulting JSONL files to find first divergent layer.

Key design properties:
- `FASTVIDEO_TRACE_ACTIVATIONS=1` master toggle. Off = single env var lookup at startup, no hooks ever registered.
- `FASTVIDEO_TRACE_LAYERS=<regex>` selective filter.
- `FASTVIDEO_TRACE_STATS=abs_mean,sum,max,...` configurable per-tensor stats.
- `FASTVIDEO_TRACE_STEPS=0,1,5` step-indexed dumps via `trace_step(idx)` context manager.
- Output: JSONL records to `FASTVIDEO_TRACE_OUTPUT` path.

E2E smoke confirmed: 28,864 records generated against the magi-human pipeline.

Documentation at `docs/contributing/activation_trace.md`. Future Extensions 1-3 (FX/AST/dispatch) designed but not implemented.

Companion skill at `~/.config/opencode/skill/add-model-trace/` (template for one-off ad-hoc port investigations) is unchanged.

### Wave 10 (2026-05-01): WanVideo-pattern dtype refactor

Removed all 7 hardcoded `.to(torch.bfloat16)` casts in `fastvideo/models/dits/magi_human.py`. These were verbatim copies of upstream `daVinci-MagiHuman/inference/model/dit/dit_module.py` (lines 619, 507, 650, 694, 696). FV now follows the canonical FastVideo dtype pattern exemplified by `fastvideo/models/dits/wanvideo.py`: model dtype is **loader-owned** via `pipeline_config.dit_precision` → `default_dtype` in `component_loader.py`. Inside DiT forward, `orig_dtype = self.linear_qkv.weight.dtype` (or equivalent) is captured and used for output preservation; no hardcoded model-dtype casts remain. The top-level block-input cast (formerly `x.to(torch.bfloat16)`) is now `x.to(<loader-owned dtype>)`.

Refactored sites:
- attention pre_norm output (line ~360)
- q/k/v post-RoPE casts (lines ~399-401)
- attention output (line ~411)
- MLP pre_norm + activation casts (lines ~444-447)
- top-level block-input cast (line ~684)

Production behavior unchanged: bf16 parity numbers identical to baseline (`diff_max=0.057, diff_mean=0.005`). Loader's `dit_precision="bf16"` default → all params/inputs bf16 → `orig_dtype = bf16` → outputs preserved as bf16 → same as before.

fp32 parity now works end-to-end on the FV side (model is dtype-agnostic in forward), but the parity test against upstream still shows bf16-noise residual drift (post-refactor: `diff_max=0.061, diff_mean=0.0068`; pre-refactor was `0.082 / 0.0079`, ~1.2x improvement). The remaining drift is from upstream `dit_module.py` itself — upstream still hardcodes `.to(torch.bfloat16)` in its forward, so even in an fp32 parity run, upstream's intermediate tensors are bf16. **Fully fp32-clean parity would require either patching the local upstream clone OR using a build of upstream where the hardcoded casts are also config-driven.**

OQ-9 (NEW): upstream `daVinci-MagiHuman/inference/model/dit/dit_module.py` has hardcoded `.to(torch.bfloat16)` casts at lines 619, 507, 650, 694, 696. For full fp32 parity validation, these would need to be patched in the local clone OR a flag added upstream. Tracked as low-priority follow-up; affects only fp32 parity testing, not production.

### Wave 14 (2026-05-02): upstream E2E coherent vs FV E2E noise (REAL bug confirmed)

Ran the upstream `daVinci-MagiHuman` pipeline end-to-end with the same prompt + seed (42) + steps (32) + resolution (480x256) used by `examples/inference/basic/basic_magi_human.py`. Required installing `magi_compiler` from the local subdir, `alias_free_torch`, and downgrading `diffusers` per upstream's pinned version.

**Result**: upstream produces a **coherent** video — young woman in a pink shirt reading a red book on a park bench surrounded by green trees, matching the prompt. Reference at `/tmp/opencode/upstream_magi_base_4s_480x256.mp4` (frames at `/tmp/opencode/upstream_frame_*.png`). FV produces **pure colorful-blob noise** at the same configuration (`outputs_video/magi_human_basic/output_magi_human_*.mp4`).

**This invalidates the Wave 13 "structural / no real bug" verdict** for OQ-6 and reopens it. The bug is in code that production exercises but the parity test bypasses — parity test still passes (~0.5% per-step drift on (2,6,6) tiny synthetic latents) yet production produces noise on real (26,16,30) latents with real text encoding.

#### Falsified candidates so far

1. **T5-Gemma fp16 cast (Candidate A)**. Upstream `t5_gemma_model.py:24-27` casts `outputs["last_hidden_state"].half()` (bf16→fp16) before pad/trim → fp32; FV keeps bf16 → fp32 (`fastvideo/pipelines/basic/magi_human/pipeline_configs.py:t5gemma_postprocess_text`). Parity test bypasses this because it uses FV's encoder for both upstream and FV sides (`tests/local_tests/magi_human/test_magi_human_pipeline_parity.py:118-147`). Applied `outputs.last_hidden_state.to(torch.float16)` in the postprocess function and reran the basic example → **still pure noise**, visually identical to before. Reverted.

2. **Local-window video→video attention (Candidate B)**. Upstream `MagiDataProxy.process_input` returns 5 args including `local_attn_handler` (`daVinci-MagiHuman/inference/pipeline/data_proxy.py:319-382`); FV's `MagiHumanDiT.forward` only takes `(x, coords, mm)` and uses full SDPA. **Verified to be inert for the base model**: upstream `local_attn_layers` config defaults to `[]` for the base BR pipeline (`daVinci-MagiHuman/inference/common/config.py:71`); only the SR_1080 pipeline sets non-empty layer indices (lines 229-241). Base-model upstream uses `flash_attn_with_cp` (full attention) at `dit_module.py:644-645`, equivalent to FV's full SDPA.

#### Root cause + fix (Oracle, 2026-05-02)

**Bug**: FV's `_img2tokens` packed video latents as **spatial-major** `(pT pH pW C)` (channels innermost) at `fastvideo/pipelines/basic/magi_human/stages/latent_preparation.py:84`. Upstream's `MagiDataProxy.process_input` uses `UnfoldNd(...)` at `daVinci-MagiHuman/inference/pipeline/data_proxy.py:287-317`, which is implemented via a grouped convolution (`groups=in_channels`) that reshapes to `(batch, in_channels * kernel_size_numel, -1)` (`unfoldNd/unfold.py:66`) — i.e. **channel-major** `(C pT pH pW)` (channels slowest). The DiT's `video_embedder` (`Linear(192, 5120)`) was trained on the channel-major layout. Spatial-major input silently permutes the in-features of every video token, scrambling the entire feature representation and producing pure noise.

**Why parity test passed**: `test_magi_human_pipeline_parity.py:222` imports FV's `build_packed_inputs` for the upstream side too, so both sides ate the same FV-spatial-major tokens and agreed on equally-wrong inputs. Production faces real DiT weights and breaks.

**Fix**: One-character rearrange-string change in `_img2tokens`:
```diff
- "B C (T pT) (H pH) (W pW) -> B (T H W) (pT pH pW C)"
+ "B C (T pT) (H pH) (W pW) -> B (T H W) (C pT pH pW)"
```

`unpack_tokens` keeps spatial-major `(pT pH pW C)` because the DiT's `final_linear_video` was trained to emit that layout, mirroring upstream's `SingleData.depack_token_sequence` at `data_proxy.py:220-228`.

**Validation**: Reran `examples/inference/basic/basic_magi_human.py` at the standard 480x256 / 32 step / seed 42 prompt. Output is **coherent video** matching the prompt — woman in teal sweater on a wooden park bench reading a book, green trees, sunny park scene. Output mp4 size dropped from ~932 KB (incompressible noise) to ~222 KB (coherent video). Frame samples at `/tmp/opencode/channelmajor_frame_*.png`.

#### Wave 14 follow-up (2026-05-02): re-running parity exposed dtype-boundary divergences

After fixing the channel-major bug, both DiT and pipeline parity tests started failing with much larger diffs than the pre-fix baseline (DiT diff_max=0.56 vs old "0.057"; pipeline video diff_mean=0.89 vs old 0.47). The pre-fix "0.057" baseline turned out to be a *garbage-in-garbage-out cancellation*: with both sides processing scrambled tokens, the kernel-level differences (TORCH_SDPA vs flash_attn) happened to converge on noise-equilibrium output. Once the inputs were correct, the underlying dtype-boundary divergences from upstream became visible.

Three additional fixes brought parity to bit-exact:

1. **Attention dtype boundary mirrors upstream**: FV now hardcodes the bf16 cast for SDPA inputs (matching `daVinci-MagiHuman/inference/model/dit/dit_module.py:508` `flash_attn_with_cp` which `q.to(bf16), k.to(bf16), v.to(bf16)` regardless of weight dtype). The attention output is upcast to fp32 before the per-head gating multiply (matching upstream's `bf16 * fp32` promotion at `dit_module.py:649`), and the gated result is cast to bf16 only for `linear_proj`. Wave 10's "dtype-agnostic" `orig_dtype` cast at the SDPA call was silently running fp32 attention whenever weights happened to be fp32 (e.g., parity-test load path). Fix in `fastvideo/models/dits/magi_human.py:MagiAttention.forward`.

2. **fp32 residual stream**: removed the `x.to(linear_qkv.weight.dtype)` cast at `MagiHumanDiT.forward` (was line 689). Upstream casts to `params_dtype` which defaults to fp32, so the residual stream stays fp32 across all 40 layers — internal compute still bf16, but the cross-layer accumulator is fp32. FV's bf16 residual was compounding ~6-7 bits of mantissa loss per layer × 40 layers = visible parity drift. Fix in `fastvideo/models/dits/magi_human.py:MagiHumanDiT.forward`.

3. **Pipeline parity test scheduler single-shift**: `_build_fastvideo_schedulers` was still constructing `FlowUniPCMultistepScheduler(shift=shift)` and then calling `set_timesteps(... shift=shift)` (double-shift), but production was migrated to single-shift in Wave 11 (`magi_human_pipeline.py:146-149` + `denoising.py:105-116`). The test helper had a stale docstring. Fix in `tests/local_tests/magi_human/test_magi_human_pipeline_parity.py:_build_fastvideo_schedulers`.

**Final parity numbers** (8 of 8 tests passing, 7 of 8 bit-exact):

| Test | diff_max | diff_mean |
|---|---|---|
| `test_magi_human_dit_parity` | 0.0 | 0.0 |
| `test_magi_human_t5gemma_parity` | 0.0 | 0.0 |
| `test_magi_human_sa_audio_parity` | 0.0 | 0.0 |
| `test_magi_human_sa_audio_official_parity` | 0.0 | 0.0 |
| `test_magi_human_vae_parity` | 8.0e-4 | 4.9e-5 |
| `test_magi_human_pipeline_latent_parity` | 0.0 | 0.0 |
| `test_magi_human_pipeline_smoke` (2 cases) | passes | passes |

Production E2E re-validated post-fix: still produces coherent video at the standard 480x256 / 32 step / seed 42 prompt; runtime unchanged (23.5s).

**OQ-6 RESOLVED** (Wave 14, full resolution including dtype boundaries).

#### Parity test fidelity follow-up (separate issue)

The pipeline parity test should be updated to use upstream's *real* `MagiDataProxy.process_input` for the upstream side (instead of importing FV's `build_packed_inputs`), so it can catch this class of "both sides use FV's helper, both consume scrambled tokens, parity passes" bypass in the future. Tracked as OQ-11.

### Potential mitigations (not investigated this session)

- Run sensitive ops (MM-layer pre-norm, attention) in fp32 instead of bf16.
- Match upstream's exact dtype boundaries around MLP activation (verify FV does
  the same fp32 cast upstream does in `_BF16ComputeLinear`).
- Use a more numerically stable scheduler (FlowUniPC may have known issues at
  certain step counts).
- Per-modality `up_gate_proj` drill to find the first diverging activation.

### Per-side layer logs and drill methodology

Layer-by-layer traces are written to:

- `/tmp/opencode/magi_dit_up_layers.log` (upstream reference)
- `/tmp/opencode/magi_dit_fv_layers.log` (FastVideo)

These are produced by `tests/local_tests/magi_human/_debug_magi_human_block_parity.py`
via forward hooks registered on each transformer block. The `add-model-trace`
skill at `~/.config/opencode/skill/add-model-trace/` generalizes this
methodology for future ports: forward-hook + monkey-patch + git-stash-cleanup
with hard rules around no-source-residue cleanup.

## Open questions / blockers

| ID | Item | Status |
|---|---|---|
| OQ-1 | **Native T5-Gemma port.** Full Phase 11 compliance requires a native FastVideo T5-Gemma implementation with no HF model-class imports in production code. Multi-week scope. | TRACKED FOLLOW-UP |
| OQ-2 | **SSIM reference videos not seeded.** `fastvideo/tests/ssim/test_magi_human_similarity.py` skips cleanly until reference videos are uploaded to `FastVideo/ssim-reference-videos` on HF via the `seed-ssim-references` skill on Modal L40S. | TRACKED FOLLOW-UP |
| OQ-3 | **Audio quality regression metric.** Mel-spectrogram L1 / multi-resolution STFT regression deferred per `tests/local_tests/stable-audio.md` precedent. | DEFERRED |
| OQ-4 | **`_find_base_shard_dir` is fragile across HF-cache configurations.** Wave 1 fixed the loader with `snapshot_download(repo_id, allow_patterns=['base/*.safetensors'])` fallback in 3 files. `MAGI_HUMAN_BASE_SHARD_DIR` still works as an override but is no longer required. | RESOLVED |
| OQ-5 | **Basic-example output mp4 visual quality is impressionistic at 256x448.** Root cause identified: OQ-6 (pre-existing compounding bf16 drift over the 32-step denoise loop). Wave 2-3 investigation confirmed the 4-step pipeline parity shows 18.85x compounding ratio vs expected 4x linear. See OQ-6 for full details and mitigation candidates. | RESOLVED-ROOT-CAUSE-IDENTIFIED (see OQ-6) |
| OQ-6 | **Video patch packing was spatial-major instead of channel-major.** Wave 14 (2026-05-02) ran upstream E2E and got coherent output; FV produced pure noise at same config. Oracle triage identified the bug in `_img2tokens` rearrange order: FV used `(pT pH pW C)` (spatial-major) but the DiT's `video_embedder` Linear weight was trained on the channel-major `(C pT pH pW)` layout that upstream's `UnfoldNd` (grouped-conv reshape, `unfoldNd/unfold.py:66`) produces. The pipeline parity test imported FV's `build_packed_inputs` for both sides at `test_magi_human_pipeline_parity.py:222`, so it consumed equally-permuted tokens on both sides and reported agreement on garbage. Fixed in `latent_preparation.py:_img2tokens` by changing the einops pattern from `(pT pH pW C)` to `(C pT pH pW)`. Validated end-to-end: `examples/inference/basic/basic_magi_human.py` now produces coherent video matching the prompt (woman on park bench reading a book, green trees). Earlier waves' production-side fixes (negative prompt, tokenizer padding, resolution defaults, silent-audio fallback) all still stand. | RESOLVED — Wave 14 |
| OQ-11 | **Pipeline parity test imports FV's `build_packed_inputs` for the upstream side.** `tests/local_tests/magi_human/test_magi_human_pipeline_parity.py:222` calls FV's packer for both sides instead of upstream's real `MagiDataProxy.process_input`. This let the channel-major-vs-spatial-major bug (OQ-6, Wave 14) sit silent for weeks because both sides agreed on the wrong layout. Update the parity test to drive the upstream side through `MagiDataProxy.process_input` so future packing-layout regressions are caught at parity time, not at production E2E. | TRACKED FOLLOW-UP |
| OQ-7 | **Wan VAE shared fp32 op-order drift (MEDIUM PRIORITY).** FV uses `z * std + mean` at decode normalization; upstream uses `z / (1/std) + mean`. Bitwise non-equivalent in fp32. Affects all Wan-family pipelines (`fastvideo/configs/pipelines/wan.py`, `turbodiffusion.py`, `longcat.py`, magi-human). Magi VAE test loosened to `atol=1e-3, rtol=1e-3` (Wave 4) to defer. Tighten back to `atol=1e-4` once the Wan VAE op-order fix lands. Fix should be validated against Wan2.1, Wan2.2, and magi-human. Estimated 0.5-1 day to fix and validate. | TRACKED FOLLOW-UP |
| OQ-9 | **Upstream `dit_module.py` hardcoded bf16 casts block full fp32 parity validation.** `daVinci-MagiHuman/inference/model/dit/dit_module.py` has hardcoded `.to(torch.bfloat16)` casts at lines 619, 507, 650, 694, 696. FV's DiT forward is now dtype-agnostic (Wave 10), but parity tests against upstream still show bf16-noise residual drift in fp32 runs because upstream's intermediate tensors are bf16. Full fp32-clean parity would require patching the local upstream clone or adding a dtype-config flag upstream. Affects only fp32 parity testing, not production. | TRACKED FOLLOW-UP (LOW PRIORITY) |

## Troubleshooting

**`RuntimeError: Upstream DiT missing 331 keys` despite shards being present.**
This happens when the upstream base shards are downloaded into one HF cache
(e.g. `~/.cache/huggingface/hub/`) but `_find_base_shard_dir` resolves the
snapshot via a different cache path (e.g. `/raid/huggingface/hub/...`) where
only `model.safetensors.index.json` is present, not the 7 shard files.

**Workaround**: explicitly set `MAGI_HUMAN_BASE_SHARD_DIR` to the snapshot dir
that actually contains the `model-0000*-of-00007.safetensors` shards:

```bash
export MAGI_HUMAN_BASE_SHARD_DIR=~/.cache/huggingface/hub/models--GAIR--daVinci-MagiHuman/snapshots/<sha>/base
```

Tracked as open question **OQ-4** for a more robust loader.

**`401 Unauthorized` on any gated repo.** Check `echo $HF_TOKEN` and confirm
you've accepted the model terms at each URL listed in §1. The four repos have
separate terms pages; accepting one doesn't cover the others.

- T5-Gemma: https://huggingface.co/google/t5gemma-9b-9b-ul2
- Stable Audio Open: https://huggingface.co/stabilityai/stable-audio-open-1.0
- Wan 2.2 TI2V-5B: https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B
- daVinci-MagiHuman: https://huggingface.co/GAIR/daVinci-MagiHuman

**Override the base shard directory.** If you have the raw MagiHuman shards
at a non-default path, point the tests at them:

```bash
export MAGI_HUMAN_BASE_SHARD_DIR=/path/to/raw/shards
```

**Override the converted weights path.** If you ran the conversion script with
a custom `--output` path, tell the tests where to find it:

```bash
export MAGI_HUMAN_DIFFUSERS_PATH=/path/to/converted_weights/magi_human_base
```

**Missing `daVinci-MagiHuman/` clone.** Tests that need the upstream reference
(`test_magi_human_parity.py`, `test_magi_human_pipeline_parity.py`) skip
cleanly with a message pointing to the clone command in §3. The VAE parity
tests and the smoke test don't need the clone.

**OOM during DiT load.** The base DiT loads in bf16 by default. If you're
tight on VRAM, use `--cast-bf16` during conversion to ensure the transformer
shards are stored in bf16 rather than fp32. The distill variant is the same
size; both fit on a single 80 GB GPU.

**Wall-clock blew up past 10 min.** The first run downloads T5-Gemma (~18 GB),
the Wan VAE, and the Stable Audio VAE if they aren't cached. See the pre-warm
step in §5.

## Adding new parity tests for this family

`tests/local_tests/helpers/magi_human_upstream.py` contains shared reference
loaders for the upstream DiT, VAE, and pipeline. Use these as the starting
point for any new parity test rather than duplicating the load logic.

The `_debug_magi_human_block_parity.py` and `_debug_magi_human_weight_diff.py`
scripts in `tests/local_tests/magi_human/` are scratch tools for divergence
investigation. They are NOT pytest tests and must NOT be promoted to formal
tests. Run them directly with `python` when you need to inspect per-block diffs
or weight mismatches during a parity-debug session.

If you need to chase per-layer divergence on a future add-model port, see the
`add-model-trace` skill at `~/.config/opencode/skill/add-model-trace/`.
Generalized from `tests/local_tests/magi_human/_debug_magi_human_block_parity.py`
(the worked magi example), it provides a forward-hook + monkey-patch +
git-stash-cleanup methodology with hard rules around no-source-residue cleanup.
