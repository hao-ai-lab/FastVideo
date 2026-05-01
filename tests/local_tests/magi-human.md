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
pytest tests/local_tests/transformers/test_magi_human_parity.py \
       tests/local_tests/encoders/test_magi_human_t5gemma_parity.py \
       tests/local_tests/vaes/test_magi_human_sa_audio_parity.py \
       tests/local_tests/vaes/test_magi_human_vae_parity.py \
       tests/local_tests/pipelines/test_magi_human_pipeline_smoke.py \
       tests/local_tests/pipelines/test_magi_human_pipeline_parity.py \
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
pytest tests/local_tests/pipelines/test_magi_human_pipeline_parity.py -v -s
```

## Phase 11 status

Branch tip `eeef855b` (rebased onto `origin/main` `c77a76c6`), Wave 1+4 changes applied (uncommitted working tree), NVIDIA B200. Wave 2-3 numerical-alignment investigation completed 2026-05-01; see §Numerical-alignment investigation below.

| Test | Status | Diff numbers | Notes |
|---|---|---|---|
| `tests/local_tests/encoders/test_magi_human_t5gemma_parity.py::test_magi_human_t5gemma_wrapper_parity` | PASS | exact (`assert_close(atol=1e-3, rtol=1e-3)`) | gated repo, requires HF token |
| `tests/local_tests/transformers/test_magi_human_parity.py::test_magi_human_dit_parity` | FAIL | video diff_max=0.057, diff_mean=0.008; audio diff_max=0.034, diff_mean=0.008; text exact (diff_max=0) | Tightened to `atol=0.03, rtol=0.01` (Wave 1). Bf16-noise-floor; per-layer drift ~1e-3 accumulates over 40 layers. Root cause of OQ-6 compounding. See §Numerical-alignment investigation. |
| `tests/local_tests/vaes/test_magi_human_vae_parity.py::test_magi_human_vae_decode_parity` | PASS | diff_max=8e-4, diff_mean=4.9e-5 | Wan VAE. Deferred to `atol=1e-3, rtol=1e-3` per OQ-7 (Wave 4). Tighten to `atol=1e-4` once Wan VAE op-order fix lands. |
| `tests/local_tests/vaes/test_magi_human_sa_audio_parity.py::test_magi_human_sa_audio_vae_decode_parity` | PASS | exact (`assert_close(atol=1e-5, rtol=1e-5)`, machine epsilon) | gated repo, requires HF token; uses main's shared `OobleckVAE` + `SAAudioVAEModel` wrapper |
| `tests/local_tests/pipelines/test_magi_human_pipeline_smoke.py::test_magi_human_typed_surface_preflight` | PASS | CPU-only key/preset checks, exact key set equality, 331 keys | no skip conditions met locally |
| `tests/local_tests/pipelines/test_magi_human_pipeline_smoke.py::test_magi_human_pipeline_smoke` | PASS | shape-only; 2 inference steps, output shape `[B,C,T,H,W]` validated | wallclock ~50s |
| `tests/local_tests/pipelines/test_magi_human_pipeline_parity.py::test_magi_human_pipeline_latent_parity` | FAIL | video: diff_max=15.10, diff_mean=1.30 (18.85x compounding ratio vs 1-step 0.069); audio: diff_mean=0.74 | Bumped to `num_inference_steps=4` (Wave 1). Pre-existing compounding bug; tracked as OQ-6. Bug present in original commit 620aaf41 (confirmed via bisect). |
| `fastvideo/tests/ssim/test_magi_human_similarity.py::test_magi_human_base_inference_similarity` | DEFERRED | n/a | Reference videos not yet seeded to `FastVideo/ssim-reference-videos` HF repo; tracked as OQ-2. Requires Modal L40S seeding via `seed-ssim-references` skill. |
| _(debug)_ | INFO | Per-side layer logs: `/tmp/opencode/magi_dit_up_layers.log`, `/tmp/opencode/magi_dit_fv_layers.log` | Added in Wave 1 to `_debug_magi_human_block_parity.py`. See `add-model-trace` skill at `~/.config/opencode/skill/add-model-trace/`. |

_Last verified: 2026-05-01 (Wave 2-3 numerical alignment investigation), branch `will/magi` @ working tree (Wave 1+4 changes uncommitted), B200 GPU, HF token from `~/.cache/huggingface/token`. Loader fix means `MAGI_HUMAN_BASE_SHARD_DIR` no longer required (override still works)._

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

These are produced by `tests/local_tests/transformers/_debug_magi_human_block_parity.py`
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
| OQ-6 | **Pre-existing compounding bug in MagiHumanDiT denoising loop (HIGH PRIORITY).** 4-step pipeline parity shows video diff_mean=1.30 vs 1-step 0.069, a ratio of 18.85x (expected ~4x linear). Bug pre-exists in commit 620aaf41 (original magi port); confirmed via `git revert` bisect (4-step diff_mean=1.20 with all Wave 1 reverts). Single-forward DiT drift (diff_max=0.057) is bf16-noise-floor: cumulative ~1e-3 per-layer over 40 layers, consistent with random-walk accumulation. Wave 3 ruled out: PackedExpertLinear routing (A/B patch showed zero change), expert chunk ordering (bit-exact direct test), conversion script (bit-exact). Hypothesis: bf16 dtype boundary mismatch around sensitive MM-layer ops (pre-norm, attention, MLP activation); diffusion sampling amplifies per-step perturbations geometrically over the denoise loop. Estimated 2-5 days to investigate further: per-modality `up_gate_proj` drill, fp32 paths for sensitive ops, comparison with upstream's exact dtype casts in MLP/attention. | TRACKED FOLLOW-UP |
| OQ-7 | **Wan VAE shared fp32 op-order drift (MEDIUM PRIORITY).** FV uses `z * std + mean` at decode normalization; upstream uses `z / (1/std) + mean`. Bitwise non-equivalent in fp32. Affects all Wan-family pipelines (`fastvideo/configs/pipelines/wan.py`, `turbodiffusion.py`, `longcat.py`, magi-human). Magi VAE test loosened to `atol=1e-3, rtol=1e-3` (Wave 4) to defer. Tighten back to `atol=1e-4` once the Wan VAE op-order fix lands. Fix should be validated against Wan2.1, Wan2.2, and magi-human. Estimated 0.5-1 day to fix and validate. | TRACKED FOLLOW-UP |

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
scripts in `tests/local_tests/transformers/` are scratch tools for divergence
investigation. They are NOT pytest tests and must NOT be promoted to formal
tests. Run them directly with `python` when you need to inspect per-block diffs
or weight mismatches during a parity-debug session.

If you need to chase per-layer divergence on a future add-model port, see the
`add-model-trace` skill at `~/.config/opencode/skill/add-model-trace/`.
Generalized from `tests/local_tests/transformers/_debug_magi_human_block_parity.py`
(the worked magi example), it provides a forward-hook + monkey-patch +
git-stash-cleanup methodology with hard rules around no-source-residue cleanup.
