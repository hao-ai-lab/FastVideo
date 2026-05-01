---
name: seed-ssim-references
description: Seed HF reference artefacts for a single newly-added SSIM test (pixel `.mp4` for `run_text_to_video_similarity_test`-style tests, or latent `.pt` for `run_text_to_latent_similarity_test`-style tests). Runs the test on Modal L40S, downloads the generated artefacts via `modal volume get`, pauses for the user to verify (visual eyeball for mp4, numerics dump for pt), then uploads only that test's files to `FastVideo/ssim-reference-videos`. Use when a new `fastvideo/tests/ssim/test_*_similarity.py` has just been added and has no references on HF yet.
---

# Seed SSIM Reference Artefacts (mp4 or pt)

## Purpose

A brand-new SSIM test in `fastvideo/tests/ssim/` fails forever until its
reference artefacts exist on the HF dataset
(`FastVideo/ssim-reference-videos`). The dataset hosts two kinds of artefacts
side-by-side per `(model_id, backend, prompt)`:

- **`.mp4`** — pixel ground-truth for tests that call
  `run_text_to_video_similarity_test` / `run_image_to_video_similarity_test`
  in `inference_similarity_utils.py`. Compared via SSIM.
- **`.pt`** — pre-VAE latent bundle (fp16 full latent + fp32 slice +
  metadata + `slice_spec` + `format_version`) for tests that call
  `run_text_to_latent_similarity_test` in `latent_similarity_utils.py`.
  Compared via cosine distance on the slice and the full tensor.

This skill:

1. Detects which artefact type the test produces (pixel vs latent).
2. Runs the test on Modal's L40S pool to generate the artefacts.
3. Downloads them to the local repo via `modal volume get`.
4. Pauses so the user can verify quality:
   - **mp4**: visual eyeball in a video player.
   - **pt**: numerics dump (shape, slice stats, NaN/Inf check, metadata).
5. Uploads only the new test's files to HF, with a guard that refuses to
   overwrite anything already present.

The skill is run **manually**, once per new test. Before invoking it, the user
has already sanity-tested the new test locally — it launches `VideoGenerator`
and writes an artefact without crashing (the missing-reference assertion at
the end is expected). The skill does not re-test locally; it goes straight
to Modal L40S (which is what CI uses).

## When to use

- A new `test_*_similarity.py` file has been added in `fastvideo/tests/ssim/`
  and the HF dataset has no `reference_videos/default/L40S_reference_videos/<model_id>/`
  subtree for it yet.

## When not to use

- Regular CI runs — once refs exist, `pytest fastvideo/tests/ssim/` downloads
  them automatically.
- Re-seeding an existing test. That requires `--force` on the upload step, and
  is out of scope here; treat as a separate, deliberate operation.

## Inputs

The skill has **one required input**: the path to the new SSIM test file.
Prompt the user for it if they didn't supply it.

| Parameter | Required | Description |
|-----------|----------|-------------|
| `test_file` | Yes | e.g. `fastvideo/tests/ssim/test_ltx2_similarity.py`. The skill's first action is to ask for this if missing. |

Everything else is fixed:

- Modal runner GPU: **L40S** (hardcoded in `fastvideo/tests/modal/ssim_test.py`).
- Device folder: `L40S_reference_videos`.
- Quality tier: `default` (the tier CI runs). The `full_quality` tier is not
  seeded by this skill.
- HF repo: `FastVideo/ssim-reference-videos` (dataset).
- Multi-model test files: all model ids in `*_MODEL_TO_PARAMS` are seeded
  together; the Modal run produces one mp4 per (model, prompt, backend) and
  the upload scopes by `--model-id`, looping if there is more than one.

## Prerequisites

The user has confirmed:

- `modal` CLI authenticated.
- `HF_API_KEY` (or `HUGGINGFACE_HUB_TOKEN` / `HF_TOKEN`) exported with write
  access to `FastVideo/ssim-reference-videos`.
- The test file runs locally end-to-end (generates an mp4; SSIM assertion
  failure due to missing reference is expected and fine).

Fail fast if the token env var is missing.

## Steps

### 1. Ask for the test file, then detect artefact type

If the user didn't name one, ask: *"Which SSIM test file do you want to seed
references for? (e.g. `fastvideo/tests/ssim/test_ltx2_similarity.py`)"*.

Validate:

- Path exists and matches `fastvideo/tests/ssim/test_*_similarity.py`.
- File defines a `*_MODEL_TO_PARAMS` dict — grep it to extract the set of
  model ids. Those ids drive step 5.

Detect artefact type by inspecting the file's imports / helper call:

- **latent** (`.pt`) — file imports `run_text_to_latent_similarity_test`
  from `fastvideo.tests.ssim.latent_similarity_utils` (or any other helper
  that ends with `_latent_similarity_test`).
- **pixel** (`.mp4`) — file imports
  `run_text_to_video_similarity_test` / `run_image_to_video_similarity_test`
  from `fastvideo.tests.ssim.inference_similarity_utils`, OR uses the
  legacy custom-inline helper pattern (see `test_gamecraft`,
  `test_longcat`, etc.). Default to pixel when both heuristics fail.

Record `ARTEFACT_TYPE ∈ {pixel, latent}` for use in step 4. Steps 2, 3, 5,
and 6 are artefact-type-agnostic — `_iter_reference_files`,
`copy_generated_to_reference`, and `upload_reference_videos` already walk
both `.mp4` and `.pt` (see `reference_videos_cli.py`).

If either check fails, stop and tell the user what's wrong.

### 2. Run the test on Modal L40S

Pick a subdir name so repeated runs don't collide:

```bash
SHORT_COMMIT=$(git rev-parse --short=12 HEAD)
TIMESTAMP=$(date -u +%Y%m%d_%H%M%S)
SUBDIR="${TIMESTAMP}_${SHORT_COMMIT}"
```

Then launch the Modal run. The `IMAGE_VERSION` and `BUILDKITE_*` env-prefix
**must** match what CI exports in `.buildkite/scripts/pr_test.sh`, otherwise
`fastvideo/tests/modal/ssim_test.py` resolves a different GHCR image tag
(default is `latest`, CI is `py3.12-latest`) and bakes different values into
the image's frozen env block (`ssim_test.py:17-18, 38-46`). Mismatched image
or env produces SSIM drift that doesn't show up until the same commit runs
in CI.

```bash
IMAGE_VERSION="py3.12-latest" \
BUILDKITE_REPO="$(git config --get remote.origin.url)" \
BUILDKITE_COMMIT="$(git rev-parse HEAD)" \
BUILDKITE_PULL_REQUEST="${BUILDKITE_PULL_REQUEST:-false}" \
modal run fastvideo/tests/modal/ssim_test.py \
    --git-repo="$(git config --get remote.origin.url)" \
    --git-commit="$(git rev-parse HEAD)" \
    --hf-api-key="$HF_API_KEY" \
    --test-files="<test_file>" \
    --sync-generated-to-volume \
    --generated-volume-subdir="$SUBDIR" \
    --skip-reference-download \
    --no-fail-fast
```

Env prefix rationale (parity with CI; see `.buildkite/pipeline.yml:1-3` and
`.buildkite/scripts/pr_test.sh:62-83`):
- `IMAGE_VERSION=py3.12-latest`: pins the Modal image tag to the same one CI
  uses. Without this, `ssim_test.py:17` falls back to `latest`, which on
  GHCR is built from `Dockerfile.python3.10` — different Python, torch, and
  flash-attn wheel than CI's `py3.12-latest` (`infra-build-image.yml:51-67`,
  `_template-build-image.yml:65-101`).
- `BUILDKITE_REPO`/`BUILDKITE_COMMIT`/`BUILDKITE_PULL_REQUEST`: mirror what
  Buildkite exports. `ssim_test.py:38-46` bakes these into the image's
  `.env(...)` block; mismatched values can perturb in-container code paths
  that branch on PR-vs-non-PR. `false` for `BUILDKITE_PULL_REQUEST` matches
  Buildkite's "non-PR build" sentinel.

Flag rationale:
- `--skip-reference-download`: no refs exist yet, so conftest must not try to
  pull them.
- `--no-fail-fast`: lets the test finish generation before `_assert_similarity`
  raises `FileNotFoundError: Reference video folder does not exist`. The
  expected failure is what we want — the mp4 has already been written.
- `--sync-generated-to-volume` + `--generated-volume-subdir`: copies the
  generated mp4s to the `hf-model-weights` Modal volume under
  `ssim_generated_videos/default/<SUBDIR>/generated_videos/` so we can pull
  them locally.

The Modal run will end with a nonzero exit (expected) and print a
`modal volume get hf-model-weights ssim_generated_videos/default/<SUBDIR>/generated_videos ./generated_videos_modal/default`
command. Capture that `<SUBDIR>` — you need it for step 3.

### 3. Download generated videos locally

```bash
modal volume get --force hf-model-weights \
    ssim_generated_videos/default/"$SUBDIR"/generated_videos \
    ./generated_videos_modal/default
```

`--force` is required when the parent `./generated_videos_modal/default`
already exists; without it, `modal volume get` errors with `[Errno 21] Is a
directory`. Safe to pass on the first run too.

After this, the mp4s live at
`./generated_videos_modal/default/generated_videos/L40S_reference_videos/<model_id>/<backend>/<prompt>.mp4`.
The extra `generated_videos/` level comes from the volume layout in
`_sync_generated_videos_to_volume` (`ssim_test.py`) — the command copies
`<repo>/fastvideo/tests/ssim/generated_videos/<tier>` to
`ssim_generated_videos/<tier>/<SUBDIR>/generated_videos/`, and `modal volume
get` preserves that trailing `generated_videos/` segment.

### 4. PAUSE — user reviews quality

Type-aware verification.

**For `ARTEFACT_TYPE = pixel`** — list the downloaded mp4s and ask the user to
open them in a video player:

> "Generated videos downloaded to `./generated_videos_modal/default/generated_videos/L40S_reference_videos/`. Please open them and confirm the quality looks correct. Reply **`upload`** to continue, or anything else to abort."

**For `ARTEFACT_TYPE = latent`** — `.pt` files are not human-watchable. Print
a numerics dump for each `.pt` so the user can sanity-check shape, distribution,
and metadata:

```python
import torch
from pathlib import Path
ROOT = Path("./generated_videos_modal/default/generated_videos/L40S_reference_videos")
for p in sorted(ROOT.rglob("*.pt")):
    d = torch.load(p, map_location="cpu", weights_only=False)
    s = d["expected_slice"]
    L = d["latent"].float()
    print(f"=== {p.relative_to(ROOT)} ===")
    print(f"  format_version: {d['format_version']}")
    print(f"  shape:          {d['shape']}")
    print(f"  dtype_original: {d['dtype_original']}")
    print(f"  slice_spec:     {d['slice_spec']}")
    print(f"  slice  shape={tuple(s.shape)}  mean={s.mean():+.4f}  std={s.std():.4f}  min={s.min():+.4f}  max={s.max():+.4f}")
    print(f"  latent shape={tuple(L.shape)}  mean={L.mean():+.4f}  std={L.std():.4f}  min={L.min():+.4f}  max={L.max():+.4f}")
    print(f"  finite: latent NaN={torch.isnan(L).any().item()} Inf={torch.isinf(L).any().item()}; "
          f"slice NaN={torch.isnan(s).any().item()} Inf={torch.isinf(s).any().item()}")
    print(f"  metadata: {d['metadata']}\n")
```

Sanity criteria:
- `format_version == 1` (matches `LATENT_REFERENCE_FORMAT_VERSION`).
- `shape` matches what the model produces (e.g. LTX-2 distilled =
  `[1, 128, T_lat, H_lat, W_lat]`; Stable Audio Open 1.0 = `[1, 64, 1024]`).
- `slice_spec.kind` matches a registered kind (`corner_3x3_first_frame`
  for video, `audio_first_8_timesteps` for audio).
- No `NaN`/`Inf`. `mean ≈ 0`, `std ≈ 1` (denoised latents stay close to
  the initial Gaussian distribution; very wide deviations suggest
  numerical drift).
- `metadata.prompt` matches the test's prompt.

Then ask:

> "Numerics look right? Reply **`upload`** to continue, or anything else to abort."

Do not proceed until the user explicitly says `upload`. If they abort, leave
everything on disk so they can inspect further — no cleanup.

### 5. Copy into the local reference layout

Scoped copy — only the new test's artefacts. Single command works for both
artefact types because `_iter_reference_files` walks `.mp4` and `.pt`:

```bash
python fastvideo/tests/ssim/reference_videos_cli.py copy-local \
    --quality-tier default \
    --device-folder L40S_reference_videos \
    --generated-dir ./generated_videos_modal/default/generated_videos/L40S_reference_videos
```

(The `--generated-dir` points at the device-folder root inside the
downloaded tree; `copy-local` walks all `<model>/<backend>/*.{mp4,pt}`
underneath it. Since the Modal run was scoped to a single test file via
`--test-files`, only that test's model(s) are present — so the copy is
implicitly per-test.)

Result for pixel: `fastvideo/tests/ssim/reference_videos/default/L40S_reference_videos/<model_id>/<backend>/<prompt>.mp4`.
Result for latent: same path with `.pt` extension.

### 6. Upload to HF — scoped per model_id, with overwrite guard

For each `<model_id>`:

```bash
python fastvideo/tests/ssim/reference_videos_cli.py upload \
    --quality-tier default \
    --device-folder L40S_reference_videos \
    --model-id "<model_id>"
```

The upload command:

- Uploads **only** `reference_videos/default/L40S_reference_videos/<model_id>/`.
- **Refuses** if any file already exists at that path on HF (this is the
  guard — seeding a new test should never clobber existing refs). To override,
  the user must re-run with `--force`. If the guard fires, stop and report
  exactly which files exist; do not silently `--force`.

Reads the HF token from `HF_API_KEY` / `HUGGINGFACE_HUB_TOKEN` / `HF_TOKEN`.

### 7. Report success

List what was uploaded (paths in repo) and remind the user to push any
related code changes. Do **not** auto-verify by re-running Modal — the user
can run `pytest fastvideo/tests/ssim/<test_file>` later to confirm end-to-end;
it will auto-download the refs they just uploaded.

## Failure modes and how to handle them

- **`HF_API_KEY` unset.** Stop before step 2. The Modal run needs it (passed
  via `--hf-api-key`), and step 6 needs it for upload. If the user
  ran `hf auth login` instead of exporting an env var, read the cached
  token via `huggingface_hub.get_token()` and forward it to Modal as
  `--hf-api-key="$CACHED_TOKEN"`.
- **Modal run fails before generation.** No artefacts on the volume — nothing
  to download. Fix the test locally (`pytest fastvideo/tests/ssim/<test_file>`)
  and retry from step 2.
- **`./generated_videos_modal/default/L40S_reference_videos/` missing after
  `modal volume get`.** The run didn't produce artefacts (most likely the
  test crashed before writing, or `REQUIRED_GPUS` exceeded the partition
  capacity — see Modal logs).
- **Latent test crashed with FSDP / inference_mode error
  (`RuntimeError: Inference tensors do not track version counter`).** The
  test must pass `init_kwargs_override={"use_fsdp_inference": False}` when
  `sp_size == 1` — see `test_stable_audio_similarity.py` for the pattern.
  Fix in the test, push, retry.
- **Upload guard fires (files already exist).** The test name / model id
  collides with something already on HF. Verify the user actually wants to
  replace existing refs; if so, re-run the upload with `--force`. If not,
  rename the model id in `*_MODEL_TO_PARAMS` and re-seed.
- **Quality looks wrong in step 4.** Abort. The artefacts stay on disk for
  inspection. The fix is usually in the test's params (resolution, steps,
  seed) — edit the test, then re-run the skill.
  - For latent: also check `slice_spec.kind` matches the latent rank
    (`corner_3x3_first_frame` requires 5-D, `audio_first_8_timesteps`
    requires 3-D); a rank/kind mismatch raises in `_extract_expected_slice`.

## Design notes (for future skill maintainers)

- The skill deliberately runs on Modal, **not** locally, because the CI
  runner is L40S. Seeding from a different GPU SKU produces refs that CI's
  L40S runs can't match (pixel SSIM drifts across SKUs; latent cosine has
  tighter cross-SKU bf16 drift but the configured tolerances assume
  same-SKU seed → same-SKU verify).
- The skill is default-tier only. `full_quality` refs are seeded by a
  separate, deliberate operation — they double runtime and aren't what CI
  gates on.
- The overwrite guard in `reference_videos_cli.py upload` is default-on
  specifically because this skill exists. Re-seeding is a distinct operation
  that requires explicit `--force`.
- Both artefact types share the same Modal flow: the orchestrator sets
  `--skip-reference-download` + `--no-fail-fast`, runs pytest, the test's
  helper writes the artefact (`.mp4` via `imageio` for pixel,
  `save_latent_reference` → `torch.save` for latent) BEFORE the
  missing-reference assertion raises. `_sync_generated_videos_to_volume` in
  `ssim_test.py` does a `shutil.copytree` of the whole `generated_videos/`
  tree, picking up `.mp4`, `.pt`, and the `*_ssim.json` / `*_latent.json`
  metric files alongside.

## References

- `fastvideo/tests/modal/ssim_test.py` — Modal orchestrator; see
  `--sync-generated-to-volume`, `--generated-volume-subdir`,
  `--skip-reference-download`, `--no-fail-fast`.
- `fastvideo/tests/ssim/reference_videos_cli.py` — `copy-local`, `upload`
  (with `--model-id`, `--force`), `download`, `ensure` subcommands.
  Extension allowlist is `REFERENCE_EXTENSIONS = VIDEO_EXTENSIONS +
  LATENT_EXTENSIONS` (`.pt`).
- `fastvideo/tests/ssim/README.md` — reference layout, HF repo conventions.
- `fastvideo/tests/ssim/inference_similarity_utils.py` — pixel helpers
  (`run_text_to_video_similarity_test`,
  `run_image_to_video_similarity_test`, `build_init_kwargs`).
- `fastvideo/tests/ssim/latent_similarity_utils.py` — latent helper
  (`run_text_to_latent_similarity_test`), slice spec dispatch
  (`_extract_expected_slice`), reference schema
  (`save_latent_reference` / `load_latent_reference`),
  `LATENT_REFERENCE_FORMAT_VERSION`.

## Changelog

| Date | Change |
|------|--------|
| 2026-04-17 | Initial version (Modal sync-to-volume flow). |
| 2026-04-21 | Rewrite: single-test scope, explicit user-review pause, per-`model_id` upload, HF overwrite guard. Dropped `scripts/seed_ssim.sh`. |
| 2026-04-21 | Post-first-run fixes: `modal volume get` needs `--force` when parent exists; download tree has an extra `generated_videos/` level so `--generated-dir` must reflect it. |
| 2026-05-01 | Latent (`*.pt`) artefact support: artefact-type detection in step 1, type-aware verification (visual eyeball for mp4, numerics dump for pt) in step 4, FSDP+inference_mode failure-mode added, design notes for the unified Modal flow. Triggered by PR #1253 (LTX-2 latent migration + Stable Audio latent test). |
