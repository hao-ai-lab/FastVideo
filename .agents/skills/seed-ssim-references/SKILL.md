---
name: seed-ssim-references
description: Seed HF reference videos for a single newly-added SSIM test. Runs the test on Modal L40S, downloads the generated mp4s via `modal volume get`, pauses for the user to eyeball quality, then uploads only that test's files to `FastVideo/ssim-reference-videos`. Use when a new `fastvideo/tests/ssim/test_*_similarity.py` has just been added and has no references on HF yet.
---

# Seed SSIM Reference Videos

## Purpose

A brand-new SSIM test in `fastvideo/tests/ssim/` fails forever until its
reference videos exist on the HF dataset (`FastVideo/ssim-reference-videos`).
This skill:

1. Runs the test on Modal's L40S pool to generate the videos.
2. Downloads them to the local repo via `modal volume get`.
3. Pauses so the user can eyeball the mp4s and confirm quality.
4. Uploads only the new test's files to HF, with a guard that refuses to
   overwrite anything already present.

The skill is run **manually**, once per new test. Before invoking it, the user
has already sanity-tested the new test locally — it launches `VideoGenerator`
and writes an mp4 without crashing. The skill does not re-test locally; it
goes straight to Modal L40S (which is what CI uses).

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

### 1. Ask for the test file

If the user didn't name one, ask: *"Which SSIM test file do you want to seed
references for? (e.g. `fastvideo/tests/ssim/test_ltx2_similarity.py`)"*.

Validate:

- Path exists and matches `fastvideo/tests/ssim/test_*_similarity.py`.
- File defines a `*_MODEL_TO_PARAMS` dict — grep it to extract the set of
  model ids. Those ids drive step 5.

If either check fails, stop and tell the user what's wrong.

### 2. Run the test on Modal L40S

Pick a subdir name so repeated runs don't collide:

```bash
SHORT_COMMIT=$(git rev-parse --short=12 HEAD)
TIMESTAMP=$(date -u +%Y%m%d_%H%M%S)
SUBDIR="${TIMESTAMP}_${SHORT_COMMIT}"
```

Then launch the Modal run:

```bash
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
modal volume get hf-model-weights \
    ssim_generated_videos/default/"$SUBDIR"/generated_videos \
    ./generated_videos_modal/default
```

After this, the mp4s live at
`./generated_videos_modal/default/L40S_reference_videos/<model_id>/<backend>/<prompt>.mp4`.

### 4. PAUSE — user reviews quality

Print the list of downloaded mp4s and their paths, then stop. Tell the user:

> "Generated videos downloaded to `./generated_videos_modal/default/L40S_reference_videos/`. Please open them and confirm the quality looks correct. Reply **`upload`** to continue, or anything else to abort."

Do not proceed until the user explicitly says `upload`. If they abort, leave
everything on disk so they can inspect further — no cleanup.

### 5. Copy into the local reference layout

Scoped copy — only the new test's mp4s. Loop over each `<model_id>` extracted
in step 1:

```bash
python fastvideo/tests/ssim/reference_videos_cli.py copy-local \
    --quality-tier default \
    --device-folder L40S_reference_videos \
    --generated-dir ./generated_videos_modal/default/L40S_reference_videos
```

(The `--generated-dir` points at the device-folder root; `copy-local` walks
all `<model>/<backend>/*.mp4` underneath it. Since the Modal run was scoped
to a single test file via `--test-files`, only that test's model(s) are
present — so the copy is implicitly per-test.)

Result: `fastvideo/tests/ssim/reference_videos/default/L40S_reference_videos/<model_id>/<backend>/<prompt>.mp4`.

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
  via `--hf-api-key`), and step 6 needs it for upload.
- **Modal run fails before generation.** No mp4s on the volume — nothing to
  download. Fix the test locally (`pytest fastvideo/tests/ssim/<test_file>`)
  and retry from step 2.
- **`./generated_videos_modal/default/L40S_reference_videos/` missing after
  `modal volume get`.** The run didn't produce videos (most likely the test
  crashed before writing, or `REQUIRED_GPUS` exceeded the partition capacity
  — see Modal logs).
- **Upload guard fires (files already exist).** The test name / model id
  collides with something already on HF. Verify the user actually wants to
  replace existing refs; if so, re-run the upload with `--force`. If not,
  rename the model id in `*_MODEL_TO_PARAMS` and re-seed.
- **Quality looks wrong in step 4.** Abort. The mp4s stay on disk for
  inspection. The fix is usually in the test's params (resolution, steps,
  seed) — edit the test, then re-run the skill.

## Design notes (for future skill maintainers)

- The skill deliberately runs on Modal, **not** locally, because the CI
  runner is L40S. Seeding from a different GPU SKU produces refs that CI's
  L40S runs can't match (SSIM drifts across SKUs).
- The skill is default-tier only. `full_quality` refs are seeded by a
  separate, deliberate operation — they double runtime and aren't what CI
  gates on.
- The overwrite guard in `reference_videos_cli.py upload` is default-on
  specifically because this skill exists. Re-seeding is a distinct operation
  that requires explicit `--force`.

## References

- `fastvideo/tests/modal/ssim_test.py` — Modal orchestrator; see
  `--sync-generated-to-volume`, `--generated-volume-subdir`,
  `--skip-reference-download`, `--no-fail-fast`.
- `fastvideo/tests/ssim/reference_videos_cli.py` — `copy-local`, `upload`
  (with `--model-id`, `--force`), `download`, `ensure` subcommands.
- `fastvideo/tests/ssim/README.md` — reference layout, HF repo conventions.
- `fastvideo/tests/ssim/inference_similarity_utils.py` —
  `run_text_to_video_similarity_test` + `_build_init_kwargs`: what each test
  config passes to `VideoGenerator.from_pretrained`.

## Changelog

| Date | Change |
|------|--------|
| 2026-04-17 | Initial version (Modal sync-to-volume flow). |
| 2026-04-21 | Rewrite: single-test scope, explicit user-review pause, per-`model_id` upload, HF overwrite guard. Dropped `scripts/seed_ssim.sh`. |
