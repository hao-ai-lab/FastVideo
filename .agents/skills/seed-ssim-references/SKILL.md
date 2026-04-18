---
name: seed-ssim-references
description: Run a new (or updated) SSIM test on Modal, capture the generated videos, and upload them to the HF reference repo so the test can act as a regression guard. Use when a `fastvideo/tests/ssim/test_*.py` file has been added and has no reference video yet on `FastVideo/ssim-reference-videos`.
---

# Seed SSIM Reference Videos

## Purpose

A brand-new SSIM test in `fastvideo/tests/ssim/` fails forever until its
reference videos exist on the HF dataset (`FastVideo/ssim-reference-videos` by
default). This skill runs the test on Modal's L40S pool, pulls the generated
videos back locally, moves them into the `reference_videos/` layout, and
uploads to HF. After it runs, re-executing the same test on CI (or locally
with the references downloaded) should pass.

Use this skill when:
- A `test_*_similarity.py` file has just landed on `main` and the HF repo
  still reports "Reference video folder does not exist" for it.
- An existing SSIM test's config changed (resolution, steps, prompt) and its
  references need to be regenerated.

Do not use this skill for regular CI runs — once the references exist, normal
`pytest` is enough.

## Prerequisites

- `modal` CLI installed and authenticated (`modal token set ...`).
- HF API token with write access to `FastVideo/ssim-reference-videos`, exposed
  via one of: `HF_API_KEY`, `HUGGINGFACE_HUB_TOKEN`, `HF_TOKEN`.
- The test file under review already:
  - declares `REQUIRED_GPUS` at module scope (so the Modal orchestrator
    schedules correctly);
  - exposes `<NAME>_MODEL_TO_PARAMS` (so CI can split one subprocess per
    model id);
  - registers in `fastvideo/tests/ssim/` using the shared helpers
    (`resolve_inference_device_reference_folder`,
    `run_text_to_video_similarity_test`, etc.).
- The `ghcr.io/hao-ai-lab/fastvideo/fastvideo-dev:<IMAGE_VERSION>` container
  tag is pullable (set `IMAGE_VERSION` or accept the `latest` default).

## Inputs

| Parameter | Required | Description |
|-----------|----------|-------------|
| `test_file` | Yes | Path to the new test file, e.g. `fastvideo/tests/ssim/test_ltx2_similarity.py`. |
| `model_ids` | No | Comma-separated subset of the test's model ids. Defaults to all. |
| `quality_tier` | No | `default` (CI-friendly) or `full_quality`. Seed both if the test ships both. Default `default`. |
| `commit` | No | Git commit to run against. Defaults to local `HEAD`. |
| `git_repo` | No | Git repo URL. Defaults to `origin` remote of the local clone. |
| `reference_repo` | No | Target HF dataset repo. Defaults to `FastVideo/ssim-reference-videos`. |
| `device_folder` | No | Reference subfolder name, e.g. `L40S_reference_videos`. Defaults to `L40S_reference_videos` (the Modal runner's GPU). |

## Steps

### 1. Verify preconditions

- Confirm `test_file` exists and matches `test_*_similarity.py`.
- Grep for `REQUIRED_GPUS` and `*_MODEL_TO_PARAMS` in the file — if missing,
  stop and direct the user to add them (see `fastvideo/tests/ssim/README.md`
  "Adding a New SSIM Test").
- Confirm `HF_API_KEY` (or equivalent) is exported.
- Confirm `modal token current` succeeds.

### 2. Run the test on Modal with video export enabled

Run from the repo root:

```bash
bash .agents/skills/seed-ssim-references/scripts/seed_ssim.sh \
  --test-file fastvideo/tests/ssim/test_ltx2_similarity.py \
  --quality-tier default
```

The script is a thin wrapper that:

1. Computes `git_repo` and `git_commit` if not supplied.
2. Invokes:
  
```
   modal run fastvideo/tests/modal/ssim_test.py \
     --test-files=<test_file> \
     --sync-generated-to-volume \
     --skip-reference-download \
     --no-fail-fast \
     [--full-quality]
   ```
  
`--skip-reference-download` prevents the conftest from pre-pulling
   (there are no references yet); `--no-fail-fast` lets generation finish
   before `_assert_similarity` raises; `--sync-generated-to-volume` copies
   the generated videos to the Modal volume `hf-model-weights`.
3. Captures the Modal volume path printed at the end of the run — it looks
   like `ssim_generated_videos/<tier>/<timestamp>_<short_commit>/generated_videos`.

### 3. Pull the generated videos back

Using the path printed in step 2:

```bash
modal volume get hf-model-weights \
  ssim_generated_videos/<tier>/<subdir>/generated_videos \
  ./generated_videos_modal/<tier>
```

### 4. Copy into the `reference_videos/` layout

```bash
python fastvideo/tests/ssim/reference_videos_cli.py copy-local \
  --quality-tier <tier> \
  --generated-dir ./generated_videos_modal/<tier>/L40S_reference_videos \
  --device-folder L40S_reference_videos
```

This produces `fastvideo/tests/ssim/reference_videos/<tier>/L40S_reference_videos/<model>/<backend>/<prompt>.mp4`.

### 5. Upload to HF

```bash
python fastvideo/tests/ssim/reference_videos_cli.py upload \
  --quality-tier <tier> \
  --device-folder L40S_reference_videos
```

The CLI reads the HF token from `HF_API_KEY` / `HUGGINGFACE_HUB_TOKEN` /
`HF_TOKEN` and fails fast if none are set. Override the target repo with
`FASTVIDEO_SSIM_REFERENCE_HF_REPO=<org/repo>` if you need a non-default
destination.

### 6. Verify

Re-run the test locally (or on Modal) **without** `--skip-reference-download`:

```bash
modal run fastvideo/tests/modal/ssim_test.py \
  --test-files=fastvideo/tests/ssim/test_ltx2_similarity.py
```

The conftest will now auto-download the references you just uploaded and the
SSIM comparison should pass. If it still fails, compare the local generated
video against the uploaded reference to see whether the threshold needs
adjusting.

### 7. Seed the full-quality tier if applicable

If the test ships both tiers, repeat steps 2-5 with `--quality-tier
full_quality` (which passes `--full-quality` to the Modal run).

## Outputs

- Generated videos on the Modal volume `hf-model-weights` under
  `ssim_generated_videos/<tier>/<subdir>/generated_videos/...`.
- Local copies under `./generated_videos_modal/<tier>/...`.
- Local references under `fastvideo/tests/ssim/reference_videos/<tier>/L40S_reference_videos/...`.
- Uploaded references on HF at `<reference_repo>` (default
  `FastVideo/ssim-reference-videos`).

## Example Usage

Seed the new LTX-2 test (PR #1240) on main, default tier only:

```
Prompt: "Seed SSIM references for fastvideo/tests/ssim/test_ltx2_similarity.py"
  test_file: fastvideo/tests/ssim/test_ltx2_similarity.py
  quality_tier: default
```

Seed both tiers for an existing test whose config just changed:

```
Prompt: "Re-seed test_wan_t2v_similarity.py; I bumped num_frames from 45 to 49"
  test_file: fastvideo/tests/ssim/test_wan_t2v_similarity.py
  quality_tier: default, then full_quality
```

## Troubleshooting

- **"Hugging Face token is required"** — export `HF_API_KEY` in the shell
  that runs `modal run`. Modal passes it through as a secret.
- **"No generated_videos directory found for quality tier"** — the test
  didn't actually run inside the Modal container. Most common cause:
  `REQUIRED_GPUS` larger than the partition has (the orchestrator skips
  oversized tasks). Check the Modal logs under
  `Partition <i>: running <x>/<y> tasks`.
- **`ensure_reference_videos_available` still fails after upload** — the
  HF dataset has caching; add a short sleep before step 6 or pass
  `--skip-reference-download` and rely on local `reference_videos/` for
  the first verification run.
- **Test passes locally but fails on Modal** — the Modal runner is L40S;
  if you generated references on a different GPU SKU, use the right
  `--device-folder` name (`A40_reference_videos`, `H100_reference_videos`,
  etc.).

## References

- `fastvideo/tests/modal/ssim_test.py` (`run_ssim_tests` local entrypoint,
  `run_ssim_partition` Modal function, `--sync-generated-to-volume` flow).
- `fastvideo/tests/ssim/reference_videos_cli.py` (`copy-local`, `upload`,
  `download` subcommands, HF repo conventions).
- `fastvideo/tests/ssim/inference_similarity_utils.py`
  (`run_text_to_video_similarity_test` + `_build_init_kwargs` — what each
  test config ends up passing to `VideoGenerator.from_pretrained`).
- `fastvideo/tests/ssim/conftest.py` (`--skip-ssim-reference-download`,
  `--ssim-full-quality`, `--ssim-reference-repo` pytest options).
- `fastvideo/tests/ssim/README.md` ("Adding a New SSIM Test" checklist,
  reference videos layout, HF repo layout).

## Changelog

| Date | Change |
|------|--------|
| 2026-04-17 | Initial version. Documents the `--sync-generated-to-volume` seeding flow introduced by `ssim_test.py`. |
