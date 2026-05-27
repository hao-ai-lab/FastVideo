# SSIM Directory Guidelines

**Generated:** 2026-05-02

## Scope
These instructions apply to everything under `fastvideo/tests/ssim/`.

## Purpose
SSIM tests are GPU-backed end-to-end quality regression checks. They generate
videos and compare them against device-specific reference videos.

## Key Files
- `test_*.py`: SSIM regression tests by model/task.
- `conftest.py`: optional filtering via `FASTVIDEO_SSIM_MODEL_ID`.
- `reference_videos/<quality-tier>/<GPU>_reference_videos/`: local cache of
  references split by quality tier and GPU type.
- `generated_videos/<quality-tier>/<GPU>_reference_videos/`: local outputs and
  `*_ssim.json` artifacts (git-ignored).
- `reference_videos_cli.py`: copy/download/upload/ensure reference videos
  (including HF sync).

## Run Commands
- Full SSIM suite: `pytest fastvideo/tests/ssim/ -vs`
- Full SSIM with full-quality params:
  `pytest fastvideo/tests/ssim/ -vs --ssim-full-quality`
- Single test file:
  `pytest fastvideo/tests/ssim/test_wan_t2v_similarity.py -vs`
- Single model split:
  `FASTVIDEO_SSIM_MODEL_ID=<model_id> pytest fastvideo/tests/ssim/test_wan_t2v_similarity.py -vs`
- Modal orchestrator (CI-style scheduling):
  `modal run fastvideo/tests/modal/ssim_test.py`

## Authoring Rules
- Name files `test_<feature>_similarity.py`.
- Set `REQUIRED_GPUS = <int>` near the top of each test module.
- For multi-model suites, keep configs in `*_MODEL_TO_PARAMS` dictionaries.
  The Modal scheduler auto-discovers these keys and runs one subprocess per
  model id.
- Keep runs deterministic when possible (fixed prompts/seeds/frames/backend).
- Persist metrics with `write_ssim_results(...)`.
- Use `pytest.skip(...)` for unsupported hardware, missing assets, or
  insufficient GPU count.

## Updating Reference Videos
1. Run the target SSIM test and inspect generated outputs + scores.
2. Update references only for intentional behavior changes.
3. Copy local generated outputs into the target reference folder:
   `python fastvideo/tests/ssim/reference_videos_cli.py copy-local ...`
4. Upload updated reference folders to HF:
   `python fastvideo/tests/ssim/reference_videos_cli.py upload ...`
5. In the PR, explain why references changed and which GPU type generated
   them.

## PR Expectations
- Include exact test commands and pass/fail evidence.
- If thresholds change, include before/after SSIM numbers and rationale.
- If references change, call out the source commit/model/backend.
