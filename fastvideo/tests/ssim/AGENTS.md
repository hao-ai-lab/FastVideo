# SSIM Directory Guidelines

## Scope
These instructions apply to everything under `fastvideo/tests/ssim/`.

## Purpose
SSIM tests are GPU-backed end-to-end quality regression checks. They generate
videos and compare them against device-specific reference videos.

## Key Files
- `test_*.py`: SSIM regression tests by model/task.
- `conftest.py`: optional filtering via `FASTVIDEO_SSIM_MODEL_ID`.
- `*_reference_videos/`: committed reference outputs by GPU type.
- `generated_videos/`: local outputs and `*_ssim.json` artifacts (git-ignored).
- `update_reference_videos.sh`: copies generated outputs into a target
  reference directory (defaults to `L40S_reference_videos`).

## Run Commands
- Full SSIM suite: `pytest fastvideo/tests/ssim/ -vs`
- Single test file:
  `pytest fastvideo/tests/ssim/test_inference_similarity.py -vs`
- Single model split:
  `FASTVIDEO_SSIM_MODEL_ID=<model_id> pytest fastvideo/tests/ssim/test_inference_similarity.py -vs`
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
3. From this directory run `bash update_reference_videos.sh`.
4. If needed, edit `REFERENCE_DIR` in `update_reference_videos.sh` first to
   match the active GPU reference folder.
5. In the PR, explain why references changed and which GPU type generated
   them.

## PR Expectations
- Include exact test commands and pass/fail evidence.
- If thresholds change, include before/after SSIM numbers and rationale.
- If references change, call out the source commit/model/backend.
