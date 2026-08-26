# Testing In FastVideo

This guide explains how to add and run tests in FastVideo. CI routing,
slash-command mappings, and workflow ownership live in
[CI/CD Architecture](ci_architecture.md).

## Test Types

| Type | Location | Purpose |
|---|---|---|
| Unit tests | `fastvideo/tests/api`, `fastvideo/tests/dataset`, `fastvideo/tests/entrypoints`, `fastvideo/tests/workflow`, CPU-safe `fastvideo/tests/train` subsets | Validate individual functions, APIs, contracts, and lightweight workflows. |
| Component tests | `fastvideo/tests/encoders`, `fastvideo/tests/transformers`, `fastvideo/tests/vaes` | Validate loading and basic behavior for model components. |
| Train framework tests | `fastvideo/tests/train/models`, `fastvideo/tests/train/methods` | Exercise the new `fastvideo/train/` framework on real checkpoints and tiny synthetic batches. |
| SSIM tests | `fastvideo/tests/ssim` | Compare generated videos against references to catch visual regressions. |
| Training tests | `fastvideo/tests/training` | Validate legacy training loops, LoRA, distillation, self-forcing, and VSA behavior. |
| Inference tests | `fastvideo/tests/inference` | Validate specialized inference paths such as LoRA inference and V-MoBA. |
| Performance tests | `fastvideo/tests/performance` | Gate latency, throughput, peak memory, and stage timings. See [Performance Benchmarks](performance_benchmarks.md). |
| Eval tests | `fastvideo/tests/eval` | Check eval metrics against pinned reference scores and assets. |
| DreamVerse app tests | `apps/dreamverse` | Validate the DreamVerse backend, frontend, and mock-backed browser flows. |

## Running Tests Locally

Run the narrowest useful suite while iterating:

```bash
pytest tests/
pytest fastvideo/tests/ -v
pytest fastvideo/tests/encoders -vs
pytest fastvideo/tests/transformers -vs
pytest fastvideo/tests/vaes -vs
```

GPU-heavy suites need the right hardware, credentials, local caches, and
sometimes custom kernels. Document those assumptions in new tests.

## SSIM Tests

SSIM tests generate videos using specific models and parameters, then compare
the output against reference videos with Structural Similarity Index Measure.
Use them for pipeline-level visual regression coverage when output quality or
generation behavior must be preserved.

!!! note
    Add enough prompts, seeds, backends, and parameter combinations to cover the
    behavior you want to protect, but keep runtime reasonable for CI.

### Directory Structure

```text
fastvideo/tests/ssim/
├── reference_videos/
│   ├── default/
│   │   └── <GPU>_reference_videos/
│   │       └── <Model_Name>/
│   │           └── <Backend>/
│   │               └── <Video_File>
│   └── full_quality/
│       └── <GPU>_reference_videos/
├── generated_videos/
├── reference_videos_cli.py
├── test_wan_t2v_similarity.py
├── test_wan_i2v_similarity.py
└── ...
```

### Adding An SSIM Test

1. Create or update a model-specific file, for example
   `test_wan_t2v_similarity.py`.
2. Define model parameters such as model path, dimensions, frame count,
   inference steps, guidance, seed, and GPU count.
3. Parametrize prompts, attention backends, and model variants where useful.
4. Generate the video with `VideoGenerator`.
5. Compare the generated video with the reference using the SSIM helpers.
6. Seed or update reference videos only after inspecting output quality.

Example shape:

```python
import pytest


MY_MODEL_PARAMS = {
    "num_gpus": 1,
    "model_path": "organization/model-name",
    "height": 480,
    "width": 832,
    "num_frames": 45,
    "num_inference_steps": 20,
}


@pytest.mark.parametrize("prompt", TEST_PROMPTS)
@pytest.mark.parametrize("attention_backend", ["FLASH_ATTN"])
def test_my_model_similarity(prompt, attention_backend):
    # Set backend, generate video, and compare against reference.
    ssim_values = compute_video_ssim_torchvision(
        reference_path,
        generated_path,
        use_ms_ssim=True,
    )
    assert ssim_values[0] >= 0.98
```

### Reference Videos

When a reference is missing, the test writes generated output under:

```text
fastvideo/tests/ssim/generated_videos/<quality-tier>/<GPU>_reference_videos
```

After inspecting the generated video, copy it into the matching reference tree:

```text
fastvideo/tests/ssim/reference_videos/<quality-tier>/<GPU>_reference_videos/<Model>/<Backend>/
```

The helper CLI can copy, upload, and download references:

```bash
python fastvideo/tests/ssim/reference_videos_cli.py copy-local \
  --quality-tier default \
  --reference-dir fastvideo/tests/ssim/reference_videos/default/L40S_reference_videos

python fastvideo/tests/ssim/reference_videos_cli.py upload --quality-tier all

python fastvideo/tests/ssim/reference_videos_cli.py download \
  --quality-tier full_quality \
  --device-folder H200_reference_videos
```

### Running SSIM Locally

```bash
pytest fastvideo/tests/ssim/ -vs
```

Use a machine whose GPU and backend match the reference folder you are testing.

## Slurm CI Runs For SSIM

Comment `/test ssim` on a pull request to run the canonical four-GPU SSIM
lane on the Slinky Slurm cluster. `fastvideo/tests/ssim/ci_runner.py`
discovers the suite without importing test modules, packs independent pytest
processes across the four assigned GPUs, and stops the lane on the first
failure.

The change-aware `/merge` planner may run only the SSIM test basenames owned
by the changed model family. Shared SSIM harness changes still select the
complete lane. Independently, `main` runs the full SSIM matrix every Sunday at
05:00 UTC so infrequently touched model families retain periodic coverage.

For a focused developer run, invoke pytest directly and optionally select one
model from a parameterized test through `FASTVIDEO_SSIM_MODEL_ID`:

```bash
pytest fastvideo/tests/ssim/test_wan_t2v_similarity.py -vs

FASTVIDEO_SSIM_MODEL_ID=Wan2.1-T2V-1.3B-Diffusers \
pytest fastvideo/tests/ssim/test_wan_t2v_similarity.py -vs
```

The files under `fastvideo/tests/modal/` are retained only as a disabled
manual rollback implementation. No active CI trigger invokes them.

### SSIM Bootstrap Mode

Normal SSIM runs are strict: if a reference video or latent is missing, the
test fails. For new-model PRs, CI can run SSIM in bootstrap mode so missing
references are uploaded as draft artifacts for review instead of immediately
blocking on a missing canonical reference.

Buildkite enables SSIM bootstrap mode when either condition is true:

- the PR title or Buildkite message contains `[new-model]`;
- `FASTVIDEO_SSIM_BOOTSTRAP_MODE=1` is set for the Buildkite job.

Bootstrap mode passes `--ssim-bootstrap-mode` to pytest. When a generated
artifact is available, the test uploads it under the `drafts/...` namespace in
the SSIM reference repo and marks that case as expected-failed. After reviewing
the draft, promote it into the canonical reference layout:

```bash
python fastvideo/tests/ssim/reference_videos_cli.py promote-draft \
  --quality-tier default \
  --device-folder L40S_reference_videos \
  --model-id <model_id>
```

## CI Integration

FastVideo GPU CI is orchestrated by Buildkite and runs only on isolated Slinky
Slurm workers. The main files are:

| File | Purpose |
|---|---|
| `.buildkite/pipeline.yml` | Static, validated 20-lane Slurm test graph. |
| `.github/scripts/plan_merge_ci.py` | Trusted path-to-lane and focused quality-test policy for `/merge`. |
| `.buildkite/scripts/unit_test.sh`, `.buildkite/scripts/lanes/*.sh` | Repository-owned test payloads executed inside Slurm containers. |
| `fastvideo/tests/ssim/ci_runner.py` | Four-GPU SSIM task discovery and scheduling. |
| `.buildkite/scripts/pr_test.sh`, `fastvideo/tests/modal/*.py` | Dormant manual rollback path; rejected in Buildkite. |

For exact tier membership, slash commands, runner isolation, and aggregate statuses,
see [CI/CD Architecture](ci_architecture.md).

### Adding A New CI Test Category

If a new test does not fit an existing lane:

1. Put the test payload in an executable `.buildkite/scripts/lanes/<lane>.sh`.
2. Add its static step to `.buildkite/pipeline.yml`, its changed-path ownership
   to `.github/scripts/plan_merge_ci.py`, and extend the CI contract tests.
3. Add the `/test` mapping in `.github/workflows/ci-slash-commands.yml`.
4. Coordinate the matching GPU, timeout, dependency, secret, and artifact
   policy in the private Slurm runner allowlist.
5. Document the new category in [CI/CD Architecture](ci_architecture.md) and add
   authoring notes here if contributors need them.

When a test only extends an existing category, update that category's tests
instead of adding a new CI lane.
