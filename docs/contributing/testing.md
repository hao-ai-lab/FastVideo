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

## Modal Runs For SSIM

For CI-like SSIM execution, use `fastvideo/tests/modal/ssim_test.py`:

```bash
python -m modal run fastvideo/tests/modal/ssim_test.py::run_ssim_tests
```

Target specific files or model ids:

```bash
python -m modal run fastvideo/tests/modal/ssim_test.py::run_ssim_tests \
  --test-files test_wan_t2v_similarity.py \
  --model-ids Wan2.1-T2V-1.3B-Diffusers
```

If `HF_API_KEY`, `HUGGINGFACE_HUB_TOKEN`, or `HF_TOKEN` is not set, the local
entrypoint fails fast.

To export raw generated videos from Modal to the shared volume:

```bash
python -m modal run fastvideo/tests/modal/ssim_test.py::run_ssim_tests \
  --sync-generated-to-volume
```

The raw export path is quality-tiered:

- default params: `ssim_generated_videos/default/<subdir>/generated_videos`
- full-quality params: `ssim_generated_videos/full_quality/<subdir>/generated_videos`

The printed `modal volume get` command downloads into
`./generated_videos_modal/<quality-tier>`. Convert those outputs into local
references with `copy-local`:

```bash
python fastvideo/tests/ssim/reference_videos_cli.py copy-local \
  --quality-tier full_quality \
  --generated-dir ./generated_videos_modal/full_quality/L40S_reference_videos \
  --device-folder L40S_reference_videos
```

## CI Integration

FastVideo CI tests are orchestrated by Buildkite and run on Modal GPU
instances. The main files are:

| File | Purpose |
|---|---|
| `.buildkite/pipeline.yml` | Buildkite test graph and path filters. |
| `.buildkite/scripts/pr_test.sh` | Dispatches `TEST_TYPE` to a Modal function. |
| `fastvideo/tests/modal/pr_test.py` | Modal functions for most test lanes. |
| `fastvideo/tests/modal/ssim_test.py` | Modal functions and partitioning for SSIM. |

For exact tier membership, path filters, slash commands, and aggregate statuses,
see [CI/CD Architecture](ci_architecture.md).

### Adding A New CI Test Category

If a new test does not fit an existing lane:

1. Add a Modal function in `fastvideo/tests/modal/pr_test.py` or a focused
   companion module.
2. Add a matching `TEST_TYPE` case in `.buildkite/scripts/pr_test.sh`.
3. Add Buildkite direct-test and path-filter entries in `.buildkite/pipeline.yml`.
4. Add the `/test` mapping in `.github/workflows/ci-slash-commands.yml`.
5. Document the new category in [CI/CD Architecture](ci_architecture.md) and add
   authoring notes here if contributors need them.

When a test only extends an existing category, update that category's tests
instead of adding a new CI lane.
