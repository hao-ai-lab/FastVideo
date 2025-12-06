# Testing in FastVideo

This guide explains how to add and run tests in FastVideo. The testing suite is divided into several categories to ensure correctness across components, training workflows, and inference quality.

## Test Types

* **Unit Tests**: Located in `fastvideo/tests/dataset`, `fastvideo/tests/entrypoints`, and `fastvideo/tests/workflow`. These test individual functions and classes.
* **Component Tests**: Located in `fastvideo/tests/encoders`, `fastvideo/tests/transformers`, and `fastvideo/tests/vaes`. These verify the loading and basic functionality of model components.
* **SSIM Tests**: Located in `fastvideo/tests/ssim`. These are regression tests that compare generated videos against reference videos using the Structural Similarity Index Measure (SSIM) to detect quality degradation.
* **Training Tests**: Located in `fastvideo/tests/training`. These validate training loops, loss calculations, and specific training techniques like LoRA, Distillation, and VSA.
* **Inference Tests**: Located in `fastvideo/tests/inference`. These test specialized inference pipelines and optimizations (e.g., STA, V-MoBA).

For now, we will focus on **SSIM Tests**.

## SSIM Tests

SSIM tests are located in `fastvideo/tests/ssim`. These tests generate videos using specific models and parameters, and compare them against reference videos to ensure that changes in the codebase do not degrade generation quality or alter the output unexpectedly.

!!! note
    If you are adding an SSIM test, this serves as a safeguard. Any future code changes that break or cause errors with the specific arguments and configurations you defined will trigger a failure. Therefore, it is important to include multiple settings and arguments that cover the core features of your new pipeline to ensure robust regression testing.

### Directory Structure

```
fastvideo/tests/ssim/
├── <GPU>_reference_videos/   # Reference videos organized by GPU type (e.g., L40S_reference_videos)
│   ├── <Model_Name>/
│   │   ├── <Backend>/        # e.g., FLASH_ATTN, TORCH_SDPA
│   │   │   └── <Video_File>
├── test_causal_similarity.py
├── test_inference_similarity.py
├── update_reference_videos.sh
└── ...
```

### Adding a New SSIM Test

To add a new SSIM test, follow these steps:

1. **Create or Update a Test File**: You can add a new test function to an existing file (like `test_inference_similarity.py`) or create a new one if testing a distinct category of models.

2. **Define Model Parameters**: Define the configuration for the model you want to test. This includes model path, dimensions, inference steps, and other generation parameters. **Note:** Consider using lower `num_inference_steps` or reduced resolution (e.g., 480p instead of 720p) to keep test execution time reasonable, provided it doesn't compromise the test's ability to detect regression.

   ```python
   MY_MODEL_PARAMS = {
       "num_gpus": 1,
       "model_path": "organization/model-name",
       "height": 480,
       "width": 832,
       "num_frames": 45,
       "num_inference_steps": 20,
       # ... other parameters
   }
   ```

3. **Implement the Test Function**:
   * Use `pytest.mark.parametrize` to run the test with different prompts, backends, and models.
   * Set the attention backend environment variable.
   * Initialize the `VideoGenerator`.
   * Generate the video.
   * Compare the generated video with the reference video using `compute_video_ssim_torchvision`.

   Example structure:

   ```python
   @pytest.mark.parametrize("prompt", TEST_PROMPTS)
   @pytest.mark.parametrize("ATTENTION_BACKEND", ["FLASH_ATTN"])
   def test_my_model_similarity(prompt, ATTENTION_BACKEND):
       # Setup output directories
       # ...

       # Initialize Generator
       generator = VideoGenerator.from_pretrained(...)
       generator.generate_video(prompt, ...)

       # Compare with Reference
        ssim_values = compute_video_ssim_torchvision(reference_path, generated_path, use_ms_ssim=True)
        assert ssim_values[0] >= 0.98  # Threshold
    ```

4. **Reference Videos**:
   * When running the test for the first time (or when updating the reference), the test will fail because the reference video is missing. The generated video will be saved in `fastvideo/tests/ssim/generated_videos`.
   * Inspect the generated video to ensure it meets quality expectations.
   * Move the generated video to the appropriate reference folder: `fastvideo/tests/ssim/<GPU>_reference_videos/<Model>/<Backend>/`.
   * You can use the helper script `update_reference_videos.sh` to automate copying videos from `generated_videos` to `L40S_reference_videos`. Note: Check the script to ensure paths match your environment (it defaults to `L40S_reference_videos`).

### Running Tests Locally

To run the SSIM tests locally:

```bash
pytest fastvideo/tests/ssim/test_inference_similarity.py -vs
```

Ensure you have the necessary GPUs available as defined in your test parameters.

## Modal Workflow

FastVideo uses [Modal](https://modal.com/) for running tests in a CI environment. The workflow scripts are located in `fastvideo/tests/modal/`.

### `pr_test.py`

The main entry point for CI tests is `fastvideo/tests/modal/pr_test.py`. This script defines Modal functions that execute the pytest suites on specific hardware (e.g., L40S, H100).

### Updating Modal Configuration

If you add a new test that requires:
* **Different GPU Hardware**: You may need to change the `@app.function(gpu=...)` decorator.
* **Longer Execution Time**: Increase the `timeout` parameter.
* **New Environment Variables/Secrets**: Add them to `secrets=[...]` or the image environment. For example, if your model is gated on Hugging Face, ensure `HF_API_KEY` is passed.

For SSIM tests, the `run_ssim_tests` function in `pr_test.py` currently runs:

```python
@app.function(gpu="L40S:2", image=image, timeout=2700, secrets=[modal.Secret.from_dict({"HF_API_KEY": os.environ.get("HF_API_KEY", "")})])
def run_ssim_tests():
    run_test("hf auth login --token $HF_API_KEY && pytest ./fastvideo/tests/ssim -vs")
```

If your new test file is inside `fastvideo/tests/ssim`, it will automatically be picked up by this command. However, ensure that the `gpu="L40S:2"` configuration is sufficient for your model. If your model requires more GPUs (e.g., 4 or 8), you might need to create a separate Modal function or update the existing one.

### Workflow Scripts

The shell script that triggers these tests in the CI pipeline is located at `.buildkite/scripts/pr_test.sh`. If you add a new test category (e.g., a new folder outside of `ssim`), you will need to:
1. Add a new function in `fastvideo/tests/modal/pr_test.py`.
2. Add a new case in `.buildkite/scripts/pr_test.sh` to handle the new test type.

!!! note
    If you are a maintainer, you'll need to finally manually update the workflow script in Buildkite. Otherwise, a maintainer will help you update.
