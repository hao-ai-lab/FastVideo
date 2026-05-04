import os

import modal

app = modal.App()

model_vol = modal.Volume.from_name("hf-model-weights")
image_version = os.getenv("IMAGE_VERSION")
image_tag = f"ghcr.io/hao-ai-lab/fastvideo/fastvideo-dev:{image_version}"
print(f"Using image: {image_tag}")

image = (modal.Image.from_registry(
    image_tag, add_python="3.12"
).run_commands("rm -rf /FastVideo").apt_install(
    "cmake", "pkg-config", "build-essential", "curl", "libssl-dev", "ffmpeg"
).run_commands(
    "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable"
).run_commands("echo 'source ~/.cargo/env' >> ~/.bashrc").env({
    "PATH":
    "/root/.cargo/bin:$PATH",
    "BUILDKITE_REPO":
    os.environ.get("BUILDKITE_REPO", ""),
    "BUILDKITE_COMMIT":
    os.environ.get("BUILDKITE_COMMIT", ""),
    "BUILDKITE_PULL_REQUEST":
    os.environ.get("BUILDKITE_PULL_REQUEST", ""),
    "BUILDKITE_BRANCH":
    os.environ.get("BUILDKITE_BRANCH", ""),
    "TEST_SCOPE":
    os.environ.get("TEST_SCOPE", ""),
    "IMAGE_VERSION":
    os.environ.get("IMAGE_VERSION", ""),
    "HF_REPO_ID":
    "FastVideo/performance-tracking",
}))


def run_test(pytest_command: str):
    """Helper function to run a test suite with custom pytest command"""
    import subprocess
    import sys
    import os

    git_repo = os.environ.get("BUILDKITE_REPO")
    git_commit = os.environ.get("BUILDKITE_COMMIT")
    pr_number = os.environ.get("BUILDKITE_PULL_REQUEST")

    print(f"Cloning repository: {git_repo}")
    print(f"Target commit: {git_commit}")
    if pr_number:
        print(f"PR number: {pr_number}")

    # For PRs (including forks), use GitHub's PR refs to get the correct commit
    if pr_number and pr_number != "false":
        checkout_command = f"git fetch --prune origin refs/pull/{pr_number}/head && git checkout FETCH_HEAD"
        print(f"Using PR ref for checkout: {checkout_command}")
    else:
        checkout_command = f"git checkout {git_commit}"
        print(f"Using direct commit checkout: {checkout_command}")

    command = f"""
    source $HOME/.local/bin/env &&
    source /opt/venv/bin/activate &&
    git clone {git_repo} /FastVideo &&
    cd /FastVideo &&
    {checkout_command} &&
    git submodule update --init --recursive &&
    cd fastvideo-kernel &&
    ./build.sh &&
    cd .. &&
    uv pip install -e ".[test]" &&
    {pytest_command}
    """

    result = subprocess.run(["/bin/bash", "-c", command],
                            stdout=sys.stdout,
                            stderr=sys.stderr,
                            check=False)

    # Modal containers crash on sys.exit(0); raise on failure, return on success.
    if result.returncode != 0:
        raise RuntimeError(
            f"Test command failed with exit code {result.returncode}")

@app.function(gpu="H100:1",
              image=image,
              timeout=1200,
              secrets=[
                  modal.Secret.from_dict(
                      {"HF_API_KEY": os.environ.get("HF_API_KEY", "")})
              ],
              volumes={"/root/data": model_vol})
def run_encoder_tests():
    run_test(
        "export HF_HOME='/root/data/.cache' && hf auth login --token $HF_API_KEY && pytest ./fastvideo/tests/encoders -vs"
    )


@app.function(gpu="L40S:1",
              image=image,
              timeout=1200,
              secrets=[
                  modal.Secret.from_dict(
                      {"HF_API_KEY": os.environ.get("HF_API_KEY", "")})
              ],
              volumes={"/root/data": model_vol})
def run_vae_tests():
    run_test(
        "export HF_HOME='/root/data/.cache' && hf auth login --token $HF_API_KEY && pytest ./fastvideo/tests/vaes -vs"
    )


@app.function(gpu="L40S:1",
              image=image,
              timeout=900,
              secrets=[
                  modal.Secret.from_dict(
                      {"HF_API_KEY": os.environ.get("HF_API_KEY", "")})
              ],
              volumes={"/root/data": model_vol})
def run_transformer_tests():
    run_test(
        "export HF_HOME='/root/data/.cache' && hf auth login --token $HF_API_KEY && pytest ./fastvideo/tests/transformers -vs"
    )


@app.function(gpu="L40S:4",
              image=image,
              timeout=900,
              secrets=[
                  modal.Secret.from_dict(
                      {"WANDB_API_KEY": os.environ.get("WANDB_API_KEY", "")})
              ],
              volumes={"/root/data": model_vol})
def run_training_tests():
    run_test(
        "export HF_HOME='/root/data/.cache' && wandb login $WANDB_API_KEY && pytest ./fastvideo/tests/training/Vanilla -srP"
    )


@app.function(gpu="L40S:2",
              image=image,
              timeout=900,
              secrets=[
                  modal.Secret.from_dict(
                      {"WANDB_API_KEY": os.environ.get("WANDB_API_KEY", "")})
              ],
              volumes={"/root/data": model_vol})
def run_training_lora_tests():
    run_test(
        "export HF_HOME='/root/data/.cache' && wandb login $WANDB_API_KEY && pytest ./fastvideo/tests/training/lora/test_lora_training.py -srP"
    )


@app.function(gpu="H100:2",
              image=image,
              timeout=900,
              secrets=[
                  modal.Secret.from_dict(
                      {"WANDB_API_KEY": os.environ.get("WANDB_API_KEY", "")})
              ])
def run_training_tests_VSA():
    run_test(
        "wandb login $WANDB_API_KEY && pytest ./fastvideo/tests/training/VSA -srP"
    )


@app.function(gpu="H100:1", image=image, timeout=900)
def run_kernel_tests():
    run_test("pytest fastvideo-kernel/tests/ -vs")


# @app.function(gpu="H100:1", image=image, timeout=900)
# def run_precision_tests_VSA():
#     # VSA correctness is covered by the same file now
#     run_test("pytest fastvideo-kernel/tests/test_correctness.py")

# @app.function(gpu="L40S:1", image=image, timeout=900)
# def run_precision_tests_vmoba():
#     run_test("pytest fastvideo-kernel/tests/test_vmoba_correctness.py")


@app.function(gpu="L40S:1", image=image, timeout=900)
def run_inference_tests_vmoba():
    run_test('python fastvideo/tests/inference/vmoba/test_vmoba_inference.py')


@app.function(gpu="L40S:1", image=image, timeout=1200)
def run_inference_lora_tests():
    run_test(
        "pytest ./fastvideo/tests/inference/lora/test_lora_inference_similarity.py -vs"
    )


@app.function(gpu="L40S:2", image=image, timeout=900)
def run_distill_dmd_tests():
    run_test(
        "pytest ./fastvideo/tests/training/distill/test_distill_dmd.py -vs")


@app.function(gpu="L40S:2",
              image=image,
              timeout=900,
              secrets=[
                  modal.Secret.from_dict(
                      {"WANDB_API_KEY": os.environ.get("WANDB_API_KEY", "")})
              ])
def run_self_forcing_tests():
    run_test(
        "wandb login $WANDB_API_KEY && pytest ./fastvideo/tests/training/self-forcing/test_self_forcing.py -vs"
    )


@app.function(gpu="L40S:1", image=image, timeout=900)
def run_unit_test():
    run_test(
        "pytest ./fastvideo/tests/api/ ./fastvideo/tests/dataset/ ./fastvideo/tests/workflow/ ./fastvideo/tests/entrypoints/ ./fastvideo/tests/train/ --ignore=./fastvideo/tests/entrypoints/test_openai_api_integration.py -vs"
    )


@app.function(gpu="L40S:1",
              image=image,
              timeout=3600,
              secrets=[
                  modal.Secret.from_dict(
                      {"HF_API_KEY": os.environ.get("HF_API_KEY", "")})
              ])
def run_lora_extraction_tests():
    run_test(
        "hf auth login --token $HF_API_KEY && pytest ./fastvideo/tests/lora_extraction/test_lora_extraction.py"
    )


@app.function(gpu="L40S:2",
              image=image,
              timeout=1800,
              secrets=[
                  modal.Secret.from_dict(
                      {"HF_API_KEY": os.environ.get("HF_API_KEY", "")})
              ],
              volumes={"/root/data": model_vol})
def run_performance_tests():
    # Dashboard runs whenever result JSONs exist, even if pytest failed after
    # writing them. Raw perf_*.json results are exported for any failed
    # performance job if result files exist; those artifacts are the reviewed
    # source input for manual performance-baseline reseeds.
    run_test(
        "export HF_HOME='/root/data/.cache' && "
        "export PERFORMANCE_TRACKING_ROOT='/tmp/perf-tracking' && "
        "hf auth login --token $HF_API_KEY && "
        "pytest ./fastvideo/tests/performance -vs; "
        "PYTEST_RC=$?; "
        "PERF_RC=0; "
        "if [ $PYTEST_RC -eq 0 ]; then "
        "python ./fastvideo/tests/performance/compare_baseline.py; "
        "PERF_RC=$?; "
        "fi; "
        "if ls ./fastvideo/tests/performance/results/perf_*.json >/dev/null 2>&1; then "
        "python ./fastvideo/tests/performance/dashboard.py || true; "
        "fi; "
        "FINAL_RC=$PYTEST_RC; "
        "if [ $FINAL_RC -eq 0 ]; then FINAL_RC=$PERF_RC; fi; "
        "if [ $FINAL_RC -ne 0 ]; then "
        "if ls ./fastvideo/tests/performance/results/perf_*.json >/dev/null 2>&1; then "
        "mkdir -p /root/data/perf_reports/results && "
        "cp ./fastvideo/tests/performance/results/perf_*.json /root/data/perf_reports/results/; "
        "else "
        "echo 'No raw performance result JSON files found to export.'; "
        "fi; "
        "fi; "
        "exit $FINAL_RC")


@app.function(gpu="L40S:1",
              image=image,
              timeout=1800,
              secrets=[
                  modal.Secret.from_dict(
                      {"HF_API_KEY": os.environ.get("HF_API_KEY", "")})
              ],
              volumes={"/root/data": model_vol})
def run_api_server_tests():
    run_test(
        "export HF_HOME='/root/data/.cache' && hf auth login --token $HF_API_KEY && pytest ./fastvideo/tests/entrypoints/test_openai_api_integration.py -vs"
    )
