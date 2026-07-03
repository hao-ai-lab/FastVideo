import os
import sys

import modal

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from modal_image_utils import resolve_image_ref  # noqa: E402
except ModuleNotFoundError:
    # Remote Modal containers re-import this module but mount only the
    # entrypoint file; the digest resolution already happened at local
    # launch time, so a passthrough is correct there.
    def resolve_image_ref(image_ref: str) -> str:
        return image_ref

app = modal.App()

model_vol = modal.Volume.from_name("hf-model-weights")
image_version = os.getenv("IMAGE_VERSION", "latest")
image_tag = f"ghcr.io/hao-ai-lab/fastvideo/fastvideo-dev:{image_version}"
image_ref = resolve_image_ref(image_tag)
print(f"Using image: {image_ref}")

# Mutable tags inherit the registry image's baked backend, keeping a latest-tag
# transition safe. Explicit CUDA tags also work with older images that predate
# the baked setting, and a caller override always wins.
uv_torch_backend_override = os.environ.get("UV_TORCH_BACKEND")
if not uv_torch_backend_override:
    if "cuda13" in image_tag.lower():
        uv_torch_backend_override = "cu130"
    elif "cuda12.6" in image_tag.lower():
        uv_torch_backend_override = "cu126"

image = (modal.Image.from_registry(
    image_ref, add_python="3.12"
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
    "BUILDKITE_BUILD_URL":
    os.environ.get("BUILDKITE_BUILD_URL", ""),
    "BUILDKITE_BUILD_ID":
    os.environ.get("BUILDKITE_BUILD_ID", ""),
    "BUILDKITE_JOB_ID":
    os.environ.get("BUILDKITE_JOB_ID", ""),
    "TEST_SCOPE":
    os.environ.get("TEST_SCOPE", ""),
    "IMAGE_VERSION":
    image_version,
    **({
        "UV_TORCH_BACKEND": uv_torch_backend_override
    } if uv_torch_backend_override else {}),
    "HF_REPO_ID":
    "FastVideo/performance-tracking",
}))

dreamverse_image = (image.run_commands(
    "curl -fsSL https://deb.nodesource.com/setup_22.x | bash -"
).apt_install("nodejs").run_commands("node --version && npm --version"))


def run_test(pytest_command: str):
    """Helper function to run a test suite with custom pytest command"""
    run_test_command(pytest_command, build_kernel=True)


def run_test_command(test_command: str,
                     build_kernel: bool,
                     install_command: str = 'uv pip install -e ".[test]"'):
    """Helper function to run a test suite with custom test command.

    Most FastVideo CI suites need the custom kernel build. App-level tests like
    DreamVerse's mock-backend UI checks do not, so keep the kernel build
    optional to avoid unrelated CUDA/kernel setup in that CI path.

    The dependency install runs BEFORE the kernel build: pyproject pins the
    PyPI fastvideo-kernel wheel, so an install after the build silently
    replaces the just-built in-tree kernel with the (older) wheel -- every
    lane would then test stale kernels. Pass install_command="" for commands
    that manage their own installs.
    """
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

    build_kernel_command = """
    cd fastvideo-kernel &&
    ./build.sh &&
    cd .. &&
    """ if build_kernel else ""

    install_clause = f"{install_command} &&" if install_command else ""

    command = f"""
    source $HOME/.local/bin/env &&
    source /opt/venv/bin/activate &&
    git clone {git_repo} /FastVideo &&
    cd /FastVideo &&
    {checkout_command} &&
    git submodule update --init --recursive &&
    {install_clause}
    {build_kernel_command}
    {test_command}
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
        "pytest ./fastvideo/tests/api/ ./fastvideo/tests/contract/ ./fastvideo/tests/dataset/ ./fastvideo/tests/workflow/ ./fastvideo/tests/entrypoints/ ./fastvideo/tests/train/ --ignore=./fastvideo/tests/entrypoints/test_openai_api_integration.py --ignore=./fastvideo/tests/train/models --ignore=./fastvideo/tests/train/methods -vs"
    )


# TODO: David: GPU only used to resolve import time requirement (not needed for this test). Maybe make those imports lazy?
@app.function(gpu="L40S:1", image=dreamverse_image, timeout=1800)
def run_dreamverse_app_tests():
    run_test_command(
        install_command="",
        build_kernel=False,
        test_command="""
        uv pip install -e ".[test,dreamverse]" &&
        export PYTHONPATH=/FastVideo/apps/dreamverse:$PYTHONPATH &&
        pytest apps/dreamverse/dreamverse/tests -q &&
        cd apps/dreamverse/web &&
        npm ci &&
        npm run typecheck &&
        npm test &&
        npx playwright install --with-deps chromium webkit firefox &&
        bash -c '
            set -e
            BACKEND_PORT="${BACKEND_PORT:-8009}"
            python -m uvicorn dreamverse.mock_server:app --host 127.0.0.1 --port "$BACKEND_PORT" &
            MOCK_SERVER_PID=$!
            trap "kill $MOCK_SERVER_PID 2>/dev/null || true" EXIT
            for i in {1..30}; do
                curl -fsS "http://127.0.0.1:$BACKEND_PORT/healthz" && break
                sleep 1
            done
            curl -fsS "http://127.0.0.1:$BACKEND_PORT/healthz"
            BACKEND_HOST=127.0.0.1 BACKEND_PORT="$BACKEND_PORT" CI=1 \
                npm run e2e -- \
                    --project=chromium \
                    --project=webkit \
                    --project=firefox \
                    --project=mobile-safari \
                    --project=mobile-chromium
        '
        """)


@app.function(gpu="L40S:1",
              image=image,
              timeout=1800,
              secrets=[
                  modal.Secret.from_dict(
                      {"HF_API_KEY": os.environ.get("HF_API_KEY", "")})
              ],
              volumes={"/root/data": model_vol})
def run_train_framework_tests():
    run_test(
        "export HF_HOME='/root/data/.cache' && hf auth login --token $HF_API_KEY && pytest ./fastvideo/tests/train/models ./fastvideo/tests/train/methods -vs"
    )


@app.function(gpu="L40S:1",
              image=image,
              timeout=1800,
              secrets=[
                  modal.Secret.from_dict(
                      {"HF_API_KEY": os.environ.get("HF_API_KEY", "")})
              ],
              volumes={"/root/data": model_vol})
def seed_grad_norm_references():
    """Record the per-method grad-norm reference for the **CI GPU (L40S only)**.

    Phase 2 / 5a-ii one-off seeding entrypoint. Pinned to ``gpu="L40S:1"`` (the
    Modal CI runner), so this function only seeds the ``L40S`` key in
    ``fastvideo/tests/train/methods/grad_norm_refs.json``.

    ``FASTVIDEO_GRADNORM_UPDATE=1`` makes ``check_grad_norm_regression`` record
    the measured norm instead of asserting; ``-rs`` surfaces the recorded value
    in the log so it can be copied into the JSON.

    To seed any other device (e.g. our local Blackwell dev box → ``GB200``
    key), run the same env-var + pytest invocation directly on that
    workstation — see the module docstring of ``grad_norm_regression.py`` for
    the local command and the ``_DEVICE_MAPPINGS`` table.
    """
    run_test(
        "export HF_HOME='/root/data/.cache' && hf auth login --token $HF_API_KEY && FASTVIDEO_GRADNORM_UPDATE=1 pytest ./fastvideo/tests/train/methods -vs -rs"
    )


@app.function(gpu="L40S:1",
              image=image,
              timeout=3600,
              secrets=[
                  modal.Secret.from_dict(
                      {"HF_API_KEY": os.environ.get("HF_API_KEY", "")})
              ],
              volumes={"/root/data": model_vol})
def run_eval_tests():
    # Eval metric regression: drives the high-level fastvideo.eval API on a
    # fixed asset and asserts each score matches the upstream reference number
    # checked into fastvideo/tests/eval/reference_scores/. Pulls several scorer
    # checkpoints (VideoScore2 VLM, VBench nets, audio models) on first run;
    # they cache on the hf-model-weights volume thereafter.
    #
    # Installs [eval-full] (eval + vbench + audio extras) on top of [test]:
    # the dev image only ships [dev], and without the extras skip_missing_deps
    # in conftest would silently drop nearly every metric and the lane would
    # pass vacuously. detectron2-backed vbench metrics remain skipped by
    # design (not pip-installable; see fastvideo/eval/README.md).
    run_test_command(
        "export HF_HOME='/root/data/.cache' && hf auth login --token $HF_API_KEY && pytest ./fastvideo/tests/eval -vs",
        build_kernel=True,
        install_command='uv pip install -e ".[test,eval-full]"')


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
    # PR/direct records are uploaded only on pass; scheduled main uploads pass
    # and fail so the dashboard records every canonical baseline attempt.
    run_test(
        "export HF_HOME='/root/data/.cache' && "
        "export PERFORMANCE_TRACKING_ROOT='/tmp/perf-tracking' && "
        "hf auth login --token $HF_API_KEY && "
        "if [ \"${BUILDKITE_BRANCH:-}\" = 'main' ] && [ \"${TEST_SCOPE:-}\" = 'full' ]; then "
        "export PERF_RUN_SOURCE='scheduled_main'; "
        "export PERF_UPLOAD_POLICY='always'; "
        "elif [ -n \"${BUILDKITE_PULL_REQUEST:-}\" ] && [ \"${BUILDKITE_PULL_REQUEST:-false}\" != 'false' ]; then "
        "export PERF_RUN_SOURCE='pr'; "
        "export PERF_UPLOAD_POLICY='pass'; "
        "elif [ \"${TEST_SCOPE:-}\" = 'direct' ]; then "
        "export PERF_RUN_SOURCE='unknown'; "
        "export PERF_UPLOAD_POLICY='pass'; "
        "else "
        "export PERF_RUN_SOURCE='unknown'; "
        "export PERF_UPLOAD_POLICY='never'; "
        "fi; "
        "pytest ./fastvideo/tests/performance -vs; "
        "PYTEST_RC=$?; "
        "PERF_RC=0; "
        "if [ $PYTEST_RC -eq 0 ] || [ \"$PERF_UPLOAD_POLICY\" = 'always' ]; then "
        "PERF_PYTEST_RC=$PYTEST_RC python ./fastvideo/tests/performance/compare_baseline.py; "
        "PERF_RC=$?; "
        "fi; "
        "python ./fastvideo/tests/performance/dashboard.py || true; "
        "FINAL_RC=$PYTEST_RC; "
        "if [ $FINAL_RC -eq 0 ]; then FINAL_RC=$PERF_RC; fi; "
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
