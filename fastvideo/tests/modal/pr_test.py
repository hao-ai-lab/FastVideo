import os

import modal

app = modal.App()

model_vol = modal.Volume.from_name("hf-model-weights")
# Build cache for Flash Attention 3 (Hopper sm90+). FA3 is not in the default
# fastvideo-dev image, and a source build from flash-attention/hopper takes
# ~90 min on the Modal container CPU. Cached artifacts are keyed on the
# torch + CUDA + python ABI (read at runtime), so an image bump that changes
# any of them misses the cache and triggers a fresh build automatically.
fa3_cache_vol = modal.Volume.from_name("fa3-build-cache", create_if_missing=True)
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
    "BUILDKITE_BUILD_URL":
    os.environ.get("BUILDKITE_BUILD_URL", ""),
    "BUILDKITE_BUILD_ID":
    os.environ.get("BUILDKITE_BUILD_ID", ""),
    "BUILDKITE_JOB_ID":
    os.environ.get("BUILDKITE_JOB_ID", ""),
    "TEST_SCOPE":
    os.environ.get("TEST_SCOPE", ""),
    "IMAGE_VERSION":
    os.environ.get("IMAGE_VERSION", ""),
    "HF_REPO_ID":
    "FastVideo/performance-tracking",
}))

dreamverse_image = (image.run_commands(
    "curl -fsSL https://deb.nodesource.com/setup_22.x | bash -"
).apt_install("nodejs").run_commands("node --version && npm --version"))


def run_test(pytest_command: str):
    """Helper function to run a test suite with custom pytest command"""
    run_test_command(f'uv pip install -e ".[test]" && {pytest_command}',
                     build_kernel=True)


def run_test_command(test_command: str, build_kernel: bool):
    """Helper function to run a test suite with custom test command.

    Most FastVideo CI suites need the custom kernel build. App-level tests like
    DreamVerse's mock-backend UI checks do not, so keep the kernel build
    optional to avoid unrelated CUDA/kernel setup in that CI path.
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

    command = f"""
    source $HOME/.local/bin/env &&
    source /opt/venv/bin/activate &&
    git clone {git_repo} /FastVideo &&
    cd /FastVideo &&
    {checkout_command} &&
    git submodule update --init --recursive &&
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
        """
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
        """,
        build_kernel=False)


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
        'uv pip install -e ".[test,eval-full]" && '
        "export HF_HOME='/root/data/.cache' && hf auth login --token $HF_API_KEY && pytest ./fastvideo/tests/eval -vs",
        build_kernel=True)


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


@app.function(
    gpu="H100:2",
    image=image,
    # Cold FA3 source build is ~90 min (293 templated nvcc kernels at
    # MAX_JOBS=8 on the Modal container CPU); cache hit drops install to ~30 s.
    timeout=10800,
    secrets=[
        modal.Secret.from_dict({
            "HF_API_KEY": os.environ.get("HF_API_KEY", ""),
            "IMAGE_VERSION": os.environ.get("IMAGE_VERSION", ""),
        })
    ],
    volumes={
        "/root/data": model_vol,
        "/fa3_cache": fa3_cache_vol,
    },
)
def run_performance_tests_h100():
    """Run inference perf tests on H100:2 with Flash Attention 3 enabled.

    FA3 (Hopper sm90+) is not in the default fastvideo-dev image. This
    function installs it via a three-tier strategy:
      1. already-present  — image baked it in (future-proof; not today)
      2. cache restore    — Modal Volume hit; ~30 s copy
      3. source build     — flash-attention/hopper HEAD; ~90 min nvcc compile,
                            then transactional save to the cache volume

    The cache key is the torch+CUDA+python ABI (read at runtime), so an image
    bump that changes the ABI triggers a rebuild. After install, an assertion
    verifies ``fa_version == '3'`` so a silent fallback to FA2 fails the job.

    Pytest then runs the entire ``fastvideo/tests/performance`` suite, so
    any benchmark whose ``required_gpus <= 2`` will exercise FA3 on Hopper.
    """
    fa3_install = (
        "SP=$(python -c 'import site; print(site.getsitepackages()[0])') && "
        # Key on the build ABI (torch+CUDA+python), not the mutable image tag, so
        # a torch/CUDA bump misses the cache instead of restoring a stale binary.
        "FA3_ABI=$(python -c 'import sys, torch; print(f\"py{sys.version_info.major}{sys.version_info.minor}-torch{torch.__version__}-cu{torch.version.cuda}\")') && "
        "CACHE_DIR=/fa3_cache/fa3-${FA3_ABI} && "
        "CACHE_MARKER=$CACHE_DIR/.INSTALLED && "
        "if python -c 'import flash_attn_interface' 2>/dev/null; then "
        "  echo '[fa3] already present in image'; "
        "elif [ -f \"$CACHE_MARKER\" ]; then "
        "  echo '[fa3] cache hit at' $CACHE_DIR && "
        "  cp -r $CACHE_DIR/. $SP/ && "
        "  python -c 'import flash_attn_interface' && "
        "  echo '[fa3] restored from cache'; "
        "else "
        "  echo '[fa3] cache miss; building flash-attention/hopper (~90 min)' && "
        "  uv pip install -q ninja && "
        "  git clone --depth 1 https://github.com/Dao-AILab/flash-attention.git /tmp/flash-attention && "
        "  ( cd /tmp/flash-attention/hopper && MAX_JOBS=8 python setup.py install ) && "
        "  python -c 'import flash_attn_interface' && "
        "  STAGE=/fa3_cache/staging-$$ && mkdir -p $STAGE && "
        # flash_attn_config is a py_module lazily imported by flash_attn_interface;
        # drop it and a restored cache imports fine but crashes at the first forward.
        "  for pat in 'flash_attn_interface*' 'flash_attn_config*' 'flash_attn_3*'; do "
        "    find $SP -maxdepth 1 -name \"$pat\" -exec cp -r {} $STAGE/ \\; 2>/dev/null; "
        "  done && "
        "  rm -rf $CACHE_DIR && mv $STAGE $CACHE_DIR && touch $CACHE_MARKER && "
        "  echo '[fa3] cache saved to' $CACHE_DIR; "
        "fi"
    )
    verify_fa3 = (
        "python -c \"from fastvideo.attention.backends.flash_attn import fa_version; "
        "assert fa_version == '3', f'expected fa_version=3, got {fa_version}'\""
    )
    try:
        run_test(
            "export HF_HOME='/root/data/.cache' && "
            "export PERFORMANCE_TRACKING_ROOT='/tmp/perf-tracking' && "
            "hf auth login --token $HF_API_KEY && "
            f"{fa3_install} && "
            f"{verify_fa3} && "
            "pytest ./fastvideo/tests/performance -vs; "
            "PYTEST_RC=$?; "
            "PERF_RC=0; "
            "if [ $PYTEST_RC -eq 0 ]; then "
            "python ./fastvideo/tests/performance/compare_baseline.py; "
            "PERF_RC=$?; "
            "fi; "
            "python ./fastvideo/tests/performance/dashboard.py || true; "
            "FINAL_RC=$PYTEST_RC; "
            "if [ $FINAL_RC -eq 0 ]; then FINAL_RC=$PERF_RC; fi; "
            "exit $FINAL_RC"
        )
    finally:
        try:
            fa3_cache_vol.commit()
        except Exception as exc:
            print(f"[fa3] cache volume commit failed: {exc}")
