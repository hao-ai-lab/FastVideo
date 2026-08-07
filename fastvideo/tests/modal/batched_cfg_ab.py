"""
A/B harness for ``use_batched_cfg`` on the main ``DenoisingStage``.

Single-container pattern: clone the patched branch, build once, then
run the same N seed-pinned prompts twice — once with
``use_batched_cfg=False`` (legacy sequential cond+uncond pair) and once
with ``use_batched_cfg=True`` (single batch=2 forward per step). Records
per-prompt wall + pairwise SSIM/LPIPS, prints a delta table.

Modeled on the overlay-pattern Kuan used for #1395
(``nccl_stream_ab.py``) — same container, same model load, isolates the
one-flag change from container/topology variance.

Usage (from FastVideo repo root):

    modal run fastvideo/tests/modal/batched_cfg_ab.py --gpu L40S \\
        --model wan2_1-t2v-1.3b --num-gpus 1

    modal run fastvideo/tests/modal/batched_cfg_ab.py --gpu H100 \\
        --model wan2_2-t2v-14b --num-gpus 2

Requires a Modal Secret named ``huggingface-token`` with key
``HF_TOKEN`` (create once: ``modal secret create huggingface-token
HF_TOKEN=hf_xxx``).
"""
import os

import modal

app = modal.App("batched-cfg-ab")

model_vol = modal.Volume.from_name("hf-model-weights")
hf_secret = modal.Secret.from_name("huggingface-token")
# Pin to py3.12-latest (built from docker/Dockerfile.python3.12). Its
# baked FA2 wheel is cu128torch2.11-cp312, which matches the torch
# version pulled by `uv pip install -e ".[test]"` (torch 2.11.0+cu128)
# — no ABI mismatch on the post-install FastVideo import. The plain
# `latest` tag is Python 3.10 + an older torch that gets bumped to
# 2.11.0 by [test], stranding the baked FA2 wheel.
image_tag = f"ghcr.io/hao-ai-lab/fastvideo/fastvideo-dev:{os.getenv('IMAGE_VERSION', 'py3.12-latest')}"

# Prebuilt FA3 wheel for the same ABI as the FA2 wheel the
# fastvideo-dev image ships (cu128torch2.11, stable cp39-abi3, works on
# cp312). Per Will Slack 2026-05-28 -> mjun0812/flash-attention-
# prebuild-wheels. ~30s install vs ~90min cold source build; sidesteps
# Kuan's #1389 Volume-cache pattern entirely on the inference path.
FA3_WHEEL_URL = (
    "https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.9.4/"
    "flash_attn_3-3.0.0%2Bcu128torch2.11gite2743ab-cp39-abi3-linux_x86_64.whl")

image = (modal.Image.from_registry(image_tag, add_python="3.12").apt_install(
    "cmake",
    "pkg-config",
    "build-essential",
    "curl",
    "libssl-dev",
    "ffmpeg",
    "libgl1",
    "libglib2.0-0",
).run_commands(
    "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable",
    "echo 'source ~/.cargo/env' >> ~/.bashrc",
).env({"PATH": "/root/.cargo/bin:$PATH"}))

# Same five prompts Kuan used for the #1395 A/B, so deltas across PRs in
# this thread (W3a/W3b/W3c/W3d) are directly comparable.
AB_PROMPTS = [
    "A curious raccoon peers through a vibrant field of yellow sunflowers, its eyes wide with interest. "
    "The playful yet serene atmosphere is complemented by soft natural light filtering through the petals. "
    "Mid-shot, warm and cheerful tones.",
    "A majestic lion strides across the golden savanna, its powerful frame glistening under the warm afternoon "
    "sun. The tall grass ripples gently in the breeze, enhancing the lion's commanding presence. Low angle, "
    "steady tracking shot, cinematic.",
    "A sailing ship cuts through dark stormy waves under a sky of rolling thunderclouds. Sea spray catches the "
    "lantern light along the hull. Dramatic, painterly tones; medium-wide tracking shot.",
    "A street vendor in Tokyo flips okonomiyaki on a sizzling iron griddle while neon shop signs reflect in "
    "rain-slicked pavement. Steam rises around her hands; warm color grade; handheld close-up.",
    "An astronaut floats slowly past the cupola of a space station, the curve of Earth glowing blue beyond "
    "the glass. Calm, contemplative pacing; smooth dolly; cinematic.",
]
AB_SEED = 42

# Per-model defaults matched to ground_truth shapes for L40S Wan 1.3B
# and H100 Wan 14B. Resolutions / frame counts are the same as Kuan's
# #1395 baseline so deltas across W3 sub-PRs compare directly.
MODEL_PRESETS = {
    "wan2_1-t2v-1.3b": {
        "model_id": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        "height": 720,
        "width": 1280,
        "num_frames": 77,
        "num_inference_steps": 30,
        "default_num_gpus": 1,
    },
    "wan2_2-t2v-14b": {
        "model_id": "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
        "height": 720,
        "width": 1280,
        "num_frames": 49,
        "num_inference_steps": 30,
        "default_num_gpus": 2,
    },
}


def _build_workspace_command(git_repo: str, git_ref: str, install_fa3: bool, full_install: bool = True) -> str:
    """Container-side bootstrap. Reads ``HF_TOKEN`` from the Modal
    Secret (mounted as env). When ``install_fa3`` is set, also installs
    the prebuilt FA3 wheel before the FastVideo install (FastVideo's
    flash_attn backend module picks FA3 at import time if importable —
    see flash_attn.py:13-19).

    With ``full_install=False`` (used by ``recover_ssim``), skip the
    [test] install, the kernel build, and the fa_version verify — those
    all import ``fastvideo`` which transitively pulls triton (via
    vmoba), which needs a CUDA driver to initialise. We just need the
    inner script file accessible on disk for CPU-only SSIM recompute.
    """
    import shlex
    fa3_step = f'uv pip install --no-cache-dir "{FA3_WHEEL_URL}"' if install_fa3 else 'echo "[fa3] skipped"'
    install_block = f"""
{fa3_step}
uv pip install -e ".[test]"
cd fastvideo-kernel && ./build.sh && cd ..
export HF_HOME=/root/data/.cache
hf auth login --token "$HF_TOKEN"
python -c "from fastvideo.attention.backends.flash_attn import fa_version; print(f'[fa] resolved fa_version={{fa_version}}')"
""" if full_install else """
# Minimal recovery bootstrap: clone only, no fastvideo install.
# The inner script's compute_ssim mode uses pytorch_msssim + torchvision/av
# directly (no fastvideo import) so the image's venv suffices. Matches the
# SSIM library fastvideo/tests/utils.py uses, so our numbers are directly
# comparable to the canonical helper.
uv pip install --no-cache-dir pytorch_msssim av || echo "[recover] pytorch_msssim/av already present"
echo "[recover] skipped fastvideo install + kernel build"
"""
    return f"""
set -euxo pipefail
source $HOME/.local/bin/env
source $HOME/.cargo/env
source /opt/venv/bin/activate
if [ -d /FastVideo/.git ]; then
  cd /FastVideo && git remote set-url origin {shlex.quote(git_repo)} && git fetch --prune origin
else
  git clone {shlex.quote(git_repo)} /FastVideo && cd /FastVideo
fi
git checkout {shlex.quote(git_ref)}
git submodule update --init --recursive
{install_block}
"""


def _run_pass(*, use_batched_cfg: bool, model_id: str, num_gpus: int, height: int, width: int, num_frames: int,
              num_inference_steps: int, output_dir: str, enable_compile: bool,
              num_prompts: int | None = None) -> list[dict]:
    """Invoke the inner pass script (``_batched_cfg_ab_inner.py``) via
    /opt/venv/bin/python so FastVideo runs inside the image's venv.

    Modal's main function process runs in Modal's own Python interpreter
    (the ``add_python="3.12"`` layer) which doesn't have FastVideo, so
    a direct ``from fastvideo import VideoGenerator`` fails with
    ``ModuleNotFoundError: No module named 'torch'``. The inner script
    is checked into the repo and runs the actual pass; we ferry the
    per-prompt records back over a JSON file.
    """
    import json
    import subprocess
    import tempfile

    selected_prompts = list(AB_PROMPTS) if num_prompts is None else list(AB_PROMPTS)[:num_prompts]
    config = {
        "model_id": model_id,
        "num_gpus": num_gpus,
        "use_batched_cfg": use_batched_cfg,
        "enable_compile": enable_compile,
        "height": height,
        "width": width,
        "num_frames": num_frames,
        "num_inference_steps": num_inference_steps,
        "output_dir": output_dir,
        "prompts": selected_prompts,
        "seed_base": AB_SEED,
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(config, f)
        config_path = f.name
    results_path = config_path.replace(".json", ".results.json")

    inner_script = "/FastVideo/fastvideo/tests/modal/_batched_cfg_ab_inner.py"
    cmd = (f"source /opt/venv/bin/activate && "
           f"exec python {inner_script} "
           f"--config-json {config_path} --results-json {results_path}")
    subprocess.run(["/bin/bash", "-lc", cmd], check=True)

    with open(results_path) as f:
        return json.load(f)


def _pairwise_ssim_lpips(baseline_records: list[dict], patched_records: list[dict]) -> list[dict]:
    """For each prompt index, compute SSIM between baseline and patched
    outputs. Same subprocess-into-venv pattern as ``_run_pass`` because
    fastvideo.tests.utils imports torch and we're in Modal's add_python
    interpreter where torch isn't installed."""
    import json
    import subprocess
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".baseline.json", delete=False) as f:
        json.dump(baseline_records, f)
        baseline_path = f.name
    with tempfile.NamedTemporaryFile(mode="w", suffix=".patched.json", delete=False) as f:
        json.dump(patched_records, f)
        patched_path = f.name
    ssim_path = baseline_path.replace(".baseline.json", ".ssim.json")

    inner_script = "/FastVideo/fastvideo/tests/modal/_batched_cfg_ab_inner.py"
    cmd = (f"source /opt/venv/bin/activate && exec python {inner_script} --mode compute_ssim "
           f"--baseline-results-json {baseline_path} --patched-results-json {patched_path} "
           f"--ssim-output-json {ssim_path}")
    subprocess.run(["/bin/bash", "-lc", cmd], check=True)

    with open(ssim_path) as f:
        return json.load(f)


def _print_table(rows: list[dict]) -> None:
    print("\n=== batched_cfg A/B results ===")
    print(f"{'prompt':>6} {'baseline_wall_s':>16} {'patched_wall_s':>16} {'delta_pct':>10} {'ssim_mean':>10} "
          f"{'ssim_worst':>11}")
    base_sum = patched_sum = 0.0
    ssim_means = []
    ssim_worsts = []
    for r in rows:
        delta = (r["patched_wall_s"] - r["baseline_wall_s"]) / r["baseline_wall_s"] * 100.0
        print(f"{r['i']:>6} {r['baseline_wall_s']:>16.3f} {r['patched_wall_s']:>16.3f} {delta:>9.2f}% "
              f"{r['ssim_mean']:>10.6f} {r['ssim_worst']:>11.6f}")
        base_sum += r["baseline_wall_s"]
        patched_sum += r["patched_wall_s"]
        ssim_means.append(r["ssim_mean"])
        ssim_worsts.append(r["ssim_worst"])
    delta_total = (patched_sum - base_sum) / base_sum * 100.0
    print(f"\nbaseline total: {base_sum:.3f}s")
    print(f"patched  total: {patched_sum:.3f}s")
    print(f"delta:          {delta_total:+.2f}%")
    print(f"SSIM mean-of-means: {sum(ssim_means) / len(ssim_means):.6f}")
    print(f"SSIM worst-of-worst: {min(ssim_worsts):.6f}")


@app.function(
    image=image,
    timeout=10800,
    volumes={"/root/data": model_vol},
    secrets=[hf_secret],
    gpu="L40S:1",
)
def run_ab(
    *,
    git_repo: str,
    git_ref: str,
    model_preset: str,
    num_gpus: int,
    install_fa3: bool = False,
    enable_compile: bool = False,
    num_prompts: int = 0,
):
    import subprocess

    if model_preset not in MODEL_PRESETS:
        raise ValueError(f"unknown model_preset: {model_preset}")
    if "HF_TOKEN" not in os.environ:
        raise RuntimeError("HF_TOKEN not set in container env — Modal Secret 'huggingface-token' missing or "
                           "missing the HF_TOKEN key.")
    preset = MODEL_PRESETS[model_preset]

    result = subprocess.run(
        ["/bin/bash", "-lc", _build_workspace_command(git_repo, git_ref, install_fa3=install_fa3)],
        env=os.environ.copy(),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr)
        raise RuntimeError(f"workspace setup failed (rc={result.returncode})")
    os.chdir("/FastVideo")

    common = dict(
        model_id=preset["model_id"],
        num_gpus=num_gpus,
        height=preset["height"],
        width=preset["width"],
        num_frames=preset["num_frames"],
        num_inference_steps=preset["num_inference_steps"],
        enable_compile=enable_compile,
        num_prompts=(num_prompts if num_prompts > 0 else None),
    )

    # Output dirs are tagged by mode so eager and compile runs don't
    # clobber each other (both legs may run in the same Volume).
    mode_tag = "compile" if enable_compile else "eager"
    baseline_dir = f"/root/data/ab_out/{model_preset}/{mode_tag}/baseline"
    patched_dir = f"/root/data/ab_out/{model_preset}/{mode_tag}/patched"

    print(f"\n--- baseline pass (use_batched_cfg=False, mode={mode_tag}) on {model_preset} ---")
    baseline = _run_pass(use_batched_cfg=False, output_dir=baseline_dir, **common)

    print(f"\n--- patched pass (use_batched_cfg=True, mode={mode_tag}) on {model_preset} ---")
    patched = _run_pass(use_batched_cfg=True, output_dir=patched_dir, **common)

    rows = _pairwise_ssim_lpips(baseline, patched)
    _print_table(rows)
    return rows


@app.function(
    image=image,
    timeout=600,
    volumes={"/root/data": model_vol},
    secrets=[hf_secret],
    gpu="H100:1",
)
def probe_fa_version() -> str:
    """5-minute FA3-availability probe on H100. Imports the
    FastVideo flash-attn backend and reports which version got picked.
    Use this once before kicking off the H100 leg of the A/B to confirm
    the container has FA3 installed (otherwise the leg silently falls
    back to FA2 and the "we tested on FA3" claim in the PR body is
    wrong).

    Run: modal run fastvideo/tests/modal/batched_cfg_ab.py::probe_fa_version
    """
    import subprocess
    import sys

    if "HF_TOKEN" not in os.environ:
        raise RuntimeError("HF_TOKEN missing — check the huggingface-token Modal Secret.")
    # Minimal probe: do NOT clone or pip-install FastVideo. Just inspect
    # what flash-attn variants are present in the image. FastVideo's
    # flash_attn backend module fails-loud (unhandled ImportError) if
    # none of FA2/FA3/FA4 are installed — so going through it conflates
    # "FA3 missing" with "all FAs missing". Probe the three packages
    # individually instead. Source ~/.local/bin/env for `uv` parity with
    # the main A/B bootstrap, though we don't use it here.
    setup = f"""
    set -euxo pipefail
    source $HOME/.local/bin/env
    source /opt/venv/bin/activate
    echo '--- before FA3 install ---'
    python -c "
import torch
print(f'cuda_device={{torch.cuda.get_device_name(0)}}')
print(f'cuda_capability={{torch.cuda.get_device_capability(0)}}')
for name in ('flash_attn', 'flash_attn_interface'):
    try:
        __import__(name)
        print(f'{{name}}=present')
    except ImportError:
        print(f'{{name}}=missing')
"
    echo
    echo '--- installing FA3 wheel ---'
    time uv pip install --no-cache-dir "{FA3_WHEEL_URL}"
    echo
    echo '--- baked /FastVideo source state (diagnostic) ---'
    cd /FastVideo && git log -1 --oneline 2>/dev/null || echo '(no git history in image)'
    echo
    echo '--- after FA3 install: verify wheel directly, skip stale FastVideo wrapper ---'
    # The fastvideo-dev image bakes FastVideo source at build time; the
    # image's flash_attn.py predates the fa_version symbol. The actual
    # A/B run (run_ab) does `git checkout` of our branch via
    # _build_workspace_command, so the wrapper there is current. For
    # the probe we only need to confirm the wheel is usable.
    python -c "
import flash_attn_interface as fa3
print(f'flash_attn_interface=present ({{fa3.__file__}})')
assert hasattr(fa3, 'flash_attn_func'), 'wheel missing flash_attn_func -- wrong wheel?'
print(f'flash_attn_func callable: {{callable(fa3.flash_attn_func)}}')
print()
print('OK: FA3 wheel installed and importable.')
print('Run main A/B (which does git checkout of perf/wan-batched-cfg)')
print('to validate fa_version=3 resolves through the current wrapper.')
"
    """
    result = subprocess.run(["/bin/bash", "-lc", setup], capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        raise RuntimeError(f"probe failed rc={result.returncode}")
    return result.stdout


@app.function(
    image=image,
    timeout=1800,
    volumes={"/root/data": model_vol},
    secrets=[hf_secret],
    cpu=4,
)
def recover_ssim(
    *,
    git_repo: str,
    git_ref: str,
    model_preset: str = "wan2_1-t2v-1.3b",
    mode_tag: str = "eager",
) -> list[dict]:
    """Recompute the A/B SSIM table from existing baseline + patched
    mp4s in the Volume. Use this after an A/B run that completed the
    passes but failed during the in-process SSIM step. ~5min, ~$0.05;
    no GPU.

    Run: modal run fastvideo/tests/modal/batched_cfg_ab.py::recover_ssim --git-repo ... --git-ref ...
    """
    import json
    import subprocess
    import tempfile

    # Need /FastVideo source so the inner script is on disk. SSIM
    # compute itself runs without importing fastvideo, so we skip the
    # heavy install (no triton init -> no CUDA driver requirement).
    if "HF_TOKEN" not in os.environ:
        raise RuntimeError("HF_TOKEN missing")
    setup_result = subprocess.run(
        ["/bin/bash", "-lc", _build_workspace_command(git_repo, git_ref, install_fa3=False,
                                                     full_install=False)],
        env=os.environ.copy(),
        capture_output=True,
        text=True,
    )
    if setup_result.returncode != 0:
        print(setup_result.stdout)
        print(setup_result.stderr)
        raise RuntimeError(f"workspace setup failed (rc={setup_result.returncode})")

    base_dir = f"/root/data/ab_out/{model_preset}/{mode_tag}/baseline"
    patched_dir = f"/root/data/ab_out/{model_preset}/{mode_tag}/patched"

    def _scan(root: str) -> list[dict]:
        """Pick the NEWEST mp4 per prompt_NN directory. FastVideo's
        generate_video doesn't overwrite — it appends _1, _2, ... to
        avoid name collisions. After multiple A/B runs the same dir
        accumulates mp4s from each run; we want the one written by the
        most recent run."""
        records = []
        for entry in sorted(os.listdir(root)):
            if not entry.startswith("prompt_"):
                continue
            i = int(entry.split("_")[1])
            prompt_dir = os.path.join(root, entry)
            mp4s = sorted(
                [os.path.join(prompt_dir, f) for f in os.listdir(prompt_dir) if f.endswith(".mp4")],
                key=os.path.getmtime,
            )
            if not mp4s:
                continue
            records.append({"i": i, "wall_s": float("nan"), "mp4": mp4s[-1]})
        return records

    baseline = _scan(base_dir)
    patched = _scan(patched_dir)
    print(f"[recover] baseline videos: {len(baseline)}; patched videos: {len(patched)}")
    if not baseline or not patched:
        raise RuntimeError(f"missing videos. baseline={base_dir} patched={patched_dir}")

    with tempfile.NamedTemporaryFile(mode="w", suffix=".baseline.json", delete=False) as f:
        json.dump(baseline, f)
        baseline_path = f.name
    with tempfile.NamedTemporaryFile(mode="w", suffix=".patched.json", delete=False) as f:
        json.dump(patched, f)
        patched_path = f.name
    ssim_path = baseline_path.replace(".baseline.json", ".ssim.json")

    cmd = (f"source /opt/venv/bin/activate && exec python "
           f"/FastVideo/fastvideo/tests/modal/_batched_cfg_ab_inner.py --mode compute_ssim "
           f"--baseline-results-json {baseline_path} --patched-results-json {patched_path} "
           f"--ssim-output-json {ssim_path}")
    subprocess.run(["/bin/bash", "-lc", cmd], check=True)

    with open(ssim_path) as f:
        rows = json.load(f)
    _print_table(rows)
    return rows


@app.local_entrypoint()
def main(
    gpu: str = "L40S",
    model: str = "wan2_1-t2v-1.3b",
    num_gpus: int = 0,
    git_repo: str = "",
    git_ref: str = "perf/wan-batched-cfg",
    enable_compile: bool = False,
    num_prompts: int = 0,
):
    """Drive the A/B from your laptop. ``gpu`` is the Modal GPU class
    string (``L40S``, ``H100``, ``A100-80GB``, ...). ``num_gpus=0``
    uses the model preset default. ``git_repo`` defaults to the ``fork``
    remote (where W3 sub-branches live) and falls back to ``origin``."""
    import subprocess

    if model not in MODEL_PRESETS:
        raise ValueError(f"unknown model: {model}; choose from {list(MODEL_PRESETS)}")
    if num_gpus == 0:
        num_gpus = MODEL_PRESETS[model]["default_num_gpus"]
    if not git_repo:
        for remote in ("fork", "origin"):
            try:
                git_repo = subprocess.check_output(
                    ["git", "config", "--get", f"remote.{remote}.url"],
                    text=True,
                    stderr=subprocess.DEVNULL,
                ).strip()
                break
            except subprocess.CalledProcessError:
                continue
        if not git_repo:
            raise RuntimeError("Could not resolve git_repo. Pass --git-repo or configure a 'fork' or 'origin' remote.")

    # FA3 install only on Hopper (capability 9.0). L40S is sm_89 and
    # can't use FA3 — stays on the image's baked FA2.
    install_fa3 = gpu.upper().startswith("H100") or gpu.upper().startswith("H200")
    print(f"GPU: {gpu}:{num_gpus}  model: {model}  ref: {git_ref}  install_fa3: {install_fa3}  "
          f"enable_compile: {enable_compile}  num_prompts: {num_prompts or 'all'}  repo: {git_repo}")

    # Re-bind the GPU class at call time (Modal supports passing gpu via
    # .with_options).
    run_ab.with_options(gpu=f"{gpu}:{num_gpus}").remote(
        git_repo=git_repo,
        git_ref=git_ref,
        model_preset=model,
        num_gpus=num_gpus,
        install_fa3=install_fa3,
        enable_compile=enable_compile,
        num_prompts=num_prompts,
    )
