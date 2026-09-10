"""
A/B harness for ``use_cachedit`` (cache-dit step caching) on the Wan DiT.

Single-container pattern: clone the branch, build once, then run the same N
seed-pinned prompts twice — once with caching off (baseline) and once with
``use_cachedit=True`` (optionally + TaylorSeer). Records per-prompt wall +
pairwise SSIM and prints a delta table.

cache-dit caching is LOSSY: the SSIM column measures the quality cost (target
>= ~0.95), it is NOT a 1.0 correctness gate. The "win" is the wall delta.
Sweep ``--threshold`` (and ``--fn`` / ``--bn`` / ``--warmup`` / ``--taylorseer``)
to trade speed for quality.

Usage (from FastVideo repo root):

    # F8B0 + TaylorSeer, threshold 0.08 (the balanced default), L40S Wan 1.3B
    modal run fastvideo/tests/modal/cachedit_ab.py --gpu L40S --taylorseer

    # More aggressive (recover more wall): higher threshold
    modal run fastvideo/tests/modal/cachedit_ab.py --gpu L40S --taylorseer --threshold 0.15

Requires a Modal Secret named ``huggingface-token`` with key ``HF_TOKEN``.
"""
import os

import modal

app = modal.App("cachedit-ab")

model_vol = modal.Volume.from_name("hf-model-weights")
hf_secret = modal.Secret.from_name("huggingface-token")
image_tag = f"ghcr.io/hao-ai-lab/fastvideo/fastvideo-dev:{os.getenv('IMAGE_VERSION', 'py3.12-latest')}"

# Prebuilt FA3 wheel matching the fastvideo-dev image ABI (Hopper leg only).
FA3_WHEEL_URL = (
    "https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.9.4/"
    "flash_attn_3-3.0.0%2Bcu128torch2.11gite2743ab-cp39-abi3-linux_x86_64.whl")

image = (modal.Image.from_registry(image_tag, add_python="3.12").apt_install(
    "cmake", "pkg-config", "build-essential", "curl", "libssl-dev", "ffmpeg", "libgl1", "libglib2.0-0",
).run_commands(
    "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable",
    "echo 'source ~/.cargo/env' >> ~/.bashrc",
).env({"PATH": "/root/.cargo/bin:$PATH"}))

# Same five prompts as the prior Wan A/Bs so deltas compare directly.
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

MODEL_PRESETS = {
    "wan2_1-t2v-1.3b": {
        "model_id": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        "height": 720, "width": 1280, "num_frames": 77, "num_inference_steps": 30, "default_num_gpus": 1,
    },
    "wan2_2-t2v-14b": {
        "model_id": "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
        "height": 720, "width": 1280, "num_frames": 49, "num_inference_steps": 30, "default_num_gpus": 2,
    },
}


def _build_workspace_command(git_repo: str, git_ref: str, install_fa3: bool) -> str:
    import shlex
    fa3_step = f'uv pip install --no-cache-dir "{FA3_WHEEL_URL}"' if install_fa3 else 'echo "[fa3] skipped"'
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
{fa3_step}
uv pip install -e ".[test,cache]"
cd fastvideo-kernel && ./build.sh && cd ..
export HF_HOME=/root/data/.cache
hf auth login --token "$HF_TOKEN"
"""


def _run_pass(*, use_cachedit: bool, model_id: str, num_gpus: int, preset: dict, output_dir: str,
              enable_compile: bool, num_prompts, fn: int, bn: int, threshold: float, warmup: int,
              taylorseer: bool, taylorseer_order: int) -> list[dict]:
    import json
    import subprocess
    import tempfile

    selected = list(AB_PROMPTS) if num_prompts is None else list(AB_PROMPTS)[:num_prompts]
    config = {
        "model_id": model_id, "num_gpus": num_gpus, "use_cachedit": use_cachedit,
        "cachedit_fn_compute_blocks": fn, "cachedit_bn_compute_blocks": bn,
        "cachedit_residual_threshold": threshold, "cachedit_max_warmup_steps": warmup,
        "cachedit_taylorseer": taylorseer, "cachedit_taylorseer_order": taylorseer_order,
        "enable_compile": enable_compile, "height": preset["height"], "width": preset["width"],
        "num_frames": preset["num_frames"], "num_inference_steps": preset["num_inference_steps"],
        "output_dir": output_dir, "prompts": selected, "seed_base": AB_SEED,
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(config, f)
        config_path = f.name
    results_path = config_path.replace(".json", ".results.json")
    inner = "/FastVideo/fastvideo/tests/modal/_cachedit_ab_inner.py"
    cmd = (f"source /opt/venv/bin/activate && exec python {inner} "
           f"--config-json {config_path} --results-json {results_path}")
    try:
        subprocess.run(["/bin/bash", "-lc", cmd], check=True)
        with open(results_path) as f:
            return json.load(f)
    finally:
        for p in (config_path, results_path):
            try:
                os.unlink(p)
            except OSError:
                pass


def _pairwise_ssim(baseline: list[dict], patched: list[dict]) -> list[dict]:
    import json
    import subprocess
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".baseline.json", delete=False) as f:
        json.dump(baseline, f)
        bpath = f.name
    with tempfile.NamedTemporaryFile(mode="w", suffix=".patched.json", delete=False) as f:
        json.dump(patched, f)
        ppath = f.name
    spath = bpath.replace(".baseline.json", ".ssim.json")
    inner = "/FastVideo/fastvideo/tests/modal/_cachedit_ab_inner.py"
    cmd = (f"source /opt/venv/bin/activate && exec python {inner} --mode compute_ssim "
           f"--baseline-results-json {bpath} --patched-results-json {ppath} --ssim-output-json {spath}")
    try:
        subprocess.run(["/bin/bash", "-lc", cmd], check=True)
        with open(spath) as f:
            return json.load(f)
    finally:
        for p in (bpath, ppath, spath):
            try:
                os.unlink(p)
            except OSError:
                pass


def _print_table(rows: list[dict], label: str) -> None:
    print(f"\n=== cache-dit A/B results ({label}) ===")
    print("NOTE: cache-dit is lossy — SSIM is a quality cost (target >= ~0.95), not a 1.0 gate.")
    print(f"{'prompt':>6} {'baseline_wall_s':>16} {'patched_wall_s':>16} {'delta_pct':>10} {'ssim_mean':>10} "
          f"{'ssim_worst':>11}")
    base_sum = patched_sum = 0.0
    means, worsts = [], []
    for r in rows:
        delta = (r["patched_wall_s"] - r["baseline_wall_s"]) / r["baseline_wall_s"] * 100.0
        print(f"{r['i']:>6} {r['baseline_wall_s']:>16.3f} {r['patched_wall_s']:>16.3f} {delta:>9.2f}% "
              f"{r['ssim_mean']:>10.6f} {r['ssim_worst']:>11.6f}")
        base_sum += r["baseline_wall_s"]
        patched_sum += r["patched_wall_s"]
        means.append(r["ssim_mean"])
        worsts.append(r["ssim_worst"])
    print(f"\nbaseline total: {base_sum:.3f}s")
    print(f"cachedit total: {patched_sum:.3f}s")
    print(f"wall delta:     {(patched_sum - base_sum) / base_sum * 100.0:+.2f}%   (negative = faster)")
    print(f"SSIM mean-of-means:  {sum(means) / len(means):.6f}")
    print(f"SSIM worst-of-worst: {min(worsts):.6f}")


@app.function(image=image, timeout=10800, volumes={"/root/data": model_vol}, secrets=[hf_secret], gpu="L40S:1")
def run_ab(*, git_repo: str, git_ref: str, model_preset: str, num_gpus: int, install_fa3: bool = False,
           enable_compile: bool = False, num_prompts: int = 0, fn: int = 8, bn: int = 0, threshold: float = 0.08,
           warmup: int = 8, taylorseer: bool = False, taylorseer_order: int = 1):
    import subprocess

    if model_preset not in MODEL_PRESETS:
        raise ValueError(f"unknown model_preset: {model_preset}")
    if "HF_TOKEN" not in os.environ:
        raise RuntimeError("HF_TOKEN not set — Modal Secret 'huggingface-token' missing the HF_TOKEN key.")
    preset = MODEL_PRESETS[model_preset]

    result = subprocess.run(["/bin/bash", "-lc", _build_workspace_command(git_repo, git_ref, install_fa3)],
                            env=os.environ.copy(), capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr)
        raise RuntimeError(f"workspace setup failed (rc={result.returncode})")
    os.chdir("/FastVideo")

    common = dict(model_id=preset["model_id"], num_gpus=num_gpus, preset=preset, enable_compile=enable_compile,
                  num_prompts=(num_prompts if num_prompts > 0 else None), fn=fn, bn=bn, threshold=threshold,
                  warmup=warmup, taylorseer=taylorseer, taylorseer_order=taylorseer_order)

    ts_tag = f"-ts{taylorseer_order}" if taylorseer else ""
    cfg_tag = f"cachedit{ts_tag}_f{fn}b{bn}t{str(threshold).replace('.', 'p')}w{warmup}"
    baseline_dir = f"/root/data/cachedit_out/{model_preset}/baseline"
    patched_dir = f"/root/data/cachedit_out/{model_preset}/{cfg_tag}"
    label = f"{model_preset} | Fn={fn} Bn={bn} threshold={threshold} warmup={warmup} taylorseer={taylorseer}"

    print(f"\n--- baseline pass (caching off) on {model_preset} ---")
    baseline = _run_pass(use_cachedit=False, output_dir=baseline_dir, **common)
    print(f"\n--- cache-dit pass ({label}) ---")
    patched = _run_pass(use_cachedit=True, output_dir=patched_dir, **common)

    rows = _pairwise_ssim(baseline, patched)
    _print_table(rows, label)
    return rows


@app.local_entrypoint()
def main(gpu: str = "L40S", model: str = "wan2_1-t2v-1.3b", num_gpus: int = 0, git_repo: str = "",
         git_ref: str = "perf/wan-cachedit", enable_compile: bool = False, num_prompts: int = 0, fn: int = 8,
         bn: int = 0, threshold: float = 0.08, warmup: int = 8, taylorseer: bool = False, taylorseer_order: int = 1):
    """Drive the cache-dit A/B from your laptop. ``--taylorseer`` enables the
    TaylorSeer calibrator. Sweep ``--threshold`` (higher = more aggressive /
    faster / lower quality). ``git_repo`` defaults to the ``fork`` remote."""
    import subprocess

    if model not in MODEL_PRESETS:
        raise ValueError(f"unknown model: {model}; choose from {list(MODEL_PRESETS)}")
    if num_gpus == 0:
        num_gpus = MODEL_PRESETS[model]["default_num_gpus"]
    if not git_repo:
        for remote in ("fork", "origin"):
            try:
                git_repo = subprocess.check_output(["git", "config", "--get", f"remote.{remote}.url"],
                                                   text=True, stderr=subprocess.DEVNULL).strip()
                break
            except subprocess.CalledProcessError:
                continue
        if not git_repo:
            raise RuntimeError("Could not resolve git_repo. Pass --git-repo or configure a 'fork'/'origin' remote.")

    install_fa3 = gpu.upper().startswith("H100") or gpu.upper().startswith("H200")
    print(f"GPU: {gpu}:{num_gpus}  model: {model}  ref: {git_ref}  install_fa3: {install_fa3}  "
          f"compile: {enable_compile}  prompts: {num_prompts or 'all'}  "
          f"cache-dit: Fn={fn} Bn={bn} threshold={threshold} warmup={warmup} "
          f"taylorseer={taylorseer}(o{taylorseer_order})  repo: {git_repo}")

    run_ab.with_options(gpu=f"{gpu}:{num_gpus}").remote(
        git_repo=git_repo, git_ref=git_ref, model_preset=model, num_gpus=num_gpus, install_fa3=install_fa3,
        enable_compile=enable_compile, num_prompts=num_prompts, fn=fn, bn=bn, threshold=threshold, warmup=warmup,
        taylorseer=taylorseer, taylorseer_order=taylorseer_order)
