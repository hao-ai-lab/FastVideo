"""Headline e2e numbers (480p 5 s, 768p 10 s) for a FastH3 HF repo on 1 / 4 / 8 RTX PRO 6000 Blackwell GPUs.

Reuses the image and kernel wheel of ``app.py`` (run ``modal run app.py --step build`` once per volume).

    MODAL_PROFILE=aryan5v modal run --detach headline_app.py --repo FastVideo/FastH3-Pruned-8Step-NVFP4-ckpt300 \\
        --profile h3_dit_ffn --gpus 1,4,8

Results land in the ``h3-pro6000-weights`` volume under ``outputs/headline/<run>/`` (``results.json`` + clips).
"""
import json
import os
import pathlib
import subprocess

import modal

from app import FAST_ENV, _install_kernel, _sh, image, volume

HERE = pathlib.Path(__file__).resolve().parent
headline_image = (image.add_local_python_source("app")
                  .add_local_file(HERE / "bench_headline.py", "/root/bench_headline.py")
                  .add_local_file(HERE / "headline_prompts.json", "/root/headline_prompts.json"))
app = modal.App("h3-pro6000-headline", image=headline_image)
SECRETS = [modal.Secret.from_name("hf-fastvideo")]


@app.function(cpu=8, memory=32768, timeout=3600, volumes={"/vol": volume}, secrets=SECRETS)
def fetch(repo: str) -> str:
    from huggingface_hub import snapshot_download
    local = f"/vol/models/{repo.split('/')[-1]}"
    snapshot_download(repo, local_dir=local, token=os.environ["HF_TOKEN"], max_workers=16)
    volume.commit()
    return _sh(f"du -sh {local}/*")


def _headline(repo: str, gpus: int, profile: str, extra_env: dict | None) -> dict:
    _install_kernel()
    model = f"/vol/models/{repo.split('/')[-1]}"
    run_name = f"pro6000x{gpus}-{repo.split('/')[-1]}"
    env = dict(os.environ, **FAST_ENV, **(extra_env or {}), HEADLINE_OUT="/vol/outputs/headline",
               HEADLINE_DEVICE=f"{gpus}x RTX PRO 6000", PYTHONPATH="/src/fastvideo")
    proc = subprocess.run(["python", "/root/bench_headline.py", run_name, model, str(gpus), profile,
                           "--prompts", "/root/headline_prompts.json"], env=env, capture_output=True, text=True,
                          cwd="/root")
    volume.commit()
    tail = (proc.stdout + proc.stderr)[-6000:]
    result_path = pathlib.Path("/vol/outputs/headline") / run_name / "results.json"
    results = json.loads(result_path.read_text()) if result_path.exists() else {}
    return {"run": run_name, "returncode": proc.returncode, "results": results,
            "log_tail": tail if proc.returncode else tail[-1500:]}


@app.function(gpu="RTX-PRO-6000", memory=131072, cpu=8, timeout=2 * 3600, volumes={"/vol": volume})
def headline1(repo: str, profile: str, extra_env: dict | None = None) -> dict:
    return _headline(repo, 1, profile, extra_env)


@app.function(gpu="RTX-PRO-6000:4", memory=196608, cpu=16, timeout=2 * 3600, volumes={"/vol": volume})
def headline4(repo: str, profile: str, extra_env: dict | None = None) -> dict:
    return _headline(repo, 4, profile, extra_env)


@app.function(gpu="RTX-PRO-6000:8", memory=262144, cpu=32, timeout=2 * 3600, volumes={"/vol": volume})
def headline8(repo: str, profile: str, extra_env: dict | None = None) -> dict:
    return _headline(repo, 8, profile, extra_env)


@app.local_entrypoint()
def main(repo: str, profile: str = "h3_dit_ffn", gpus: str = "1,4,8", skip_fetch: bool = False):
    if not skip_fetch:
        print(fetch.remote(repo))
    fns = {"1": headline1, "4": headline4, "8": headline8}
    calls = [fns[g].spawn(repo, profile) for g in gpus.split(",")]
    out = HERE / "headline_results"
    out.mkdir(exist_ok=True)
    for call in calls:
        res = call.get()
        print(json.dumps({k: v for k, v in res.items() if k != "log_tail"}, indent=1)[:3000])
        if res["returncode"]:
            print(res["log_tail"])
        (out / f"{res['run']}.json").write_text(json.dumps(res, indent=1))
