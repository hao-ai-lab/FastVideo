"""FastH3 V2 NVFP4 on one RTX PRO 6000 (sm_120): build, convert, time generations."""
import json
import os
import pathlib
import subprocess
import time

import modal

_HERE = pathlib.Path(__file__).resolve()
# Inside the container this file is /root/app.py; the repo is already baked into the image there.
WORKTREE = _HERE.parents[3] if len(_HERE.parents) > 3 else pathlib.Path("/src/fastvideo")
CUTLASS_COMMIT = "e67e63c331d6e4b729047c95cf6b92c8454cba89"
volume = modal.Volume.from_name("h3-pro6000-weights", create_if_missing=True)

image = (
    modal.Image.from_registry("nvidia/cuda:13.0.1-devel-ubuntu24.04", add_python="3.12")
    .apt_install("git", "build-essential", "ffmpeg", "libgl1", "libglib2.0-0")
    .pip_install("uv")
    .add_local_dir(WORKTREE, "/src/fastvideo", copy=True,
                   ignore=[".git", "**/__pycache__", "fastvideo-kernel/include/cutlass/**",
                           "fastvideo-kernel/include/tk/**", "fastvideo/third_party/eval/**", "docs/**",
                           "assets/**", "comfyui/**", "apps/**", "**/*.mp4", "**/*.log"])
    .run_commands("cd /src/fastvideo && UV_TORCH_BACKEND=cu130 uv pip install --system -e . --no-sources")
    .run_commands("uv pip install --system 'cmake==3.31.6' ninja 'scikit-build-core>=0.10' pybind11 hf_transfer")
    .run_commands(f"git clone --filter=blob:none https://github.com/NVIDIA/cutlass.git /cutlass && "
                  f"git -C /cutlass checkout {CUTLASS_COMMIT}")
    .env({"FLASHINFER_CUDA_ARCH_LIST": "12.0a", "FLASHINFER_WORKSPACE_BASE": "/vol/cache/flashinfer",
          "TORCHINDUCTOR_CACHE_DIR": "/vol/cache/inductor", "TRITON_CACHE_DIR": "/vol/cache/triton",
          "FASTVIDEO_VSA_SM100A": "0", "FASTVIDEO_FA4": "0", "FASTVIDEO_STAGE_LOGGING": "1",
          "HF_HUB_ENABLE_HF_TRANSFER": "1"})
)
app = modal.App("h3-pro6000-fastest", image=image)


def _sh(cmd: str, **kw) -> str:
    proc = subprocess.run(cmd, shell=True, capture_output=True, text=True, **kw)
    out = (proc.stdout + proc.stderr)[-8000:]
    if proc.returncode != 0:
        raise RuntimeError(f"command failed ({proc.returncode}): {cmd}\n{out}")
    return out


@app.function(cpu=16, memory=32768, timeout=3600, volumes={"/vol": volume})
def build_kernel() -> str:
    _sh("rm -rf /src/fastvideo/fastvideo-kernel/include/cutlass && "
        "ln -s /cutlass /src/fastvideo/fastvideo-kernel/include/cutlass && mkdir -p /vol/wheels/cu130")
    _sh("rm -f /vol/wheels/cu130/*.whl")
    env = dict(os.environ, TORCH_CUDA_ARCH_LIST="12.0a", MAX_JOBS="16", CC="gcc", CXX="g++", CUDAHOSTCXX="g++",
               CMAKE_ARGS="-DFASTVIDEO_KERNEL_BUILD_ATTN_QAT_INFER=ON -DFASTVIDEO_KERNEL_BUILD_TK=OFF "
               "-DGPU_BACKEND=CUDA -DCMAKE_CUDA_ARCHITECTURES=120a")
    _sh("cd /src/fastvideo/fastvideo-kernel && pip wheel . --no-build-isolation --no-deps -w /vol/wheels/cu130",
        env=env)
    volume.commit()
    return _sh("ls -la /vol/wheels/cu130")


def _install_kernel():
    _sh("pip install --no-deps --force-reinstall /vol/wheels/cu130/*.whl")


def _build_light_int8_vae() -> str:
    """26-block LynnReal light decoder: dense fp16 decoder + official encoder, Kijai int8-convrot overlay."""
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open
    from safetensors.torch import save_file

    target = pathlib.Path("/vol/fv/vae_light_int8")
    if (target / "config.json").exists():
        return "exists"
    target.mkdir(parents=True, exist_ok=True)
    overlay = hf_hub_download("Kijai/MiniMax-H3-experimental", "minimax_h3_lynnreal_light_vae_int8_convrot.safetensors",
                              local_dir="/vol/kijai")
    official = pathlib.Path("/vol/official/vae")
    weight_map = json.loads((official / "diffusion_pytorch_model.safetensors.index.json").read_text())["weight_map"]
    tensors = {}
    for shard in sorted(set(weight_map.values())):
        with safe_open(str(official / shard), framework="pt") as reader:
            for key in reader.keys():
                if not key.startswith("decoder.transformer_blocks."):
                    tensors[key] = reader.get_tensor(key)
    with safe_open("/vol/light-vae/lynnreal_light_vae_decoder_fp16.safetensors", framework="pt") as reader:
        light_keys = list(reader.keys())
        for key in light_keys:
            tensors[key] = reader.get_tensor(key)
    blocks = {int(k.split(".")[2]) for k in tensors if k.startswith("decoder.transformer_blocks.")}
    assert blocks == set(range(26)), sorted(blocks)
    save_file(tensors, str(target / "diffusion_pytorch_model.safetensors"))
    config = json.loads((official / "config.json").read_text())
    config["decoder_num_layers"] = 26
    (target / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (target / "minimax_h3_video_vae_int8_convrot.safetensors").symlink_to(overlay)
    return f"light decoder keys={len(light_keys)} total={len(tensors)} blocks={len(blocks)}"


@app.function(gpu="RTX-PRO-6000", memory=131072, cpu=8, timeout=10800, volumes={"/vol": volume})
def convert(minimal: bool = True) -> dict:
    _install_kernel()
    report = {}
    os.makedirs("/vol/fv", exist_ok=True)
    if not os.path.exists("/vol/fv/text_encoder_nvfp4/config.json"):
        report["text_encoder"] = _sh(
            "cd /src/fastvideo && python scripts/checkpoint_conversion/convert_minimax_h3_text_encoder_nvfp4.py "
            "--src /vol/v2-nvfp4/text_encoder --dst /vol/fv/text_encoder_nvfp4")[-1500:]
        volume.commit()
    for name, src, flag in (("transformer_full", "v2-nvfp4", "--quantize-attention"),
                            ("transformer_ffn", "v2-nvfp4", ""),
                            ("transformer_vsa", "v2-nvfp4", "--quantize-attention --quantize-gate"),
                            ("v4_transformer_vsa", "v4-nvfp4", "--quantize-attention --quantize-gate")):
        if not os.path.exists(f"/vol/{src}/transformer"):
            continue
        if minimal and name not in ("transformer_vsa", "v4_transformer_vsa"):
            continue
        if not os.path.exists(f"/vol/fv/{name}/config.json"):
            report[name] = _sh(
                "cd /src/fastvideo && python scripts/checkpoint_conversion/convert_minimax_h3_modelopt_nvfp4_dit.py "
                f"--src /vol/{src}/transformer --dst /vol/fv/{name} {flag}")[-1500:]
            volume.commit()
    # VAE folders: official dense shards, plus Comfy's int8-convrot overlay variant.
    for vae_name in (() if minimal else ("vae_dense", "vae_int8")):
        target = pathlib.Path(f"/vol/fv/{vae_name}")
        target.mkdir(parents=True, exist_ok=True)
        for item in pathlib.Path("/vol/official/vae").iterdir():
            link = target / item.name
            if not link.exists():
                link.symlink_to(item)
    int8 = pathlib.Path("/vol/fv/vae_int8/minimax_h3_video_vae_int8_convrot.safetensors")
    if not minimal and not int8.exists():
        int8.symlink_to("/vol/comfy/vae/minimax_h3_video_vae_int8_convrot.safetensors")
    report["light_vae"] = _build_light_int8_vae()
    # Model folders: V2 small components + chosen transformer / text encoder / VAE.
    # 4-step VSA-0.9 model: its own manifest/schedulers, shared text encoder, VAEs and audio VAE.
    root = pathlib.Path("/vol/fv/v4_vsa_light")
    if pathlib.Path("/vol/v4-nvfp4/fastvideo_inference.json").exists():
        root.mkdir(parents=True, exist_ok=True)
        for item in pathlib.Path("/vol/v4-nvfp4").iterdir():
            if item.name in ("transformer", "text_encoder", "vae", "audio_vae") or item.name.startswith("."):
                continue
            if not (root / item.name).exists():
                (root / item.name).symlink_to(item)
        for comp, src in (("transformer", "/vol/fv/v4_transformer_vsa"), ("text_encoder", "/vol/fv/text_encoder_nvfp4"),
                          ("vae", "/vol/fv/vae_light_int8"), ("audio_vae", "/vol/v2-nvfp4/audio_vae")):
            if not (root / comp).exists():
                (root / comp).symlink_to(src)
    model_sets = (("v2_vsa_light", "transformer_vsa", "vae_light_int8"),
                  ("v2_full_light", "transformer_full", "vae_light_int8"),
                                    ("v2_full_int8", "transformer_full", "vae_int8"),
                                    ("v2_full_dense", "transformer_full", "vae_dense"),
                                    ("v2_ffn_int8", "transformer_ffn", "vae_int8"))
    for model, transformer, vae in (model_sets[:1] if minimal else model_sets):
        root = pathlib.Path(f"/vol/fv/{model}")
        root.mkdir(parents=True, exist_ok=True)
        for item in pathlib.Path("/vol/v2-nvfp4").iterdir():
            if item.name in ("transformer", "text_encoder", "vae") or item.name.startswith("."):
                continue
            link = root / item.name
            if not link.exists():
                link.symlink_to(item)
        for comp, src in (("transformer", transformer), ("text_encoder", "text_encoder_nvfp4"), ("vae", vae)):
            link = root / comp
            if not link.exists():
                link.symlink_to(f"/vol/fv/{src}")
    volume.commit()
    report["models"] = _sh("ls -la /vol/fv /vol/fv/v2_vsa_light")
    # Are V2's VSA compression gates trained (nonzero)?
    import torch
    from safetensors import safe_open
    index = json.loads(pathlib.Path("/vol/v2-nvfp4/transformer/diffusion_pytorch_model.safetensors.index.json").read_text())
    gate_stats = {}
    for blk in (0, 25, 49):
        key = f"transformer_blocks.{blk}.attn.to_gate_compress.weight"
        with safe_open(f"/vol/v2-nvfp4/transformer/{index['weight_map'][key]}", framework="pt") as reader:
            w = reader.get_tensor(key).float()
        gate_stats[blk] = {"abs_mean": w.abs().mean().item(), "nonzero_frac": (w != 0).float().mean().item()}
    report["gate_stats"] = gate_stats
    return report


PROMPTS = {
    "kitesurf": ("A kite surfer carves hard across choppy bay water while the camera dives alongside; spray hisses off "
                 "the board edge, the sail flaps and snaps in the wind, and gulls cry overhead."),
    "chef": ("(S1) In a bright home kitchen, a chef looks straight at the camera and says <d>[English] Fold the eggs "
             "gently and taste before you salt.</d> A pot simmers behind her with soft bubbling and no music."),
}


@app.function(gpu="RTX-PRO-6000", memory=131072, cpu=8, timeout=5400, volumes={"/vol": volume})
def run_variant(name: str, model: str, profile: str, attention: str, decode: str, vae_compile: bool,
                height: int = 480, width: int = 832, warmups: int = 1, num_frames: int = 124,
                env: dict | None = None, prompts: tuple = ("kitesurf", "chef"), sparsity: float = 0.8,
                steps: int = 9, num_gpus: int = 1, parallel_decode: bool = False, pre_runs: tuple = (),
                prompt_texts: dict | None = None, offload: dict | None = None,
                experimental_extra: dict | None = None) -> dict:
    os.environ.update(env or {})
    texts = {**PROMPTS, **(prompt_texts or {})}
    _install_kernel()
    import torch
    from fastvideo import VideoGenerator

    if attention == "VIDEO_SPARSE_ATTN_H3":
        os.environ["FASTVIDEO_VSA_TRITON"] = "1"
    experimental = {"attention_backend": attention, "h3_sequential_load": False, "inference_torch_compile": False,
                    "vae_parallel_decode": parallel_decode, "video_decode_backend": decode}
    if attention == "VIDEO_SPARSE_ATTN_H3":
        experimental.update({"VSA_sparsity": sparsity, "VSA_tile_size": 64})
    experimental.update(experimental_extra or {})
    config = {
        "model_path": f"/vol/fv/{model}",
        "engine": {"num_gpus": num_gpus, "use_fsdp_inference": False,
                   "quantization": {"transformer_quant": "NVFP4", "layer_profile": profile},
                   "parallelism": {"tp_size": 1, "sp_size": num_gpus},
                   "offload": {"dit": False, "dit_layerwise": False, "text_encoder": False, "vae": False,
                               "pin_cpu_memory": num_gpus == 1, "lazy_module_load": False, **(offload or {})},
                   "compile": {"enabled": False, "vae_enabled": vae_compile}},
        "pipeline": {"experimental": experimental},
    }
    t0 = time.perf_counter()
    generator = VideoGenerator.from_config(config)
    load_s = time.perf_counter() - t0
    out_dir = pathlib.Path(f"/vol/outputs/{name}")
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {"name": name, "load_s": round(load_s, 1), "env": env or {}, "shape": [height, width, num_frames],
               "num_gpus": num_gpus,
               "runs": []}
    try:
        # Optional runs at other shapes first (e.g. a 480p correctness clip), same loaded model.
        for j, (ph, pw, pf, pid) in enumerate(pre_runs):
            request = {"prompt": texts[pid], "negative_prompt": "",
                       "sampling": {"seed": 20260929, "height": ph, "width": pw, "num_frames": pf, "fps": 24,
                                    "num_inference_steps": steps, "guidance_scale": 1.0, "batch_cfg": False},
                       "output": {"output_path": str(out_dir / f"pre{j:02d}_{pid}_{ph}p.mp4"), "save_video": True,
                                  "return_frames": False}}
            t = time.perf_counter()
            result = generator.generate(request)
            results.setdefault("pre_runs", []).append({"shape": [ph, pw, pf], "prompt": pid,
                                                       "wall_s": round(time.perf_counter() - t, 2),
                                                       "video": getattr(result, "video_path", None)})
            print("PRE_RUN", json.dumps(results["pre_runs"][-1]), flush=True)
        if pre_runs:
            (out_dir / "results.json").write_text(json.dumps(results, indent=1))
            volume.commit()
        # Warm every distinct prompt: prompt length changes the packed sequence, and shape-specialized
        # compiled kernels would otherwise recompile inside the first timed run of each prompt.
        distinct = list(dict.fromkeys(prompts))
        order = [distinct[i % len(distinct)] for i in range(warmups)] + list(prompts)
        for i, pid in enumerate(order):
            prompt = texts[pid]
            request = {"prompt": prompt, "negative_prompt": "",
                       "sampling": {"seed": 20260929, "height": height, "width": width, "num_frames": num_frames, "fps": 24,
                                    "num_inference_steps": steps, "guidance_scale": 1.0, "batch_cfg": False},
                       "output": {"output_path": str(out_dir / f"{i:02d}_{pid}.mp4"), "save_video": True,
                                  "return_frames": False}}
            torch.cuda.synchronize()
            t = time.perf_counter()
            result = generator.generate(request)
            torch.cuda.synchronize()
            wall = time.perf_counter() - t
            results["runs"].append({"prompt": pid, "warmup": i < warmups, "wall_s": round(wall, 2),
                                    "generation_time_s": getattr(result, "generation_time", None),
                                    "video": getattr(result, "video_path", None)})
        results["peak_mem_gb_device"] = _sh("nvidia-smi --query-gpu=memory.used --format=csv,noheader").strip()
    finally:
        generator.shutdown()
    (out_dir / "results.json").write_text(json.dumps(results, indent=1))
    volume.commit()
    return results


bench_image = image.add_local_file(pathlib.Path(__file__).parent / "bench_code.py", "/root/bench_code.py")

SHAPES = [
    # name, prefix segments (text, audio rows), video latent tokens (t, h, w) after 1x2x2 patching
    ("480p_124f", [256, 414], [37, 15, 26]),
    ("768p_243f", [256, 810], [72, 24, 42]),
]


@app.function(gpu="RTX-PRO-6000", memory=65536, cpu=8, timeout=3600, volumes={"/vol": volume}, image=bench_image)
def bench_block() -> dict:
    _install_kernel()
    import sys
    sys.path.insert(0, "/root")
    import bench_code
    report = {"check": bench_code.check_tile64()}
    report.update(bench_code.run(SHAPES))
    return json.loads(json.dumps(report, default=str))


@app.function(gpu="RTX-PRO-6000", memory=65536, cpu=8, timeout=1800, volumes={"/vol": volume}, image=bench_image)
def density_fn() -> dict:
    _install_kernel()
    import sys
    sys.path.insert(0, "/root")
    import bench_code
    return json.loads(json.dumps(bench_code.density_study(), default=str))


@app.function(gpu="RTX-PRO-6000", memory=65536, cpu=8, timeout=1800, volumes={"/vol": volume}, image=bench_image)
def kcheck_fn() -> dict:
    _install_kernel()
    import sys
    sys.path.insert(0, "/root")
    import bench_code
    out = {"check": bench_code.check_tile64()}
    out["density"] = bench_code.density_study()
    return json.loads(json.dumps(out, default=str))


@app.function(gpu="RTX-PRO-6000:2", memory=229376, cpu=16, timeout=5400, volumes={"/vol": volume})
def run_variant2(*args, **kwargs) -> dict:
    return run_variant.local(*args, **kwargs)


@app.function(gpu="RTX-PRO-6000:8", memory=196608, cpu=16, timeout=5400, volumes={"/vol": volume})
def run_variant8(*args, **kwargs) -> dict:
    return run_variant.local(*args, **kwargs)


@app.function(gpu="RTX-PRO-6000:4", memory=131072, cpu=8, timeout=5400, volumes={"/vol": volume})
def run_variant4(*args, **kwargs) -> dict:
    return run_variant.local(*args, **kwargs)


@app.function(gpu="RTX-PRO-6000:8", memory=196608, cpu=16, timeout=2 * 3600, volumes={"/vol": volume})
def bench8_pair(prompt_texts: dict, timed: tuple, warmups: int, models: tuple = ("v2", "v2_profile")) -> list:
    """8-GPU 10 s 768p V2 8-step benchmark, then a stage-profiled pass, in one container (shared compile caches)."""
    specs = {"v2": ("sp8_v2_8step_768p10s_1k", "v2_vsa_light", 0.8, 9, {}, timed, warmups),
             # One warm + one profiled generation; CUDA-event spans log per DiT forward as H3_STAGE_MS.
             "v2_profile": ("sp8_v2_8step_profile", "v2_vsa_light", 0.8, 9, {"FASTVIDEO_H3_SP_PROFILE": "1"},
                            timed[:1], 1)}
    out = []
    for m in models:
        name, model, sparsity, steps, extra_env, prompts, warm = specs[m]
        r = run_variant.local(name, model, "h3_dit_vsa", attention="VIDEO_SPARSE_ATTN_H3", decode="h3-vae",
                              vae_compile=True, height=768, width=1344, num_frames=243, warmups=warm,
                              env={**FAST_ENV, **extra_env}, prompts=prompts, sparsity=sparsity, steps=steps,
                              num_gpus=8, parallel_decode=True, prompt_texts=prompt_texts)
        print("RESULT", json.dumps(r), flush=True)
        out.append(r)
    return out


@app.function(gpu="RTX-PRO-6000", memory=196608, cpu=16, timeout=3 * 3600, volumes={"/vol": volume})
def memladder(prompt_texts: dict, pid: str, configs: dict | None = None) -> list:
    """V2 8-step 10 s 768p on one GPU under memory placements; stage logs carry per-stage peaks.

    ``configs`` maps a name to (offload overrides, experimental overrides, extra env). The env can set
    FASTVIDEO_CUDA_MEMORY_CAP_GIB to emulate a smaller card.
    """
    seq = {"h3_sequential_load": True}
    lw = {"text_encoder": True, "vae": True, "dit_layerwise": True}
    configs = configs or {
        "A_resident": ({}, {}, {}),
        "B_seq_encoder_vae_offload": ({"text_encoder": True, "vae": True}, seq, {}),
        "C_plus_dit_layerwise": (lw, seq, {}),
    }
    out = []
    for name, (offload, extra, env_extra) in configs.items():
        try:
            r = run_variant.local(f"mem_{name}", "v2_vsa_light", "h3_dit_vsa", attention="VIDEO_SPARSE_ATTN_H3",
                                  decode="h3-vae", vae_compile=False, height=768, width=1344, num_frames=243,
                                  warmups=1, env={**FAST_ENV, **env_extra}, prompts=(pid,), sparsity=0.8, steps=9,
                                  num_gpus=1, prompt_texts=prompt_texts, offload=offload, experimental_extra=extra)
        except Exception as e:  # keep the ladder going; record the failure
            r = {"name": f"mem_{name}", "error": repr(e)[:3000]}
        print("RESULT", json.dumps(r), flush=True)
        out.append(r)
        for k in env_extra:
            os.environ.pop(k, None)
    return out


@app.function(cpu=8, memory=32768, timeout=1800, volumes={"/vol": volume})
def compare_videos(a: str, b: str) -> dict:
    """Frame PSNR between two MP4s on the volume (same seed and prompt)."""
    import imageio.v3 as iio
    import numpy as np
    fa = iio.imread(a, plugin="pyav").astype(np.float32)
    fb = iio.imread(b, plugin="pyav").astype(np.float32)
    n = min(len(fa), len(fb))
    mse = ((fa[:n] - fb[:n]) ** 2).reshape(n, -1).mean(axis=1)
    psnr = 10 * np.log10(255.0**2 / np.maximum(mse, 1e-9))
    return {"frames": [len(fa), len(fb)], "psnr_mean": float(psnr.mean()), "psnr_min": float(psnr.min())}


FAST_ENV = {"FASTVIDEO_H3_VSA_FP4": "1", "FASTVIDEO_MINIMAX_H3_FUSIONS": "all", "FASTVIDEO_NVFP4_MM_BACKEND": "cutlass",
            "FASTVIDEO_H3_VAE_TILE_BATCH": "28"}


@app.local_entrypoint()
def main(step: str = "all", ladder: str = "base"):
    if step == "prep_personal":
        print("KERNEL", build_kernel.remote())
        print("CONVERT", json.dumps(convert.remote(minimal=True), indent=1)[:6000])
        return
    if step == "sp2check":
        common = dict(attention="VIDEO_SPARSE_ATTN_H3", decode="h3-vae", vae_compile=False, height=480, width=832,
                      num_frames=124, warmups=0, env=FAST_ENV, prompts=("kitesurf",), sparsity=0.8, steps=9)
        one = run_variant.spawn("sp1_480p", "v2_vsa_light", "h3_dit_vsa", **common)
        two = run_variant2.spawn("sp2_480p", "v2_vsa_light", "h3_dit_vsa", num_gpus=2, **common)
        r1, r2 = one.get(), two.get()
        print("RESULT", json.dumps(r1))
        print("RESULT", json.dumps(r2))
        print("COMPARE", json.dumps(compare_videos.remote(r1["runs"][0]["video"], r2["runs"][0]["video"])))
        return
    if step == "sp8":
        r = run_variant8.remote(
            "sp8_v2_8step", "v2_vsa_light", "h3_dit_vsa", attention="VIDEO_SPARSE_ATTN_H3", decode="h3-vae",
            vae_compile=True, height=768, width=1344, num_frames=243, warmups=1, env=FAST_ENV,
            prompts=("kitesurf", "chef", "kitesurf", "chef"), sparsity=0.8, steps=9, num_gpus=8, parallel_decode=True,
            pre_runs=((480, 832, 124, "kitesurf"), ))
        print("RESULT", json.dumps(r))
        if r.get("pre_runs"):
            print("COMPARE", json.dumps(compare_videos.remote("/vol/outputs/sp1_480p/00_kitesurf.mp4",
                                                              r["pre_runs"][0]["video"])))
        return
    if step == "simfp8":
        # SP=1 with and without the SP exchange's FP8 rounding, same settings as sp1_480p.
        common = dict(attention="VIDEO_SPARSE_ATTN_H3", decode="h3-vae", vae_compile=False, height=480, width=832,
                      num_frames=124, warmups=0, prompts=("kitesurf",), sparsity=0.8, steps=9)
        plain = run_variant.spawn("sp1_480p_rerun", "v2_vsa_light", "h3_dit_vsa", env=FAST_ENV, **common)
        sim = run_variant.spawn("sp1_480p_simfp8", "v2_vsa_light", "h3_dit_vsa",
                                env={**FAST_ENV, "FASTVIDEO_H3_SIM_SP_FP8": "1"}, **common)
        r_plain, r_sim = plain.get(), sim.get()
        print("RESULT", json.dumps(r_plain))
        print("RESULT", json.dumps(r_sim))
        return
    if step == "memreport":
        r = run_variant.remote("memreport", "v2_vsa_light", "h3_dit_vsa", attention="VIDEO_SPARSE_ATTN_H3",
                               decode="h3-vae", vae_compile=False, height=480, width=832, num_frames=124, warmups=0,
                               env={**FAST_ENV, "FASTVIDEO_MEMORY_REPORT": "1"}, prompts=("kitesurf",), sparsity=0.8,
                               steps=9, offload={"text_encoder": True, "vae": True},
                               experimental_extra={"h3_sequential_load": False})
        print("RESULT", json.dumps(r))
        return
    if step == "memladder":
        rows = [json.loads(line) for line in open(WORKTREE.parent / "UniServe-sm120fp4" / "uniserve_eval" / "workloads"
                                                  / "fast_h3" / "latency.jsonl")]
        pid = "latency-ceramics-005"
        text = {r["id"]: r["prompt"] for r in rows if r["id"] == pid}
        configs = None
        if ladder == "caps":
            seq = {"h3_sequential_load": True}
            lw = {"text_encoder": True, "vae": True, "dit_layerwise": True}
            configs = {
                "D32_C_cap32": (lw, seq, {"FASTVIDEO_CUDA_MEMORY_CAP_GIB": "32"}),
                "D24_buffers_cap24": (lw, seq, {"FASTVIDEO_CUDA_MEMORY_CAP_GIB": "24",
                                                "FASTVIDEO_LAYERWISE_OFFLOAD_BUFFERS": "1"}),
                "D16_buffers_tile8_cap16": (lw, seq, {"FASTVIDEO_CUDA_MEMORY_CAP_GIB": "16",
                                                      "FASTVIDEO_LAYERWISE_OFFLOAD_BUFFERS": "1",
                                                      "FASTVIDEO_H3_VAE_TILE_BATCH": "8"}),
            }
        for r in memladder.remote(text, pid, configs):
            print("RESULT", json.dumps(r))
        return
    if step == "bench8_pair":
        # UniServe's 10 s / ~1K-token latency prompts, so the numbers line up with its published protocol.
        rows = [json.loads(line) for line in open(WORKTREE.parent / "UniServe-sm120fp4" / "uniserve_eval" / "workloads"
                                                  / "fast_h3" / "latency.jsonl")]
        ten = {r["id"]: r["prompt"] for r in rows if r["seconds"] == 10 and r["prompt_len"] == 1000}
        ids = ("latency-ceramics-005", "latency-harbor-005")
        texts = {i: ten[i] for i in ids}
        for r in bench8_pair.remote(texts, (ids[0], ids[1], ids[0], ids[1]), 2):
            timed = sorted(x["wall_s"] for x in r["runs"] if not x["warmup"])
            print("SUMMARY", r["name"], "timed", timed, "median", timed[len(timed) // 2] if len(timed) % 2
                  else (timed[len(timed) // 2 - 1] + timed[len(timed) // 2]) / 2)
        return
    if step == "sp4":
        r = run_variant4.remote(
            "sp4_v2_8step", "v2_vsa_light", "h3_dit_vsa", attention="VIDEO_SPARSE_ATTN_H3", decode="h3-vae",
            vae_compile=True, height=768, width=1344, num_frames=243, warmups=1, env=FAST_ENV,
            prompts=("kitesurf", "chef", "kitesurf", "chef"), sparsity=0.8, steps=9, num_gpus=4, parallel_decode=True,
            pre_runs=((480, 832, 124, "kitesurf"), ))
        print("RESULT", json.dumps(r))
        if r.get("pre_runs"):
            print("COMPARE", json.dumps(compare_videos.remote("/vol/outputs/sp1_480p/00_kitesurf.mp4",
                                                              r["pre_runs"][0]["video"])))
        return
    if step in ("bench8_v2", "bench8_v4"):
        v2 = step == "bench8_v2"
        r = run_variant8.remote(
            f"sp8_{'v2_8step' if v2 else 'v4_4step'}_768p10s", "v2_vsa_light" if v2 else "v4_vsa_light", "h3_dit_vsa",
            attention="VIDEO_SPARSE_ATTN_H3", decode="h3-vae", vae_compile=True, height=768, width=1344,
            num_frames=243, warmups=1, env=FAST_ENV, prompts=("kitesurf", "chef", "kitesurf", "chef"),
            sparsity=0.8 if v2 else 0.9, steps=9 if v2 else 5, num_gpus=8, parallel_decode=True)
        print("RESULT", json.dumps(r))
        return
    if step == "build_bench":
        print("KERNEL", build_kernel.remote())
        step = "bench"
    if step in ("prep768", "e2e768"):
        if step == "prep768":
            print("KERNEL", build_kernel.remote())
            print("CONVERT", json.dumps(convert.remote(), indent=1)[:6000])
        fast = {"FASTVIDEO_H3_VSA_FP4": "1", "FASTVIDEO_MINIMAX_H3_FUSIONS": "all",
                "FASTVIDEO_NVFP4_MM_BACKEND": "cutlass", "FASTVIDEO_H3_VAE_TILE_BATCH": "28"}
        common = dict(attention="VIDEO_SPARSE_ATTN_H3", decode="h3-vae", vae_compile=True, height=768, width=1344,
                      num_frames=243, warmups=1, env=fast, prompts=("kitesurf", "chef"))
        calls = {
            "v4_4step_vsa90_768p10s": run_variant.spawn("v4_4step_vsa90_768p10s", "v4_vsa_light", "h3_dit_vsa",
                                                        sparsity=0.9, steps=5, **common),
            "v2_8step_vsa80_768p10s": run_variant.spawn("v2_8step_vsa80_768p10s", "v2_vsa_light", "h3_dit_vsa",
                                                        sparsity=0.8, steps=9, **common),
        }
        for name, call in calls.items():
            try:
                print("RESULT", json.dumps(call.get()))
            except Exception as exc:  # noqa: BLE001
                print("FAILED", name, repr(exc)[:3000])
        return
    if step == "kcheck":
        print("KERNEL", build_kernel.remote())
        print("KCHECK", json.dumps(kcheck_fn.remote(), indent=1))
        return
    if step == "density":
        print("DENSITYRESULT", json.dumps(density_fn.remote(), indent=1))
        return
    if step == "bench":
        print("BENCHRESULT", json.dumps(bench_block.remote(), indent=1))
        return
    if step in ("all", "build"):
        print("KERNEL", build_kernel.remote())
    if step in ("all", "convert"):
        print("CONVERT", json.dumps(convert.remote(), indent=1)[:6000])
    if step in ("all", "run"):
        variants = [
            ("full_qatinfer_light", "v2_full_light", "h3_dit", "ATTN_QAT_INFER", "h3-vae", True),
            ("full_vsa_light", "v2_full_light", "h3_dit", "VIDEO_SPARSE_ATTN_H3", "h3-vae", True),
            ("full_qatinfer_int8", "v2_full_int8", "h3_dit", "ATTN_QAT_INFER", "h3-vae", True),
            ("full_qatinfer_taeh3", "v2_full_light", "h3_dit", "ATTN_QAT_INFER", "taeh3", False),
            ("ffn_vsa_int8", "v2_ffn_int8", "h3_dit_ffn", "VIDEO_SPARSE_ATTN_H3", "h3-vae", True),
        ]
        # One RTX PRO 6000 per variant, all at once; failures are returned, not raised.
        for v, result in zip(variants, run_variant.starmap(variants, return_exceptions=True)):
            if isinstance(result, Exception):
                print("FAILED", v[0], repr(result)[:3000])
            else:
                print("RESULT", json.dumps(result))
