# SPDX-License-Identifier: Apache-2.0
"""Real-weight Kandinsky T2V parity against kandinskylab/kandinsky-5, then SP.

Opt in with KANDINSKY5_RUN_GPU_TESTS=1. Set KANDINSKY5_DIFFUSERS_PATH to
an existing Lite SFT 5s snapshot and KANDINSKY5_OFFICIAL_ROOT to the official
source checkout. Each pipeline runs in a fresh process; no requirements from
the official checkout are installed by this test. Default: 512x512, 25 frames,
two denoising steps, math SDPA, and two GPUs for SP. Compare final latents
(before VAE scaling/decoding), with BF16 reference parity and both BF16/FP32
SP parity. This is not a decoded-video SSIM or performance test.

Example from the repository root::

    KANDINSKY5_RUN_GPU_TESTS=1 KANDINSKY5_DIFFUSERS_PATH=/path/to/snapshot \
    CUDA_VISIBLE_DEVICES=2,3 PYTHONPATH=. python -m pytest \
        tests/local_tests/pipelines/test_kandinsky5_pipeline_parity.py -vs

Set KANDINSKY5_REFERENCE_PYTHON to a reference interpreter if the official
loader needs dependencies incompatible with FastVideo (e.g. Accelerate).
KANDINSKY5_PARITY_OUTPUT optionally selects a persistent log/latent directory.
The reference checkout tested during bring-up was 3a74261b957df0446c1969c5cedf8bbb07969ac2.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
PROMPT = "Ocean waves breaking against a rocky coastline at sunset."
NEGATIVE = ("Static, 2D cartoon, cartoon, 2d animation, paintings, images, worst quality, "
            "low quality, ugly, deformed, walking backwards")


def _backend_settings(_worker=None):
    import torch

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    return {"math": torch.backends.cuda.math_sdp_enabled(),
            "flash": torch.backends.cuda.flash_sdp_enabled(),
            "efficient": torch.backends.cuda.mem_efficient_sdp_enabled()}


def _component_view(destination, *sources):
    """Combine split Diffusers weights/tokenizers without copying weight files."""
    destination.mkdir(parents=True, exist_ok=True)
    for source in sources:
        for item in source.iterdir():
            target = destination / item.name
            if target.exists():
                assert target.resolve() == item.resolve(), f"Stale component view: {target}"
            else:
                target.symlink_to(item.resolve())
    return str(destination)


def _official(config, output):
    sys.path.insert(0, config["official_root"])
    from omegaconf import OmegaConf
    from kandinsky.utils import get_video_pipeline
    import kandinsky.generation_utils as generation

    model = Path(config["model"])
    workspace = output.parent / "official_assets"
    conf = OmegaConf.load(Path(config["official_root"]) / "configs/k5_lite_t2v_5s_sft_sd.yaml")
    conf.model.checkpoint_path = str(model / "transformer/diffusion_pytorch_model.safetensors")
    conf.model.vae.checkpoint_path = str(model)
    conf.model.text_embedder.qwen.checkpoint_path = _component_view(
        workspace / "qwen", model / "text_encoder", model / "tokenizer")
    conf.model.text_embedder.clip.checkpoint_path = _component_view(
        workspace / "clip", model / "text_encoder_2", model / "tokenizer_2")
    config_path = workspace / "config.yaml"
    OmegaConf.save(conf, config_path)
    pipe = get_video_pipeline("cuda:0", conf_path=str(config_path), cache_dir=str(workspace),
                              offload=True, magcache=False, quantized_qwen=False,
                              text_token_padding=False, attention_engine="sdpa", mode="t2v")
    _backend_settings()
    # Observe the released sampling loop without replacing its computations.
    captured = {}
    original_generate = generation.generate

    def record_generate(*args, **kwargs):
        captured["initial"] = args[2].detach().cpu().clone().unsqueeze(0)
        value = original_generate(*args, **kwargs)
        captured["latent"] = value.detach().cpu().clone().unsqueeze(0)
        return value

    generation.generate = record_generate
    try:
        pipe(PROMPT, time_length=config["seconds"], height=config["height"], width=config["width"],
             seed=42, num_steps=config["steps"], guidance_weight=5.0, scheduler_scale=5.0,
             negative_caption=NEGATIVE, expand_prompts=False, save_path=None, progress=False)
        assert "initial" in captured and "latent" in captured
        torch.save(captured, output)
    finally:
        generation.generate = original_generate


def _fastvideo(config, output, sp):
    import cloudpickle
    from fastvideo import VideoGenerator
    from fastvideo.configs.pipelines.kandinsky5 import Kandinsky5T2VConfig

    initial = torch.load(output.parent / "official.pt", weights_only=True)["initial"]
    # Match the official config's 256 Qwen tokens and FP32 CLIP weights.
    pipeline_config = Kandinsky5T2VConfig(dit_precision=config.get("dit_precision", "bf16"),
                                         text_encoder_max_lengths=(129 + 256, 77),
                                         text_encoder_precisions=("bf16", "fp32"))
    generator = VideoGenerator.from_pretrained(
        config["model"], pipeline_config=pipeline_config, num_gpus=sp, sp_size=sp, tp_size=1,
        use_fsdp_inference=False, dit_cpu_offload=False, dit_layerwise_offload=False,
        text_encoder_cpu_offload=True, vae_cpu_offload=True, pin_cpu_memory=False,
        output_type="latent", inference_torch_compile=False, enable_torch_compile=False,
        enable_torch_compile_text_encoder=False, enable_torch_compile_vae=False)
    try:
        backends = generator.executor.collective_rpc(cloudpickle.dumps(_backend_settings))
        assert len(backends) == sp
        assert all(b == {"math": True, "flash": False, "efficient": False} for b in backends)
        result = generator.generate_video(
            PROMPT, negative_prompt=NEGATIVE, height=config["height"], width=config["width"],
            num_frames=config["seconds"] * 24 + 1, num_inference_steps=config["steps"],
            # Native sampling uses linspace(1, 0, steps + 1), excluding the terminal zero.
            sigmas=torch.linspace(1, 0, config["steps"] + 1)[:-1].tolist(),
            guidance_scale=5.0, seed=42, latents=initial.clone(), save_video=False, return_frames=True)
        value = result["samples"].detach().cpu()
        # FastVideo exports channel-first latents; official sampling is channel-last.
        expected = (1, 16, config["seconds"] * 6 + 1, config["height"] // 8, config["width"] // 8)
        assert tuple(value.shape) == expected
        torch.save({"latent": value.permute(0, 2, 3, 4, 1).contiguous()}, output)
    finally:
        generator.shutdown()


def _run(config, directory, mode, name=None):
    name = name or mode
    config_path = directory / f"{name}_config.json"
    config_path.write_text(json.dumps(config, indent=2))
    env = os.environ.copy()
    env.update(PYTHONPATH=str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", ""),
               FASTVIDEO_ATTENTION_BACKEND="TORCH_SDPA", FASTVIDEO_FA4="0", TORCHDYNAMO_DISABLE="1")
    with (directory / f"{name}.log").open("w") as log:
        interpreter = os.environ.get("KANDINSKY5_REFERENCE_PYTHON", sys.executable) if mode == "official" else sys.executable
        process = subprocess.run([interpreter, str(Path(__file__).resolve()), "--worker", mode,
                                  "--config", str(config_path), "--output", str(directory / f"{name}.pt")],
                                 env=env, stdout=log, stderr=subprocess.STDOUT, timeout=1800)
    assert process.returncode == 0, (directory / f"{name}.log").read_text()[-16000:]
    return torch.load(directory / f"{name}.pt", weights_only=True)["latent"]


@pytest.fixture(scope="module")
def parity_runs(tmp_path_factory):
    if os.environ.get("KANDINSKY5_RUN_GPU_TESTS") != "1":
        pytest.skip("Set KANDINSKY5_RUN_GPU_TESTS=1 to run real-weight pipeline parity")
    if torch.cuda.device_count() < 2:
        pytest.skip("Requires two CUDA GPUs")
    default_model = REPO_ROOT / "official_weights/kandinskylab/Kandinsky-5.0-T2V-Lite-sft-5s-Diffusers"
    model = Path(os.environ.get("KANDINSKY5_DIFFUSERS_PATH", default_model)).resolve()
    official = Path(os.environ.get("KANDINSKY5_OFFICIAL_ROOT", REPO_ROOT / "kandinsky-5")).resolve()
    assert (model / "model_index.json").is_file(), f"Missing checkpoint: {model}"
    assert (official / "kandinsky/t2v_pipeline.py").is_file(), f"Missing official checkout: {official}"
    directory = Path(os.environ.get("KANDINSKY5_PARITY_OUTPUT", tmp_path_factory.mktemp("kandinsky5-parity"))).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    config = {"model": str(model), "official_root": str(official),
              "height": int(os.environ.get("KANDINSKY5_PARITY_HEIGHT", "512")),
              "width": int(os.environ.get("KANDINSKY5_PARITY_WIDTH", "512")),
              "seconds": int(os.environ.get("KANDINSKY5_PARITY_SECONDS", "1")),
              "steps": int(os.environ.get("KANDINSKY5_PARITY_STEPS", "2"))}
    expected = _run(config, directory, "official")
    single = _run(config, directory, "single")
    return config, directory, expected, single


def _compare(actual, expected, directory, name, max_relative_rmse, min_cosine=0.999):
    assert actual.shape == expected.shape
    assert torch.isfinite(actual).all() and torch.isfinite(expected).all()
    actual, expected = actual.float(), expected.float()
    difference = actual - expected
    metrics = {"max_abs": difference.abs().max().item(),
               "relative_rmse": (difference.norm() / expected.norm().clamp_min(1e-12)).item(),
               "cosine": torch.nn.functional.cosine_similarity(actual.flatten(), expected.flatten(), dim=0).item()}
    (directory / f"{name}.json").write_text(json.dumps(metrics, indent=2))
    print(name, metrics, flush=True)
    assert metrics["relative_rmse"] <= max_relative_rmse, metrics
    assert metrics["cosine"] >= min_cosine, metrics


# BF16 control runs differed from the FP32 DiT by 3.2-3.5%. Use a 5%
# aggregate bound plus cosine >= 0.999, and a separate tight FP32 SP check.
# Passing does not establish bitwise parity or explain all native-reference drift.
def test_pipeline_matches_official(parity_runs):
    _, directory, expected, single = parity_runs
    _compare(single, expected, directory, "official_vs_single", max_relative_rmse=0.05)


def test_pipeline_sp_matches_single(parity_runs):
    config, directory, _, single = parity_runs
    parallel = _run(config, directory, "sp")
    _compare(parallel, single, directory, "single_vs_sp", max_relative_rmse=0.05)


def test_pipeline_sp_matches_single_fp32(parity_runs):
    config, directory, _, _ = parity_runs
    config = dict(config, dit_precision="fp32")
    single = _run(config, directory, "single", name="single_fp32")
    parallel = _run(config, directory, "sp", name="sp_fp32")
    _compare(parallel, single, directory, "single_vs_sp_fp32",
             max_relative_rmse=0.001, min_cosine=0.99999)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", choices=("official", "single", "sp"), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    configuration = json.loads(args.config.read_text())
    if args.worker == "official":
        _official(configuration, args.output)
    else:
        _fastvideo(configuration, args.output, 1 if args.worker == "single" else 2)
