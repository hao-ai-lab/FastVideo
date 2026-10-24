# SPDX-License-Identifier: Apache-2.0
"""Run native/reference in separate processes, then compare saved boundaries.

Example (same arguments for both implementations):
  python tests/local_tests/bernini/run_pipeline_parity.py --implementation native ...
  python tests/local_tests/bernini/run_pipeline_parity.py --implementation official ...
  python tests/local_tests/bernini/run_pipeline_parity.py --compare OUTPUT
"""
import argparse
import gc
import importlib
import json
import time
from pathlib import Path
from types import SimpleNamespace

import imageio.v3 as iio
import torch

from reference import ROOT, REFERENCE, WEIGHTS, official_modules


def save_result(batch, output, implementation, started):
    output.mkdir(parents=True, exist_ok=True)
    artifacts = {key: value.detach().float().cpu() for key, value in batch.items()}
    torch.save(artifacts, output / f"{implementation}.pt")
    frames = artifacts["frames"][0].permute(1, 2, 3, 0)
    iio.imwrite(output / f"{implementation}.mp4", (frames.clamp(0, 1) * 255).round().byte().numpy(), fps=16)
    selected = torch.cat([frames[i] for i in (0, len(frames) // 2, len(frames) - 1)], dim=1)
    iio.imwrite(output / f"{implementation}_frames.png", (selected.clamp(0, 1) * 255).round().byte().numpy())
    print(json.dumps({
        "implementation": implementation,
        "seconds": time.perf_counter() - started,
        "peak_allocated_GiB": torch.cuda.max_memory_allocated() / 2**30,
        "frames": list(frames.shape),
    }, indent=2))


def run_native(args):
    from fastvideo.pipelines.basic.bernini.bernini_pipeline import BerniniRPipeline
    from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
    pipe = BerniniRPipeline.from_pretrained(
        str(ROOT / "converted_weights/bernini-r-1.3b"), num_gpus=1, sp_size=1, tp_size=1,
        attention_backend="TORCH_SDPA", use_fsdp_inference=False, dit_layerwise_offload=False,
        dit_cpu_offload=True, text_encoder_cpu_offload=True, vae_cpu_offload=True,
        enable_stage_verification=True,
    )
    batch = ForwardBatch(data_type="video", prompt=args.prompt, negative_prompt="", video_path=str(args.video),
                         height=args.height, width=args.width, num_frames=args.frames, fps=16, seed=42,
                         num_inference_steps=args.steps, guidance_scale=4.0)
    result = pipe.forward(batch, pipe.fastvideo_args)
    return {
        "frames": result.output,
        "latents": result.latents,
        "source_latents": result.video_latent,
        "prompt": result.prompt_embeds[0],
        "negative_prompt": result.negative_prompt_embeds[0],
    }


@torch.no_grad()
def run_official(args):
    import diffusers
    from diffusers import AutoencoderKLWan
    from transformers import AutoTokenizer, UMT5EncoderModel
    assert diffusers.__version__ == "0.35.2", "Run the oracle with PYTHONPATH=reference_repos/bernini/.oracle-deps"
    official_modules()
    data = importlib.import_module("_bernini_oracle.data_utils")
    renderer = importlib.import_module("_bernini_oracle.models.renderer")
    decoder = importlib.import_module("_bernini_oracle.models.wan_diffusion")
    config = SimpleNamespace(**json.loads((WEIGHTS / "config.json").read_text()))
    config.wan22_base = str(WEIGHTS)
    vae = AutoencoderKLWan.from_pretrained(str(WEIGHTS / "vae"), torch_dtype=torch.float32).cuda().eval()
    pixels = data.preprocess_video(str(args.video), fps=16, max_image_size=max(args.height, args.width),
                                    max_image_num=args.frames, device="cuda")
    assert pixels.shape[-2:] == (args.height, args.width), pixels.shape
    latent = vae.encode(pixels).latent_dist.mode()
    mean = torch.tensor(vae.config.latents_mean, device="cuda").view(1, -1, 1, 1, 1)
    std = torch.tensor(vae.config.latents_std, device="cuda").view(1, -1, 1, 1, 1)
    source_latents = (latent - mean) / std
    vae.cpu()
    tokenizer = AutoTokenizer.from_pretrained(str(WEIGHTS / "tokenizer"))
    text_encoder = UMT5EncoderModel.from_pretrained(str(WEIGHTS / "text_encoder"),
                                                  torch_dtype=torch.bfloat16).cuda().eval()
    encoder = SimpleNamespace(t5_text_encoder=text_encoder, max_sequence_length=512)
    embeds = []
    for text in [args.prompt, ""]:
        encoded = tokenizer(text, padding="max_length", max_length=512, truncation=True,
                             add_special_tokens=True, return_tensors="pt").to("cuda")
        embeds.append(renderer.BerniniRendererModel.encode_prompt(
            encoder, encoded.input_ids, encoded.attention_mask))
    text_encoder.cpu()
    del text_encoder
    gc.collect()
    torch.cuda.empty_cache()
    model = decoder.GEN_Wanx22(config).eval()
    latents = model.sample(prompt_embeds=embeds[0], uncond_prompt_embeds=embeds[1],
                           multi_video_vae_latents=[source_latents], num_frames=pixels.shape[2],
                           height=args.height, width=args.width, num_inference_steps=args.steps,
                           guidance_mode="v2v_apg", omega_txt=4.0, eta=0.5, momentum=0.0,
                           norm_threshold=(50.0,50.0), seed=42, device="cuda")
    model.cpu()
    vae.cuda()
    frames = (vae.decode(latents * std + mean, return_dict=False)[0] / 2 + 0.5).clamp(0, 1)
    return {
        "frames": frames,
        "latents": latents,
        "source_latents": source_latents,
        "prompt": embeds[0],
        "negative_prompt": embeds[1],
    }


def compare(directory):
    from pytorch_msssim import ms_ssim
    native = torch.load(directory / "native.pt", map_location="cpu", weights_only=True)
    official = torch.load(directory / "official.pt", map_location="cpu", weights_only=True)
    report = {}
    for key in native:
        actual, expected = native[key], official[key]
        assert actual.shape == expected.shape, (key, actual.shape, expected.shape)
        report[key] = {
            "rel_l2": ((actual - expected).norm() / expected.norm().clamp_min(1e-8)).item(),
            "max_abs": (actual - expected).abs().max().item(),
        }
    score = ms_ssim(native["frames"][0].transpose(0, 1), official["frames"][0].transpose(0, 1), data_range=1).item()
    report["MS_SSIM"] = score
    print(json.dumps(report, indent=2))
    (directory / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    assert report["source_latents"]["rel_l2"] < 1e-5, "Source preprocessing/VAE encode drifted"
    for key in ("prompt", "negative_prompt"):
        assert report[key]["rel_l2"] < 1e-3, f"Text encoding drifted: {key}"
    assert score >= 0.98, f"Pipeline MS-SSIM {score} < 0.98"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--implementation", choices=["native", "official"])
    parser.add_argument("--video", type=Path, default=REFERENCE / "assets/testcases/v2v/source_case1.mp4")
    parser.add_argument("--prompt", default=(
        "Change the scene to a watercolor painting, preserving the dog and its motion."))
    parser.add_argument("--prompt-file", type=Path, help="Official testcase JSON containing a prompt field")
    parser.add_argument("--height", type=int, default=192)
    parser.add_argument("--width", type=int, default=352)
    parser.add_argument("--frames", type=int, default=17)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs_bernini_parity")
    parser.add_argument("--compare", type=Path)
    args = parser.parse_args()
    if args.prompt_file:
        args.prompt = json.loads(args.prompt_file.read_text())["prompt"]
    if args.compare:
        compare(args.compare)
        return
    if args.implementation is None:
        parser.error("--implementation is required unless comparing artifacts")
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    try:
        result = run_native(args) if args.implementation == "native" else run_official(args)
        save_result(result, args.output, args.implementation, started)
    finally:
        if args.implementation == "native":
            from fastvideo.distributed import cleanup_dist_env_and_memory
            cleanup_dist_env_and_memory()


if __name__ == "__main__":
    main()
