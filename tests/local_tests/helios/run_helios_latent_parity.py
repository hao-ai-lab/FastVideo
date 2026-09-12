# SPDX-License-Identifier: Apache-2.0
"""Compare official and FastVideo Helios latents with identical CPU RNG."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import re
import time

import torch
from diffusers import AutoModel, HeliosPyramidPipeline as OfficialHeliosPipeline
import torch.nn.functional as F

PROMPT = (
    "A vibrant tropical fish swims gracefully through a colorful coral reef "
    "in clear turquoise water, cinematic close-up, fluid motion, vivid detail."
)
NEGATIVE_PROMPT = (
    "Bright tones, overexposed, static, blurred details, subtitles, paintings, "
    "images, overall gray, worst quality, low quality, JPEG artifacts, ugly, "
    "deformed, disfigured, still picture, messy background."
)


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=repo_root / "official_weights" / "helios",
    )
    parser.add_argument("--output-dir", type=Path, default=repo_root / "outputs")
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--width", type=int, default=192)
    parser.add_argument("--num-frames", type=int, default=33)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--share-official-prompt-embeds", action="store_true")
    parser.add_argument("--reuse-saved-official-artifacts", action="store_true")
    parser.add_argument("--official-trace-output", type=Path)
    return parser.parse_args()


def attach_official_trace(model, output_path: Path):
    pattern = re.compile(r"^(blocks\.(0|9|19|29|39)|proj_out)$")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sink = output_path.open("w", encoding="utf-8")
    call_counts: dict[str, int] = {}
    handles = []

    def make_hook(module_name: str):

        def hook(module, inputs, output):
            del module, inputs
            tensor = output[0] if isinstance(output, tuple | list) else output
            if not isinstance(tensor, torch.Tensor):
                return
            step = call_counts.get(module_name, 0)
            call_counts[module_name] = step + 1
            value = tensor.detach().float()
            sink.write(
                json.dumps({
                    "module": module_name,
                    "tensor": "out",
                    "step": step,
                    "abs_mean": value.abs().mean().item(),
                    "mean": value.mean().item(),
                    "std": value.std().item(),
                    "max": value.max().item(),
                    "shape": list(value.shape),
                    "dtype": str(tensor.dtype),
                }) + "\n")

        return hook

    for name, module in model.named_modules():
        if pattern.fullmatch(name):
            handles.append(module.register_forward_hook(make_hook(name)))
    if len(handles) != 6:
        sink.close()
        raise RuntimeError(f"Expected 6 official trace modules, attached {len(handles)}")
    return handles, sink


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.cuda.set_device(args.gpu)

    official_path = args.output_dir / "helios_official_cpu_seed42_latents.pt"
    prompt_embeds_path = args.output_dir / "helios_official_prompt_embeds.pt"
    if args.reuse_saved_official_artifacts:
        official_latents = torch.load(official_path, map_location="cpu")
        shared_prompt_embeds_cpu = (torch.load(prompt_embeds_path, map_location="cpu").detach()
                                    if args.share_official_prompt_embeds else None)
        official_seconds = 0.0
    else:
        official_started = time.perf_counter()
        vae = AutoModel.from_pretrained(
            str(args.model_dir),
            subfolder="vae",
            local_files_only=True,
            torch_dtype=torch.float32,
        )
        official_pipeline = OfficialHeliosPipeline.from_pretrained(
            str(args.model_dir),
            vae=vae,
            local_files_only=True,
            torch_dtype=torch.bfloat16,
        )
        official_pipeline.enable_model_cpu_offload(gpu_id=args.gpu)
        trace_handles = []
        trace_sink = None
        if args.official_trace_output is not None:
            trace_handles, trace_sink = attach_official_trace(
                official_pipeline.transformer,
                args.official_trace_output,
            )
        shared_prompt_embeds = None
        if args.share_official_prompt_embeds:
            shared_prompt_embeds, _ = official_pipeline.encode_prompt(
                prompt=PROMPT,
                negative_prompt=None,
                do_classifier_free_guidance=False,
                num_videos_per_prompt=1,
                max_sequence_length=512,
                dtype=torch.bfloat16,
            )
        try:
            official_latents = (official_pipeline(
                prompt=None if shared_prompt_embeds is not None else PROMPT,
                negative_prompt=(None if shared_prompt_embeds is not None else NEGATIVE_PROMPT),
                prompt_embeds=shared_prompt_embeds,
                height=args.height,
                width=args.width,
                num_frames=args.num_frames,
                pyramid_num_inference_steps_list=[args.steps] * 3,
                guidance_scale=1.0,
                is_amplify_first_chunk=True,
                generator=torch.Generator("cpu").manual_seed(42),
                output_type="latent",
            ).frames.float().cpu())
        finally:
            for handle in trace_handles:
                handle.remove()
            if trace_sink is not None:
                trace_sink.close()
        official_seconds = time.perf_counter() - official_started
        torch.save(official_latents, official_path)
        shared_prompt_embeds_cpu = shared_prompt_embeds.detach().cpu() if shared_prompt_embeds is not None else None
        if shared_prompt_embeds_cpu is not None:
            torch.save(shared_prompt_embeds_cpu, prompt_embeds_path)

        del official_pipeline, vae
        gc.collect()
        torch.cuda.empty_cache()

    from fastvideo import VideoGenerator

    fastvideo_load_started = time.perf_counter()
    generator = VideoGenerator.from_pretrained(
        str(args.model_dir),
        num_gpus=1,
        use_fsdp_inference=False,
        dit_cpu_offload=False,
        dit_layerwise_offload=True,
        text_encoder_cpu_offload=True,
        vae_cpu_offload=True,
        pin_cpu_memory=False,
        enable_stage_verification=True,
        output_type="latent",
    )
    fastvideo_load_seconds = time.perf_counter() - fastvideo_load_started
    fastvideo_started = time.perf_counter()
    try:
        generation_kwargs = {
            "negative_prompt": NEGATIVE_PROMPT,
            "output_path": str(args.output_dir),
            "save_video": False,
            "return_frames": True,
            "height": args.height,
            "width": args.width,
            "num_frames": args.num_frames,
            "fps": 24,
            "num_inference_steps": args.steps,
            "pyramid_num_inference_steps_list": [args.steps] * 3,
            "history_sizes": [16, 2, 1],
            "num_latent_frames_per_chunk": 9,
            "keep_first_frame": True,
            "is_skip_first_chunk": False,
            "use_zero_init": True,
            "zero_steps": 1,
            "guidance_scale": 1.0,
            "is_amplify_first_chunk": True,
            "seed": 42,
        }
        if shared_prompt_embeds_cpu is None:
            result = generator.generate_video(prompt=PROMPT, **generation_kwargs)
        else:
            result = generator._generate_video_impl(
                prompt=PROMPT,
                sampling_param=None,
                fastvideo_args=generator.fastvideo_args,
                prompt_embeds=[shared_prompt_embeds_cpu],
                **generation_kwargs,
            )
    finally:
        generator.shutdown()
    fastvideo_seconds = time.perf_counter() - fastvideo_started
    fastvideo_latents = result["samples"].float().cpu()
    fastvideo_path = args.output_dir / "helios_fastvideo_cpu_seed42_latents.pt"
    torch.save(fastvideo_latents, fastvideo_path)

    if official_latents.shape != fastvideo_latents.shape:
        raise AssertionError(
            f"Latent shape mismatch: official={official_latents.shape}, FastVideo={fastvideo_latents.shape}")
    difference = official_latents - fastvideo_latents
    summary = {
        "shape":
        list(official_latents.shape),
        "official_path":
        str(official_path),
        "fastvideo_path":
        str(fastvideo_path),
        "official_abs_mean":
        official_latents.abs().mean().item(),
        "fastvideo_abs_mean":
        fastvideo_latents.abs().mean().item(),
        "diff_max":
        difference.abs().max().item(),
        "diff_mean":
        difference.abs().mean().item(),
        "rmse":
        difference.square().mean().sqrt().item(),
        "cosine":
        F.cosine_similarity(
            official_latents.flatten().unsqueeze(0),
            fastvideo_latents.flatten().unsqueeze(0),
        ).item(),
        "official_seconds":
        round(official_seconds, 3),
        "fastvideo_load_seconds":
        round(fastvideo_load_seconds, 3),
        "fastvideo_seconds":
        round(fastvideo_seconds, 3),
        "seed":
        42,
        "generator_device":
        "cpu",
        "conditioning_source":
        ("shared_official_prompt_embeds" if shared_prompt_embeds_cpu is not None else "each_pipeline_text_encoder"),
        "steps": [args.steps] * 3,
    }
    print("HELIOS_LATENT_PARITY=" + json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
