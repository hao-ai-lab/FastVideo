# SPDX-License-Identifier: Apache-2.0
"""Generate full Flux2 Diffusers/FastVideo image comparison artifacts.

This is intended for Modal GPU runs. The parent process tries requested square
resolutions in order; each attempt runs in a fresh child process so a CUDA OOM
can fall back cleanly to the next resolution.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw


DEFAULT_PROMPT = "a photo of a banana on a wooden table, studio lighting"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate full Flux2 image comparison artifacts.")
    parser.add_argument("--model-dir", default=os.getenv("FLUX2_FULL_MODEL_DIR", ""))
    parser.add_argument("--output-root", default="/root/data/flux2_full_image_compare")
    parser.add_argument("--run-name", default="20260526_h100_full_t2i")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sizes", default="1024,768,512,256")
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--guidance-scale", type=float, default=4.0)
    parser.add_argument("--max-sequence-length", type=int, default=64)
    parser.add_argument("--child-size", type=int, default=0)
    parser.add_argument("--child-mode", choices=("diffusers", "fastvideo"), default="")
    return parser.parse_args()


def _load_prompt_embeds(
    model_dir: Path,
    prompt: str,
    dtype: torch.dtype,
    max_sequence_length: int,
) -> torch.Tensor:
    from diffusers import Flux2Pipeline

    prompt_pipe = Flux2Pipeline.from_pretrained(
        str(model_dir),
        local_files_only=True,
        torch_dtype=dtype,
    )
    try:
        with torch.no_grad():
            return prompt_pipe._get_mistral_3_small_prompt_embeds(  # noqa: SLF001
                text_encoder=prompt_pipe.text_encoder,
                tokenizer=prompt_pipe.tokenizer,
                prompt=[prompt],
                device=torch.device("cpu"),
                max_sequence_length=max_sequence_length,
                system_message=prompt_pipe.system_message,
                hidden_states_layers=(10, 20, 30),
            )
    finally:
        del prompt_pipe
        gc.collect()
        torch.cuda.empty_cache()


def _generate_diffusers_image(
    model_dir: Path,
    output_path: Path,
    *,
    prompt: str,
    seed: int,
    size: int,
    steps: int,
    guidance_scale: float,
    max_sequence_length: int,
) -> Image.Image:
    from diffusers import Flux2Pipeline

    device = torch.device("cuda:0")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    prompt_embeds = _load_prompt_embeds(model_dir, prompt, dtype, max_sequence_length)
    max_memory = {idx: "78GiB" for idx in range(torch.cuda.device_count())}
    pipe = Flux2Pipeline.from_pretrained(
        str(model_dir),
        local_files_only=True,
        torch_dtype=dtype,
        text_encoder=None,
        tokenizer=None,
        device_map="balanced",
        max_memory=max_memory,
    )
    pipe.set_progress_bar_config(disable=True)
    try:
        with torch.no_grad():
            output = pipe(
                prompt=None,
                prompt_embeds=prompt_embeds.to(device=device, dtype=dtype),
                height=size,
                width=size,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                max_sequence_length=max_sequence_length,
                generator=torch.Generator(device="cpu").manual_seed(seed),
                output_type="pil",
                return_dict=True,
            )
        image = output.images[0].convert("RGB")
        image.save(output_path)
        return image
    finally:
        del pipe
        del prompt_embeds
        gc.collect()
        torch.cuda.empty_cache()


def _generate_fastvideo_image(
    model_dir: Path,
    output_path: Path,
    *,
    prompt: str,
    seed: int,
    size: int,
    steps: int,
    guidance_scale: float,
    max_sequence_length: int,
) -> Image.Image:
    from fastvideo import VideoGenerator
    from fastvideo.api.sampling_param import SamplingParam

    generator = VideoGenerator.from_pretrained(
        str(model_dir),
        num_gpus=1,
        tp_size=1,
        sp_size=1,
        workload_type="t2i",
        use_fsdp_inference=False,
        dit_cpu_offload=False,
        vae_cpu_offload=True,
        text_encoder_cpu_offload=True,
        pin_cpu_memory=False,
        override_pipeline_cls_name="Flux2Pipeline",
    )
    try:
        sampling = SamplingParam.from_pretrained(str(model_dir))
        sampling.prompt = prompt
        sampling.height = size
        sampling.width = size
        sampling.num_frames = 1
        sampling.fps = 1
        sampling.num_inference_steps = steps
        sampling.guidance_scale = guidance_scale
        sampling.max_sequence_length = max_sequence_length
        sampling.seed = seed
        sampling.output_path = str(output_path)
        sampling.save_video = True
        sampling.return_frames = False
        generator.generate_video(prompt, sampling_param=sampling, output_path=str(output_path))
    finally:
        generator.shutdown()
        gc.collect()
        torch.cuda.empty_cache()

    return Image.open(output_path).convert("RGB")


def _save_comparison(
    output_dir: Path,
    upstream: Image.Image,
    fastvideo: Image.Image,
    metadata: dict[str, Any],
) -> None:
    upstream_arr = np.asarray(upstream.convert("RGB"), dtype=np.int16)
    fastvideo_arr = np.asarray(fastvideo.convert("RGB"), dtype=np.int16)
    diff = np.abs(upstream_arr - fastvideo_arr).astype(np.uint8)
    max_diff = int(diff.max())
    metrics = {
        **metadata,
        "max_abs_diff": max_diff,
        "mean_abs_diff": float(diff.mean()),
        "median_abs_diff": float(np.median(diff)),
        "rmse": float(np.sqrt(np.mean((upstream_arr - fastvideo_arr).astype(np.float32) ** 2))),
    }

    Image.fromarray(diff, mode="RGB").save(output_dir / "abs_diff.png")
    scale = 1 if max_diff == 0 else min(255.0 / max_diff, 64.0)
    Image.fromarray(np.clip(diff.astype(np.float32) * scale, 0, 255).astype(np.uint8), mode="RGB").save(
        output_dir / "abs_diff_scaled.png"
    )

    label_height = 34
    gap = 8
    side_by_side = Image.new(
        "RGB",
        (upstream.width + fastvideo.width + gap, upstream.height + label_height),
        "white",
    )
    draw = ImageDraw.Draw(side_by_side)
    draw.text((0, 8), "Diffusers", fill=(0, 0, 0))
    draw.text((upstream.width + gap, 8), "FastVideo", fill=(0, 0, 0))
    side_by_side.paste(upstream, (0, label_height))
    side_by_side.paste(fastvideo, (upstream.width + gap, label_height))
    side_by_side.save(output_dir / "side_by_side.png")

    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)
    with (output_dir / "README.txt").open("w", encoding="utf-8") as f:
        f.write(
            "Full Flux2 H100 image comparison\n"
            f"prompt: {metadata['prompt']}\n"
            f"seed: {metadata['seed']}\n"
            f"size: {metadata['size']}x{metadata['size']}\n"
            f"steps: {metadata['steps']}\n"
            f"guidance_scale: {metadata['guidance_scale']}\n"
            f"max_sequence_length: {metadata['max_sequence_length']}\n"
            f"max_abs_diff: {metrics['max_abs_diff']}\n"
            f"mean_abs_diff: {metrics['mean_abs_diff']}\n"
            f"median_abs_diff: {metrics['median_abs_diff']}\n"
            f"rmse: {metrics['rmse']}\n"
        )


def _attempt_dir(args: argparse.Namespace, size: int) -> Path:
    return Path(args.output_root) / args.run_name / f"{size}x{size}"


def _run_diffusers_child(args: argparse.Namespace) -> None:
    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        raise RuntimeError("Set --model-dir or FLUX2_FULL_MODEL_DIR to the full Flux2 checkpoint directory")

    output_dir = _attempt_dir(args, args.child_size)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[image-compare] writing Diffusers image to {output_dir}", flush=True)
    _generate_diffusers_image(
        model_dir,
        output_dir / "upstream_diffusers.png",
        prompt=args.prompt,
        seed=args.seed,
        size=args.child_size,
        steps=args.steps,
        guidance_scale=args.guidance_scale,
        max_sequence_length=args.max_sequence_length,
    )
    print(f"[image-compare] Diffusers success size={args.child_size}", flush=True)


def _run_fastvideo_child(args: argparse.Namespace) -> None:
    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        raise RuntimeError("Set --model-dir or FLUX2_FULL_MODEL_DIR to the full Flux2 checkpoint directory")

    output_dir = _attempt_dir(args, args.child_size)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[image-compare] writing FastVideo image to {output_dir}", flush=True)
    _generate_fastvideo_image(
        model_dir,
        output_dir / "fastvideo.png",
        prompt=args.prompt,
        seed=args.seed,
        size=args.child_size,
        steps=args.steps,
        guidance_scale=args.guidance_scale,
        max_sequence_length=args.max_sequence_length,
    )
    print(f"[image-compare] FastVideo success size={args.child_size}", flush=True)


def _compare_attempt(args: argparse.Namespace, size: int) -> None:
    output_dir = _attempt_dir(args, size)
    upstream = Image.open(output_dir / "upstream_diffusers.png").convert("RGB")
    fastvideo = Image.open(output_dir / "fastvideo.png").convert("RGB")
    _save_comparison(
        output_dir,
        upstream,
        fastvideo,
        {
            "prompt": args.prompt,
            "seed": args.seed,
            "size": size,
            "steps": args.steps,
            "guidance_scale": args.guidance_scale,
            "max_sequence_length": args.max_sequence_length,
        },
    )
    print(f"[image-compare] comparison success size={size} output_dir={output_dir}", flush=True)


def _copy_successful_attempt(attempt_dir: Path, final_dir: Path) -> None:
    final_dir.mkdir(parents=True, exist_ok=True)
    for path in attempt_dir.iterdir():
        if path.is_file():
            shutil.copy2(path, final_dir / path.name)
    with (final_dir / "selected_attempt.txt").open("w", encoding="utf-8") as f:
        f.write(str(attempt_dir) + "\n")


def _run_parent(args: argparse.Namespace) -> None:
    final_dir = Path(args.output_root) / args.run_name
    final_dir.mkdir(parents=True, exist_ok=True)
    sizes = [int(item.strip()) for item in args.sizes.split(",") if item.strip()]
    errors: list[str] = []
    for size in sizes:
        print(f"[image-compare] trying size={size}", flush=True)
        base_child_cmd = [sys.executable, __file__, *sys.argv[1:], "--child-size", str(size)]
        diffusers_result = subprocess.run([*base_child_cmd, "--child-mode", "diffusers"], check=False)
        if diffusers_result.returncode != 0:
            errors.append(f"{size}/diffusers: exit_code={diffusers_result.returncode}")
            print(
                f"[image-compare] size={size} Diffusers failed with exit_code={diffusers_result.returncode}",
                flush=True,
            )
            continue
        fastvideo_result = subprocess.run([*base_child_cmd, "--child-mode", "fastvideo"], check=False)
        if fastvideo_result.returncode == 0:
            _compare_attempt(args, size)
            attempt_dir = final_dir / f"{size}x{size}"
            _copy_successful_attempt(attempt_dir, final_dir)
            print(f"[image-compare] selected size={size}", flush=True)
            print(f"[image-compare] final output_dir={final_dir}", flush=True)
            return
        errors.append(f"{size}/fastvideo: exit_code={fastvideo_result.returncode}")
        print(
            f"[image-compare] size={size} FastVideo failed with exit_code={fastvideo_result.returncode}",
            flush=True,
        )
    raise RuntimeError("All image comparison attempts failed: " + "; ".join(errors))


def main() -> None:
    args = _parse_args()
    if args.child_size and args.child_mode == "diffusers":
        _run_diffusers_child(args)
    elif args.child_size and args.child_mode == "fastvideo":
        _run_fastvideo_child(args)
    else:
        _run_parent(args)


if __name__ == "__main__":
    main()
