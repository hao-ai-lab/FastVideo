# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import contextlib
import os
import re

DEFAULT_PROMPTS = [
    "a photo of a cat",
    (
        "a cinematic photo of a red panda wearing a tiny backpack, standing on a "
        "rainy neon-lit street at night, shallow depth of field, sharp focus, "
        "35mm, bokeh"
    ),
]


def _safe_filename(text: str, max_len: int = 100) -> str:
    """Make a stable, filesystem-friendly filename base."""
    s = text[:max_len].strip()
    s = s.replace(os.sep, "_")
    if os.altsep:
        s = s.replace(os.altsep, "_")
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^A-Za-z0-9 .,_-]", "_", s)
    s = s.strip(" .")
    return s or "prompt"


def _remove_existing_outputs(out_dir: str, filename_base: str) -> None:
    """Delete prior outputs so reruns do not get _1, _2 suffixes."""
    if not os.path.isdir(out_dir):
        return

    pattern = re.compile(rf"^{re.escape(filename_base)}(_\d+)?\.(mp4|png)$")
    for fn in os.listdir(out_dir):
        if pattern.match(fn):
            with contextlib.suppress(FileNotFoundError):
                os.remove(os.path.join(out_dir, fn))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run FLUX.1-dev text-to-image with FastVideo VideoGenerator.",
    )
    p.add_argument(
        "--model-path",
        default="official_weights/FLUX.1-dev",
        help="Local Diffusers checkpoint dir or HF repo id.",
    )
    p.add_argument(
        "--out-dir",
        "--outdir",
        default="outputs/flux_dev/samples",
        help="Directory for saved PNG outputs.",
    )
    p.add_argument(
        "--prompt",
        action="append",
        default=None,
        help="Prompt. Repeat for multiple images.",
    )
    p.add_argument(
        "--backend",
        default=None,
        help="Set FASTVIDEO_ATTENTION_BACKEND (e.g. TORCH_SDPA).",
    )
    p.add_argument("--seed", type=int, default=42, help="Base seed; each prompt uses seed + index.")
    p.add_argument("--height", type=int, default=1024, help="Output height.")
    p.add_argument("--width", type=int, default=1024, help="Output width.")
    p.add_argument("--steps", type=int, default=28, help="Number of inference steps.")
    p.add_argument("--guidance", type=float, default=3.5, help="Guidance scale.")
    p.add_argument("--num-gpus", type=int, default=1, help="GPU count.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    prompts: list[str] = args.prompt if args.prompt else DEFAULT_PROMPTS

    if args.backend:
        os.environ["FASTVIDEO_ATTENTION_BACKEND"] = args.backend

    from fastvideo import VideoGenerator

    os.makedirs(args.out_dir, exist_ok=True)

    init_kwargs = {
        "num_gpus": args.num_gpus,
        "workload_type": "t2i",
        "sp_size": 1,
        "tp_size": 1,
        "dit_cpu_offload": False,
        "dit_layerwise_offload": False,
        "text_encoder_cpu_offload": False,
        "vae_cpu_offload": False,
        "image_encoder_cpu_offload": False,
        "pin_cpu_memory": False,
        "use_fsdp_inference": False,
    }

    generator = VideoGenerator.from_pretrained(
        model_path=args.model_path,
        **init_kwargs,
    )
    try:
        for i, prompt in enumerate(prompts):
            seed = args.seed + i
            filename_base = (
                f"flux_dev_{i:02d}_seed{seed}_{_safe_filename(prompt, max_len=80)}"
            )
            _remove_existing_outputs(args.out_dir, filename_base)
            output_path = os.path.join(args.out_dir, f"{filename_base}.png")
            print(f"[flux] prompt_idx={i} seed={seed} output_path={output_path}")

            generation_kwargs = {
                "output_path": output_path,
                "height": args.height,
                "width": args.width,
                "num_frames": 1,
                "fps": 1,
                "num_inference_steps": args.steps,
                "guidance_scale": args.guidance,
                "use_embedded_guidance": True,
                "true_cfg_scale": 1.0,
                "seed": seed,
                "save_video": True,
            }

            generator.generate_video(prompt, **generation_kwargs)

        print(f"[flux] done. outputs written to: {args.out_dir}")
    finally:
        generator.shutdown()


if __name__ == "__main__":
    main()
