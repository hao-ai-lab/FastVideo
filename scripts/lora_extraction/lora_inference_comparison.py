#!/usr/bin/env python3
"""Compare Fine-Tuned vs Merged/Base+LoRA inference outputs.

Generates two videos with the same seed and computes SSIM.

Usage examples:
python lora_inference_comparison.py \
  --base ./merged_model \
  --ft FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers \
  --adapter NONE \
  --output-dir ./inference_comparison \
  --compute-ssim \
  --seed 41

or

python lora_inference_comparison.py \
  --base Wan-AI/Wan2.2-TI2V-5B-Diffusers \
  --ft FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers \
  --adapter adapter.safetensors \
  --output-dir ./inference_comparison \
  --compute-ssim \
  --seed 41

"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any

import logging

# minimal distributed env defaults (kept for compatibility)
os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29500")
os.environ.setdefault("WORLD_SIZE", "1")
os.environ.setdefault("RANK", "0")
os.environ.setdefault("LOCAL_RANK", "0")

# allow running from repo root where fastvideo is located
_FASTVIDEO_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "fastvideo_pr", "FastVideo"))
if _FASTVIDEO_PATH not in sys.path:
    sys.path.insert(0, _FASTVIDEO_PATH)

logger = logging.getLogger("inference_comparison")


def configure_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler()
    fmt = "%(asctime)s %(levelname)s %(message)s"
    handler.setFormatter(logging.Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(handler)
    logger.setLevel(level)


def _validate_adapter(adapter: Optional[str]) -> Optional[str]:
    if not adapter:
        return None
    if adapter.upper() == "NONE":
        return None
    p = Path(adapter).expanduser()
    if not p.exists() or not p.is_file():
        raise FileNotFoundError(f"Adapter not found: {p}")
    if p.suffix != ".safetensors":
        raise ValueError(f"Adapter must be a .safetensors file, got: {p.suffix}")
    if p.stat().st_size == 0:
        raise ValueError(f"Adapter file is empty: {p}")
    return str(p.resolve())


def generate_with_model(
    model_path: str,
    output_dir: str,
    output_name: str,
    prompt: str,
    seed: int,
    lora_path: Optional[str],
    height: int,
    width: int,
    num_frames: int,
    num_inference_steps: int,
    guidance_scale: float,
    flow_shift: Optional[float] = None,
    embedded_guidance_scale: Optional[float] = None,
) -> str:
    """Produce a video with VideoGenerator.from_pretrained; returns video path."""
    try:
        from fastvideo import VideoGenerator  # lazy import
    except Exception as exc:
        raise RuntimeError(f"Failed to import fastvideo.VideoGenerator: {exc}") from exc

    init_kwargs: Dict[str, Any] = {
        "num_gpus": 1,
        "dit_cpu_offload": True,
        "vae_cpu_offload": True,
        "text_encoder_cpu_offload": True,
        "pin_cpu_memory": True,
    }
    if lora_path:
        init_kwargs["lora_path"] = lora_path
        init_kwargs["lora_nickname"] = "extracted"

    generator = VideoGenerator.from_pretrained(model_path, **init_kwargs)

    gen_kwargs = {
        "height": height,
        "width": width,
        "num_frames": num_frames,
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale,
        "seed": seed,
        "output_path": output_dir,
        "output_video_name": output_name,
        "save_video": True,
    }
    if flow_shift is not None:
        gen_kwargs["flow_shift"] = flow_shift
    if embedded_guidance_scale is not None:
        gen_kwargs["embedded_guidance_scale"] = embedded_guidance_scale

    result = generator.generate_video(prompt, **gen_kwargs)

    # best-effort cleanup of internal executors
    try:
        if hasattr(generator, "executor") and hasattr(generator.executor, "shutdown"):
            generator.executor.shutdown()
    except Exception:
        pass

    # determine saved video path
    expected = Path(output_dir) / f"{output_name}.mp4"
    if expected.exists():
        return str(expected)
    # fallback: check result dict
    if isinstance(result, dict) and "video_path" in result:
        return str(result["video_path"])
    raise FileNotFoundError(f"Video not found at expected path: {expected}")


def compute_ssim_if_requested(output_dir: str, ft_video: str, other_video: str, num_inference_steps: int, prompt: str) -> Optional[float]:
    try:
        from fastvideo.tests.utils import compute_video_ssim_torchvision, write_ssim_results  # type: ignore
    except Exception:
        logger.warning("SSIM utilities unavailable; skipping SSIM computation")
        return None

    ssim_values = compute_video_ssim_torchvision(ft_video, other_video, use_ms_ssim=True)
    mean_ssim = float(ssim_values[0])
    write_ssim_results(output_dir, ssim_values, ft_video, other_video, num_inference_steps, prompt)
    return mean_ssim


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare Fine-Tuned vs Merged/Base+LoRA inference outputs")
    p.add_argument("--base", required=True, help="Base model ID or merged model path")
    p.add_argument("--ft", required=True, help="Fine-tuned model ID or path (reference)")
    p.add_argument("--adapter", default="NONE", help="Path to .safetensors adapter, or NONE to use merged model")
    p.add_argument("--output-dir", default="./inference_comparison")
    p.add_argument("--prompt", default="A cat playing with a ball of yarn in a cozy living room, cinematic")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width", type=int, default=832)
    p.add_argument("--num-frames", type=int, default=49)
    p.add_argument("--num-inference-steps", type=int, default=32)
    p.add_argument("--guidance-scale", type=float, default=6.0)
    p.add_argument("--compute-ssim", action="store_true")
    p.add_argument("--flow-shift", type=float, default=None)
    p.add_argument("--embedded-guidance-scale", type=float, default=None)
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)

    os.makedirs(args.output_dir, exist_ok=True)
    try:
        adapter_path = _validate_adapter(args.adapter)
    except Exception as exc:
        logger.error("Adapter validation failed: %s", exc)
        sys.exit(2)

    # 1) generate with fine-tuned model (reference)
    logger.info("Generating reference (fine-tuned): %s", args.ft)
    try:
        ft_video = generate_with_model(
            model_path=args.ft,
            output_dir=args.output_dir,
            output_name="fine_tuned",
            prompt=args.prompt,
            seed=args.seed,
            lora_path=None,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            flow_shift=args.flow_shift,
            embedded_guidance_scale=args.embedded_guidance_scale,
        )
        logger.info("Reference video saved: %s", ft_video)
    except Exception as exc:
        logger.error("Reference generation failed: %s", exc)
        sys.exit(3)

    # 2) generate with merged model OR base + adapter
    use_merged = adapter_path is None
    mode = "merged model" if use_merged else "base+adapter"
    logger.info("Generating target (%s): %s", mode, args.base)
    try:
        target_video = generate_with_model(
            model_path=args.base,
            output_dir=args.output_dir,
            output_name="merged_model" if use_merged else "base_plus_lora",
            prompt=args.prompt,
            seed=args.seed,
            lora_path=None if use_merged else adapter_path,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            flow_shift=args.flow_shift,
            embedded_guidance_scale=args.embedded_guidance_scale,
        )
        logger.info("Target video saved: %s", target_video)
    except Exception as exc:
        logger.error("Target generation failed: %s", exc)
        sys.exit(4)

    # 3) optional SSIM
    if args.compute_ssim:
        logger.info("Computing SSIM...")
        mean_ssim = compute_ssim_if_requested(args.output_dir, ft_video, target_video, args.num_inference_steps, args.prompt)
        if mean_ssim is not None:
            logger.info("Mean SSIM: %.4f", mean_ssim)
        else:
            logger.info("SSIM computation not available.")

    logger.info("Comparison complete. Videos in: %s", args.output_dir)


if __name__ == "__main__":
    main()
