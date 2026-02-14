# SPDX-License-Identifier: Apache-2.0
"""
Run official Overworld Waypoint-1-Small with diffusers (ModularPipeline).

Parameterized for comparison runs: same prompt, seed, num_frames as FastVideo.

Usage:
  python scripts/run_diffusers_waypoint.py --prompt "..." --seed 1024 --num-frames 16 --output diffusers.mp4

Requirements:
  pip install "diffusers>=0.36.0" "transformers>=4.57.1" einops tensordict imageio imageio-ffmpeg tqdm
"""
import argparse
import os
from dataclasses import dataclass, field
from typing import Set, Tuple

# RunPod: avoid disk quota
_workspace = "/workspace"
if os.path.isdir(_workspace):
    _hf = os.path.join(_workspace, ".cache", "huggingface")
    _hub = os.path.join(_hf, "hub")
    for _d in (_hub,):
        os.makedirs(_d, exist_ok=True)
    os.environ.setdefault("HF_HOME", _hf)
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", _hub)
    os.environ.setdefault("TMPDIR", os.path.join(_workspace, "tmp"))

os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")


def main():
    import torch
    from tqdm import tqdm

    try:
        from diffusers.modular_pipelines import ModularPipelineBlocks
        from diffusers.utils import export_to_video
    except ImportError as e:
        raise SystemExit(
            "Install diffusers and deps:\n"
            "  pip install 'diffusers>=0.36.0' 'transformers>=4.57.1' "
            "einops tensordict imageio imageio-ffmpeg tqdm"
        ) from e

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompt",
        default="A first-person view of walking through a grassy field.",
        help="Text prompt",
    )
    parser.add_argument("--seed", type=int, default=1024, help="Random seed")
    parser.add_argument(
        "--num-frames",
        type=int,
        default=16,
        help="Number of frames to generate",
    )
    parser.add_argument(
        "--output",
        default="diffusers_waypoint.mp4",
        help="Output video path",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model path or HF id (default: Overworld/Waypoint-1-Small)",
    )
    parser.add_argument(
        "--debug-output",
        default=None,
        help="Save debug stats (VAE config, frame image stats) to JSON",
    )
    args = parser.parse_args()

    model_id = args.model or os.environ.get(
        "WAYPOINT_DIFFUSERS_MODEL_PATH", "Overworld/Waypoint-1-Small"
    )

    @dataclass
    class CtrlInput:
        button: Set[int] = field(default_factory=set)
        mouse: Tuple[float, float] = (0.0, 0.0)

    KEY_W = 17
    ctrl = lambda: CtrlInput(button={KEY_W}, mouse=(0.0, 0.0))

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print(f"Loading {model_id}...")
    blocks = ModularPipelineBlocks.from_pretrained(
        model_id, trust_remote_code=True
    )
    pipe = blocks.init_pipeline(pretrained_model_name_or_path=model_id)
    pipe.load_components(
        device_map="cuda",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    if pipe.transformer is None:
        raise RuntimeError("Failed to load transformer.")
    pipe.transformer.apply_inference_patches()

    print(f"Generating {args.num_frames} frames...")
    state = pipe(
        prompt=args.prompt,
        image=None,
        button=ctrl().button,
        mouse=ctrl().mouse,
        output_type="pil",
    )
    outputs = [state.values["images"]]
    state.values["image"] = None

    for _ in tqdm(range(1, args.num_frames)):
        state = pipe(
            state,
            prompt=args.prompt,
            button=ctrl().button,
            mouse=ctrl().mouse,
            output_type="pil",
        )
        outputs.append(state.values["images"])

    out_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    export_to_video(outputs, out_path, fps=60)
    print(f"Saved: {out_path}")

    if args.debug_output:
        import json
        import numpy as np
        debug_data = {
            "source": "diffusers",
            "model_id": model_id,
            "seed": args.seed,
            "prompt": args.prompt,
            "num_frames": args.num_frames,
            "vae_config": {},
            "frames": [],
        }
        vae = getattr(pipe, "vae", None)
        if vae is not None:
            vae_config = getattr(vae, "config", None)
            if vae_config is not None:
                sf = getattr(vae_config, "scaling_factor", None)
                shift = getattr(vae_config, "shift_factor", None)
                debug_data["vae_config"]["scaling_factor"] = (
                    float(sf) if sf is not None else 1.0)
                shift_val = None
                if shift is not None:
                    try:
                        shift_val = (
                            float(shift.item())
                            if hasattr(shift, "numel") and shift.numel() == 1
                            else float(shift))
                    except (TypeError, ValueError):
                        pass
                debug_data["vae_config"]["shift_factor"] = shift_val
        for frame_idx, imgs in enumerate(outputs):
            frame_stat = {"frame": frame_idx}
            img_list = imgs if isinstance(imgs, (list, tuple)) else [imgs]
            if img_list and img_list[0] is not None:
                img = img_list[0]
                try:
                    arr = torch.from_numpy(
                        np.array(img, dtype=np.float32)
                    ) / 255.0
                except Exception:
                    arr = None
                if arr is not None and arr.numel() > 0:
                    frame_stat["vae_out"] = {
                        "mean": float(arr.mean()),
                        "std": float(arr.std()) if arr.numel() > 1 else 0.0,
                        "min": float(arr.min()),
                        "max": float(arr.max()),
                    }
            debug_data["frames"].append(frame_stat)
        debug_path = os.path.abspath(args.debug_output)
        os.makedirs(os.path.dirname(debug_path) or ".", exist_ok=True)
        with open(debug_path, "w") as f:
            json.dump(debug_data, f, indent=2)
        print(f"Debug stats saved to {debug_path}")


if __name__ == "__main__":
    main()
