# SPDX-License-Identifier: Apache-2.0
"""Generate fixed official LingBot-Video TI2V tensors for exact parity tests."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from tests.local_tests.lingbot_video.hf_assets import (
    OFFICIAL_DENSE,
    OFFICIAL_MOE,
    download_components,
    materialize_component_view,
)

os.environ.setdefault("DIFFUSERS_ATTN_BACKEND", "native")
os.environ.setdefault("LINGBOT_QWEN_ATTN_IMPLEMENTATION", "sdpa")

HEIGHT = 480
WIDTH = 832
NUM_FRAMES = 121
GUIDANCE_SCALE = 3.0
FLOW_SHIFT = 3.0
SEED = 42


def _prompt(case_dir: Path) -> str:
    """Serialize example 4 exactly as the official runner does."""
    sample = json.loads((case_dir / "prompt.json").read_text(encoding="utf-8"))
    return json.dumps(sample["caption"], ensure_ascii=False, separators=(",", ":"))


def _as_channel_first(value: Any) -> torch.Tensor:
    """Normalize official latent or pixel output to a CPU float tensor."""
    tensor = value if torch.is_tensor(value) else torch.as_tensor(value)
    if tensor.ndim == 5 and tensor.shape[-1] == 3:
        tensor = tensor.permute(0, 4, 1, 2, 3)
    return tensor.detach().float().cpu()


def _configure_backends() -> None:
    """Use one deterministic CUDA policy for baseline generation and comparison."""
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.enabled = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.enable_cudnn_sdp(False)


def _model_root(variant: str, scratch: Path) -> Path:
    """Materialize the official base-generation component layout for one variant."""
    components = ("scheduler", "text_encoder", "processor", "transformer", "vae")
    if variant == "dense":
        return download_components(OFFICIAL_DENSE, *components)
    return materialize_component_view(OFFICIAL_MOE, scratch / "official_moe_base", *components)


def _initial_latents() -> torch.Tensor:
    """Create the fixed CPU noise tensor supplied to both implementations."""
    return torch.randn(
        (1, 16, (NUM_FRAMES - 1) // 4 + 1, HEIGHT // 8, WIDTH // 8),
        generator=torch.Generator(device="cpu").manual_seed(SEED),
        dtype=torch.float32,
    )


def _run_reference(pipe: Any, case_dir: Path) -> dict[str, torch.Tensor]:
    """Capture real one-step DiT calls, the 40-step trajectory, and decoded pixels."""
    from lingbot_video.pipeline_lingbot_video import DEFAULT_NEGATIVE_PROMPT

    prompt = _prompt(case_dir)
    initial_latents = _initial_latents()
    with Image.open(case_dir / "first_frame.png") as image_file:
        image = image_file.convert("RGB")

    dit_calls: list[dict[str, torch.Tensor]] = []
    text_encoder_calls: list[dict[str, torch.Tensor]] = []

    def capture_text_encoder_call(
        _module: Any,
        _args: tuple[Any, ...],
        kwargs: dict[str, Any],
        _output: Any,
    ) -> None:
        """Retain the exact processor tensors supplied to both CFG text-encoder calls."""
        if len(text_encoder_calls) < 2:
            text_encoder_calls.append({
                name: value.detach().cpu()
                for name, value in kwargs.items()
                if torch.is_tensor(value)
            })

    def capture_dit_call(_module: Any, args: tuple[Any, ...], kwargs: dict[str, Any], output: Any) -> None:
        """Retain the two sequential-CFG transformer calls from the one-step run."""
        output_tensor = output[0] if isinstance(output, tuple) else output.sample
        dit_calls.append({
            "hidden_states": args[0].detach().cpu(),
            "timestep": args[1].detach().cpu(),
            "encoder_hidden_states": args[2].detach().cpu(),
            "encoder_attention_mask": kwargs["encoder_attention_mask"].detach().cpu(),
            "output": output_tensor.detach().float().cpu(),
        })

    text_encoder_hook = pipe.text_encoder.register_forward_hook(capture_text_encoder_call, with_kwargs=True)
    hook = pipe.transformer.register_forward_hook(capture_dit_call, with_kwargs=True)
    with torch.inference_mode():
        one_step = pipe(
            prompt=prompt,
            image=image,
            negative_prompt=DEFAULT_NEGATIVE_PROMPT,
            height=HEIGHT,
            width=WIDTH,
            num_frames=NUM_FRAMES,
            num_inference_steps=1,
            guidance_scale=GUIDANCE_SCALE,
            shift=FLOW_SHIFT,
            generator=torch.Generator(device="cuda").manual_seed(SEED),
            latents=initial_latents.clone(),
            output_type="latent",
            batch_cfg=False,
            return_dict=True,
        )
    hook.remove()
    text_encoder_hook.remove()
    if len(dit_calls) != 2:
        raise AssertionError(f"expected two sequential-CFG DiT calls, got {len(dit_calls)}")
    if len(text_encoder_calls) != 2:
        raise AssertionError(f"expected two sequential-CFG text-encoder calls, got {len(text_encoder_calls)}")

    trajectory: list[torch.Tensor] = []
    scheduler_step = pipe.scheduler.step

    def capture_scheduler_step(*args: Any, **kwargs: Any) -> Any:
        """Retain every official scheduler output before clean-frame reinsertion."""
        result = scheduler_step(*args, **kwargs)
        trajectory.append(result[0].detach().float().cpu())
        return result

    pipe.scheduler.step = capture_scheduler_step
    try:
        with torch.inference_mode():
            full = pipe(
                prompt=prompt,
                image=image,
                negative_prompt=DEFAULT_NEGATIVE_PROMPT,
                height=HEIGHT,
                width=WIDTH,
                num_frames=NUM_FRAMES,
                num_inference_steps=40,
                guidance_scale=GUIDANCE_SCALE,
                shift=FLOW_SHIFT,
                generator=torch.Generator(device="cuda").manual_seed(SEED),
                latents=initial_latents.clone(),
                output_type="np",
                batch_cfg=False,
                return_dict=True,
            )
    finally:
        pipe.scheduler.step = scheduler_step
    if len(trajectory) != 40:
        raise AssertionError(f"expected 40 scheduler outputs, got {len(trajectory)}")
    conditional, unconditional = dit_calls
    guided = unconditional["output"] + GUIDANCE_SCALE * (conditional["output"] - unconditional["output"])
    return {
        "initial_latents": initial_latents,
        "condition_latent": conditional["hidden_states"][:, :, :1],
        "dit_hidden_states": conditional["hidden_states"],
        "dit_timestep": conditional["timestep"],
        "dit_prompt_embeds": conditional["encoder_hidden_states"],
        "dit_prompt_mask": conditional["encoder_attention_mask"],
        "dit_negative_embeds": unconditional["encoder_hidden_states"],
        "dit_negative_mask": unconditional["encoder_attention_mask"],
        "dit_conditional": conditional["output"],
        "dit_unconditional": unconditional["output"],
        "dit_guided": guided,
        "text_encoder_calls": text_encoder_calls,
        "one_step_latent": _as_channel_first(one_step.frames),
        "trajectory_before_condition": torch.stack(trajectory, dim=1),
        "pixels": _as_channel_first(full.frames),
    }


def main() -> None:
    """Load one pinned official variant and write its immutable parity baseline."""
    parser = argparse.ArgumentParser()
    parser.add_argument("variant", choices=("dense", "moe"))
    parser.add_argument("case_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _configure_backends()
    from lingbot_video.runner import _load_diffusers_pipe

    dtype_map = {
        "default": torch.bfloat16,
        "transformer": torch.bfloat16,
        "text_encoder": torch.bfloat16,
        "vae": torch.float32,
    }
    with tempfile.TemporaryDirectory(dir=args.output_dir) as temporary:
        model_root = _model_root(args.variant, Path(temporary))
        pipe = _load_diffusers_pipe(model_root, dtype_map, mode="ti2v", transformer_subfolder="transformer")
        try:
            tensors = _run_reference(pipe, args.case_dir)
        finally:
            del pipe
            gc.collect()
            torch.cuda.empty_cache()
    torch.save(tensors, args.output_dir / "baseline.pt")
    image_sha256 = hashlib.sha256((args.case_dir / "first_frame.png").read_bytes()).hexdigest()
    manifest = {
        "variant": args.variant,
        "height": HEIGHT,
        "width": WIDTH,
        "num_frames": NUM_FRAMES,
        "steps": [1, 40],
        "guidance_scale": GUIDANCE_SCALE,
        "flow_shift": FLOW_SHIFT,
        "seed": SEED,
        "qwen_attention": os.environ["LINGBOT_QWEN_ATTN_IMPLEMENTATION"],
        "prompt": _prompt(args.case_dir),
        "image_sha256": image_sha256,
        "baseline": "baseline.pt",
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
