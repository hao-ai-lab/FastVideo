# SPDX-License-Identifier: Apache-2.0
"""LTX-2 SR (refine) numerical alignment harness — public vs internal.

Runs the LTX-2 stage-2 spatial-refine pipeline with a fixed prompt and
seed and writes the decoded video frames + intermediate latents to a
torch save file. Intended to be invoked twice with different
``PYTHONPATH`` prefixes — once against ``FastVideo-internal`` and once
against the public ``FastVideo`` package — so a third pass can diff the
two outputs tensor-by-tensor.

Usage::

    # 1) Internal reference run.
    PYTHONPATH=/home/william5lin/FastVideo-internal \
        python scripts/ltx2_sr_alignment.py --label internal \
            --output /tmp/ltx2_sr_internal.pt

    # 2) Public run against the new typed API.
    PYTHONPATH=/home/william5lin/FastVideo \
        python scripts/ltx2_sr_alignment.py --label public --use-typed-api \
            --output /tmp/ltx2_sr_public.pt

    # 3) Diff.
    python scripts/ltx2_sr_alignment.py --diff \
        --reference /tmp/ltx2_sr_internal.pt \
        --candidate /tmp/ltx2_sr_public.pt

The harness deliberately matches `examples/inference/basic/basic_ltx2_upscale.py`
defaults (LTX2-Distilled, 8 base steps, 3 refine steps, 1088x1920, seed=10)
but skips FP4 and Dreamverse-specific knobs to keep the bf16 path clean.
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _bootstrap_fastvideo_source() -> None:
    """If FASTVIDEO_FROM is set in the environment, prepend that path to
    sys.path **and** strip any editable-install finders from
    ``sys.meta_path`` so the named source wins over a .pth-installed
    sibling. This lets us point the same harness at either
    ``../FastVideo`` or ``../FastVideo-internal`` without juggling
    venvs."""
    src = os.environ.get("FASTVIDEO_FROM")
    if not src:
        return
    src = os.path.abspath(src)
    if not os.path.isdir(src):
        raise SystemExit(f"FASTVIDEO_FROM={src!r} is not a directory")

    # Remove editable-install finders that would otherwise win over
    # PYTHONPATH. uv-installed editables register a custom finder
    # named like ``__editable___fastvideo_0_1_7_finder``.
    sys.meta_path = [
        finder for finder in sys.meta_path
        if "__editable___fastvideo" not in type(finder).__module__
    ]
    # Drop pre-resolved fastvideo modules from any earlier import.
    for name in [k for k in sys.modules if k == "fastvideo" or k.startswith("fastvideo.")]:
        sys.modules.pop(name, None)
    # Drop the corresponding .pth-pointed paths from sys.path so the
    # editable repo doesn't shadow the explicit source.
    sys.path = [p for p in sys.path if not p.endswith(".egg-info")]
    sys.path.insert(0, src)


_bootstrap_fastvideo_source()


import numpy as np  # noqa: E402  (after bootstrap so torch picks up the right env)
import torch  # noqa: E402

# Pinned alignment fixture — matches basic_ltx2_upscale.py upstream.
PROMPT = (
    "A warm sunny backyard. The camera starts in a tight cinematic close-up "
    "of a woman and a man in their 30s, facing each other with serious "
    "expressions. The woman, emotional and dramatic, says softly, \"That's "
    "it... Dad's lost it. And we've lost Dad.\" The man exhales, slightly "
    "annoyed: \"Stop being so dramatic, Jess.\" A beat. He glances aside, "
    "then mutters defensively, \"He's just having fun.\" The camera slowly "
    "pans right, revealing the grandfather in the garden wearing enormous "
    "butterfly wings, waving his arms in the air like he's trying to take "
    "off. He shouts, \"Wheeeew!\" as he flaps his wings with full commitment. "
    "The woman covers her face, on the verge of tears. The tone is deadpan, "
    "absurd, and quietly tragic.")

DEFAULT_MODEL = "FastVideo/LTX2-Distilled-Diffusers"
DEFAULT_SEED = 10
DEFAULT_HEIGHT = 1088
DEFAULT_WIDTH = 1920
DEFAULT_FRAMES = 121
DEFAULT_FPS = 24
DEFAULT_BASE_STEPS = 8
DEFAULT_REFINE_STEPS = 3


@dataclass
class RunResult:
    label: str
    fastvideo_repo: str
    output_frames: torch.Tensor
    metadata: dict[str, Any]


def _detect_fastvideo_repo() -> str:
    import fastvideo  # noqa: PLC0415
    return os.path.realpath(os.path.dirname(fastvideo.__file__))


def _run_legacy(args: argparse.Namespace) -> RunResult:
    """Legacy ``VideoGenerator.from_pretrained(**flat_kwargs)`` path —
    used by the internal reference run because the internal API is
    still flat-kwarg-shaped."""
    from fastvideo import VideoGenerator  # noqa: PLC0415

    generator = VideoGenerator.from_pretrained(
        args.model,
        num_gpus=1,
        ltx2_refine_enabled=True,
        ltx2_refine_lora_path="",  # disable refine LoRA — distilled needs none
        ltx2_refine_num_inference_steps=args.refine_steps,
        ltx2_refine_guidance_scale=1.0,
        ltx2_refine_add_noise=True,
        dit_cpu_offload=False,
        vae_cpu_offload=False,
        text_encoder_cpu_offload=False,
        pin_cpu_memory=True,
        dit_layerwise_offload=False,
        enable_torch_compile=False,
        ltx2_vae_tiling=False,
    )

    # Pin the LTX-2-specific sampling knobs explicitly so both
    # internal and public runs use identical denoising mechanics —
    # otherwise the public LTX2_BASE preset's modality/rescale/stg
    # defaults (3.0 / 0.7 / 1.0) would diverge from internal's
    # ForwardBatch defaults (1.0 / 0.0 / 0.0).
    result = generator.generate_video(
        prompt=PROMPT,
        output_path=str(args.output) + ".legacy.mp4",
        fps=args.fps,
        seed=args.seed,
        num_inference_steps=args.base_steps,
        guidance_scale=1.0,
        save_video=False,
        return_frames=True,
        height=args.height,
        width=args.width,
        num_frames=args.frames,
        ltx2_cfg_scale_video=1.0,
        ltx2_cfg_scale_audio=1.0,
        ltx2_modality_scale_video=1.0,
        ltx2_modality_scale_audio=1.0,
        ltx2_rescale_scale=0.0,
        ltx2_stg_scale_video=0.0,
        ltx2_stg_scale_audio=0.0,
    )
    generator.shutdown()
    frames = _result_frames(result)
    return RunResult(
        label=args.label,
        fastvideo_repo=_detect_fastvideo_repo(),
        output_frames=frames,
        metadata={
            "api": "legacy_from_pretrained",
            "prompt": PROMPT,
            "seed": args.seed,
            "height": args.height,
            "width": args.width,
            "num_frames": args.frames,
            "fps": args.fps,
            "base_steps": args.base_steps,
            "refine_steps": args.refine_steps,
        },
    )


def _run_typed(args: argparse.Namespace) -> RunResult:
    """New typed public API: GeneratorConfig + GenerationRequest with
    pipeline.preset_overrides["refine"] driving the SR path."""
    from fastvideo import VideoGenerator  # noqa: PLC0415
    from fastvideo.api import (  # noqa: PLC0415
        ComponentConfig, EngineConfig, GenerationRequest, GeneratorConfig,
        InputConfig, OutputConfig, PipelineSelection, SamplingConfig)

    config = GeneratorConfig(
        model_path=args.model,
        engine=EngineConfig(num_gpus=1, ),
        pipeline=PipelineSelection(
            components=ComponentConfig(),
            preset_overrides={
                "refine": {
                    "enabled": True,
                    "add_noise": True,
                    "num_inference_steps": args.refine_steps,
                    "guidance_scale": 1.0,
                },
            },
        ),
    )

    generator = VideoGenerator.from_pretrained(config=config)

    # The typed SamplingConfig doesn't expose the LTX-2-specific
    # modality/rescale/stg knobs, so route them through experimental
    # so the legacy pipeline batch builder picks them up. This keeps
    # the typed run's denoising mechanics identical to the legacy
    # run's (and therefore to the internal reference).
    request = GenerationRequest(
        prompt=PROMPT,
        sampling=SamplingConfig(
            num_frames=args.frames,
            height=args.height,
            width=args.width,
            fps=args.fps,
            num_inference_steps=args.base_steps,
            guidance_scale=1.0,
            seed=args.seed,
        ),
        inputs=InputConfig(),
        output=OutputConfig(
            save_video=False,
            return_frames=True,
        ),
    )
    # Set the LTX-2 sampling overrides on the request via attribute so
    # the legacy SamplingParam translation picks them up alongside the
    # typed fields. (Until the typed SamplingConfig grows these
    # fields, this is the canonical override path on the public side.)
    for attr, value in {
            "ltx2_cfg_scale_video": 1.0,
            "ltx2_cfg_scale_audio": 1.0,
            "ltx2_modality_scale_video": 1.0,
            "ltx2_modality_scale_audio": 1.0,
            "ltx2_rescale_scale": 0.0,
            "ltx2_stg_scale_video": 0.0,
            "ltx2_stg_scale_audio": 0.0,
    }.items():
        setattr(request, attr, value)

    result = generator.generate(request)
    generator.shutdown()
    frames = _result_frames(result)
    return RunResult(
        label=args.label,
        fastvideo_repo=_detect_fastvideo_repo(),
        output_frames=frames,
        metadata={
            "api": "typed",
            "prompt": PROMPT,
            "seed": args.seed,
            "height": args.height,
            "width": args.width,
            "num_frames": args.frames,
            "fps": args.fps,
            "base_steps": args.base_steps,
            "refine_steps": args.refine_steps,
        },
    )


def _result_frames(result: Any) -> torch.Tensor:
    """Coerce whatever generate() returned into ``[F, H, W, C]`` uint8."""
    frames = getattr(result, "frames", None)
    if frames is None and isinstance(result, dict):
        frames = result.get("frames")
    if frames is None:
        raise RuntimeError(
            "Generation returned no frames; ensure return_frames=True.")
    if isinstance(frames, list):
        frames = np.stack(frames, axis=0)
    if isinstance(frames, np.ndarray):
        return torch.from_numpy(frames)
    if torch.is_tensor(frames):
        return frames.detach().cpu()
    raise TypeError(f"Unsupported frames container: {type(frames)!r}")


def _run(args: argparse.Namespace) -> None:
    if args.use_typed_api:
        result = _run_typed(args)
    else:
        result = _run_legacy(args)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "frames": result.output_frames,
            "metadata": result.metadata,
            "label": result.label,
            "fastvideo_repo": result.fastvideo_repo,
        },
        output_path,
    )
    print(
        f"[align] saved {result.output_frames.shape} frames "
        f"({result.fastvideo_repo}) -> {output_path}",
        flush=True,
    )


def _diff(args: argparse.Namespace) -> None:
    ref_path = Path(args.reference)
    cand_path = Path(args.candidate)
    if not ref_path.is_file() or not cand_path.is_file():
        raise FileNotFoundError(
            f"Need both --reference and --candidate to exist: {ref_path}, "
            f"{cand_path}")
    ref = torch.load(ref_path, map_location="cpu")
    cand = torch.load(cand_path, map_location="cpu")

    ref_frames = ref["frames"]
    cand_frames = cand["frames"]
    print(f"[align] reference: {ref['label']} {ref_frames.shape} from "
          f"{ref['fastvideo_repo']}")
    print(f"[align] candidate: {cand['label']} {cand_frames.shape} from "
          f"{cand['fastvideo_repo']}")

    if ref_frames.shape != cand_frames.shape:
        print(f"[align] shape MISMATCH: ref={tuple(ref_frames.shape)} "
              f"vs cand={tuple(cand_frames.shape)}")
        return

    diff = (ref_frames.float() - cand_frames.float()).abs()
    max_abs = float(diff.max().item())
    mean_abs = float(diff.mean().item())
    rms = float(torch.sqrt((diff**2).mean()).item())

    # PSNR over uint8 [0,255] data range.
    if rms == 0.0:
        psnr = float("inf")
    else:
        psnr = 20.0 * float(torch.log10(torch.tensor(255.0 / rms)).item())

    print("[align] frame-level uint8 metrics (per-pixel):")
    print(f"        max_abs_diff = {max_abs:.4f}")
    print(f"        mean_abs_diff = {mean_abs:.6f}")
    print(f"        rms_diff = {rms:.4f}")
    print(f"        psnr_db = {psnr:.2f}")

    # Bucket pixels by exactness so we have a sense of how close we are.
    eq = (diff == 0).float().mean().item()
    within_1 = (diff <= 1).float().mean().item()
    within_2 = (diff <= 2).float().mean().item()
    within_4 = (diff <= 4).float().mean().item()
    print("[align] pixel buckets:")
    print(f"        exact = {eq * 100:.2f}%")
    print(f"        within 1 = {within_1 * 100:.2f}%")
    print(f"        within 2 = {within_2 * 100:.2f}%")
    print(f"        within 4 = {within_4 * 100:.2f}%")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="mode", required=False)

    run = parser
    run.add_argument("--label",
                     default="run",
                     help="Identifier saved alongside the output (e.g. internal/public).")
    run.add_argument("--output",
                     default="/tmp/ltx2_sr_alignment.pt",
                     help="Destination .pt file.")
    run.add_argument("--use-typed-api",
                     action="store_true",
                     help="Use the new typed GeneratorConfig API "
                          "(public side); otherwise use legacy from_pretrained.")
    run.add_argument("--model", default=DEFAULT_MODEL)
    run.add_argument("--seed", type=int, default=DEFAULT_SEED)
    run.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    run.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    run.add_argument("--frames", type=int, default=DEFAULT_FRAMES)
    run.add_argument("--fps", type=int, default=DEFAULT_FPS)
    run.add_argument("--base-steps", type=int, default=DEFAULT_BASE_STEPS)
    run.add_argument("--refine-steps", type=int, default=DEFAULT_REFINE_STEPS)

    run.add_argument("--diff",
                     action="store_true",
                     help="Diff two .pt outputs instead of running.")
    run.add_argument("--reference")
    run.add_argument("--candidate")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.diff:
        if not (args.reference and args.candidate):
            parser.error("--diff requires --reference and --candidate")
        _diff(args)
        return 0
    _run(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
