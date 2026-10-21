# SPDX-License-Identifier: Apache-2.0
"""Reference-conditioned FastH3 OmniRef (Ref2VA) generation on Apple Silicon MLX.

A FastH3 OmniRef PDD export carries only what distillation changed:
``transformer_ref/``, the two scheduler configs and ``fastvideo_inference.json``.
The text encoder, tokenizer, processor and both VAEs are base MiniMax-H3's, so
this script composes the two into one directory of symlinks - the same shape the
CUDA example builds - and then runs the MLX runtime over it.

There is no FP4 on Apple Silicon, so the DiT converts to INT8 (or INT6/INT4),
the same formats the text-to-video-audio MLX path uses. The DiT has to be in
the packed MLX layout first:

    python scripts/checkpoint_conversion/convert_minimax_h3_mlx.py \\
      --model-root <composed>/transformer_ref --out ~/models/FastH3-OmniRef-MLX \\
      --formats int8 --include-vsa

    python examples/inference/basic/mlx_fasth3_omniref.py \\
      --export ./FastVideo-FastH3-Omni-8-Step-V1 --composed-dir ./FastH3-OmniRef-composed \\
      --base-model-path ./MiniMax-H3 --mlx-checkpoint ~/models/FastH3-OmniRef-MLX/int8 \\
      --image reference.png --audio voice.wav \\
      --prompt "a person on a sofa, lit by a warm lamp, speaking to camera" \\
      --output-path ~/fasth3_outputs/omniref.mp4

References are ordered: pass them in the order the model should read them, with
``--image`` / ``--video`` / ``--audio`` (for example ``--video dance.mp4
--image outfit.png --audio voice.wav``). At least one image or video reference
is required; audio references only accompany them.

Add ``--compose-only`` to build the composed directory (the conversion above
needs it) and stop, with no checkpoint, prompt or reference.

The PDD ladder comes from the export's contract, so ``--steps`` is optional:
without it the trained fused-block count runs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

CONTRACT = "fastvideo_inference.json"
MANIFESTS = ("modular_model_index.json", "model_index.json")
# Components the distilled export replaces, and the ones it shares with base MiniMax-H3.
EXPORT_COMPONENTS = ("transformer_ref", "scheduler", "audio_scheduler")
BASE_COMPONENTS = ("text_encoder", "tokenizer", "processor", "vae", "audio_vae")


class _AppendReference(argparse.Action):
    """Collect --image/--video/--audio into one list that keeps command-line order."""

    def __call__(self, parser, namespace, value, option_string=None):
        references = list(getattr(namespace, self.dest, None) or [])
        references.append((self.const, value))
        setattr(namespace, self.dest, references)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--export", type=Path,
                        help="FastH3 OmniRef PDD export directory; composes it with --base-model-path")
    parser.add_argument("--base-model-path", type=Path,
                        help="base MiniMax-H3 snapshot (vae/, audio_vae/, text_encoder/, tokenizer/)")
    parser.add_argument("--composed-dir", type=Path, default=Path("outputs/fasth3_omniref_mlx_model"),
                        help="where the composed directory of symlinks is written")
    parser.add_argument("--model-root", type=Path,
                        help="an already-composed directory; skips composing")
    parser.add_argument("--compose-only", action="store_true",
                        help="compose --export with --base-model-path into --composed-dir and exit")
    parser.add_argument("--mlx-checkpoint", type=Path,
                        help="packed MLX checkpoint converted from the composed transformer_ref/")
    for flag in ("image", "video", "audio"):
        parser.add_argument(f"--{flag}", dest="references", action=_AppendReference, const=flag, metavar="PATH",
                            help=f"an ordered {flag} reference (repeatable)")
    parser.add_argument("--prompt")
    parser.add_argument("--output-path", type=Path, default=Path("outputs/mlx_fasth3_omniref.mp4"))
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--num-frames", type=int, default=124)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--steps", type=int, default=0,
                        help="transformer forwards; 0 uses the export contract's trained count")
    parser.add_argument("--prompt-cache-dir", type=Path)
    parser.add_argument("--vae-dtype", default="fp32")
    parser.add_argument("--vae-decoder", type=Path,
                        help="standalone video decoder (FastH3 decoder[-nvfp4].safetensors or a LightVAE decoder) "
                        "replacing the base vae/ decoder")
    parser.add_argument("--vae-encoder", type=Path,
                        help="standalone video encoder (LightVAE) replacing the base vae/ encoder; with both flags "
                        "vae/ only needs config.json")
    parser.add_argument("--vae-decoder-quantization", choices=("nvfp4", "mxfp8", "int8", "int4"),
                        help="quantize the decoder block linears of a dense decoder file as it loads")
    parser.add_argument("--vsa", action=argparse.BooleanOptionalAction, default=None,
                        help="VSA sparse attention (needs a checkpoint converted with --include-vsa). "
                        "On by default when the contract is a VSA contract; --no-vsa runs dense.")
    parser.add_argument("--vsa-sparsity", type=float, default=None,
                        help="generated-video sparsity; default is the contract's trained value")
    parser.add_argument("--vsa-tile-size", type=int, default=None,
                        help="VSA tile: 64, 128 or 256. Must equal the contract's trained tile; "
                        "default is the contract's.")
    parser.add_argument("--vsa-ref-keep-rate", type=float, default=None,
                        help="fraction of each reference video's tiles a query keeps (p2_multi_region); "
                        "default is the contract's trained value")
    parser.add_argument("--inter-step-cooldown-s", type=float, default=0.0)
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.compose_only:
        if args.export is None or args.base_model_path is None:
            parser.error("--compose-only needs --export and --base-model-path")
        return args
    if args.mlx_checkpoint is None or args.prompt is None:
        parser.error("--mlx-checkpoint and --prompt are required unless --compose-only is set")
    if not args.references:
        parser.error("pass at least one ordered reference with --image, --video, or --audio")
    if not any(kind != "audio" for kind, _ in args.references):
        parser.error("Ref2VA needs at least one image or video reference")
    if args.model_root is None and (args.export is None or args.base_model_path is None):
        parser.error("give --model-root, or --export together with --base-model-path")
    return args


def _link(destination: Path, source: Path) -> None:
    if not source.exists():
        raise FileNotFoundError(f"Missing checkpoint component: {source}")
    source = source.resolve()
    if destination.is_symlink():
        if destination.resolve() == source:
            return
        destination.unlink()
    elif destination.exists():
        raise FileExistsError(f"{destination} exists and is not a symlink; choose another --composed-dir.")
    destination.symlink_to(source, target_is_directory=source.is_dir())


def compose_model_dir(export_dir: Path, base_dir: Path, composed_dir: Path) -> Path:
    """One MiniMax-H3 model directory: distilled components from the export, the rest from the base."""
    contract = export_dir / CONTRACT
    if not contract.is_file():
        raise FileNotFoundError(f"{export_dir} has no {CONTRACT}; it is not a FastH3 distilled export.")
    composed_dir.mkdir(parents=True, exist_ok=True)
    for name in (*EXPORT_COMPONENTS, CONTRACT):
        _link(composed_dir / name, export_dir / name)
    for name in MANIFESTS:
        for root in (export_dir, base_dir):
            if (root / name).is_file():
                _link(composed_dir / name, root / name)
                break
    for name in BASE_COMPONENTS:
        _link(composed_dir / name, base_dir / name)
    return composed_dir


def load_contract(export_dir: Path) -> dict[str, Any]:
    """The export's PDD contract, validated as a Ref2VA one."""
    path = export_dir / CONTRACT
    if not path.is_file():
        raise FileNotFoundError(f"{export_dir} has no {CONTRACT}")
    contract = json.loads(path.read_text(encoding="utf-8"))
    if contract.get("model_type") != "ref2va":
        raise ValueError(f"{path} is not a Ref2VA contract; use the example matching the checkpoint.")
    return contract


def resolve_model_root(args: argparse.Namespace) -> Path:
    if args.model_root is not None:
        return args.model_root
    return compose_model_dir(args.export, args.base_model_path, args.composed_dir)


def run(args: argparse.Namespace) -> None:
    if args.compose_only:
        print(f"Composed model directory: {resolve_model_root(args)}")
        return
    from fastvideo.mlx_runtime.minimax_h3_pipeline import MiniMaxH3MLXPipeline
    from fastvideo.mlx_runtime.minimax_h3_reference import MiniMaxH3Reference

    model_root = resolve_model_root(args)
    export_dir = args.export if args.export is not None else model_root
    contract = load_contract(export_dir)
    if args.export is not None:
        print(f"Composed model directory: {model_root}")
        print(f"Contract: {contract['num_inference_steps']} fused blocks of a {contract['pdd_steps']}-interval grid, "
              f"{contract['attention_backend']} with sparsity {contract['vsa_sparsity']}, "
              f"{contract['vsa_tile_size']}-token tiles")
    references = [MiniMaxH3Reference(source=path, media_type=kind) for kind, path in args.references]
    args.output_path.parent.mkdir(parents=True, exist_ok=True)

    pipeline = MiniMaxH3MLXPipeline(
        model_root=model_root,
        mlx_dit_checkpoint=args.mlx_checkpoint,
        vae_dtype=args.vae_dtype,
        vae_decoder_path=args.vae_decoder,
        vae_encoder_path=args.vae_encoder,
        vae_decoder_quantization=args.vae_decoder_quantization,
        prompt_cache_dir=args.prompt_cache_dir,
    )
    try:
        kwargs: dict[str, Any] = {}
        if args.steps:
            kwargs["num_steps"] = args.steps
        elif "transformer_forwards" in contract:
            # The contract's fused-block count is the ladder the student was distilled on.
            kwargs["num_steps"] = int(contract["transformer_forwards"])
        result = pipeline.generate(
            args.prompt,
            output_path=args.output_path,
            references=references,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            seed=args.seed,
            vsa=args.vsa,
            vsa_sparsity=args.vsa_sparsity,
            vsa_tile_size=args.vsa_tile_size,
            vsa_ref_keep_rate=args.vsa_ref_keep_rate,
            inter_step_cooldown_s=args.inter_step_cooldown_s,
            **kwargs,
        )
    finally:
        pipeline.close()
    print(json.dumps({
        "video_path": result.video_path,
        "references": [kind for kind, _ in args.references],
        "timings_s": {key: round(value, 2) for key, value in result.timings.items()},
        "peak_memory_gib": {key: round(value, 2) for key, value in result.peak_memory_gib.items()},
        "vsa": result.vsa,
        "audio_samples": int(result.waveform.shape[-1]),
    }, indent=2))


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
