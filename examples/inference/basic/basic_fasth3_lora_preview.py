# SPDX-License-Identifier: Apache-2.0
"""Run a FastH3 four-step Preview LoRA with the measured FastVideo defaults.

This is the LoRA counterpart of ``basic_fasth3.py``. Both routes share the
same performance profile: four DiT forwards, regional fullgraph DiT compile,
H3 fusions, compiled and sequence-parallel video VAE decode, replicated DiT,
pinned CPU offload, FA4, and the sm100a tile-64 kernel for VSA adapters.

The FastH3 adapters include low-rank factors plus exact dense deltas. Some also
provide the VSA compression gate that is absent from the base checkpoint. Pass
the adapter at construction so all three payload types receive the same
``--lora-strength``. The attention backend is inferred from that payload unless
``--vsa`` or ``--no-vsa`` is specified explicitly.
"""

from __future__ import annotations

import argparse
import math
from collections.abc import Sequence

try:
    from . import basic_fasth3
except ImportError:
    # Direct script execution puts this directory, rather than ``examples``, on
    # sys.path. Keep both ``python file.py`` and module/importlib use working.
    import basic_fasth3  # type: ignore[no-redef]

BASE_MODEL = "MiniMaxAI/MiniMax-H3"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = basic_fasth3.build_parser(description=__doc__)
    parser.set_defaults(model_path=BASE_MODEL, output="outputs/fasth3_lora_preview")
    parser.add_argument(
        "--lora-path",
        required=True,
        help="FastH3 adapter safetensors file or local adapter directory",
    )
    parser.add_argument(
        "--lora-strength",
        type=float,
        default=1.0,
        help="adapter strength; 0 zeros its weights but keeps its backend, and 1 applies its published scale",
    )
    parser.add_argument(
        "--vsa",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="select VSA explicitly; by default it is inferred from the adapter's compression-gate payload",
    )
    args = basic_fasth3.validate_args(parser, parser.parse_args(argv))
    if not math.isfinite(args.lora_strength):
        parser.error("--lora-strength must be finite")
    return _resolve_attention_backend(parser, args)


def _resolve_attention_backend(parser: argparse.ArgumentParser, args: argparse.Namespace) -> argparse.Namespace:
    # Header-only inspection keeps the payload on disk. A replacement compression
    # gate is an unambiguous VSA requirement; adapters without one default to dense.
    from fastvideo.models.loader.lora_patch import DenseLoRAPatch

    patch = DenseLoRAPatch.from_adapter(args.lora_path, strength=args.lora_strength)
    needs_vsa = bool(patch and any("gate_compress" in name for name in patch.replacement_parameters))
    if args.vsa is None:
        args.vsa = needs_vsa
    elif needs_vsa and not args.vsa:
        parser.error(f"{args.lora_path} provides to_gate_compress and must run with VSA; drop --no-vsa")
    return args


def main() -> None:
    args = parse_args()
    print(f"FastH3 adapter: {args.lora_path}")
    print(f"LoRA strength: {args.lora_strength:g}")
    print(f"Attention: {'VSA-H3' if args.vsa else 'dense FA4'}")
    basic_fasth3.run(args)


if __name__ == "__main__":
    main()
