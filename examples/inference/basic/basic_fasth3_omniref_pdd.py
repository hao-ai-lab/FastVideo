# SPDX-License-Identifier: Apache-2.0
"""Eight-forward Ref2VA video+audio generation with a FastH3 OmniRef PDD student.

A FastH3 OmniRef PDD export is a Parallel Decoding Distillation (PDD) student of
MiniMax-H3's reference-conditioned ``transformer_ref`` partition. Its directory
(local, or a Hugging Face repo) carries only what distillation changed:
``transformer_ref/`` (output heads widened to the fine grid, trained VSA
compression gates), the two scheduler configs, and ``fastvideo_inference.json``.
The text encoder, tokenizer, processor, and both VAEs are base MiniMax-H3's.

This script composes the two into one local model directory of symlinks and
runs the recipe the contract records: fused blocks of the fine grid (one
transformer forward each), the video/audio shifts, and VIDEO_SPARSE_ATTN_H3
with its sparsity and tile size, where every reference video is its own
sparse region. The contract's 128-token tiles run only on the sm_100a/sm_103a
CUDA kernel (B200/B300/GB200/GB300) of a fastvideo-kernel build with the
Blackwell VSA extension; the VSA-H3 backend raises at the first attention call
when either is missing.
With ``--num-gpus`` above 1 the DiT is sharded across the GPUs (FSDP) and the
sequence is split across them (sequence parallelism).

References are ordered. Pass them in order with --image / --video / --audio,
for example ``--video dance.mp4 --image outfit.png --audio voice.wav``.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from fastvideo import VideoGenerator
from fastvideo.api import (
    ComponentConfig,
    EngineConfig,
    GenerationRequest,
    GeneratorConfig,
    InputConfig,
    OffloadConfig,
    OutputConfig,
    ParallelismConfig,
    PipelineSelection,
    SamplingConfig,
)
from fastvideo.configs.pipelines.minimax_h3 import parse_base_model_revision
from fastvideo.pipelines.basic.minimax_h3 import MiniMaxH3Reference

CONTRACT = "fastvideo_inference.json"
# Components the distilled export replaces, and the ones it shares with base MiniMax-H3.
EXPORT_COMPONENTS = ("transformer_ref", "scheduler", "audio_scheduler")
BASE_COMPONENTS = ("text_encoder", "tokenizer", "processor", "vae", "audio_vae")
MANIFESTS = ("modular_model_index.json", "model_index.json")
# Components the composed directory's manifest must declare, besides its transformer_ref.
MANIFEST_COMPONENTS = ("scheduler", "audio_scheduler", *BASE_COMPONENTS)


class _AppendReference(argparse.Action):
    """Collect --image/--video/--audio into one list that keeps command-line order."""

    def __call__(self, parser, namespace, value, option_string=None):
        references = list(getattr(namespace, self.dest, None) or [])
        references.append((self.const, value))
        setattr(namespace, self.dest, references)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-path",
                        required=True,
                        help="FastH3 OmniRef PDD export: a local directory or a Hugging Face repo id")
    parser.add_argument("--revision", default=None, help="revision of --model-path when it is a Hugging Face repo")
    parser.add_argument("--base-model-path",
                        default=None,
                        help="base MiniMax-H3 snapshot (local directory or repo id). Default: the repo and revision "
                        "pinned by the export's base_model_revision")
    parser.add_argument("--base-revision",
                        default=None,
                        help="revision of --base-model-path when it is a repo id (the export's pinned revision "
                        "applies only to its own base repo); without --base-model-path, overrides that pin")
    parser.add_argument("--composed-dir",
                        default=None,
                        help="where to write the composed model directory of symlinks (default: "
                        "<output>/fasth3_omniref_model)")
    for flag in ("image", "video", "audio"):
        parser.add_argument(f"--{flag}",
                            dest="references",
                            action=_AppendReference,
                            const=flag,
                            metavar="PATH",
                            help=f"an ordered {flag} reference (repeatable)")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", default="outputs/fasth3_omniref_pdd")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--num-frames", type=int, default=124)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-gpus", type=int, default=1, help="sequence-parallel GPUs (must divide 56 heads)")
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.references:
        parser.error("pass at least one ordered reference with --image, --video, or --audio")
    if not any(kind != "audio" for kind, _ in args.references):
        parser.error("Ref2VA needs at least one image or video reference")
    return args


def _snapshot(path_or_repo: str, revision: str | None, allow_patterns: list[str]) -> Path:
    local = Path(path_or_repo).expanduser()
    if local.is_dir():
        return local
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo_id=path_or_repo, revision=revision, allow_patterns=allow_patterns))


def load_contract(export_dir: Path) -> dict[str, Any]:
    path = export_dir / CONTRACT
    if not path.is_file():
        raise FileNotFoundError(f"{export_dir} has no {CONTRACT}; it is not a FastH3 distilled export.")
    contract = json.loads(path.read_text(encoding="utf-8"))
    if "pdd_steps" not in contract or contract.get("model_type") != "ref2va":
        raise ValueError(f"{path} is not a Ref2VA PDD contract; use the example matching the checkpoint.")
    return contract


def base_model_source(args: argparse.Namespace, contract: dict[str, Any]) -> tuple[str, str | None]:
    """The base snapshot and revision: --base-model-path as given, else the export's pin."""
    base_repo, base_revision = base_model_from_contract(contract)
    if args.base_model_path:
        return args.base_model_path, args.base_revision
    return base_repo, args.base_revision or base_revision


def base_model_from_contract(contract: dict[str, Any]) -> tuple[str, str]:
    """The base repo id and revision that the export pins as ``hf://<repo id>@<revision>``."""
    return parse_base_model_revision(contract.get("base_model_revision"))


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


def _declares_components(manifest: Path) -> bool:
    """Whether a Diffusers manifest declares every component of the composed directory."""
    declared = {
        name
        for name, spec in json.loads(manifest.read_text(encoding="utf-8")).items()
        if isinstance(spec, list) and spec and spec[0] is not None
    }
    return set(MANIFEST_COMPONENTS) <= declared and bool({"transformer", "transformer_ref"} & declared)


def select_manifest(export_dir: Path, base_dir: Path) -> Path:
    """The export's manifest when it declares every composed component, else the base's."""
    for root in (export_dir, base_dir):
        for name in MANIFESTS:
            if (root / name).is_file() and _declares_components(root / name):
                return root / name
    raise FileNotFoundError("Neither the export nor the base snapshot has a Diffusers model manifest that declares "
                            f"{', '.join(MANIFEST_COMPONENTS)} and transformer_ref.")


def compose_model_dir(export_dir: Path, base_dir: Path, composed_dir: Path) -> Path:
    """One MiniMax-H3 model directory: distilled components from the export, the rest from the base."""
    composed_dir.mkdir(parents=True, exist_ok=True)
    for name in (*EXPORT_COMPONENTS, CONTRACT):
        _link(composed_dir / name, export_dir / name)
    manifest_source = select_manifest(export_dir, base_dir)
    for name in MANIFESTS:
        stale = composed_dir / name
        if name != manifest_source.name and stale.is_symlink():
            stale.unlink()
    _link(composed_dir / manifest_source.name, manifest_source)
    for name in BASE_COMPONENTS:
        _link(composed_dir / name, base_dir / name)
    return composed_dir


def resolve_model(args: argparse.Namespace) -> tuple[Path, dict[str, Any]]:
    export_dir = _snapshot(args.model_path, args.revision,
                           [CONTRACT, *MANIFESTS, *(f"{name}/**" for name in EXPORT_COMPONENTS)])
    contract = load_contract(export_dir)
    base_source, base_revision = base_model_source(args, contract)
    base_dir = _snapshot(base_source, base_revision, [*MANIFESTS, *(f"{name}/**" for name in BASE_COMPONENTS)])
    composed_dir = Path(args.composed_dir) if args.composed_dir else Path(args.output) / "fasth3_omniref_model"
    return compose_model_dir(export_dir, base_dir, composed_dir), contract


def build_generator_config(model_dir: Path, num_gpus: int) -> GeneratorConfig:
    return GeneratorConfig(
        model_path=str(model_dir),
        engine=EngineConfig(
            num_gpus=num_gpus,
            # Shard the DiT across the GPUs rather than holding a full copy on each.
            use_fsdp_inference=num_gpus > 1,
            parallelism=ParallelismConfig(tp_size=1, sp_size=num_gpus),
            offload=OffloadConfig(dit=False, dit_layerwise=False, text_encoder=True, vae=True, pin_cpu_memory=False),
        ),
        pipeline=PipelineSelection(
            workload_type="i2v",
            # FastVideo reads the trained fused-block partition and attention
            # settings from the composed directory's fastvideo_inference.json.
            components=ComponentConfig(override_pipeline_cls_name="MiniMaxH3Ref2VAModularPipeline"),
        ),
    )


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir, contract = resolve_model(args)
    print(f"Composed model directory: {model_dir}")
    print(f"Contract: {contract['num_inference_steps']} fused blocks of a {contract['pdd_steps']}-interval grid, "
          f"{contract['attention_backend']} with sparsity {contract['vsa_sparsity']}, "
          f"{contract['vsa_tile_size']}-token tiles, reference keep rate {contract['vsa_ref_keep_rate']}")
    references = [MiniMaxH3Reference(source=path, media_type=kind) for kind, path in args.references]

    generator = VideoGenerator.from_config(build_generator_config(model_dir, args.num_gpus))
    try:
        result = generator.generate(
            GenerationRequest(
                prompt=args.prompt,
                negative_prompt="",
                inputs=InputConfig(references=references),
                sampling=SamplingConfig(
                    height=args.height,
                    width=args.width,
                    num_frames=args.num_frames,
                    fps=24,
                    # PDD counts fused blocks: one transformer forward each.
                    num_inference_steps=contract["num_inference_steps"],
                    guidance_scale=1.0,
                    batch_cfg=False,
                    seed=args.seed,
                ),
                output=OutputConfig(
                    output_path=str(output_dir / "fasth3_omniref_pdd.mp4"),
                    save_video=True,
                    return_frames=False,
                ),
            ))
        print(f"Output written to: {result.video_path}")
        if result.generation_time is not None:
            print(f"Generation time: {result.generation_time:.1f} s")
        if result.peak_memory_mb is not None:
            print(f"Peak GPU memory allocated (rank 0): {result.peak_memory_mb / 1024:.1f} GiB")
    finally:
        generator.shutdown()


if __name__ == "__main__":
    main()
