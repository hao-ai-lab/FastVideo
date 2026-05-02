# SPDX-License-Identifier: Apache-2.0
"""Convert daVinci-MagiHuman (GAIR-NLP) weights to a Diffusers-format repo.

MagiHuman publishes weights in a raw layout on HuggingFace
(https://huggingface.co/GAIR/daVinci-MagiHuman). The layout is:

    base/                       <- DiT safetensors (sharded)
    distill/                    <- distilled DiT (out of scope for the base port)
    540p_sr/, 1080p_sr/         <- super-resolution DiTs (out of scope)
    turbo_vae/                  <- optional fast VAE decoder (out of scope for first cut)

The base DiT uses Wan-AI/Wan2.2-TI2V-5B's VAE and google/t5gemma-9b-9b-ul2's
encoder at inference time; neither is bundled upstream.

This converter takes the raw MagiHuman base DiT and emits a Diffusers-style
directory so `VideoGenerator.from_pretrained(...)` can load it standalone:

    <output>/
        model_index.json
        transformer/
            config.json
            diffusion_pytorch_model-00001-of-00N.safetensors (+ index)
        scheduler/
            scheduler_config.json                    (FlowUniPC default)
        vae/                                         (optional; --bundle-vae)
        audio_vae/                                   (optional; --bundle-audio-vae)
        text_encoder/, tokenizer/                    (optional; --bundle-text-encoder)

By default the converted repo is MINIMAL: only `transformer/`,
`scheduler/`, and `model_index.json` are emitted (~5-30 GB depending on
variant). The four cross-variant shared components — Wan VAE, Stable
Audio VAE, T5-Gemma encoder, and tokenizer — are lazy-loaded by
`MagiHumanPipeline.load_modules` from their canonical upstream HF repos
on first build, so all MagiHuman variants share a single ~25 GB cache
of upstream weights. Pass the `--bundle-*` flags only if you want to
ship a self-contained snapshot.

The DiT key names pass through unchanged — the FastVideo `MagiHumanDiT` module
mirrors the reference module tree (`adapter.*`, `block.layers.*`, `final_*`),
so no regex remapping is needed. The conversion is effectively a reshard +
Diffusers wrapper.

    Example (minimal artifact, ~5-30 GB):
    python scripts/checkpoint_conversion/convert_magi_human_to_diffusers.py \\
        --source GAIR/daVinci-MagiHuman \\
        --subfolder base \\
        --output converted_weights/magi_human_base

Example (self-contained SR-540p artifact with base + SR DiTs):
    python scripts/checkpoint_conversion/convert_magi_human_to_diffusers.py \
        --source GAIR/daVinci-MagiHuman \
        --subfolder base \
        --sr-source GAIR/daVinci-MagiHuman \
        --sr-subfolder 540p_sr \
        --output converted_weights/magi_human_sr_540p

Example (self-contained snapshot with shared components bundled):
    python scripts/checkpoint_conversion/convert_magi_human_to_diffusers.py \\
        --source GAIR/daVinci-MagiHuman \\
        --subfolder base \\
        --output converted_weights/magi_human_base \\
        --bundle-vae --bundle-audio-vae --bundle-text-encoder
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import OrderedDict
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download, snapshot_download
from safetensors.torch import load_file, save_file


# MagiHuman base arch — keys must be valid `MagiHumanArchConfig` fields.
# FastVideo's `TransformerLoader.load` calls `ArchConfig.update_model_arch`
# with this dict (minus `_class_name`, `_diffusers_version`) and rejects
# any key that isn't a declared field. Pipeline-level knobs (steps, CFG,
# guidance scales, flow_shift) live on `MagiHumanBaseConfig` and do NOT
# belong here — they'd silently shadow the ArchConfig loader otherwise.
MAGI_HUMAN_BASE_ARCH: dict = {
    "_class_name": "MagiHumanDiT",
    "_diffusers_version": "0.33.0",
    # Transformer shape (upstream `ModelConfig`, `inference/common/config.py`).
    "num_layers": 40,
    "hidden_size": 5120,
    "head_dim": 128,
    "num_query_groups": 8,
    # Modality channels.
    "video_in_channels": 192,            # 48 (VAE z_dim) * patch_size product 1*2*2
    "audio_in_channels": 64,
    "text_in_channels": 3584,            # T5Gemma-9B encoder hidden size
    # Block-level switches.
    "mm_layers": [0, 1, 2, 3, 36, 37, 38, 39],
    "local_attn_layers": [],
    "gelu7_layers": [0, 1, 2, 3],
    "post_norm_layers": [],
    "enable_attn_gating": True,
    "activation_type": "swiglu7",
    # DiT patching / positional.
    "patch_size": [1, 2, 2],
    "spatial_rope_interpolation": "extra",
    # TReAD (flattened; upstream nests as `tread_config`).
    "tread_selection_rate": 0.5,
    "tread_start_layer_idx": 2,
    "tread_end_layer_idx": 25,
}


SCHEDULER_CONFIG: dict = {
    "_class_name": "FlowUniPCMultistepScheduler",
    "_diffusers_version": "0.33.0",
    "num_train_timesteps": 1000,
    "solver_order": 2,
    "prediction_type": "flow_prediction",
    "shift": 5.0,
    "predict_x0": True,
    "solver_type": "bh2",
    "lower_order_final": True,
    "disable_corrector": [],
    "flow_shift": 5.0,
}


MAX_SHARD_BYTES = 5 * 1024 * 1024 * 1024  # 5 GB shards, matches HF defaults


def _download_dit_shards(source: Path | str, subfolder: str = "base") -> list[Path]:
    """Return local paths to all safetensors shards for the DiT."""
    source = str(source)
    if os.path.isdir(source):
        shard_dir = Path(source) / subfolder
        shards = sorted(shard_dir.glob("*.safetensors"))
        if not shards:
            raise FileNotFoundError(f"No safetensors under {shard_dir}")
        return shards

    # Remote HF repo — pull just the base subfolder.
    local_dir = snapshot_download(
        repo_id=source,
        allow_patterns=[f"{subfolder}/*.safetensors", f"{subfolder}/*.json"],
    )
    shard_dir = Path(local_dir) / subfolder
    return sorted(shard_dir.glob("*.safetensors"))


def _load_all_shards(
    shards: list[Path],
    cast_bf16: bool = False,
) -> "OrderedDict[str, torch.Tensor]":
    """Load all safetensors shards into a single state dict.

    When `cast_bf16` is True, fp32 tensors whose names match the transformer
    core (attention / mlp / final_linear_* / adapter.{video,text,audio}_embedder)
    are cast to bfloat16. fp32 is preserved for norms, rope bands, and any
    other tensor where precision matters. This is the right default for
    the distill checkpoint, which upstream ships as fp32 master weights
    (61 GB) — casting yields a 30 GB Diffusers artifact that matches the
    base checkpoint format.
    """
    # Tensors that must stay in float32 regardless of cast_bf16. These are
    # the dtypes that appear as fp32 in the BASE checkpoint, which is the
    # ground-truth shape of a "runtime-loadable" MagiHuman repo. The list
    # includes:
    #   - all RMSNorm weights (norms always run fp32 in upstream
    #     MultiModalityRMSNorm and FV's mirror)
    #   - the rope band buffer
    #   - the adapter embedders (video/text/audio: weight + bias) which
    #     upstream's Adapter declares as `dtype=torch.float32` and FV's
    #     MagiAdapter mirrors at `magi_human.py:519-527`
    #   - the final_linear_{video,audio} heads which upstream/FV both
    #     declare as `dtype=torch.float32` (`magi_human.py:645-648`,
    #     `dit_module.py:896-900`)
    # Forgetting any of these makes `--cast-bf16` lossy for the distill
    # checkpoint (which ships everything as fp32) and produces parity
    # drift vs upstream that base does not exhibit (because base already
    # ships with the right mixed-dtype layout).
    _FP32_KEEP_SUFFIXES = (
        ".pre_norm.weight",
        ".q_norm.weight",
        ".k_norm.weight",
        ".attn_post_norm.weight",
        ".mlp_post_norm.weight",
        "final_norm_video.weight",
        "final_norm_audio.weight",
        "final_linear_video.weight",
        "final_linear_audio.weight",
        "adapter.video_embedder.weight",
        "adapter.video_embedder.bias",
        "adapter.text_embedder.weight",
        "adapter.text_embedder.bias",
        "adapter.audio_embedder.weight",
        "adapter.audio_embedder.bias",
        "adapter.rope.bands",
    )
    _FP32_KEEP_FULL = {"adapter.rope.bands"}

    def _keep_fp32(k: str) -> bool:
        if k in _FP32_KEEP_FULL:
            return True
        return any(k.endswith(s) for s in _FP32_KEEP_SUFFIXES)

    state: OrderedDict[str, torch.Tensor] = OrderedDict()
    for shard in shards:
        piece = load_file(str(shard))
        for k, v in piece.items():
            if k in state:
                raise RuntimeError(f"Duplicate key across shards: {k}")
            if cast_bf16 and v.dtype == torch.float32 and not _keep_fp32(k):
                v = v.to(torch.bfloat16)
            state[k] = v
        print(f"  loaded {shard.name} ({len(piece)} tensors)")
    return state


def _validate_state(state: dict[str, torch.Tensor]) -> None:
    """Sanity-check required top-level modules are present."""
    required_prefixes = (
        "adapter.video_embedder.",
        "adapter.text_embedder.",
        "adapter.audio_embedder.",
        "adapter.rope.bands",
        "final_norm_video.",
        "final_norm_audio.",
        "final_linear_video.",
        "final_linear_audio.",
    )
    for pref in required_prefixes:
        if not any(k.startswith(pref) for k in state):
            raise RuntimeError(f"Missing expected key prefix: {pref}")
    # Layer count
    layer_ids = {int(k.split(".")[2]) for k in state if k.startswith("block.layers.")}
    if layer_ids != set(range(40)):
        raise RuntimeError(f"Expected layers 0..39, got {sorted(layer_ids)}")


def _shard_state_dict(
    state: dict[str, torch.Tensor],
    max_bytes: int = MAX_SHARD_BYTES,
) -> tuple[list[dict[str, torch.Tensor]], dict[str, str]]:
    """Greedy shard-packing: produce N shards of <= max_bytes, plus index."""
    shards: list[dict[str, torch.Tensor]] = []
    index: dict[str, str] = {}
    cur: dict[str, torch.Tensor] = {}
    cur_bytes = 0
    shard_idx = 0
    total = len(state)
    for k, v in state.items():
        t_bytes = v.numel() * v.element_size()
        if cur and cur_bytes + t_bytes > max_bytes:
            shards.append(cur)
            cur = {}
            cur_bytes = 0
            shard_idx += 1
        cur[k] = v
        cur_bytes += t_bytes
    if cur:
        shards.append(cur)
    n = len(shards)
    for i, shard in enumerate(shards, start=1):
        shard_name = f"diffusion_pytorch_model-{i:05d}-of-{n:05d}.safetensors"
        for k in shard:
            index[k] = shard_name
    assert sum(len(s) for s in shards) == total
    return shards, index


def _write_transformer(
    out_dir: Path,
    state: dict[str, torch.Tensor],
    arch: dict,
    subdir: str = "transformer",
) -> None:
    transformer_dir = out_dir / subdir
    transformer_dir.mkdir(parents=True, exist_ok=True)

    shards, weight_map = _shard_state_dict(state)
    n = len(shards)
    total_bytes = sum(v.numel() * v.element_size() for v in state.values())
    for i, shard in enumerate(shards, start=1):
        shard_name = f"diffusion_pytorch_model-{i:05d}-of-{n:05d}.safetensors"
        save_file(shard, str(transformer_dir / shard_name))
        print(f"  wrote {shard_name} ({len(shard)} tensors)")

    index = {"metadata": {"total_size": total_bytes}, "weight_map": weight_map}
    with (transformer_dir / "diffusion_pytorch_model.safetensors.index.json").open("w") as f:
        json.dump(index, f, indent=2)
        f.write("\n")

    with (transformer_dir / "config.json").open("w") as f:
        json.dump(arch, f, indent=2)
        f.write("\n")
    print(f"  wrote {subdir}/config.json ({len(arch)} keys)")


def _write_scheduler(out_dir: Path) -> None:
    scheduler_dir = out_dir / "scheduler"
    scheduler_dir.mkdir(parents=True, exist_ok=True)
    with (scheduler_dir / "scheduler_config.json").open("w") as f:
        json.dump(SCHEDULER_CONFIG, f, indent=2)
        f.write("\n")
    print(f"  wrote scheduler/scheduler_config.json")


def _write_model_index(
    out_dir: Path,
    bundle_vae: bool,
    bundle_text: bool,
    bundle_audio_vae: bool = False,
    include_sr_transformer: bool = False,
) -> None:
    index = {
        "_class_name": "MagiHumanSRPipeline" if include_sr_transformer else "MagiHumanPipeline",
        "_diffusers_version": "0.33.0",
        "transformer": ["diffusers", "MagiHumanDiT"],
        "scheduler": ["diffusers", "FlowUniPCMultistepScheduler"],
    }
    if include_sr_transformer:
        index["sr_transformer"] = ["diffusers", "MagiHumanDiT"]
    if bundle_vae:
        index["vae"] = ["diffusers", "AutoencoderKLWan"]
    if bundle_audio_vae:
        index["audio_vae"] = ["diffusers", "AutoencoderOobleck"]
    if bundle_text:
        index["text_encoder"] = ["transformers", "T5GemmaEncoderModel"]
        index["tokenizer"] = ["transformers", "GemmaTokenizer"]
    with (out_dir / "model_index.json").open("w") as f:
        json.dump(index, f, indent=2)
        f.write("\n")
    print(f"  wrote model_index.json")


def _bundle_wan_vae(out_dir: Path, source_repo: str = "Wan-AI/Wan2.2-TI2V-5B-Diffusers") -> None:
    """Download the Wan 2.2 TI2V 5B VAE component into <out_dir>/vae/.

    The `-Diffusers` variant has the canonical `vae/config.json` +
    `vae/diffusion_pytorch_model.safetensors` layout. The plain
    `Wan-AI/Wan2.2-TI2V-5B` repo ships the VAE as a single `.pth` at the
    root, which is not `from_pretrained`-friendly.
    """
    print(f"  fetching VAE from {source_repo} ...")
    local = snapshot_download(
        repo_id=source_repo,
        allow_patterns=["vae/*"],
    )
    src_vae = Path(local) / "vae"
    if not src_vae.exists():
        raise FileNotFoundError(f"No vae/ subdir in {source_repo}")
    dst_vae = out_dir / "vae"
    if dst_vae.exists():
        shutil.rmtree(dst_vae)
    shutil.copytree(src_vae, dst_vae)
    print(f"  copied {src_vae} -> {dst_vae}")


def _bundle_sa_audio_vae(out_dir: Path, source_repo: str = "stabilityai/stable-audio-open-1.0") -> None:
    """Download the Stable Audio Open 1.0 VAE component into <out_dir>/audio_vae/.

    Stability ships the VAE at `vae/config.json` +
    `vae/diffusion_pytorch_model.safetensors` inside the main repo, so
    the bundle is just a copy of that subdir. The repo is gated — the
    caller's HF token must have accepted terms on
    https://huggingface.co/stabilityai/stable-audio-open-1.0.
    """
    print(f"  fetching audio VAE from {source_repo} (gated) ...")
    token = (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        or os.environ.get("HF_API_KEY")
    )
    local = snapshot_download(
        repo_id=source_repo, token=token, allow_patterns=["vae/*"],
    )
    src = Path(local) / "vae"
    if not src.exists():
        raise FileNotFoundError(f"No vae/ subdir in {source_repo}")
    dst = out_dir / "audio_vae"
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    print(f"  copied {src} -> {dst}")


def _bundle_text_encoder(out_dir: Path, source_repo: str = "google/t5gemma-9b-9b-ul2") -> None:
    """Download the T5Gemma encoder + tokenizer.

    T5Gemma is a Google gated repo; this step requires a write-scoped token with
    accepted terms of use for the repo.
    """
    print(f"  fetching text encoder from {source_repo} (gated) ...")
    token = (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        or os.environ.get("HF_API_KEY")
    )
    local = snapshot_download(
        repo_id=source_repo,
        token=token,
        allow_patterns=[
            "*.json",
            "*.model",
            "*.safetensors",
            "*.safetensors.index.json",
        ],
    )
    # Encoder-only bundling: keep tokenizer at the root and encoder weights
    # under text_encoder/. HF's T5GemmaEncoderModel.from_pretrained(<dir>) on
    # the whole repo works, but we split to match Diffusers layout.
    src = Path(local)
    dst_encoder = out_dir / "text_encoder"
    dst_tokenizer = out_dir / "tokenizer"
    if dst_encoder.exists():
        shutil.rmtree(dst_encoder)
    if dst_tokenizer.exists():
        shutil.rmtree(dst_tokenizer)
    dst_encoder.mkdir(parents=True, exist_ok=True)
    dst_tokenizer.mkdir(parents=True, exist_ok=True)

    for fname in src.iterdir():
        if fname.name in {"tokenizer.model", "tokenizer.json", "tokenizer_config.json",
                          "special_tokens_map.json", "spiece.model"}:
            shutil.copy(fname, dst_tokenizer / fname.name)
        else:
            shutil.copy(fname, dst_encoder / fname.name)
    print(f"  staged text_encoder and tokenizer from {src}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--source",
        default="GAIR/daVinci-MagiHuman",
        help="HF repo id or local directory containing base/*.safetensors shards.",
    )
    parser.add_argument(
        "--subfolder",
        default="base",
        choices=["base", "distill", "540p_sr", "1080p_sr"],
        help="Which MagiHuman variant to convert (scope of this skill: base).",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Destination directory for the Diffusers-format repo.",
    )
    parser.add_argument(
        "--bundle-vae",
        action="store_true",
        help="Download Wan-AI/Wan2.2-TI2V-5B VAE into <output>/vae/.",
    )
    parser.add_argument(
        "--cast-bf16",
        action="store_true",
        help=(
            "Cast fp32 DiT weights to bfloat16 on save. Recommended for the "
            "distill subfolder (61 GB fp32 upstream -> 30 GB bf16 artifact). "
            "Keeps norms, RoPE bands, and other precision-sensitive tensors "
            "in fp32."
        ),
    )
    parser.add_argument(
        "--bundle-text-encoder",
        action="store_true",
        help="Download google/t5gemma-9b-9b-ul2 into <output>/text_encoder/ and tokenizer/. "
             "Requires a write-scoped HF token with accepted terms of use.",
    )
    parser.add_argument(
        "--bundle-audio-vae",
        action="store_true",
        help="Download stabilityai/stable-audio-open-1.0 VAE into <output>/audio_vae/. "
             "Requires HF terms accepted for the Stability AI gated repo.",
    )
    parser.add_argument(
        "--sr-source",
        default=None,
        help="Optional HF repo id or local directory containing SR DiT shards. When set, writes <output>/sr_transformer/.",
    )
    parser.add_argument(
        "--sr-subfolder",
        default="540p_sr",
        choices=["540p_sr", "1080p_sr"],
        help="SR source subfolder to convert into <output>/sr_transformer/.",
    )
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"-> DiT shards from {args.source}/{args.subfolder}")
    shards = _download_dit_shards(args.source, subfolder=args.subfolder)
    print(f"  found {len(shards)} shard(s)")

    print(f"-> loading DiT state dict (cast_bf16={args.cast_bf16})")
    state = _load_all_shards(shards, cast_bf16=args.cast_bf16)
    print(f"  total keys: {len(state)}")
    _validate_state(state)
    print(f"  state dict validation passed")

    print(f"-> writing {out_dir}/transformer/")
    _write_transformer(out_dir, state, MAGI_HUMAN_BASE_ARCH)

    include_sr_transformer = args.sr_source is not None
    if include_sr_transformer:
        print(f"-> SR DiT shards from {args.sr_source}/{args.sr_subfolder}")
        sr_shards = _download_dit_shards(args.sr_source, subfolder=args.sr_subfolder)
        print(f"  found {len(sr_shards)} SR shard(s)")
        print(f"-> loading SR DiT state dict (cast_bf16={args.cast_bf16})")
        sr_state = _load_all_shards(sr_shards, cast_bf16=args.cast_bf16)
        print(f"  total SR keys: {len(sr_state)}")
        _validate_state(sr_state)
        print("  SR state dict validation passed")
        print(f"-> writing {out_dir}/sr_transformer/")
        _write_transformer(
            out_dir,
            sr_state,
            MAGI_HUMAN_BASE_ARCH,
            subdir="sr_transformer",
        )

    print(f"-> writing {out_dir}/scheduler/")
    _write_scheduler(out_dir)

    if args.bundle_vae:
        print(f"-> bundling video VAE (Wan 2.2 TI2V-5B)")
        _bundle_wan_vae(out_dir)

    if args.bundle_audio_vae:
        print(f"-> bundling audio VAE (Stable Audio Open 1.0)")
        _bundle_sa_audio_vae(out_dir)

    if args.bundle_text_encoder:
        print(f"-> bundling text encoder")
        _bundle_text_encoder(out_dir)

    print(f"-> writing model_index.json")
    _write_model_index(
        out_dir,
        bundle_vae=args.bundle_vae,
        bundle_text=args.bundle_text_encoder,
        bundle_audio_vae=args.bundle_audio_vae,
        include_sr_transformer=include_sr_transformer,
    )

    print(f"\nDone. Output at: {out_dir}")
    if not args.bundle_vae:
        print("  (remember to fetch Wan-AI/Wan2.2-TI2V-5B VAE separately or re-run with --bundle-vae)")
    if not args.bundle_text_encoder:
        print("  (remember to fetch google/t5gemma-9b-9b-ul2 separately or re-run with --bundle-text-encoder)")


if __name__ == "__main__":
    main()
