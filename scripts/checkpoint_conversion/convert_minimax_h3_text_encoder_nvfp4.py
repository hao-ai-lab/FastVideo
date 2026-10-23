# SPDX-License-Identifier: Apache-2.0
"""Serialize the MiniMax-H3 Qwen3-VL text encoder as NVFP4.

H3 conditions on hidden state 50 of a 64-layer Qwen3-VL and never predicts a
token, so the conditioner builds 50 decoder layers, no ``lm_head``, and reads
nothing above (#1711). Every linear in those layers is quantized with
``flashinfer.nvfp4_quantize`` exactly as ``convert_model_to_nvfp4`` would at
runtime and stored as ``weight_packed`` / ``weight_scale`` /
``weight_global_scale``; the byte layout is documented in
``fastvideo/models/encoders/minimax_h3_checkpoint_nvfp4.py``. Everything else
(token embedding, norms, the vision tower) is copied unchanged. ``config.json``
gains the ``quantization_config`` block that makes the loader select the
serialized NVFP4 path and a ``num_hidden_layers_override`` equal to the kept
layer count, so the loader builds exactly the layers the file holds.

For the FastH3 conditioner: 50 layers x 487.6M values = 24.4B values, 12.2 GB
packed plus 1.5 GB of scales, next to about 3 GB of unquantized tensors,
against 48.8 GB of bf16 language layers (63 GB on disk with the 14 dropped
layers and ``lm_head``).

Needs a Blackwell GPU with FlashInfer: the quantizer is a CUDA kernel, and the
error report runs the same ``mm_fp4`` path the loader executes.

Usage::

    python scripts/checkpoint_conversion/convert_minimax_h3_text_encoder_nvfp4.py \
        --src /path/to/FastH3/text_encoder \
        --dst /path/to/FastH3-nvfp4/text_encoder --keep-bf16 mlp.down_proj

    # use it with any FastH3 model directory
    ComponentConfig(text_encoder_weights="/path/to/FastH3-nvfp4/text_encoder")

``--keep-bf16 mlp.down_proj`` leaves one projection kind in bf16 in every
layer and records it in ``modules_to_not_convert`` so the loader builds those
linears unquantized. The SwiGLU product feeding ``down_proj`` carries the
widest activation outliers in the stack; measured on one GB10 it is the
projection whose 4-bit form moves the layer-50 output most.

Every quantized linear is probed: random bf16 rows go through the same
``mm_fp4`` path the loader runs and the relative error against the bf16
product must stay under ``--max-probe-error``, otherwise the conversion stops
and removes what it wrote. The genuine W4A4 noise on random inputs is about
0.13; a wrong scale layout reads as 1.0. ``--report-only`` runs the probe and
writes nothing. The kept layer count is the one the conditioner builds, 50
for H3, and is also written into ``config.json`` for a directory used as the
``text_encoder`` component; when the checkpoint is supplied through
``text_encoder_weights`` the model structure comes from the model directory,
which builds the same 50 layers.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import time
from collections import Counter
from importlib import metadata as importlib_metadata
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from fastvideo.configs.models.encoders.minimax_h3_qwen3_vl import MiniMaxH3Qwen3VLConfig
from fastvideo.layers.quantization.nvfp4_config import _nvfp4_quantize, _require_flashinfer
from fastvideo.models.encoders.minimax_h3_checkpoint_nvfp4 import (
    LANGUAGE_PROJECTIONS,
    NVFP4_TENSOR_SUFFIXES,
    _nvfp4_linear,
    _quantize_activation_nvfp4,
    nvfp4_packed_weight_shape,
    nvfp4_scale_shape,
    nvfp4_weight_global_scale,
    serialized_nvfp4_quantization_config,
    validate_nvfp4_geometry,
)
from fastvideo.models.loader.weight_utils import SAFE_WEIGHTS_INDEX_NAME, resolve_safetensors_files

LANGUAGE_LINEAR = re.compile(r"^model\.language_model\.layers\.(?P<layer>\d+)\.(?P<proj>" +
                             "|".join(re.escape(name) for name in LANGUAGE_PROJECTIONS) + r")\.weight$")
LANGUAGE_LAYER = re.compile(r"^model\.language_model\.layers\.(?P<layer>\d+)\.")

# Keys the converter drops on purpose: id -> (pattern, reason).
SKIPPED = {
    "lm_head": ("lm_head.weight", "the conditioner never predicts tokens; H3 reads a hidden state"),
    "upper_layers": ("model.language_model.layers.{N >= kept}.*",
                     "layers above the one H3 reads are never built (#1711)"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", type=Path, required=True, help="text_encoder directory of the bf16 checkpoint")
    parser.add_argument("--dst", type=Path, help="text_encoder directory to write (required unless --report-only)")
    parser.add_argument("--keep-bf16", action="append", default=[], metavar="PROJ",
                        help="projection kind to leave in bf16 in every layer, e.g. mlp.down_proj; "
                        "repeat or comma-separate for several")
    parser.add_argument("--device", default="cuda", help="CUDA device that runs the quantizer")
    parser.add_argument("--shard-size-gb", type=float, default=4.0, help="safetensors shard size")
    parser.add_argument("--probe-rows", type=int, default=512,
                        help="rows of random activations per linear for the error probe, 0 disables it")
    parser.add_argument("--max-probe-error", type=float, default=0.5,
                        help="stop when a linear's FP4 GEMM relative error exceeds this; W4A4 noise is about 0.13")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--report-only", action="store_true", help="quantize and report errors, write nothing")
    args = parser.parse_args()
    if not args.report_only and args.dst is None:
        parser.error("--dst is required unless --report-only is given")
    if args.probe_rows < 0:
        parser.error("--probe-rows must be 0 or positive")
    if args.shard_size_gb <= 0:
        parser.error("--shard-size-gb must be positive")
    if args.max_probe_error <= 0:
        parser.error("--max-probe-error must be positive")
    keep = [name.strip() for entry in args.keep_bf16 for name in entry.split(",") if name.strip()]
    unknown = sorted(set(keep) - set(LANGUAGE_PROJECTIONS))
    if unknown:
        parser.error(f"--keep-bf16 got unknown projection(s) {unknown}; choose from {list(LANGUAGE_PROJECTIONS)}")
    args.keep_bf16 = tuple(dict.fromkeys(keep))
    if set(args.keep_bf16) == set(LANGUAGE_PROJECTIONS):
        parser.error("--keep-bf16 names every projection kind, so nothing would be quantized; "
                     "use the bf16 checkpoint as is")
    return args


def layer_plan(source_config: dict) -> tuple[int, int, int]:
    """Return (kept, total, tap): the language layers the conditioner builds from
    this config.json, how many the source has, and the hidden state H3 reads,
    derived the way the model derives them."""
    model_config = MiniMaxH3Qwen3VLConfig()
    model_config.update_model_arch(source_config)
    arch = model_config.arch_config
    total = int(arch.num_hidden_layers)
    tap = int(arch.output_hidden_state_index)
    override = getattr(arch, "num_hidden_layers_override", None)
    kept = min(total, int(override)) if override else total
    return kept, total, tap


def scan_source(shards: list[str], kept: int) -> tuple[Counter, int, dict[str, set[int]]]:
    """Count the language linears the conversion will touch before any work starts.

    Returns per-projection counts below ``kept``, the highest language layer index
    seen, and the layer indexes present per projection for diagnostics.
    """
    counts: Counter = Counter()
    layers_seen: dict[str, set[int]] = {proj: set() for proj in LANGUAGE_PROJECTIONS}
    highest_layer = -1
    for shard in shards:
        with safe_open(shard, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                layer_match = LANGUAGE_LAYER.match(key)
                if layer_match is not None:
                    highest_layer = max(highest_layer, int(layer_match["layer"]))
                linear_match = LANGUAGE_LINEAR.match(key)
                if linear_match is not None and int(linear_match["layer"]) < kept:
                    counts[linear_match["proj"]] += 1
                    layers_seen[linear_match["proj"]].add(int(linear_match["layer"]))
    return counts, highest_layer, layers_seen


class ShardWriter:
    """Accumulate tensors and flush them as fixed-size safetensors shards plus an index.

    Shards are renamed to the ``model-00001-of-0000N.safetensors`` convention
    once the count is known, so the output looks like any Hub checkpoint.
    """

    def __init__(self, dst: Path | None, shard_bytes: int) -> None:
        self.dst = dst
        self.shard_bytes = shard_bytes
        self.pending: dict[str, torch.Tensor] = {}
        self.pending_bytes = 0
        self.weight_map: dict[str, str] = {}
        self.total_bytes = 0
        self.shard_index = 0
        self.written: list[Path] = []

    def add(self, name: str, tensor: torch.Tensor) -> None:
        nbytes = tensor.numel() * tensor.element_size()
        if self.pending and self.pending_bytes + nbytes > self.shard_bytes:
            self.flush()
        self.pending[name] = tensor
        self.pending_bytes += nbytes
        self.total_bytes += nbytes

    def flush(self) -> None:
        if not self.pending:
            return
        self.shard_index += 1
        filename = f"model-{self.shard_index:05d}.partial.safetensors"
        if self.dst is not None:
            save_file(self.pending, str(self.dst / filename), metadata={"format": "pt"})
            self.written.append(self.dst / filename)
        for name in self.pending:
            self.weight_map[name] = filename
        self.pending = {}
        self.pending_bytes = 0

    def finish(self) -> None:
        self.flush()
        if self.dst is None:
            return
        final_names = {}
        for index in range(1, self.shard_index + 1):
            partial = f"model-{index:05d}.partial.safetensors"
            final = f"model-{index:05d}-of-{self.shard_index:05d}.safetensors"
            (self.dst / partial).rename(self.dst / final)
            final_names[partial] = final
        self.written = [self.dst / final for final in final_names.values()]
        weight_map = {name: final_names[partial] for name, partial in self.weight_map.items()}
        index_payload = {"metadata": {"total_size": self.total_bytes}, "weight_map": weight_map}
        (self.dst / SAFE_WEIGHTS_INDEX_NAME).write_text(json.dumps(index_payload, indent=2, sort_keys=True),
                                                        encoding="utf-8")

    def abort(self) -> None:
        """Remove every shard this writer produced so a failed conversion leaves no half checkpoint."""
        for path in self.written:
            path.unlink(missing_ok=True)
        self.written = []


def probe_relative_error(
    weight: torch.Tensor,
    packed: torch.Tensor,
    scale: torch.Tensor,
    global_scale: torch.Tensor,
    rows: int,
    generator: torch.Generator,
) -> float:
    """||fp4(x) @ fp4(W).T - x @ W.T|| / ||x @ W.T|| on random bf16 rows, through the loader's own GEMM path.

    The reference is the bf16 tensor-core product with fp32 accumulation; its
    rounding sits two orders of magnitude below the FP4 error being measured.
    """
    x = torch.randn(rows, weight.shape[1], generator=generator, device=weight.device,
                    dtype=torch.float32).to(torch.bfloat16)
    reference = (x @ weight.t()).float()
    x_fp4, x_scale = _quantize_activation_nvfp4(x, torch.ones((), dtype=torch.float32, device=weight.device))
    output = _nvfp4_linear(x_fp4, x_scale, packed, scale, (1.0 / global_scale).reshape(()))
    return ((output.float() - reference).norm() / reference.norm().clamp_min(1e-12)).item()


def quantize_language_linear(
    weight: torch.Tensor,
    device: torch.device,
    sf_layout: object,
    probe_rows: int,
    generator: torch.Generator | None,
    max_probe_error: float,
    key: str = "",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float | None]:
    if weight.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError(f"Expected a bf16, fp16 or fp32 source weight, got {weight.dtype}; "
                         "the source text_encoder must be unquantized")
    output_size, input_size = weight.shape
    validate_nvfp4_geometry(output_size, input_size)
    weight_device = weight.to(device=device, dtype=torch.bfloat16)
    global_scale = nvfp4_weight_global_scale(weight_device)
    packed, scale = _nvfp4_quantize(weight_device, global_scale, sfLayout=sf_layout.layout_128x4, do_shuffle=False)
    if packed.dtype != torch.uint8:
        packed = packed.view(torch.uint8)
    if scale.dtype != torch.uint8:
        scale = scale.view(torch.uint8)
    expected_packed = nvfp4_packed_weight_shape(output_size, input_size)
    expected_scale = nvfp4_scale_shape(output_size, input_size)
    if tuple(packed.shape) != expected_packed or tuple(scale.shape) != expected_scale:
        raise RuntimeError("flashinfer.nvfp4_quantize returned an unexpected layout for a "
                           f"{output_size}x{input_size} weight: packed {tuple(packed.shape)} "
                           f"(expected {expected_packed}), scale {tuple(scale.shape)} (expected {expected_scale}). "
                           "The loader allocates exactly these shapes; check the FlashInfer version.")
    error = None
    if probe_rows and generator is not None:
        error = probe_relative_error(weight_device, packed, scale, global_scale, probe_rows, generator)
        if not error <= max_probe_error:
            raise SystemExit(f"FP4 GEMM relative error {error:.3e} on {key or 'a language linear'} exceeds "
                             f"--max-probe-error {max_probe_error}; the packed layout or scales do not reproduce "
                             "the bf16 product, nothing was written")
    return packed.cpu().contiguous(), scale.cpu().contiguous(), global_scale.reshape(1).cpu(), error


def flashinfer_version() -> str:
    try:
        return importlib_metadata.version("flashinfer-python")
    except importlib_metadata.PackageNotFoundError:
        import flashinfer
        return str(getattr(flashinfer, "__version__", "unknown"))


def main() -> None:
    args = parse_args()
    src: Path = args.src
    config_path = src / "config.json"
    if not config_path.is_file():
        raise SystemExit(f"{src} has no config.json; point --src at the text_encoder directory")
    source_config = json.loads(config_path.read_text(encoding="utf-8"))
    if source_config.get("quantization_config"):
        raise SystemExit(f"{src} already carries a quantization_config "
                         f"({source_config['quantization_config'].get('quant_method')!r}); this converter "
                         "quantizes the bf16 text_encoder, not a serialized one")
    kept, total, tap = layer_plan(source_config)

    shards = resolve_safetensors_files(str(src))
    counts, highest_layer, layers_seen = scan_source(shards, kept)
    if not counts:
        raise SystemExit(f"No language linear in {src} matches model.language_model.layers.N.<proj>.weight; "
                         "this converter expects the Qwen3-VL checkpoint layout")
    if highest_layer + 1 < kept:
        raise SystemExit(f"{src} holds {highest_layer + 1} language layers but the conditioner builds {kept}")
    short = {}
    for proj in LANGUAGE_PROJECTIONS:
        missing = sorted(set(range(kept)) - layers_seen[proj])
        if missing:
            short[proj] = f"{len(missing)} missing, e.g. model.language_model.layers.{missing[0]}.{proj}.weight"
    if short:
        raise SystemExit(f"Expected {kept} of every language projection below layer {kept}: {short}")

    device = torch.device(args.device)
    if device.type != "cuda":
        raise SystemExit("The NVFP4 quantizer is a CUDA kernel; pass --device cuda")
    sf_layout, _, _ = _require_flashinfer()

    dst: Path | None = None
    if not args.report_only:
        dst = args.dst
        dst.mkdir(parents=True, exist_ok=True)
        existing = sorted(path.name for path in dst.glob("*.safetensors")) + \
            [name for name in ("config.json", SAFE_WEIGHTS_INDEX_NAME) if (dst / name).exists()]
        if existing:
            raise SystemExit(f"{dst} already holds {existing[:4]}{' and more' if len(existing) > 4 else ''}; "
                             "refusing to overwrite or mix outputs")
    writer = ShardWriter(dst, int(args.shard_size_gb * (1 << 30)))
    generator = torch.Generator(device=device).manual_seed(args.seed) if args.probe_rows else None

    quantized = 0
    kept_bf16 = 0
    kept_bf16_bytes = 0
    copied = 0
    skipped: Counter = Counter()
    source_language_bytes = 0
    written_language_bytes = 0
    errors: dict[str, list[float]] = {}
    started = time.perf_counter()
    print(f"keeping {kept} of {total} language layers, H3 reads hidden state {tap}; "
          f"{sum(counts.values())} language linears to visit", flush=True)
    packed_name, scale_name, global_scale_name = NVFP4_TENSOR_SUFFIXES

    try:
        for shard in shards:
            with safe_open(shard, framework="pt", device="cpu") as handle:
                for key in sorted(handle.keys()):
                    layer_match = LANGUAGE_LAYER.match(key)
                    if key == "lm_head.weight":
                        skipped["lm_head"] += 1
                        continue
                    if layer_match is not None and int(layer_match["layer"]) >= kept:
                        skipped["upper_layers"] += 1
                        continue
                    tensor = handle.get_tensor(key)
                    linear_match = LANGUAGE_LINEAR.match(key)
                    if linear_match is None:
                        writer.add(key, tensor.contiguous())
                        copied += 1
                        continue
                    if linear_match["proj"] in args.keep_bf16:
                        # Kept linears are stored as bf16 whatever the source dtype, matching the
                        # precision the conditioner runs them in.
                        writer.add(key, tensor.to(torch.bfloat16).contiguous())
                        kept_bf16 += 1
                        kept_bf16_bytes += tensor.numel() * 2
                        continue
                    packed, scale, global_scale, error = quantize_language_linear(
                        tensor, device, sf_layout, args.probe_rows, generator, args.max_probe_error, key)
                    prefix = key[:-len(".weight")]
                    writer.add(f"{prefix}.{packed_name}", packed)
                    writer.add(f"{prefix}.{scale_name}", scale)
                    writer.add(f"{prefix}.{global_scale_name}", global_scale)
                    source_language_bytes += tensor.numel() * tensor.element_size()
                    written_language_bytes += packed.numel() + scale.numel() + 4
                    quantized += 1
                    if error is not None:
                        errors.setdefault(linear_match["proj"], []).append(error)
                    if quantized % 35 == 0:
                        print(f"  quantized {quantized} language linears, {time.perf_counter() - started:.0f}s",
                              flush=True)
        writer.finish()
    except BaseException:
        writer.abort()
        raise

    if dst is not None:
        config = dict(source_config)
        config["num_hidden_layers_override"] = kept
        config["quantization_config"] = serialized_nvfp4_quantization_config(
            keep_bf16=args.keep_bf16,
            producer={
                "converter": Path(__file__).name,
                "flashinfer": flashinfer_version(),
                "kept_language_layers": kept,
            })
        (dst / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
        for extra in src.iterdir():
            if extra.is_file() and extra.name not in {SAFE_WEIGHTS_INDEX_NAME, "config.json"} \
                    and not extra.name.endswith(".safetensors"):
                shutil.copy2(extra, dst / extra.name)

    print(f"language linears quantized: {quantized}")
    if args.keep_bf16:
        print(f"language linears kept bf16: {kept_bf16} ({', '.join(args.keep_bf16)}; "
              f"{kept_bf16_bytes / 1e9:.2f} GB)")
    print(f"tensors copied unchanged:   {copied}")
    for skip_id, count in sorted(skipped.items()):
        pattern, reason = SKIPPED[skip_id]
        print(f"skipped {count:>4} keys: {pattern}: {reason}")
    print(f"language linear bytes: {source_language_bytes / 1e9:.2f} GB bf16 -> "
          f"{written_language_bytes / 1e9:.2f} GB NVFP4 (packed values + E4M3 scales)")
    print(f"artifact total: {writer.total_bytes / 1e9:.2f} GB in {writer.shard_index} shard(s)"
          f"{'' if dst is not None else ' (not written, --report-only)'}")
    if errors:
        print(f"FP4 GEMM relative error vs bf16, {args.probe_rows} random rows per linear, "
              f"every linear under --max-probe-error {args.max_probe_error}:")
        print(f"  {'projection':<20}{'linears':>8}{'max':>12}{'mean':>12}")
        for proj in LANGUAGE_PROJECTIONS:
            values = errors.get(proj)
            if values:
                print(f"  {proj:<20}{len(values):>8}{max(values):>12.3e}{sum(values) / len(values):>12.3e}")
    elif quantized:
        print("FP4 GEMM error probe skipped because --probe-rows is 0; nothing verified the written layout")
    print(f"done in {time.perf_counter() - started:.0f}s")


if __name__ == "__main__":
    main()
