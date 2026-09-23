# SPDX-License-Identifier: Apache-2.0
"""Serialize the MiniMax-H3 transformer with per-row int8 block linears.

Reads a FastVideo transformer directory, quantizes the seven large linears of
every main transformer block (``attn.to_q``, ``attn.to_k``, ``attn.to_v``,
``attn.to_out``, ``attn.to_gate_compress``, ``ff.fc_in``, ``ff.fc_out``) to
symmetric per-row int8 with one float32 scale per output row, and copies every
other tensor unchanged: the token refiner, the rank-reduced AdaLN factors, the
patch, audio and context projections, the time embedder and the output head.
Keys keep the checkpoint's own names, so the loader's ``param_names_mapping``
applies to ``weight`` and ``weight_scale`` alike. ``config.json`` gains the
``quantization_config`` block that makes the transformer loader build the
serialized int8 linears; see ``fastvideo/layers/quantization/minimax_h3_int8.py``
for the byte contract and the GEMM.

For the rank-16 FastH3 transformer: 50 blocks x 7 linears = 350 linears holding
21.2B values, 21.2 GB as int8 against 42.4 GB as bf16, next to about 2 GB of
tensors left in their released dtypes.

Every quantized linear is probed: random bf16 rows go through ``int8_linear``,
the same function the loader executes, and the relative error against the bf16
product must stay under ``--max-probe-error``, otherwise the conversion stops
and removes what it wrote. Genuine W8A8 noise on random rows is about 0.013; a
transposed or misaligned layout reads as 1.0. ``--report-only`` runs the probe
and writes nothing.

Usage::

    python scripts/checkpoint_conversion/convert_minimax_h3_transformer_int8.py \
        --src /path/to/FastH3-r16/transformer \
        --dst /path/to/FastH3-r16-int8/transformer

Place the output directory as the ``transformer`` component of a model
directory whose other components are unchanged.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import time
from collections import Counter
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from fastvideo.layers.quantization.minimax_h3_int8 import (
    INT8_TENSOR_SUFFIXES,
    TRANSFORMER_LINEARS,
    int8_linear,
    quantize_rows_int8,
    serialized_int8_quantization_config,
    validate_int8_geometry,
)
from fastvideo.models.loader.weight_utils import SAFE_WEIGHTS_INDEX_NAME, resolve_safetensors_files

# Checkpoint key of each quantized linear -> its FastVideo-native name. The
# checkpoint follows the Diffusers layout the loader's param_names_mapping maps.
SOURCE_LINEARS = {
    "attn.to_q": "attn.to_q",
    "attn.to_k": "attn.to_k",
    "attn.to_v": "attn.to_v",
    "attn.to_out.0": "attn.to_out",
    "attn.to_gate_compress": "attn.to_gate_compress",
    "ff.net.0.proj": "ff.fc_in",
    "ff.net.2": "ff.fc_out",
}
BLOCK_LINEAR = re.compile(r"^transformer_blocks\.(?P<block>\d+)\.(?P<name>" +
                          "|".join(re.escape(name) for name in SOURCE_LINEARS) + r")\.weight$")
BLOCK_KEY = re.compile(r"^transformer_blocks\.(?P<block>\d+)\.")

# Keys the converter leaves alone on purpose: id -> (pattern, reason).
KEPT = {
    "refiner": ("token_refiner.*", "two blocks over the text stream only; 1.7 GB, not worth a second GEMM path"),
    "adaln": ("transformer_blocks.N.adaln_proj.*, norm_out.linear.*",
              "rank-16 factors pinned to fp16 by the model; int8 would undo the r16 precision choice"),
    "projections": ("proj_in, audio_proj_in, context_embedder, time_embedder, proj_out, audio_proj_out",
                    "released in fp32 or tiny; the model keeps them in their own dtype"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", type=Path, required=True, help="transformer directory of the bf16 checkpoint")
    parser.add_argument("--dst", type=Path, help="transformer directory to write (required unless --report-only)")
    parser.add_argument("--device", default="cuda", help="device that runs the quantizer and the probe")
    parser.add_argument("--shard-size-gb", type=float, default=4.0, help="safetensors shard size")
    parser.add_argument("--probe-rows", type=int, default=512,
                        help="rows of random activations per linear for the error probe, 0 disables it")
    parser.add_argument("--max-probe-error", type=float, default=0.1,
                        help="stop when a linear's int8 GEMM relative error exceeds this; W8A8 noise is about 0.013")
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
    return args


def scan_source(shards: list[str]) -> tuple[Counter, int]:
    """Count the block linears the conversion will touch, keyed by native name, and the block count seen."""
    counts: Counter = Counter()
    highest_block = -1
    for shard in shards:
        with safe_open(shard, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                block_match = BLOCK_KEY.match(key)
                if block_match is not None:
                    highest_block = max(highest_block, int(block_match["block"]))
                linear_match = BLOCK_LINEAR.match(key)
                if linear_match is not None:
                    counts[SOURCE_LINEARS[linear_match["name"]]] += 1
    return counts, highest_block + 1


class ShardWriter:
    """Accumulate tensors and flush them as fixed-size safetensors shards plus an index.

    Shards are renamed to the ``diffusion_pytorch_model-00001-of-0000N.safetensors``
    convention once the count is known, matching the Diffusers transformer layout.
    """

    STEM = "diffusion_pytorch_model"

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
        filename = f"{self.STEM}-{self.shard_index:05d}.partial.safetensors"
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
            partial = f"{self.STEM}-{index:05d}.partial.safetensors"
            final = f"{self.STEM}-{index:05d}-of-{self.shard_index:05d}.safetensors"
            (self.dst / partial).rename(self.dst / final)
            final_names[partial] = final
        self.written = [self.dst / final for final in final_names.values()]
        weight_map = {name: final_names[partial] for name, partial in self.weight_map.items()}
        index_payload = {"metadata": {"total_size": self.total_bytes}, "weight_map": weight_map}
        (self.dst / f"{self.STEM}.safetensors.index.json").write_text(
            json.dumps(index_payload, indent=2, sort_keys=True), encoding="utf-8")

    def abort(self) -> None:
        """Remove every shard this writer produced so a failed conversion leaves no half checkpoint."""
        for path in self.written:
            path.unlink(missing_ok=True)
        self.written = []


def probe_relative_error(
    weight: torch.Tensor,
    weight_int8: torch.Tensor,
    weight_scale: torch.Tensor,
    rows: int,
    generator: torch.Generator,
) -> float:
    """||int8(x) @ int8(W).T - x @ W.T|| / ||x @ W.T|| on random bf16 rows, through the loader's own GEMM."""
    x = torch.randn(rows, weight.shape[1], generator=generator, device=weight.device,
                    dtype=torch.float32).to(torch.bfloat16)
    reference = (x.float() @ weight.float().t())
    output = int8_linear(x, weight_int8, weight_scale, out_dtype=torch.float32)
    return ((output - reference).norm() / reference.norm().clamp_min(1e-12)).item()


def quantize_block_linear(
    weight: torch.Tensor,
    device: torch.device,
    probe_rows: int,
    generator: torch.Generator | None,
    max_probe_error: float,
    key: str = "",
) -> tuple[torch.Tensor, torch.Tensor, float | None]:
    if weight.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError(f"Expected a bf16, fp16 or fp32 source weight for {key}, got {weight.dtype}; "
                         "the source transformer must be unquantized")
    output_size, input_size = weight.shape
    validate_int8_geometry(output_size, input_size)
    weight_device = weight.to(device=device, dtype=torch.bfloat16)
    weight_int8, weight_scale = quantize_rows_int8(weight_device)
    error = None
    if probe_rows and generator is not None:
        error = probe_relative_error(weight_device, weight_int8, weight_scale, probe_rows, generator)
        if not error <= max_probe_error:
            raise SystemExit(f"int8 GEMM relative error {error:.3e} on {key or 'a block linear'} exceeds "
                             f"--max-probe-error {max_probe_error}; the stored layout or scales do not reproduce "
                             "the bf16 product, nothing was written")
    return weight_int8.cpu().contiguous(), weight_scale.cpu().contiguous(), error


def main() -> None:
    args = parse_args()
    src: Path = args.src
    config_path = src / "config.json"
    if not config_path.is_file():
        raise SystemExit(f"{src} has no config.json; point --src at the transformer directory")
    source_config = json.loads(config_path.read_text(encoding="utf-8"))
    if source_config.get("quantization_config"):
        raise SystemExit(f"{src} already carries a quantization_config "
                         f"({source_config['quantization_config'].get('quant_method')!r}); this converter "
                         "quantizes the bf16 transformer, not a serialized one")
    num_layers = int(source_config.get("num_layers", 0))

    shards = resolve_safetensors_files(str(src))
    counts, blocks_seen = scan_source(shards)
    if not counts:
        raise SystemExit(f"No block linear in {src} matches transformer_blocks.N.<linear>.weight; "
                         "this converter expects the FastVideo MiniMax-H3 transformer layout")
    if num_layers and blocks_seen != num_layers:
        raise SystemExit(f"{src} holds {blocks_seen} transformer blocks but config.json declares {num_layers}")
    short = {name: counts.get(name, 0) for name in TRANSFORMER_LINEARS if counts.get(name, 0) != blocks_seen}
    # A dense checkpoint has no compression gate at all; anything else must be complete.
    if short.get("attn.to_gate_compress") == 0:
        short.pop("attn.to_gate_compress")
    if short:
        raise SystemExit(f"Expected one of every block linear in each of {blocks_seen} blocks; "
                         f"incomplete: {short}")
    has_gate = counts.get("attn.to_gate_compress", 0) > 0

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda was requested but CUDA is not available")

    dst: Path | None = None
    if not args.report_only:
        dst = args.dst
        dst.mkdir(parents=True, exist_ok=True)
        existing = sorted(path.name for path in dst.glob("*.safetensors")) + \
            [name for name in ("config.json", SAFE_WEIGHTS_INDEX_NAME, f"{ShardWriter.STEM}.safetensors.index.json")
             if (dst / name).exists()]
        if existing:
            raise SystemExit(f"{dst} already holds {existing[:4]}{' and more' if len(existing) > 4 else ''}; "
                             "refusing to overwrite or mix outputs")
    writer = ShardWriter(dst, int(args.shard_size_gb * (1 << 30)))
    generator = torch.Generator(device=device).manual_seed(args.seed) if args.probe_rows else None

    quantized = 0
    copied = 0
    copied_bytes = 0
    source_linear_bytes = 0
    written_linear_bytes = 0
    errors: dict[str, list[float]] = {}
    started = time.perf_counter()
    print(f"{blocks_seen} transformer blocks, {sum(counts.values())} block linears to quantize"
          f"{'' if has_gate else ' (dense checkpoint, no attn.to_gate_compress)'}", flush=True)
    weight_name, scale_name = INT8_TENSOR_SUFFIXES

    try:
        for shard in shards:
            with safe_open(shard, framework="pt", device="cpu") as handle:
                for key in sorted(handle.keys()):
                    tensor = handle.get_tensor(key)
                    linear_match = BLOCK_LINEAR.match(key)
                    if linear_match is None:
                        writer.add(key, tensor.contiguous())
                        copied += 1
                        copied_bytes += tensor.numel() * tensor.element_size()
                        continue
                    weight_int8, weight_scale, error = quantize_block_linear(
                        tensor, device, args.probe_rows, generator, args.max_probe_error, key)
                    prefix = key[:-len(".weight")]
                    writer.add(f"{prefix}.{weight_name}", weight_int8)
                    writer.add(f"{prefix}.{scale_name}", weight_scale)
                    source_linear_bytes += tensor.numel() * tensor.element_size()
                    written_linear_bytes += weight_int8.numel() + weight_scale.numel() * 4
                    quantized += 1
                    if error is not None:
                        errors.setdefault(SOURCE_LINEARS[linear_match["name"]], []).append(error)
                    if quantized % 35 == 0:
                        print(f"  quantized {quantized} block linears, {time.perf_counter() - started:.0f}s",
                              flush=True)
        writer.finish()
    except BaseException:
        writer.abort()
        raise

    if dst is not None:
        config = dict(source_config)
        config["quantization_config"] = serialized_int8_quantization_config(producer={
            "converter": Path(__file__).name,
            "torch": torch.__version__,
            "blocks": blocks_seen,
            "gate_compress": has_gate,
        })
        (dst / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
        for extra in src.iterdir():
            if extra.is_file() and extra.name != "config.json" and not extra.name.endswith(".safetensors") \
                    and not extra.name.endswith(".safetensors.index.json"):
                shutil.copy2(extra, dst / extra.name)

    print(f"block linears quantized:  {quantized}")
    print(f"tensors copied unchanged: {copied} ({copied_bytes / 1e9:.2f} GB)")
    for kept_id, (pattern, reason) in KEPT.items():
        print(f"kept {kept_id:<12} {pattern}: {reason}")
    print(f"block linear bytes: {source_linear_bytes / 1e9:.2f} GB source -> "
          f"{written_linear_bytes / 1e9:.2f} GB int8 (values + float32 row scales)")
    print(f"artifact total: {writer.total_bytes / 1e9:.2f} GB in {writer.shard_index} shard(s)"
          f"{'' if dst is not None else ' (not written, --report-only)'}")
    if errors:
        print(f"int8 GEMM relative error vs bf16, {args.probe_rows} random rows per linear, "
              f"every linear under --max-probe-error {args.max_probe_error}:")
        print(f"  {'linear':<24}{'count':>8}{'max':>12}{'mean':>12}")
        for name in TRANSFORMER_LINEARS:
            values = errors.get(name)
            if values:
                print(f"  {name:<24}{len(values):>8}{max(values):>12.3e}{sum(values) / len(values):>12.3e}")
    elif quantized:
        print("int8 GEMM error probe skipped because --probe-rows is 0; nothing verified the written layout")
    print(f"done in {time.perf_counter() - started:.0f}s")


if __name__ == "__main__":
    main()
