# SPDX-License-Identifier: Apache-2.0
"""Export only the Kandinsky 6 transformer in FastVideo's packed NVFP4 format."""

import argparse
import json
import math
import shutil
import tempfile
from pathlib import Path


def export_checkpoint(
    source_model: Path,
    destination: Path,
    device_name: str,
    dense_shard_bytes: int = 2_000_000_000,
) -> None:
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    from fastvideo.configs.models.dits.kandinsky6 import Kandinsky6ArchConfig
    from fastvideo.layers.quantization.nvfp4_config import (
        H3_NVFP4_DIT_EXPORT_FILENAME,
        H3_NVFP4_DIT_KEY_SEP,
        _nvfp4_quantize,
        _require_flashinfer,
        is_kandinsky6_nvfp4_linear_prefix,
    )
    from fastvideo.models.loader.utils import get_param_names_mapping

    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Output already exists: {destination}")
    source_root = source_model.resolve()
    output_root = destination.resolve()
    if output_root.is_relative_to(source_root) and output_root != source_root / "nvfp4" / "transformer":
        raise ValueError("Use <source model>/nvfp4/transformer or an output directory outside the source model.")
    if dense_shard_bytes <= 0:
        raise ValueError("The dense shard size must be positive.")

    index_path = source_model / "model_index.json"
    if not index_path.is_file():
        raise ValueError(f"A complete Kandinsky 6 checkpoint with model_index.json is required: {source_model}")
    model_index = json.loads(index_path.read_text())
    if model_index.get("_class_name") != "Kandinsky6TI2VAPipeline":
        raise ValueError("Expected a Kandinsky6TI2VAPipeline Diffusers directory.")

    transformer = source_model / "transformer"
    config_path = transformer / "config.json"
    config = json.loads(config_path.read_text())
    component = model_index.get("transformer")
    index_class = component[1] if isinstance(component, list) and len(component) >= 2 else None
    class_name = config.get("_class_name") or index_class
    if class_name != "Kandinsky6Transformer3DModel":
        raise ValueError(f"Expected a Kandinsky6Transformer3DModel transformer, got {class_name!r}.")
    shards = sorted(transformer.glob("*.safetensors"))
    if not shards or (transformer / H3_NVFP4_DIT_EXPORT_FILENAME).exists():
        raise ValueError("The input must contain dense transformer safetensors.")

    device = torch.device(device_name)
    if device.type != "cuda":
        raise ValueError("FlashInfer weight packing requires a CUDA device.")
    sf_layout, _, _ = _require_flashinfer()
    mapper = get_param_names_mapping(Kandinsky6ArchConfig().param_names_mapping)
    packed = {}
    seen = set()

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".k6-nvfp4-", dir=destination.parent) as temp:
        output = Path(temp) / "transformer"
        output.mkdir()
        dense = {}
        pending_bytes = 0
        total_dense_bytes = 0
        shard_keys: list[tuple[str, ...]] = []

        def flush_dense_shard() -> None:
            nonlocal pending_bytes
            number = len(shard_keys) + 1
            save_file(dense, output / f"part-{number:05d}.safetensors", metadata={"format": "pt"})
            shard_keys.append(tuple(dense))
            dense.clear()
            pending_bytes = 0

        with torch.inference_mode():
            for shard in shards:
                with safe_open(shard, framework="pt", device="cpu") as reader:
                    for source_key in reader.keys():  # noqa: SIM118
                        mapped, merge_index, _ = mapper(source_key)
                        if merge_index is not None or mapped in seen:
                            raise ValueError(f"Unexpected merged or duplicate checkpoint tensor: {source_key}")
                        seen.add(mapped)
                        module, _, field = mapped.rpartition(".")
                        value = reader.get_tensor(source_key)

                        # Preserve the original keys and dtypes of biases, norms and other dense tensors.
                        if field != "weight" or not is_kandinsky6_nvfp4_linear_prefix(module):
                            size = value.numel() * value.element_size()
                            if dense and pending_bytes + size > dense_shard_bytes:
                                flush_dense_shard()
                            dense[source_key] = value
                            pending_bytes += size
                            total_dense_bytes += size
                            continue

                        if value.dtype not in (torch.bfloat16, torch.float16, torch.float32):
                            raise ValueError(f"Expected a dense floating-point weight for {source_key}: {value.dtype}")
                        if value.ndim != 2 or value.shape[-1] % 16:
                            raise ValueError(f"Invalid NVFP4 linear shape for {source_key}: {value.shape}")
                        weight = value.to(device=device, dtype=torch.bfloat16)
                        multiplier = (448 * 6) / weight.abs().amax().float().clamp_min(1e-12)
                        fp4, scale = _nvfp4_quantize(weight, multiplier, sfLayout=sf_layout.layout_128x4,
                                                    do_shuffle=False)
                        buffers = {"_nvfp4_weight": fp4, "_nvfp4_weight_scale": scale,
                                   "_nvfp4_alpha": multiplier.reciprocal(),
                                   "_weight_global_sf": multiplier.to(torch.bfloat16)}
                        for name, tensor in buffers.items():
                            packed[f"{module}{H3_NVFP4_DIT_KEY_SEP}{name}"] = tensor.cpu()
                        del weight, fp4, scale, buffers
                print(f"Processed {shard.name}", flush=True)
        if dense:
            flush_dense_shard()
        if not packed or not shard_keys:
            raise ValueError("Expected both NVFP4 linear weights and remaining dense tensors.")

        save_file(packed, output / H3_NVFP4_DIT_EXPORT_FILENAME, metadata={"layer_profile": "kandinsky6"})
        weight_map = {}
        for number, keys in enumerate(shard_keys, start=1):
            name = f"diffusion_pytorch_model-{number:05d}-of-{len(shard_keys):05d}.safetensors"
            (output / f"part-{number:05d}.safetensors").rename(output / name)
            weight_map.update(dict.fromkeys(keys, name))
        (output / "diffusion_pytorch_model.safetensors.index.json").write_text(
            json.dumps({"metadata": {"total_size": total_dense_bytes}, "weight_map": weight_map}, indent=2) + "\n"
        )
        shutil.copy2(config_path, output / "config.json")
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(destination)
        output.rename(destination)
    print(f"Exported {len(packed) // 4} packed NVFP4 linears and {len(shard_keys)} dense shards to {destination}",
          flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, required=True, help="Diffusers model folder or its transformer folder")
    parser.add_argument("--dst", type=Path, help="Transformer output (default: <source model>/nvfp4/transformer)")
    parser.add_argument("--dense-shard-gb", type=float, default=2.0, help="Target dense shard size in GB")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if not math.isfinite(args.dense_shard_gb) or args.dense_shard_gb < 1e-9:
        parser.error("--dense-shard-gb must be a finite positive size")

    source = args.src.expanduser().absolute()
    if source.name == "transformer" and not (source / "model_index.json").is_file():
        source = source.parent
    destination = args.dst.expanduser().absolute() if args.dst else source / "nvfp4" / "transformer"
    export_checkpoint(source, destination, args.device, dense_shard_bytes=int(args.dense_shard_gb * 1e9))


if __name__ == "__main__":
    main()
