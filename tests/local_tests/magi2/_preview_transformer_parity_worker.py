# SPDX-License-Identifier: Apache-2.0
"""Run one isolated side of MAGI-2 preview transformer numerical parity."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import random
import sys
from typing import Any

import numpy as np
import torch
import torch.distributed as dist


REPO_ROOT = Path(__file__).resolve().parents[3]
OFFICIAL_ROOT = Path(
    os.environ.get("MAGI2_OFFICIAL_REF_DIR", REPO_ROOT.parent / "MAGI-2-preview")
)
WEIGHTS_ROOT = Path(
    os.environ.get("MAGI2_LOCAL_WEIGHTS_DIR", REPO_ROOT / "official_weights" / "magi2")
)
WORLD_SIZE = 8
SEED = 42


def _parse_args() -> argparse.Namespace:
    """Parse the implementation and artifact directory for one torchrun job."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--implementation",
        choices=("official", "fastvideo"),
        required=True,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _enable_determinism() -> None:
    """Apply the deterministic controls used by the official entry point."""
    os.environ["MAGI2_DETERMINISTIC"] = "1"
    os.environ["MAGI_ATTENTION_DETERMINISTIC_MODE"] = "1"
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.use_deterministic_algorithms(True)


def _initialize_distributed(implementation: str) -> tuple[int, torch.device]:
    """Initialize the official eight-rank CP and EP process-group topology."""
    if dist.is_initialized():
        raise RuntimeError("The parity worker requires a fresh distributed process")
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    if dist.get_world_size() != WORLD_SIZE:
        raise RuntimeError(
            f"MAGI-2 preview parity requires {WORLD_SIZE} ranks, "
            f"received {dist.get_world_size()}"
        )
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)

    if implementation == "official":
        sys.path.insert(0, str(OFFICIAL_ROOT))
        from inference.infra.distributed import (
            initialize_expert_parallel,
            initialize_model_parallel,
        )
    else:
        sys.path.insert(0, str(REPO_ROOT))
        from fastvideo.models.dits.magi2_runtime import (
            initialize_expert_parallel,
            initialize_model_parallel,
        )

    initialize_model_parallel(cp_size=WORLD_SIZE)
    initialize_expert_parallel(ep_size=WORLD_SIZE)
    return rank, torch.device("cuda", local_rank)


def _load_runtime(implementation: str, device: torch.device) -> tuple[Any, Any, type]:
    """Load one implementation's production transformer, proxy, and input type."""
    if implementation == "official":
        from inference.common.magi2_config import load_config
        from inference.infra.checkpoint.load_checkpoint import load_magi2_model
        from inference.pipeline.preview_data_proxy import Magi2DataProxy, ModelInput

        config = load_config(str(OFFICIAL_ROOT / "configs" / "magi2_preview.json"))
        config.engine_config.load = str(WEIGHTS_ROOT / "preview")
        config.engine_config.cp_size = WORLD_SIZE
        config.engine_config.ep_size = WORLD_SIZE
        model = load_magi2_model(config)
        proxy = Magi2DataProxy(config.evaluation_config.data_proxy_config)
        return model, proxy, ModelInput

    from fastvideo.configs.models.dits.magi2 import Magi2PreviewVideoConfig
    from fastvideo.models.dits.magi2_loader import load_magi2_preview_model
    from fastvideo.pipelines.basic.magi2.stages.preview_data_proxy import (
        Magi2DataProxy,
        Magi2DataProxyConfig,
        ModelInput,
    )

    with (OFFICIAL_ROOT / "configs" / "magi2_preview.json").open(
        encoding="utf-8"
    ) as config_file:
        proxy_values = json.load(config_file)["evaluation_config"]["data_proxy_config"]
    model = load_magi2_preview_model(
        checkpoint_dir=str(WEIGHTS_ROOT / "preview"),
        config=Magi2PreviewVideoConfig(),
        device=device,
    )
    proxy = Magi2DataProxy(Magi2DataProxyConfig(**proxy_values))
    return model, proxy, ModelInput


def _pattern(
    shape: tuple[int, ...],
    *,
    offset: int,
    scale: float,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Create a stable, nonuniform tensor without consuming random numbers."""
    element_count = math.prod(shape)
    values = torch.arange(element_count, dtype=torch.float32)
    values = ((values + offset).remainder(127) - 63) * scale
    return values.reshape(shape).to(dtype=dtype)


def _build_model_input(
    model_input_type: type,
    mode: str,
    device: torch.device,
) -> Any:
    """Build a compact CFG input that exercises every preview modality."""
    video = _pattern((1, 48, 2, 2, 2), offset=3, scale=1 / 64).repeat(2, 1, 1, 1, 1)
    audio = _pattern((1, 4, 64), offset=11, scale=1 / 32).repeat(2, 1, 1)
    text = _pattern(
        (2, 3, 5120),
        offset=29,
        scale=1 / 128,
        dtype=torch.bfloat16,
    )
    normalized_time = torch.tensor([0.75, 0.75], dtype=torch.float32)
    input_values: dict[str, Any] = {
        "x_t": video.to(device),
        "audio_x_t": audio.to(device),
        "audio_feat_len": torch.tensor([4, 4], dtype=torch.int32, device=device),
        "txt_feat": text.to(device),
        "txt_feat_len": torch.tensor([3, 2], dtype=torch.int32, device=device),
        "t": normalized_time.to(device),
        "per_token_video_t": normalized_time.view(2, 1, 1, 1, 1)
        .expand(2, 1, 2, 2, 2)
        .clone()
        .to(device),
        "per_token_audio_t": normalized_time.view(2, 1, 1)
        .expand(2, 4, 1)
        .clone()
        .to(device),
    }
    if mode == "i2v":
        conditioned_image = _pattern(
            (1, 1, 48, 1, 2, 2),
            offset=47,
            scale=1 / 64,
        )
        conditioned_token = _pattern(
            (1, 1, 5120),
            offset=71,
            scale=1 / 128,
            dtype=torch.bfloat16,
        )
        input_values.update(
            {
                "ref_image_feat": torch.cat(
                    [conditioned_image, torch.zeros_like(conditioned_image)],
                    dim=0,
                ).to(device),
                "ref_image_feat_len": torch.tensor(
                    [[[2, 2]], [[2, 2]]],
                    dtype=torch.int32,
                    device=device,
                ),
                "ref_image_special_token_embedding": torch.cat(
                    [conditioned_token, torch.zeros_like(conditioned_token)],
                    dim=0,
                ).to(device),
            }
        )
    elif mode != "t2v":
        raise ValueError(f"Unknown preview parity mode: {mode}")
    return model_input_type(**input_values)


def _tensor_record(tensor: torch.Tensor) -> dict[str, Any]:
    """Copy tensor values to CPU and retain the source tensor metadata."""
    return {
        "shape": tuple(tensor.shape),
        "dtype": str(tensor.dtype),
        "stride": tuple(tensor.stride()),
        "layout": str(tensor.layout),
        "device_type": tensor.device.type,
        "device_index": tensor.device.index,
        "requires_grad": tensor.requires_grad,
        "is_contiguous": tensor.is_contiguous(),
        "value": tensor.detach().to(device="cpu").contiguous(),
    }


def _packed_input_record(packed_input: tuple[Any, ...]) -> dict[str, Any]:
    """Record every tensor and sequence boundary passed into the transformer."""
    tokens, coords, modalities, varlen_handler, time_tokens = packed_input
    return {
        "tokens": _tensor_record(tokens),
        "coords": _tensor_record(coords),
        "modalities": _tensor_record(modalities),
        "time_tokens": _tensor_record(time_tokens),
        "cu_seqlens_q": _tensor_record(varlen_handler.cu_seqlens_q),
        "cu_seqlens_k": _tensor_record(varlen_handler.cu_seqlens_k),
        "max_seqlen_q": varlen_handler.max_seqlen_q,
        "max_seqlen_k": varlen_handler.max_seqlen_k,
    }


def _capture_model_boundaries(
    model: torch.nn.Module,
) -> tuple[dict[str, Any], list[torch.utils.hooks.RemovableHandle]]:
    """Attach eager hooks at the adapters and every transformer layer boundary."""
    capture: dict[str, Any] = {
        "pre_adapter": {},
        "layer_boundaries": [
            {"layer_index": layer_index} for layer_index in range(len(model.block.layers))
        ],
    }
    handles: list[torch.utils.hooks.RemovableHandle] = []

    def capture_pre_adapter(
        _module: torch.nn.Module,
        _inputs: tuple[Any, ...],
        output: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        """Record projected multimodal tokens and rotary embeddings."""
        hidden_states, rope = output
        capture["pre_adapter"] = {
            "hidden_states": _tensor_record(hidden_states),
            "rope": _tensor_record(rope),
        }

    def make_layer_input_hook(layer_index: int):
        """Create a hook that records one layer's input hidden states."""

        def capture_layer_input(
            _module: torch.nn.Module,
            inputs: tuple[Any, ...],
        ) -> None:
            """Record the hidden states entering one transformer layer."""
            capture["layer_boundaries"][layer_index]["input"] = _tensor_record(
                inputs[0]
            )

        return capture_layer_input

    def make_layer_output_hook(layer_index: int):
        """Create a hook that records one layer's output hidden states."""

        def capture_layer_output(
            _module: torch.nn.Module,
            _inputs: tuple[Any, ...],
            output: torch.Tensor,
        ) -> None:
            """Record the hidden states leaving one transformer layer."""
            capture["layer_boundaries"][layer_index]["output"] = _tensor_record(output)

        return capture_layer_output

    def capture_post_adapter(
        _module: torch.nn.Module,
        _inputs: tuple[Any, ...],
        output: torch.Tensor,
    ) -> None:
        """Record the local video-and-audio projection before CP gathering."""
        capture["post_adapter"] = _tensor_record(output)

    handles.append(model.pre_adapter.register_forward_hook(capture_pre_adapter))
    for layer_index, layer in enumerate(model.block.layers):
        handles.append(
            layer.register_forward_pre_hook(make_layer_input_hook(layer_index))
        )
        handles.append(
            layer.register_forward_hook(make_layer_output_hook(layer_index))
        )
    handles.append(model.post_adapter.register_forward_hook(capture_post_adapter))
    return capture, handles


def _run_case(
    model: torch.nn.Module,
    proxy: Any,
    model_input_type: type,
    mode: str,
    device: torch.device,
) -> dict[str, Any]:
    """Run one T2V or I2V forward pass and capture all parity boundaries."""
    model_input = _build_model_input(model_input_type, mode, device)
    packed_input = proxy.process_input(model_input)
    capture, handles = _capture_model_boundaries(model)
    capture["packed_input"] = _packed_input_record(packed_input)
    try:
        with torch.inference_mode():
            model_output = model(*packed_input)
    finally:
        for handle in handles:
            handle.remove()
    video_output, audio_output = proxy.process_output(model_output)
    capture["model_output"] = _tensor_record(model_output)
    capture["depacked_output"] = {
        "video": _tensor_record(video_output),
        "audio": _tensor_record(audio_output),
    }
    if any(
        set(layer_capture) != {"layer_index", "input", "output"}
        for layer_capture in capture["layer_boundaries"]
    ):
        raise RuntimeError("A preview transformer layer hook did not execute")
    return capture


def main() -> None:
    """Load one implementation, run both modalities, and save rank-local captures."""
    args = _parse_args()
    os.environ.setdefault("MAGI_COMPILE_COMPILE_MODE", "NONE")
    os.environ.pop("SKIP_LOAD_MODEL", None)
    os.environ["MAGI2_CKPT_ROOT"] = str(WEIGHTS_ROOT)
    _enable_determinism()
    rank, device = _initialize_distributed(args.implementation)
    model, proxy, model_input_type = _load_runtime(args.implementation, device)
    artifact = {
        "schema_version": 1,
        "implementation": args.implementation,
        "rank": rank,
        "world_size": dist.get_world_size(),
        "cases": {
            mode: _run_case(model, proxy, model_input_type, mode, device)
            for mode in ("t2v", "i2v")
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = args.output_dir / f"rank_{rank}.pt"
    temporary_path = args.output_dir / f"rank_{rank}.pt.tmp"
    torch.save(artifact, temporary_path)
    os.replace(temporary_path, artifact_path)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
