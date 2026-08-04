# SPDX-License-Identifier: Apache-2.0
"""Production-loader parity for both MiniMax H3 DiT partitions.

This gated CUDA test loads the published ``transformer/`` and
``transformer_ref/`` checkpoints through FastVideo's production
``TransformerLoader`` and compares both output heads against the pinned
Diffusers implementation.  It is intentionally excluded from routine local
and CI runs because each partition contains roughly 33B parameters.
"""

from __future__ import annotations

import gc
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch.testing import assert_close


REPO_ROOT = Path(__file__).resolve().parents[3]
OFFICIAL_REF_DIR = Path(os.environ.get("MINIMAX_H3_OFFICIAL_REF_DIR", REPO_ROOT / "DiffusersMiniMaxH3"))
OFFICIAL_SRC = OFFICIAL_REF_DIR / "src"
MODEL_ROOT = Path(os.environ.get("MINIMAX_H3_MODEL_ROOT", REPO_ROOT / "official_weights" / "MiniMax-H3"))
RUN_ENV = "MINIMAX_H3_RUN_DIT_PARITY"
PARITY_SCOPE = "production_loader"

if os.environ.get(RUN_ENV) == "1":
    os.environ.setdefault("FASTVIDEO_ATTENTION_BACKEND", "TORCH_SDPA")
    os.environ.setdefault("DIFFUSERS_ATTN_BACKEND", "native")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29659")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    # Put the pinned checkout ahead of the environment's Diffusers installation
    # before any FastVideo import can populate ``sys.modules``.
    if OFFICIAL_SRC.is_dir():
        sys.path.insert(0, str(OFFICIAL_SRC))


@pytest.fixture(scope="module", autouse=True)
def _distributed_runtime():
    if os.environ.get(RUN_ENV) != "1":
        yield
        return

    from fastvideo.distributed import cleanup_dist_env_and_memory, maybe_init_distributed_environment_and_model_parallel

    maybe_init_distributed_environment_and_model_parallel(1, 1)
    yield
    cleanup_dist_env_and_memory()


def _require_assets(partition: str) -> tuple[torch.device, Path]:
    if os.environ.get(RUN_ENV) != "1":
        pytest.skip(f"set {RUN_ENV}=1 on an allocated CUDA node")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        pytest.fail("MiniMax H3 DiT parity requires a bf16-capable CUDA GPU", pytrace=False)
    if not OFFICIAL_SRC.is_dir():
        pytest.fail(f"pinned Diffusers MiniMax H3 checkout is missing: {OFFICIAL_REF_DIR}", pytrace=False)

    component_dir = MODEL_ROOT / partition
    required = (component_dir / "config.json", component_dir / "diffusion_pytorch_model.safetensors.index.json")
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        pytest.fail(f"MiniMax H3 {partition} assets are missing: {missing}", pytrace=False)
    return torch.device("cuda:0"), component_dir


def _make_inputs() -> dict[str, torch.Tensor]:
    """Create a small padless packed layout that exercises all three modalities."""
    generator = torch.Generator(device="cpu").manual_seed(20260803)
    num_text_tokens = 2
    num_audio_tokens = 2
    num_video_tokens = 4
    sequence_length = num_text_tokens + num_audio_tokens + num_video_tokens

    text_indices = torch.arange(num_text_tokens)
    audio_indices = torch.arange(num_text_tokens, num_text_tokens + num_audio_tokens)
    video_indices = torch.arange(num_text_tokens + num_audio_tokens, sequence_length)

    token_tags = torch.empty(sequence_length, dtype=torch.long)
    token_tags[text_indices] = 1
    token_tags[audio_indices] = 2
    token_tags[video_indices] = 0

    timestep_indices = torch.zeros(sequence_length, dtype=torch.long)
    timestep_indices[audio_indices] = 1
    timestep_indices[video_indices[:2]] = 2

    position_ids = torch.zeros(sequence_length, 3, dtype=torch.float32)
    position_ids[:, 0] = torch.arange(sequence_length, dtype=torch.float32)
    position_ids[video_indices, 1] = torch.arange(num_video_tokens, dtype=torch.float32) % 2
    position_ids[video_indices, 2] = torch.arange(num_video_tokens, dtype=torch.float32) // 2

    return {
        "hidden_states": torch.randn(1, num_video_tokens, 96, generator=generator),
        "audio_hidden_states": torch.randn(1, num_audio_tokens, 32, generator=generator),
        "encoder_hidden_states": torch.randn(1, num_text_tokens, 5120, generator=generator),
        "timestep": torch.tensor([0.7, 0.3, 0.999], dtype=torch.float32),
        "timestep_indices": timestep_indices,
        "token_tags": token_tags,
        "position_ids": position_ids,
        "video_indices": video_indices,
        "audio_indices": audio_indices,
        "text_indices": text_indices,
    }


def _to_device(inputs: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {name: tensor.to(device) for name, tensor in inputs.items()}


def _assert_mixed_dtype_contract(model: torch.nn.Module) -> None:
    parameters = dict(model.named_parameters())
    assert parameters["proj_in.weight"].dtype == torch.float32
    assert parameters["audio_proj_in.weight"].dtype == torch.float32
    assert parameters["time_embedder.fc_in.weight" if "time_embedder.fc_in.weight" in parameters
                      else "time_embedder.linear_1.weight"].dtype == torch.float32
    assert parameters["proj_out.weight"].dtype == torch.float32
    assert parameters["audio_proj_out.weight"].dtype == torch.float32
    assert parameters["context_embedder.weight"].dtype == torch.bfloat16
    assert parameters["transformer_blocks.0.attn.to_q.weight"].dtype == torch.bfloat16


def _load_official(component_dir: Path, device: torch.device) -> torch.nn.Module:
    from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3Transformer3DModel
    from tests.local_tests.minimax_h3._reference import assert_reference_source

    assert_reference_source(
        MiniMaxH3Transformer3DModel,
        "src/diffusers/models/transformers/transformer_minimax_h3.py",
    )

    model, loading_info = MiniMaxH3Transformer3DModel.from_pretrained(
        component_dir,
        local_files_only=True,
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
        device_map=device,
        output_loading_info=True,
    )
    load_errors = {
        name: loading_info.get(name, [])
        for name in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
        if loading_info.get(name)
    }
    assert not load_errors, f"official MiniMax H3 checkpoint did not load strictly: {load_errors}"
    model = model.eval()
    _assert_mixed_dtype_contract(model)
    return model


def _production_loader_args() -> SimpleNamespace:
    from fastvideo.configs.pipelines.minimax_h3 import MiniMaxH3PipelineConfig

    return SimpleNamespace(
        pipeline_config=MiniMaxH3PipelineConfig(),
        model_paths={},
        override_transformer_cls_name=None,
        init_weights_from_safetensors=None,
        hsdp_replicate_dim=1,
        hsdp_shard_dim=1,
        dit_cpu_offload=False,
        pin_cpu_memory=False,
        use_fsdp_inference=True,
        training_mode=False,
        enable_torch_compile=False,
        torch_compile_kwargs={},
        inference_mode=True,
        dit_layerwise_offload=False,
    )


def _load_fastvideo(component_dir: Path) -> torch.nn.Module:
    from fastvideo.models.loader.component_loader import TransformerLoader

    model = TransformerLoader().load(str(component_dir), _production_loader_args())
    _assert_mixed_dtype_contract(model)
    return model


def _run_official(
    model: torch.nn.Module,
    inputs: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.inference_mode():
        video, audio = model(**inputs, return_dict=False)
    return video.detach().float().cpu(), audio.detach().float().cpu()


def _run_fastvideo(
    model: torch.nn.Module,
    inputs: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    from fastvideo.forward_context import set_forward_context

    with torch.inference_mode(), set_forward_context(current_timestep=0, attn_metadata=None):
        video, audio = model(**inputs)
    return video.detach().float().cpu(), audio.detach().float().cpu()


def _reclaim_vram() -> None:
    gc.collect()
    torch.cuda.empty_cache()


def _assert_head_parity(name: str, actual: torch.Tensor, expected: torch.Tensor) -> None:
    assert actual.shape == expected.shape
    drift = (actual - expected).abs()
    reference_abs_mean = expected.abs().mean().item()
    relative_mean_drift = drift.mean().item() / max(reference_abs_mean, 1e-8)
    print(
        f"{name}: reference_abs_mean={reference_abs_mean:.8f} "
        f"max_abs={drift.max().item():.8f} mean_abs={drift.mean().item():.8f} "
        f"relative_mean={relative_mean_drift:.8f}",
        flush=True,
    )
    assert relative_mean_drift == 0.0
    assert_close(actual, expected, atol=0.0, rtol=0.0)


@pytest.mark.parametrize("partition", ("transformer", "transformer_ref"))
def test_minimax_h3_transformer_parity(partition: str) -> None:
    """Match both DiT partitions against the pinned official implementation."""
    device, component_dir = _require_assets(partition)
    cpu_inputs = _make_inputs()

    official = _load_official(component_dir, device)
    expected_video, expected_audio = _run_official(official, _to_device(cpu_inputs, device))
    del official
    _reclaim_vram()

    fastvideo = _load_fastvideo(component_dir)
    actual_video, actual_audio = _run_fastvideo(fastvideo, _to_device(cpu_inputs, device))
    del fastvideo
    _reclaim_vram()

    _assert_head_parity(f"{partition}.video", actual_video, expected_video)
    _assert_head_parity(f"{partition}.audio", actual_audio, expected_audio)
