# SPDX-License-Identifier: Apache-2.0
"""Helios transformer config, weight-schema, and tiny forward parity.

Coverage scope: both. The small model exercises all three history terms and
frame-indexed 3D RoPE without allocating the 40-layer checkpoint. Official
Diffusers weights are loaded strictly into the native FastVideo key schema
before comparing deterministic float32 outputs.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import socket
import subprocess
import sys
from dataclasses import fields
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
from diffusers import HeliosTransformer3DModel as OfficialHeliosTransformer3DModel
from torch.testing import assert_close

from fastvideo.forward_context import set_forward_context
from fastvideo.models.loader.fsdp_load import load_model_from_full_model_state_dict
from fastvideo.models.loader.utils import get_param_names_mapping, set_default_torch_dtype
from fastvideo.models.loader.weight_utils import (
    resolve_safetensors_files,
    safetensors_weights_iterator,
)

os.environ.setdefault("DISABLE_SP", "1")
os.environ.setdefault("FASTVIDEO_ATTENTION_BACKEND", "TORCH_SDPA")

REPO_ROOT = Path(__file__).resolve().parents[3]
TRANSFORMER_DIR = Path(
    os.getenv(
        "HELIOS_TRANSFORMER_DIR",
        REPO_ROOT / "official_weights" / "helios" / "transformer",
    ))
HF_REVISION = "1999182614cb08d3bdcc46b9827504af2914b87b"
PARITY_SCOPE = "both"
SP_WORLD_SIZE = 2


def _native_types():
    try:
        from fastvideo.configs.models.dits.helios import (
            HeliosArchConfig,
            HeliosConfig,
        )
        from fastvideo.models.dits.helios import HeliosTransformer3DModel
    except ImportError as exc:
        raise AssertionError("Native FastVideo Helios transformer/config have not been implemented yet") from exc
    return HeliosArchConfig, HeliosConfig, HeliosTransformer3DModel


def _tiny_kwargs() -> dict:
    return {
        "patch_size": (1, 2, 2),
        "num_attention_heads": 2,
        "attention_head_dim": 32,
        "in_channels": 4,
        "out_channels": 4,
        "text_dim": 48,
        "freq_dim": 32,
        "ffn_dim": 128,
        "num_layers": 1,
        "cross_attn_norm": True,
        "qk_norm": "rms_norm_across_heads",
        "eps": 1e-6,
        "added_kv_proj_dim": None,
        "rope_dim": (12, 10, 10),
        "rope_theta": 10000.0,
        "guidance_cross_attn": True,
        "zero_history_timestep": True,
        "has_multi_term_memory_patch": True,
        "is_amplify_history": False,
        "history_scale_mode": "per_head",
    }


def _make_inputs() -> dict[str, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(3535)
    return {
        "hidden_states": torch.randn(1, 4, 2, 8, 8, generator=generator),
        "timestep": torch.tensor([517], dtype=torch.long),
        "encoder_hidden_states": torch.randn(1, 5, 48, generator=generator),
        "indices_hidden_states": torch.tensor([[19, 20]]),
        "indices_latents_history_short": torch.tensor([[17, 18]]),
        "indices_latents_history_mid": torch.tensor([[15, 16]]),
        "indices_latents_history_long": torch.tensor([[11, 12, 13, 14]]),
        "latents_history_short": torch.randn(1, 4, 2, 8, 8, generator=generator),
        "latents_history_mid": torch.randn(1, 4, 2, 8, 8, generator=generator),
        "latents_history_long": torch.randn(1, 4, 4, 8, 8, generator=generator),
    }


def _make_pyramid_inputs() -> dict[str, torch.Tensor]:
    inputs = _make_inputs()
    inputs["hidden_states"] = inputs["hidden_states"][:, :, :, :2, :2]
    return inputs


def _make_real_inputs(device: torch.device) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device=device).manual_seed(1999)
    dtype = torch.bfloat16
    return {
        "hidden_states": torch.randn(1, 16, 1, 8, 8, device=device, dtype=dtype, generator=generator),
        "timestep": torch.tensor([517], device=device, dtype=torch.long),
        "encoder_hidden_states": torch.randn(1, 8, 4096, device=device, dtype=dtype, generator=generator),
        "indices_hidden_states": torch.tensor([[20]], device=device),
        "indices_latents_history_short": torch.tensor([[18, 19]], device=device),
        "indices_latents_history_mid": torch.tensor([[16, 17]], device=device),
        "indices_latents_history_long": torch.tensor([[12, 13, 14, 15]], device=device),
        "latents_history_short": torch.randn(1, 16, 2, 8, 8, device=device, dtype=dtype, generator=generator),
        "latents_history_mid": torch.randn(1, 16, 2, 8, 8, device=device, dtype=dtype, generator=generator),
        "latents_history_long": torch.randn(1, 16, 4, 8, 8, device=device, dtype=dtype, generator=generator),
    }


def _make_real_pyramid_inputs(device: torch.device) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device=device).manual_seed(2001)
    dtype = torch.bfloat16
    return {
        "hidden_states": torch.randn(1, 16, 9, 4, 6, device=device, dtype=dtype, generator=generator),
        "timestep": torch.tensor([999], device=device, dtype=torch.long),
        "encoder_hidden_states": torch.randn(1, 512, 4096, device=device, dtype=dtype, generator=generator),
        "indices_hidden_states": torch.arange(20, 29, device=device).unsqueeze(0),
        "indices_latents_history_short": torch.tensor([[0, 19]], device=device),
        "indices_latents_history_mid": torch.tensor([[17, 18]], device=device),
        "indices_latents_history_long": torch.arange(1, 17, device=device).unsqueeze(0),
        "latents_history_short": torch.randn(1, 16, 2, 16, 24, device=device, dtype=dtype, generator=generator),
        "latents_history_mid": torch.randn(1, 16, 2, 16, 24, device=device, dtype=dtype, generator=generator),
        "latents_history_long": torch.randn(1, 16, 16, 16, 24, device=device, dtype=dtype, generator=generator),
    }


def _move_inputs(
    inputs: dict[str, torch.Tensor],
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    return {
        name: value.to(device=device, dtype=dtype if value.is_floating_point() else value.dtype)
        for name, value in inputs.items()
    }


def _load_tiny_fastvideo(
    backend,
    device: torch.device,
    dtype: torch.dtype,
):
    from fastvideo.attention.selector import global_force_attn_backend_context_manager

    HeliosArchConfig, HeliosConfig, FastVideoHeliosTransformer = _native_types()
    kwargs = _tiny_kwargs()
    torch.manual_seed(3535)
    official = OfficialHeliosTransformer3DModel(**kwargs).to(dtype=dtype).eval()
    with set_default_torch_dtype(dtype), global_force_attn_backend_context_manager(backend):
        native = FastVideoHeliosTransformer(
            config=HeliosConfig(arch_config=HeliosArchConfig(**kwargs)),
            hf_config={},
        ).eval()
    incompatible = load_model_from_full_model_state_dict(
        native,
        iter(official.state_dict().items()),
        device=device,
        param_dtype=dtype,
        strict=True,
        param_names_mapping=get_param_names_mapping(native.param_names_mapping),
        training_mode=False,
    )
    assert incompatible.missing_keys == []
    assert incompatible.unexpected_keys == []
    native.materialize_non_persistent_buffers(device, dtype)
    return native


def _has_real_weights() -> bool:
    return (TRANSFORMER_DIR / "diffusion_pytorch_model.safetensors.index.json").is_file() and any(
        TRANSFORMER_DIR.glob("*.safetensors"))


def _load_fastvideo_production():
    from fastvideo.configs.models.dits.helios import HeliosConfig
    from fastvideo.configs.pipelines.base import PipelineConfig
    from fastvideo.fastvideo_args import FastVideoArgs
    from fastvideo.models.dits.helios import HeliosTransformer3DModel
    from fastvideo.models.loader.component_loader import TransformerLoader

    args = FastVideoArgs(
        model_path=str(TRANSFORMER_DIR),
        dit_cpu_offload=False,
        dit_layerwise_offload=False,
        use_fsdp_inference=False,
        pipeline_config=PipelineConfig(
            dit_config=HeliosConfig(),
            dit_precision="bf16",
        ),
    )
    model = TransformerLoader().load(str(TRANSFORMER_DIR), args).eval()
    assert isinstance(model, HeliosTransformer3DModel)
    assert args.model_paths["transformer"] == str(TRANSFORMER_DIR)
    assert next(model.parameters()).device.type == "cuda"
    return model


def _assert_real_bf16_parity(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    scope: str,
) -> None:
    diff = (expected - actual).abs()
    expected_abs_mean = expected.abs().mean()
    actual_abs_mean = actual.abs().mean()
    abs_mean_drift = (actual_abs_mean - expected_abs_mean).abs() / expected_abs_mean.clamp_min(1e-6)
    print(f"{scope} official_abs_mean={expected_abs_mean.item():.8f} "
          f"fastvideo_abs_mean={actual_abs_mean.item():.8f} "
          f"abs_mean_drift={abs_mean_drift.item():.4%} "
          f"diff_max={diff.max().item():.8f} diff_mean={diff.mean().item():.8f}")
    assert abs_mean_drift < 0.01
    assert diff.mean() < 0.01
    assert_close(actual, expected, atol=5e-2, rtol=5e-2)


def test_helios_distilled_config_matches_pinned_checkpoint():
    HeliosArchConfig, _, _ = _native_types()
    config = HeliosArchConfig()
    assert config.patch_size == (1, 2, 2)
    assert config.num_attention_heads == 40
    assert config.attention_head_dim == 128
    assert config.hidden_size == 5120
    assert config.in_channels == config.out_channels == 16
    assert config.text_dim == 4096
    assert config.freq_dim == 256
    assert config.ffn_dim == 13824
    assert config.num_layers == 40
    assert config.rope_dim == (44, 42, 42)
    assert config.zero_history_timestep is True
    assert config.has_multi_term_memory_patch is True
    assert config.guidance_cross_attn is True

    config_path = TRANSFORMER_DIR / "config.json"
    if config_path.is_file():
        checkpoint_config = json.loads(config_path.read_text(encoding="utf-8"))
        assert checkpoint_config.pop("_class_name") == "HeliosTransformer3DModel"
        checkpoint_config.pop("_diffusers_version", None)
        arch_fields = {field.name for field in fields(HeliosArchConfig)}
        assert set(checkpoint_config) <= arch_fields
        for name, expected in checkpoint_config.items():
            actual = getattr(config, name)
            if isinstance(actual, tuple):
                expected = tuple(expected)
            assert actual == expected, f"unexpected Helios config {name}={actual!r}"


@pytest.mark.parametrize(
    ("unsupported_override", "message"),
    [
        ({"cross_attn_norm": False}, "cross_attn_norm"),
        ({"qk_norm": None}, "qk_norm"),
        ({"added_kv_proj_dim": 64}, "added_kv_proj_dim"),
        ({"guidance_cross_attn": False}, "guidance_cross_attn"),
        ({"zero_history_timestep": False}, "zero_history_timestep"),
        ({"has_multi_term_memory_patch": False}, "has_multi_term_memory_patch"),
        ({"is_amplify_history": True}, "is_amplify_history"),
        ({"history_scale_mode": "scalar"}, "history_scale_mode"),
    ],
)
def test_helios_arch_config_rejects_unverified_variants(
    unsupported_override: dict,
    message: str,
):
    """Variant knobs without parity evidence must fail instead of silently drifting."""
    HeliosArchConfig, _, _ = _native_types()
    kwargs = _tiny_kwargs()
    kwargs.update(unsupported_override)
    with pytest.raises(ValueError, match=message):
        HeliosArchConfig(**kwargs)


def test_helios_transformer_registry_resolves_native_class():
    from fastvideo.models.dits.helios import HeliosTransformer3DModel
    from fastvideo.models.registry import ModelRegistry

    model_cls, architecture = ModelRegistry.resolve_model_cls("HeliosTransformer3DModel")
    assert model_cls is HeliosTransformer3DModel
    assert architecture == "HeliosTransformer3DModel"


def test_helios_real_checkpoint_uses_identity_key_mapping(monkeypatch):
    HeliosArchConfig, HeliosConfig, FastVideoHeliosTransformer = _native_types()
    del HeliosArchConfig
    import fastvideo.models.dits.helios as fastvideo_helios

    monkeypatch.setattr(fastvideo_helios, "get_sp_world_size", lambda: 1)
    index_path = TRANSFORMER_DIR / "diffusion_pytorch_model.safetensors.index.json"
    if not index_path.exists():
        pytest.skip(
            "Pinned Helios transformer index is absent; set HELIOS_TRANSFORMER_DIR "
            f"to BestWishYsh/Helios-Distilled@{HF_REVISION}/transformer")
    with torch.device("meta"):
        native = FastVideoHeliosTransformer(config=HeliosConfig(), hf_config={})
    official_keys = set(json.loads(index_path.read_text(encoding="utf-8"))["weight_map"])
    native_keys = set(native.state_dict())
    assert native_keys == official_keys
    assert native.param_names_mapping == {}


def test_helios_real_checkpoint_strict_loads(monkeypatch):
    _, HeliosConfig, FastVideoHeliosTransformer = _native_types()
    import fastvideo.models.dits.helios as fastvideo_helios

    monkeypatch.setattr(fastvideo_helios, "get_sp_world_size", lambda: 1)
    if not _has_real_weights():
        pytest.skip(f"Pinned Helios transformer shards missing: {TRANSFORMER_DIR}")
    files = resolve_safetensors_files(str(TRANSFORMER_DIR))
    with torch.device("meta"):
        native = FastVideoHeliosTransformer(config=HeliosConfig(), hf_config={})
    incompatible = load_model_from_full_model_state_dict(
        native,
        safetensors_weights_iterator(files, to_cpu=True),
        device=torch.device("cpu"),
        param_dtype=torch.bfloat16,
        strict=True,
        param_names_mapping=get_param_names_mapping(native.param_names_mapping),
        training_mode=False,
    )
    assert incompatible.missing_keys == []
    assert incompatible.unexpected_keys == []
    native.materialize_non_persistent_buffers(torch.device("cpu"), torch.bfloat16)
    assert not any(parameter.is_meta for parameter in native.parameters())
    assert not any(buffer.is_meta for buffer in native.buffers())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for full transformer parity.")
def test_helios_real_transformer_forward_parity(monkeypatch):
    _, HeliosConfig, FastVideoHeliosTransformer = _native_types()
    import fastvideo.models.dits.helios as fastvideo_helios

    monkeypatch.setattr(fastvideo_helios, "get_sp_world_size", lambda: 1)
    if not _has_real_weights():
        pytest.skip(
            "Pinned Helios transformer weights are absent; set HELIOS_TRANSFORMER_DIR "
            f"to BestWishYsh/Helios-Distilled@{HF_REVISION}/transformer")
    device = torch.device("cuda:0")
    inputs = _make_real_inputs(device)
    pyramid_inputs = _make_real_pyramid_inputs(device)

    official = (OfficialHeliosTransformer3DModel.from_pretrained(
        str(TRANSFORMER_DIR),
        local_files_only=True,
        torch_dtype=torch.bfloat16,
    ).to(device).eval())
    with torch.inference_mode():
        official_output = official(**inputs, return_dict=False)[0].float().cpu()
        official_pyramid_output = official(**pyramid_inputs, return_dict=False)[0].float().cpu()
    del official
    gc.collect()
    torch.cuda.empty_cache()

    del HeliosConfig, FastVideoHeliosTransformer
    native = _load_fastvideo_production()
    with torch.inference_mode(), set_forward_context(current_timestep=0, attn_metadata=None):
        fastvideo_output = native(**inputs).float().cpu()
        fastvideo_pyramid_output = native(**pyramid_inputs).float().cpu()

    assert fastvideo_output.shape == official_output.shape == (1, 16, 1, 8, 8)
    _assert_real_bf16_parity(fastvideo_output, official_output, scope="real transformer")

    assert (official_pyramid_output.shape == fastvideo_pyramid_output.shape == (
        1,
        16,
        9,
        4,
        6,
    ))
    _assert_real_bf16_parity(
        fastvideo_pyramid_output,
        official_pyramid_output,
        scope="real pyramid transformer",
    )


def test_helios_tiny_transformer_strict_load_and_forward_parity(monkeypatch):
    HeliosArchConfig, HeliosConfig, FastVideoHeliosTransformer = _native_types()
    import fastvideo.models.dits.helios as fastvideo_helios

    monkeypatch.setattr(fastvideo_helios, "get_sp_world_size", lambda: 1)
    kwargs = _tiny_kwargs()
    torch.manual_seed(17)
    official = OfficialHeliosTransformer3DModel(**kwargs).float().eval()
    fastvideo = (FastVideoHeliosTransformer(
        config=HeliosConfig(arch_config=HeliosArchConfig(**kwargs)),
        hf_config={},
    ).float().eval())

    incompatible = load_model_from_full_model_state_dict(
        fastvideo,
        iter(official.state_dict().items()),
        device=torch.device("cpu"),
        param_dtype=torch.float32,
        strict=True,
        param_names_mapping=get_param_names_mapping(fastvideo.param_names_mapping),
        training_mode=False,
    )
    assert incompatible.missing_keys == []
    assert incompatible.unexpected_keys == []

    inputs = _make_inputs()
    with torch.inference_mode():
        official_output = official(**inputs, return_dict=False)[0]
        with set_forward_context(current_timestep=0, attn_metadata=None):
            fastvideo_output = fastvideo(**inputs)

    assert official_output.shape == fastvideo_output.shape == (1, 4, 2, 8, 8)
    diff = (official_output - fastvideo_output).abs()
    print(f"tiny transformer diff_max={diff.max().item():.8f} diff_mean={diff.mean().item():.8f}")
    assert_close(fastvideo_output, official_output, atol=1e-5, rtol=1e-5)


def test_helios_tiny_transformer_pyramid_geometry_parity(monkeypatch):
    """Current latents shrink per stage while history stays full resolution."""
    HeliosArchConfig, HeliosConfig, FastVideoHeliosTransformer = _native_types()
    import fastvideo.models.dits.helios as fastvideo_helios

    monkeypatch.setattr(fastvideo_helios, "get_sp_world_size", lambda: 1)
    kwargs = _tiny_kwargs()
    torch.manual_seed(23)
    official = OfficialHeliosTransformer3DModel(**kwargs).float().eval()
    fastvideo = (FastVideoHeliosTransformer(
        config=HeliosConfig(arch_config=HeliosArchConfig(**kwargs)),
        hf_config={},
    ).float().eval())
    incompatible = load_model_from_full_model_state_dict(
        fastvideo,
        iter(official.state_dict().items()),
        device=torch.device("cpu"),
        param_dtype=torch.float32,
        strict=True,
        param_names_mapping=get_param_names_mapping(fastvideo.param_names_mapping),
        training_mode=False,
    )
    assert incompatible.missing_keys == []
    assert incompatible.unexpected_keys == []

    inputs = _make_pyramid_inputs()
    cross_attention_query_lengths = []

    def record_cross_attention_query_length(module, args):
        del module
        cross_attention_query_lengths.append(args[0].shape[1])

    handle = fastvideo.blocks[0].attn2.register_forward_pre_hook(record_cross_attention_query_length)
    with torch.inference_mode():
        official_output = official(**inputs, return_dict=False)[0]
        try:
            with set_forward_context(current_timestep=0, attn_metadata=None):
                fastvideo_output = fastvideo(**inputs)
        finally:
            handle.remove()

    assert official_output.shape == fastvideo_output.shape == (1, 4, 2, 2, 2)
    assert cross_attention_query_lengths == [2]
    diff = (official_output - fastvideo_output).abs()
    print(f"pyramid transformer diff_max={diff.max().item():.8f} diff_mean={diff.mean().item():.8f}")
    assert_close(fastvideo_output, official_output, atol=1e-5, rtol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for FlashAttention parity.")
def test_helios_tiny_flash_attention_matches_sdpa(monkeypatch):
    """Both supported attention backends must execute with bounded BF16 drift."""
    pytest.importorskip("flash_attn", reason="Install optional flash-attn to verify the FLASH_ATTN backend.")
    from fastvideo.platforms import AttentionBackendEnum
    import fastvideo.models.dits.helios as fastvideo_helios

    monkeypatch.setattr(fastvideo_helios, "get_sp_world_size", lambda: 1)
    device = torch.device("cuda:0")
    inputs = _move_inputs(_make_inputs(), device, torch.bfloat16)
    sdpa = _load_tiny_fastvideo(AttentionBackendEnum.TORCH_SDPA, device, torch.bfloat16)
    flash = _load_tiny_fastvideo(AttentionBackendEnum.FLASH_ATTN, device, torch.bfloat16)
    assert sdpa.blocks[0].attn1.attn.backend is AttentionBackendEnum.TORCH_SDPA
    assert flash.blocks[0].attn1.attn.backend is AttentionBackendEnum.FLASH_ATTN
    assert flash.blocks[0].attn2.attn.backend is AttentionBackendEnum.FLASH_ATTN

    with torch.inference_mode(), set_forward_context(current_timestep=0, attn_metadata=None):
        sdpa_output = sdpa(**inputs).float()
    with torch.inference_mode(), set_forward_context(current_timestep=0, attn_metadata=None):
        flash_output = flash(**inputs).float()

    assert torch.isfinite(flash_output).all()
    diff = (flash_output - sdpa_output).abs()
    print(f"flash-vs-sdpa diff_max={diff.max().item():.8f} diff_mean={diff.mean().item():.8f}")
    assert diff.mean() < 1e-2
    assert_close(flash_output, sdpa_output, atol=5e-2, rtol=5e-2)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _run_sp_worker(mode: str, output_path: Path) -> None:
    from fastvideo.distributed import (
        cleanup_dist_env_and_memory,
        maybe_init_distributed_environment_and_model_parallel,
    )
    from fastvideo.platforms import AttentionBackendEnum

    if mode not in {"single", "sp"}:
        raise ValueError(f"Unsupported Helios SP worker mode: {mode}")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    sp_size = 1 if mode == "single" else SP_WORLD_SIZE
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    torch.manual_seed(3535)
    torch.cuda.manual_seed_all(3535)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    try:
        maybe_init_distributed_environment_and_model_parallel(1, sp_size)
        model = _load_tiny_fastvideo(AttentionBackendEnum.TORCH_SDPA, device, torch.float32)
        inputs = _move_inputs(_make_inputs(), device, torch.float32)
        with torch.inference_mode(), set_forward_context(current_timestep=0, attn_metadata=None):
            output = model(**inputs)
        assert torch.isfinite(output).all()
        if rank == 0:
            torch.save({"output": output.detach().cpu()}, output_path)
        dist.barrier()
    finally:
        cleanup_dist_env_and_memory()


def _run_torchrun(
    script_path: Path,
    mode: str,
    nproc_per_node: int,
    output_path: Path,
) -> None:
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nnodes",
        "1",
        "--nproc_per_node",
        str(nproc_per_node),
        "--master_port",
        str(_free_port()),
        str(script_path),
        "--helios-sp-worker",
        "--mode",
        mode,
        "--output",
        str(output_path),
    ]
    environment = os.environ.copy()
    environment["DISABLE_SP"] = "0"
    environment["FASTVIDEO_ATTENTION_BACKEND"] = "TORCH_SDPA"
    process = subprocess.run(command, capture_output=True, text=True, env=environment)
    if process.returncode != 0:
        raise RuntimeError(f"{mode} worker failed with code {process.returncode}\n"
                           f"STDOUT:\n{process.stdout}\n"
                           f"STDERR:\n{process.stderr}")


def test_helios_tiny_sp2_matches_single_rank(tmp_path: Path):
    """SP=2 must preserve the unpadded full output for unequal history geometry."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for Helios SP parity.")
    if torch.cuda.device_count() < SP_WORLD_SIZE:
        pytest.skip(f"Helios SP parity requires at least {SP_WORLD_SIZE} CUDA devices.")
    script_path = Path(__file__).resolve()
    single_path = tmp_path / "helios_single.pt"
    sp_path = tmp_path / "helios_sp2.pt"
    _run_torchrun(script_path, "single", 1, single_path)
    _run_torchrun(script_path, "sp", SP_WORLD_SIZE, sp_path)

    single_output = torch.load(single_path, map_location="cpu", weights_only=True)["output"]
    sp_output = torch.load(sp_path, map_location="cpu", weights_only=True)["output"]
    assert single_output.shape == sp_output.shape == (1, 4, 2, 8, 8)
    diff = (sp_output - single_output).abs()
    print(f"sp2-vs-single diff_max={diff.max().item():.8f} diff_mean={diff.mean().item():.8f}")
    assert_close(sp_output, single_output, atol=1e-5, rtol=1e-5)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--helios-sp-worker", action="store_true")
    parser.add_argument("--mode", choices=["single", "sp"], default=None)
    parser.add_argument("--output", type=str, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if not args.helios_sp_worker:
        raise SystemExit("This module is intended to be run by pytest.")
    if args.mode is None or args.output is None:
        raise SystemExit("--mode and --output are required in worker mode.")
    _run_sp_worker(args.mode, Path(args.output))
