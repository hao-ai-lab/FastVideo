# SPDX-License-Identifier: Apache-2.0
"""PAN2 transformer weight-schema and forward parity.

The reference is ``tests/local_tests/pan2/reference.py``: diffusers building blocks arranged as the (not yet public)
diffusers ``PAN2Transformer3DModel``, following the vLLM-Omni port. Its module names are the checkpoint's, so the
public tiny checkpoint loads into it strictly.

Set ``PAN2_TINY_MODEL_DIR`` to a local copy of ``wuqing157/tiny-pan2-modular-pipe``; otherwise the HF cache is
used and the checkpoint test skips when it is absent.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
from dataclasses import fields
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from torch.testing import assert_close

os.environ.setdefault("DISABLE_SP", "1")
os.environ.setdefault("FASTVIDEO_ATTENTION_BACKEND", "TORCH_SDPA")

from fastvideo.forward_context import set_forward_context  # noqa: E402
from fastvideo.models.loader.fsdp_load import load_model_from_full_model_state_dict  # noqa: E402
from fastvideo.models.loader.utils import get_param_names_mapping  # noqa: E402
from tests.local_tests.pan2.reference import TINY_REPO_ID, ReferencePAN2Transformer, tiny_model_dir  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[3]
SP_WORLD_SIZE = 2


# --------------------------------------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------------------------------------


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _small_kwargs() -> dict:
    """Larger than the tiny checkpoint: several heads, asymmetric RoPE axes, more blocks, distinct MLP ratios."""
    return {
        "patch_size": (1, 2, 2),
        "in_channels": 9,
        "out_channels": 4,
        "num_attention_heads": 4,
        "attention_head_dim": 32,
        "num_layers": 3,
        "num_refiner_layers": 2,
        "mlp_ratio": 3.0,
        "context_mlp_ratio": 2.0,
        "refiner_mlp_ratio": 1.5,
        "text_embed_dim": 48,
        "rope_axes_dim": (8, 12, 12),
        "rope_theta": 256.0,
        "rope_temporal_scale": 16 / 24,
    }


@pytest.fixture
def single_rank_dist():
    from fastvideo.distributed import cleanup_dist_env_and_memory, maybe_init_distributed_environment_and_model_parallel

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(_free_port()))
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    maybe_init_distributed_environment_and_model_parallel(1, 1)
    yield
    cleanup_dist_env_and_memory()


def _build_native(arch_kwargs: dict):
    from fastvideo.configs.models.dits.pan2 import PAN2VideoConfig
    from fastvideo.models.dits.pan2 import PAN2Transformer3DModel

    config = PAN2VideoConfig()
    config.update_model_arch(arch_kwargs)
    return PAN2Transformer3DModel(config=config, hf_config=arch_kwargs).float().eval()


def _load_into_native(native, weights) -> None:
    incompatible = load_model_from_full_model_state_dict(
        native,
        iter(weights),
        device=torch.device("cpu"),
        param_dtype=torch.float32,
        strict=True,
        param_names_mapping=get_param_names_mapping(native.param_names_mapping),
        training_mode=False,
    )
    assert incompatible.missing_keys == []
    assert incompatible.unexpected_keys == []


def _randomize(module: nn.Module, seed: int) -> None:
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for param in module.parameters():
            param.copy_(torch.randn(param.shape, generator=generator) * 0.2)


def _make_inputs(in_channels: int, text_dim: int, batch_size: int = 2, seed: int = 0) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    return {
        # 9 frames at 64x96 in latent space: 3 latent frames of 4x6.
        "hidden_states": torch.randn(batch_size, in_channels, 3, 4, 6, generator=generator),
        "timestep": torch.tensor([700.0, 125.5][:batch_size]),
        "encoder_hidden_states": torch.randn(batch_size, 13, text_dim, generator=generator),
    }


def _compare(native, reference, inputs: dict[str, torch.Tensor], scope: str) -> None:
    with torch.inference_mode():
        expected = reference(**inputs)
        with set_forward_context(current_timestep=0, attn_metadata=None):
            actual = native(**inputs)
    assert actual.shape == expected.shape
    diff = (actual - expected).abs()
    print(f"{scope}: shape={tuple(actual.shape)} ref_abs_mean={expected.abs().mean().item():.3e} "
          f"diff_max={diff.max().item():.3e} diff_mean={diff.mean().item():.3e}")
    assert torch.isfinite(expected).all() and expected.abs().mean() > 1e-3
    assert_close(actual, expected, atol=2e-5, rtol=2e-5)


# --------------------------------------------------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------------------------------------------------


def test_pan2_transformer_registry_resolves_native_class():
    from fastvideo.models.dits.pan2 import PAN2Transformer3DModel
    from fastvideo.models.registry import ModelRegistry

    model_cls, architecture = ModelRegistry.resolve_model_cls("PAN2Transformer3DModel")
    assert model_cls is PAN2Transformer3DModel
    assert architecture == "PAN2Transformer3DModel"


@pytest.mark.parametrize(("module_name", "expected"), [
    ("transformer_blocks.0", True),
    ("transformer_blocks.43", True),
    ("context_refiner.3", True),
    ("transformer_blocks.0.attn.to_qkv", False),
    ("transformer_blocks.0.ff.fc_in", False),
])
def test_pan2_fsdp_shard_conditions_match_only_blocks(module_name: str, expected: bool):
    from fastvideo.configs.models.dits.pan2 import PAN2ArchConfig

    conditions = PAN2ArchConfig()._fsdp_shard_conditions
    assert any(condition(module_name, None) for condition in conditions) is expected


def test_pan2_default_arch_config_is_the_29b_model():
    from fastvideo.configs.models.dits.pan2 import PAN2ArchConfig

    config = PAN2ArchConfig()
    assert config.hidden_size == 4096
    assert config.num_channels_latents == 48
    assert config.in_channels == 2 * 48 + 1
    assert sum(config.rope_axes_dim) == config.attention_head_dim


def test_pan2_tiny_checkpoint_config_fields_are_arch_fields():
    model_dir = tiny_model_dir()
    if model_dir is None or not (model_dir / "transformer" / "config.json").is_file():
        pytest.skip(f"{TINY_REPO_ID} is not available locally; set PAN2_TINY_MODEL_DIR")
    from fastvideo.configs.models.dits.pan2 import PAN2ArchConfig

    checkpoint_config = json.loads((model_dir / "transformer" / "config.json").read_text())
    assert checkpoint_config.pop("_class_name") == "PAN2Transformer3DModel"
    checkpoint_config.pop("_diffusers_version", None)
    assert set(checkpoint_config) <= {field.name for field in fields(PAN2ArchConfig)}


def test_pan2_tiny_checkpoint_strict_load_and_forward_parity(single_rank_dist):
    model_dir = tiny_model_dir()
    if model_dir is None or not (model_dir / "transformer").is_dir():
        pytest.skip(f"{TINY_REPO_ID} is not available locally; set PAN2_TINY_MODEL_DIR")
    from safetensors.torch import load_file

    transformer_dir = model_dir / "transformer"
    arch_kwargs = json.loads((transformer_dir / "config.json").read_text())
    state_dict = load_file(str(transformer_dir / "diffusion_pytorch_model.safetensors"))

    reference = ReferencePAN2Transformer(**arch_kwargs).float().eval()
    reference.load_state_dict(state_dict, strict=True)
    native = _build_native(arch_kwargs)
    _load_into_native(native, state_dict.items())

    inputs = _make_inputs(arch_kwargs["in_channels"], arch_kwargs["text_embed_dim"])
    _compare(native, reference, inputs, "tiny checkpoint")


@pytest.mark.parametrize("batch_size", [1, 2])
def test_pan2_random_weights_forward_parity(single_rank_dist, batch_size: int):
    kwargs = _small_kwargs()
    reference = ReferencePAN2Transformer(**kwargs).float().eval()
    # Zero-initialized tables (text modulation, type embedding) must be exercised too.
    _randomize(reference, seed=1234)
    native = _build_native(kwargs)
    _load_into_native(native, reference.state_dict().items())

    inputs = _make_inputs(kwargs["in_channels"], kwargs["text_embed_dim"], batch_size=batch_size, seed=batch_size)
    _compare(native, reference, inputs, f"random weights, batch {batch_size}")


# --------------------------------------------------------------------------------------------------------------------
# Sequence parallel: SP=2 must match one rank (video tokens are padded to a multiple of the SP size)
# --------------------------------------------------------------------------------------------------------------------


def _run_sp_worker(mode: str, output_path: Path) -> None:
    from fastvideo.configs.models.dits.pan2 import PAN2VideoConfig
    from fastvideo.distributed import cleanup_dist_env_and_memory, maybe_init_distributed_environment_and_model_parallel
    from fastvideo.models.dits.pan2 import PAN2Transformer3DModel
    import torch.distributed as dist

    sp_size = 1 if mode == "single" else SP_WORLD_SIZE
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{int(os.environ.get('LOCAL_RANK', '0'))}")
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    else:
        # GPU-less host: gloo ranks on CPU. fastvideo resolves the local device to mps there; swapping the code
        # object reaches every module that imported get_local_torch_device by name.
        import fastvideo.distributed.parallel_state as parallel_state

        def _cpu_device() -> torch.device:
            return torch.device("cpu")

        parallel_state.get_local_torch_device.__code__ = _cpu_device.__code__
        device = torch.device("cpu")
    try:
        maybe_init_distributed_environment_and_model_parallel(1, sp_size)
        kwargs = _small_kwargs()
        reference = ReferencePAN2Transformer(**kwargs).float()
        _randomize(reference, seed=1234)
        config = PAN2VideoConfig()
        config.update_model_arch(kwargs)
        native = PAN2Transformer3DModel(config=config, hf_config=kwargs).float().eval()
        _load_into_native(native, reference.state_dict().items())
        native = native.to(device)
        # 5x2x6 latents patchify to 15 video tokens, which SP=2 pads to 16.
        inputs = _make_inputs(kwargs["in_channels"], kwargs["text_embed_dim"])
        inputs["hidden_states"] = torch.randn(2, kwargs["in_channels"], 5, 2, 6, generator=torch.Generator().manual_seed(5))
        inputs = {name: value.to(device) for name, value in inputs.items()}
        with torch.inference_mode(), set_forward_context(current_timestep=0, attn_metadata=None):
            output = native(**inputs)
        if int(os.environ.get("RANK", "0")) == 0:
            torch.save({"output": output.detach().cpu()}, output_path)
        dist.barrier()
    finally:
        cleanup_dist_env_and_memory()


def _run_torchrun(mode: str, nproc_per_node: int, output_path: Path) -> None:
    command = [
        sys.executable, "-m", "torch.distributed.run", "--nnodes", "1", "--nproc_per_node",
        str(nproc_per_node), "--master_port",
        str(_free_port()),
        str(Path(__file__).resolve()), "--pan2-sp-worker", "--mode", mode, "--output",
        str(output_path)
    ]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(filter(None, [str(REPO_ROOT), environment.get("PYTHONPATH")]))
    environment["DISABLE_SP"] = "0"
    environment["FASTVIDEO_ATTENTION_BACKEND"] = "TORCH_SDPA"
    process = subprocess.run(command, capture_output=True, text=True, env=environment)
    if process.returncode != 0:
        raise RuntimeError(f"{mode} worker failed with code {process.returncode}\n"
                           f"STDOUT:\n{process.stdout}\nSTDERR:\n{process.stderr}")


def test_pan2_sp2_matches_single_rank(tmp_path: Path):
    """Ulysses SP=2 on an odd video token count (padded to the SP size) must match one rank. Runs on CPU with gloo
    when no GPU is present."""
    if torch.cuda.is_available() and torch.cuda.device_count() < SP_WORLD_SIZE:
        pytest.skip(f"PAN2 SP parity requires {SP_WORLD_SIZE} CUDA devices.")
    single_path, sp_path = tmp_path / "single.pt", tmp_path / "sp2.pt"
    _run_torchrun("single", 1, single_path)
    _run_torchrun("sp", SP_WORLD_SIZE, sp_path)
    single = torch.load(single_path, map_location="cpu", weights_only=True)["output"]
    sp = torch.load(sp_path, map_location="cpu", weights_only=True)["output"]
    diff = (sp - single).abs()
    print(f"sp2-vs-single diff_max={diff.max().item():.3e}")
    assert_close(sp, single, atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pan2-sp-worker", action="store_true")
    parser.add_argument("--mode", choices=["single", "sp"], required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()
    if not args.pan2_sp_worker:
        raise SystemExit("This module is intended to be run by pytest.")
    _run_sp_worker(args.mode, Path(args.output))
