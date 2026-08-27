# SPDX-License-Identifier: Apache-2.0
"""Real-weight Cosmos Predict2.5 student DiT parity (implementation_subcomponent)."""

from __future__ import annotations

import gc
import os
import sys
import types
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, TypeVar

import pytest
import torch
from torch.testing import assert_close

os.environ.setdefault("FASTVIDEO_ATTENTION_BACKEND", "TORCH_SDPA")

from fastvideo.configs.models.dits.cosmos2_5 import (  # noqa: E402
    Cosmos25ArchConfig,
    Cosmos25VideoConfig,
)
from fastvideo.distributed import (  # noqa: E402
    cleanup_dist_env_and_memory,
    maybe_init_distributed_environment_and_model_parallel,
)
from fastvideo.forward_context import set_forward_context  # noqa: E402
from fastvideo.models.dits.cosmos2_5 import Cosmos25Transformer3DModel  # noqa: E402
from fastvideo.models.loader.utils import (  # noqa: E402
    get_param_names_mapping,
    hf_to_custom_state_dict,
)
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch  # noqa: E402
from scripts.checkpoint_conversion.cosmos25_distilled_to_diffusers import (  # noqa: E402
    extract_student_state_dict,
)
from tests.local_tests.cosmos25._reference import reference_root  # noqa: E402

CHECKPOINT_ENV = "COSMOS25_DISTILLED_CHECKPOINT"
ModuleT = TypeVar("ModuleT", bound=torch.nn.Module)


@pytest.fixture
def distributed_setup():
    maybe_init_distributed_environment_and_model_parallel(1, 1)
    yield
    cleanup_dist_env_and_memory()


def _checkpoint_path() -> Path:
    value = os.environ.get(CHECKPOINT_ENV)
    if not value:
        pytest.skip(f"Set {CHECKPOINT_ENV} to NVIDIA's released distilled .pt checkpoint")
    assert value is not None
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        pytest.fail(f"{CHECKPOINT_ENV} does not point to a file: {path}")
    return path


def _load_student_checkpoint() -> dict[str, torch.Tensor]:
    checkpoint = torch.load(_checkpoint_path(), map_location="cpu", weights_only=True)
    return extract_student_state_dict(checkpoint)


def _official_model_class() -> type[torch.nn.Module]:
    root = reference_root()
    if not root.is_dir():
        pytest.skip("Set COSMOS25_OFFICIAL_REF_DIR to the NVIDIA Cosmos-Predict2.5 checkout")
    root_string = str(root)
    if root_string not in sys.path:
        sys.path.insert(0, root_string)
    # The published package root enforces installation of its full CUDA extra.
    # Component parity only needs the checked-out Python modules, so install a
    # namespace package that preserves normal submodule imports without running
    # that unrelated environment guard.
    if "cosmos_predict2" not in sys.modules:
        package = types.ModuleType("cosmos_predict2")
        package.__path__ = [str(root / "cosmos_predict2")]  # type: ignore[attr-defined]
        package.__package__ = "cosmos_predict2"
        sys.modules["cosmos_predict2"] = package
    try:
        from cosmos_predict2._src.predict2.networks.minimal_v1_lvg_dit import (
            MinimalV1LVGDiT,
        )
    except ImportError as error:
        pytest.fail(f"Could not import NVIDIA's MinimalV1LVGDiT from {root}: {error}")
    return MinimalV1LVGDiT


def _arch_config() -> Cosmos25ArchConfig:
    return Cosmos25ArchConfig(
        num_attention_heads=16,
        attention_head_dim=128,
        in_channels=16,
        out_channels=16,
        num_layers=28,
        patch_size=(1, 2, 2),
        max_size=(128, 240, 240),
        rope_scale=(1.0, 3.0, 3.0),
        text_embed_dim=1024,
        mlp_ratio=4.0,
        adaln_lora_dim=256,
        use_adaln_lora=True,
        concat_padding_mask=True,
        extra_pos_embed_type=None,
        use_crossattn_projection=True,
        crossattn_proj_in_channels=100352,
        rope_enable_fps_modulation=False,
        qk_norm="rms_norm",
    )


def _construct_bf16(factory: Callable[[], ModuleT]) -> ModuleT:
    previous_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        return factory()
    finally:
        torch.set_default_dtype(previous_dtype)


def _load_official_model(student: Mapping[str, torch.Tensor], device: torch.device):
    model_class = _official_model_class()
    model = _construct_bf16(
        lambda: model_class(
            max_img_h=240,
            max_img_w=240,
            max_frames=128,
            in_channels=16,
            out_channels=16,
            patch_spatial=2,
            patch_temporal=1,
            model_channels=2048,
            num_blocks=28,
            num_heads=16,
            mlp_ratio=4.0,
            crossattn_emb_channels=1024,
            pos_emb_cls="rope3d",
            pos_emb_learnable=True,
            pos_emb_interpolation="crop",
            use_adaln_lora=True,
            adaln_lora_dim=256,
            rope_h_extrapolation_ratio=3.0,
            rope_w_extrapolation_ratio=3.0,
            rope_t_extrapolation_ratio=1.0,
            extra_per_block_abs_pos_emb=False,
            rope_enable_fps_modulation=False,
            use_crossattn_projection=True,
            crossattn_proj_in_channels=100352,
            concat_padding_mask=True,
            atten_backend="torch",
        )
    )

    expected_keys = set(model.state_dict())
    wrapped_blocks = any("._checkpoint_wrapped_module." in key for key in expected_keys)
    official_state: dict[str, torch.Tensor] = {}
    for source_key, tensor in student.items():
        target_key = source_key.removeprefix("net.")
        if wrapped_blocks and target_key.startswith("blocks."):
            parts = target_key.split(".", 2)
            if len(parts) == 3 and parts[1].isdigit():
                target_key = f"blocks.{parts[1]}._checkpoint_wrapped_module.{parts[2]}"
        if target_key in expected_keys:
            official_state[target_key] = tensor

    missing, unexpected = model.load_state_dict(official_state, strict=False)
    important_missing = [key for key in missing if not key.endswith("._extra_state")]
    assert not important_missing, f"Official model missing inference keys: {important_missing[:20]}"
    assert not unexpected, f"Official model received unexpected keys: {unexpected[:20]}"
    return model.to(device=device, dtype=torch.bfloat16).eval()


def _load_fastvideo_model(student: Mapping[str, torch.Tensor], device: torch.device):
    arch = _arch_config()
    config = Cosmos25VideoConfig(arch_config=arch)
    hf_config: dict[str, Any] = {
        "in_channels": arch.in_channels,
        "out_channels": arch.out_channels,
        "num_attention_heads": arch.num_attention_heads,
        "attention_head_dim": arch.attention_head_dim,
        "num_layers": arch.num_layers,
        "patch_size": arch.patch_size,
        "max_size": arch.max_size,
        "rope_scale": arch.rope_scale,
        "text_embed_dim": arch.text_embed_dim,
        "mlp_ratio": arch.mlp_ratio,
        "adaln_lora_dim": arch.adaln_lora_dim,
        "use_adaln_lora": arch.use_adaln_lora,
        "concat_padding_mask": arch.concat_padding_mask,
        "extra_pos_embed_type": arch.extra_pos_embed_type,
        "use_crossattn_projection": arch.use_crossattn_projection,
        "crossattn_proj_in_channels": arch.crossattn_proj_in_channels,
        "rope_enable_fps_modulation": arch.rope_enable_fps_modulation,
        "qk_norm": arch.qk_norm,
    }
    model = _construct_bf16(lambda: Cosmos25Transformer3DModel(config=config, hf_config=hf_config))
    mapping = get_param_names_mapping(arch.param_names_mapping)
    mapped, _ = hf_to_custom_state_dict(student.items(), mapping)
    expected_keys = set(model.state_dict())
    filtered = {key: value for key, value in mapped.items() if key in expected_keys}
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    important_missing = [key for key in missing if not key.endswith("._extra_state")]
    assert not important_missing, f"FastVideo model missing inference keys: {important_missing[:20]}"
    assert not unexpected, f"FastVideo model received unexpected keys: {unexpected[:20]}"
    return model.to(device=device, dtype=torch.bfloat16).eval()


def _inputs() -> dict[str, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(20260827)
    return {
        "latents": torch.randn((1, 16, 2, 16, 16), generator=generator, dtype=torch.float32).to(torch.bfloat16),
        "text": torch.randn((1, 4, 100352), generator=generator, dtype=torch.float32).to(torch.bfloat16),
        "condition_mask": torch.zeros((1, 1, 2, 16, 16), dtype=torch.bfloat16),
        "padding_mask": torch.ones((1, 16, 16), dtype=torch.bfloat16),
        "timestep": torch.full((1, 2), 15 / 16, dtype=torch.bfloat16),
        "fps": torch.tensor([24], dtype=torch.bfloat16),
    }


@pytest.mark.usefixtures("distributed_setup")
def test_distilled_student_forward_matches_official() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the real Cosmos Predict2.5 DiT parity gate")

    device = torch.device("cuda:0")
    student = _load_student_checkpoint()
    inputs = _inputs()

    official = _load_official_model(student, device)
    from cosmos_predict2._src.predict2.conditioner import DataType

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        official_output = (
            official(
                x_B_C_T_H_W=inputs["latents"].to(device),
                timesteps_B_T=inputs["timestep"].to(device),
                crossattn_emb=inputs["text"].to(device),
                condition_video_input_mask_B_C_T_H_W=inputs["condition_mask"].to(device),
                fps=inputs["fps"].to(device),
                padding_mask=inputs["padding_mask"].to(device),
                data_type=DataType.VIDEO,
            )
            .float()
            .cpu()
        )
    del official
    gc.collect()
    torch.cuda.empty_cache()

    fastvideo = _load_fastvideo_model(student, device)
    forward_batch = ForwardBatch(data_type="dummy")
    with (
        torch.inference_mode(),
        torch.autocast("cuda", dtype=torch.bfloat16),
        set_forward_context(current_timestep=937, attn_metadata=None, forward_batch=forward_batch),
    ):
        fastvideo_output = (
            fastvideo(
                hidden_states=inputs["latents"].to(device),
                timestep=inputs["timestep"].to(device),
                encoder_hidden_states=inputs["text"].to(device),
                fps=inputs["fps"].to(device),
                condition_mask=inputs["condition_mask"].to(device),
                padding_mask=inputs["padding_mask"].unsqueeze(1).to(device),
            )
            .float()
            .cpu()
        )

    absolute = (fastvideo_output - official_output).abs()
    mean_abs = float(absolute.mean())
    max_abs = float(absolute.max())
    reference_abs_mean = float(official_output.abs().mean())
    relative_mean = mean_abs / max(reference_abs_mean, 1e-8)
    print(
        "Cosmos25 distilled DiT parity: "
        f"max_abs={max_abs:.8f}, mean_abs={mean_abs:.8f}, relative_mean={relative_mean:.8f}"
    )

    assert relative_mean < 0.05
    assert_close(fastvideo_output, official_output, atol=0.1, rtol=0.1)
