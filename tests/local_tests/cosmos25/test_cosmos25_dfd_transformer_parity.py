# SPDX-License-Identifier: Apache-2.0
"""Real-weight DFD V2W student parity against NVIDIA's FastGen release."""

from __future__ import annotations

import gc
import os
import sys
from collections.abc import Mapping
from pathlib import Path

import pytest
import torch
from torch.testing import assert_close

os.environ.setdefault("FASTVIDEO_ATTENTION_BACKEND", "TORCH_SDPA")

from fastvideo.forward_context import set_forward_context  # noqa: E402
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch  # noqa: E402
from scripts.checkpoint_conversion.cosmos25_dfd_to_diffusers import (  # noqa: E402
    load_dfd_dcp_state_dict,
    normalize_dfd_student_state_dict,
)
from tests.local_tests.cosmos25.test_cosmos25_distilled_transformer_parity import (  # noqa: E402
    _construct_bf16,
    _drift,
    _load_fastvideo_model,
    _print_capture_drift,
    _register_forward_captures,
    distributed_setup,  # noqa: F401 - registers the fixture in this test module.
)

REFERENCE_ENV = "COSMOS25_DFD_REF_DIR"
CHECKPOINT_ENV = "COSMOS25_DFD_CHECKPOINT_DIR"
DEFAULT_REFERENCE_ROOT = Path(__file__).resolve().parents[3] / "DFDReference"


def _reference_root() -> Path:
    root = Path(os.environ.get(REFERENCE_ENV, DEFAULT_REFERENCE_ROOT)).expanduser().resolve()
    if not (root / "fastgen/networks/cosmos_predict2/network.py").is_file():
        pytest.skip(f"Set {REFERENCE_ENV} to NVIDIA's DFD checkout")
    return root


def _checkpoint_path() -> Path:
    value = os.environ.get(CHECKPOINT_ENV)
    if not value:
        pytest.skip(f"Set {CHECKPOINT_ENV} to the extracted 0000040.net_model directory")
    checkpoint = Path(value).expanduser().resolve()
    if not (checkpoint / ".metadata").is_file():
        pytest.fail(f"{CHECKPOINT_ENV} does not point to an extracted DFD checkpoint: {checkpoint}")
    return checkpoint


def _official_model_class() -> type[torch.nn.Module]:
    root = _reference_root()
    root_string = str(root)
    if root_string not in sys.path:
        sys.path.insert(0, root_string)
    try:
        from fastgen.networks.cosmos_predict2.network import CosmosPredict2
    except ImportError as error:
        pytest.fail(f"Could not import DFD's CosmosPredict2 from {root}: {error}")
    return CosmosPredict2


def _load_student_checkpoint() -> dict[str, torch.Tensor]:
    raw_state = load_dfd_dcp_state_dict(_checkpoint_path())
    return normalize_dfd_student_state_dict(raw_state)


def _load_official_model(student: Mapping[str, torch.Tensor], device: torch.device):
    model_class = _official_model_class()
    model = _construct_bf16(
        lambda: model_class(
            model_channels=2048,
            num_blocks=28,
            num_heads=16,
            sac_config=None,
            fps=24,
            is_video2world=True,
            num_conditioning_frames=1,
            enable_logvar_linear=False,
        )
    )
    missing, unexpected = model.load_state_dict(student, strict=False)
    important_missing = [
        key
        for key in missing
        if not key.endswith("._extra_state") and "pos_embedder" not in key and not key.startswith("accum_")
    ]
    assert not important_missing, f"Official DFD model missing inference keys: {important_missing[:20]}"
    assert not unexpected, f"Official DFD model received unexpected keys: {unexpected[:20]}"
    return model.to(device=device, dtype=torch.bfloat16).eval()


def _inputs() -> dict[str, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(20260910)
    latents = torch.randn((1, 16, 2, 16, 16), generator=generator, dtype=torch.float32).to(torch.bfloat16)
    conditioning_latents = torch.randn(
        (1, 16, 1, 16, 16),
        generator=generator,
        dtype=torch.float32,
    ).to(torch.bfloat16)
    condition_mask = torch.zeros((1, 1, 2, 16, 16), dtype=torch.bfloat16)
    condition_mask[:, :, :1] = 1
    model_input = latents.clone()
    model_input[:, :, :1] = conditioning_latents
    # DFD's published t_list is materialized with RFNoiseSchedule.t_precision
    # (float64); each DiT implementation casts only after sinusoidal embedding.
    timestep = torch.full((1,), 0.937, dtype=torch.float64)
    timestep_per_frame = timestep[:, None].expand(1, 2).clone()
    timestep_per_frame[:, :1] = 0
    return {
        "latents": latents,
        "model_input": model_input,
        "conditioning_latents": conditioning_latents,
        "condition_mask": condition_mask,
        "text": torch.randn((1, 4, 100352), generator=generator, dtype=torch.float32).to(torch.bfloat16),
        "padding_mask": torch.zeros((1, 16, 16), dtype=torch.bfloat16),
        "timestep": timestep,
        "timestep_per_frame": timestep_per_frame,
        "fps": torch.tensor([24], dtype=torch.float32),
    }


@pytest.mark.usefixtures("distributed_setup")
def test_dfd_v2w_student_forward_matches_official() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the real Cosmos Predict2.5 DFD DiT parity gate")

    device = torch.device("cuda:0")
    student = _load_student_checkpoint()
    inputs = _inputs()

    official = _load_official_model(student, device)
    official_dit = official.transformer
    official_modules = {
        "patch": official_dit.x_embedder,
        "rope_angles": official_dit.pos_embedder,
        "time_norm": official_dit.t_embedding_norm,
        "text": official_dit.crossattn_proj,
        **{f"block.{index}": block for index, block in enumerate(official_dit.blocks)},
        "final": official_dit.final_layer,
    }
    official_captures, official_handles = _register_forward_captures(official_modules)
    condition = {
        "text_embeds": inputs["text"].to(device),
        "conditioning_latents": inputs["conditioning_latents"].to(device),
        "condition_mask": inputs["condition_mask"].to(device),
    }
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        official_output = (
            official(
                inputs["latents"].to(device),
                inputs["timestep"].to(device),
                condition=condition,
                fwd_pred_type="flow",
                fps=inputs["fps"].to(device),
                padding_mask=inputs["padding_mask"].to(device),
            )
            .float()
            .cpu()
        )
    for handle in official_handles:
        handle.remove()
    del official, official_dit
    gc.collect()
    torch.cuda.empty_cache()

    fastvideo = _load_fastvideo_model(
        student,
        device,
        rope_enable_fps_modulation=True,
    )
    fastvideo_modules = {
        "patch": fastvideo.patch_embed,
        "rope": fastvideo.rope,
        "time_norm": fastvideo.time_embed.norm,
        "text": fastvideo.crossattn_proj,
        **{f"block.{index}": block for index, block in enumerate(fastvideo.transformer_blocks)},
        "final": fastvideo.final_layer,
    }
    fastvideo_captures, fastvideo_handles = _register_forward_captures(fastvideo_modules)
    forward_batch = ForwardBatch(data_type="dummy")
    with (
        torch.inference_mode(),
        torch.autocast("cuda", dtype=torch.bfloat16),
        set_forward_context(current_timestep=937, attn_metadata=None, forward_batch=forward_batch),
    ):
        fastvideo_output = (
            fastvideo(
                hidden_states=inputs["model_input"].to(device),
                timestep=inputs["timestep_per_frame"].to(device),
                encoder_hidden_states=inputs["text"].to(device),
                fps=inputs["fps"].to(device),
                condition_mask=inputs["condition_mask"].to(device),
                padding_mask=inputs["padding_mask"].unsqueeze(1).to(device),
            )
            .float()
            .cpu()
        )
    for handle in fastvideo_handles:
        handle.remove()

    _print_capture_drift(official_captures, fastvideo_captures)
    for component in ("patch", "time_norm", "text"):
        assert_close(fastvideo_captures[component], official_captures[component], atol=0, rtol=0)

    reference_angles = official_captures["rope_angles"][:, 0, 0, :]
    for index, function in enumerate((torch.cos, torch.sin)):
        _, _, rope_relative_mean = _drift(
            function(reference_angles),
            fastvideo_captures[f"rope.{index}"],
        )
        assert rope_relative_mean < 0.002

    _, _, first_block_relative_mean = _drift(
        official_captures["block.0"],
        fastvideo_captures["block.0"],
    )
    # The FastGen DFD reference applies RoPE angles inside its own attention
    # implementation while FastVideo materializes cos/sin before attention.
    # BF16 drift begins at that bounded boundary and remains smooth through
    # the residual stack; keep this local guard much tighter than the final
    # aggregate tolerance without requiring the T2W path's exact threshold.
    assert first_block_relative_mean < 0.006

    mean_abs, max_abs, relative_mean = _drift(official_output, fastvideo_output)
    print(
        "Cosmos25 DFD V2W DiT parity: "
        f"max_abs={max_abs:.8f}, mean_abs={mean_abs:.8f}, relative_mean={relative_mean:.8f}"
    )
    assert relative_mean < 0.05
    assert_close(fastvideo_output, official_output, atol=0.5, rtol=0)
