# SPDX-License-Identifier: Apache-2.0
"""Wan-VACE DiT component parity: single-step forward vs Diffusers."""

from __future__ import annotations

import pytest
import torch

from fastvideo.utils import get_open_port
from tests.local_tests.wan.parity_stats import assert_dit_parity
from tests.local_tests.wan.vace_parity_helpers import (
    load_fv_transformer,
    resolve_model_dir,
    set_parity_cuda_flags,
    restore_parity_cuda_flags,
)


@pytest.fixture
def parity_runtime(monkeypatch):
    from fastvideo.distributed import cleanup_dist_env_and_memory, maybe_init_distributed_environment_and_model_parallel

    monkeypatch.setenv("DISABLE_SP", "1")
    monkeypatch.setenv("FASTVIDEO_ATTENTION_BACKEND", "TORCH_SDPA")
    monkeypatch.setenv("MASTER_ADDR", "localhost")
    monkeypatch.setenv("MASTER_PORT", str(get_open_port()))
    tf32_previous = set_parity_cuda_flags()
    maybe_init_distributed_environment_and_model_parallel(1, 1)
    try:
        yield
    finally:
        restore_parity_cuda_flags(tf32_previous)
        cleanup_dist_env_and_memory()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Wan-VACE parity requires CUDA")
@pytest.mark.parametrize("dtype", (torch.float32, torch.bfloat16), ids=("fp32", "bf16"))
def test_wan_vace_transformer_forward_parity(parity_runtime, dtype) -> None:
    model_dir = resolve_model_dir("WAN_VACE_MODEL_DIR")
    transformer_dir = model_dir / "transformer"
    if not (transformer_dir / "config.json").is_file():
        pytest.skip(f"transformer config missing under {transformer_dir}")

    from diffusers.models.transformers.transformer_wan_vace import (
        WanVACETransformer3DModel as DiffusersWanVACETransformer3DModel,
    )
    from fastvideo.forward_context import set_forward_context
    from fastvideo.models.wan.pipeline_config import WanVACE1_3B_Config
    from fastvideo.pipelines.pipeline_batch_info import ForwardBatch

    device = torch.device("cuda")
    official = DiffusersWanVACETransformer3DModel.from_pretrained(
        str(model_dir),
        subfolder="transformer",
        torch_dtype=dtype,
    ).to(device=device).eval()
    fastvideo = load_fv_transformer(model_dir, device, WanVACE1_3B_Config, dtype=dtype)

    official_block = official.vace_blocks[0]
    fastvideo_block = fastvideo.vace_blocks[0]
    for name, official_parameter, fastvideo_parameter in (
        ("vace_blocks.0.scale_shift_table", official_block.scale_shift_table, fastvideo_block.scale_shift_table),
        ("scale_shift_table", official.scale_shift_table, fastvideo.scale_shift_table),
    ):
        assert fastvideo_parameter.dtype == official_parameter.dtype, (
            f"{name} parameter dtype: FastVideo {fastvideo_parameter.dtype}, Diffusers {official_parameter.dtype}")
        assert torch.equal(fastvideo_parameter, official_parameter), f"{name} values differ"

    generator = torch.Generator(device=device).manual_seed(42)
    hidden_states = torch.randn(1, 16, 2, 8, 8, generator=generator, dtype=dtype, device=device)
    control_hidden_states = torch.randn(1, 96, 2, 8, 8, generator=generator, dtype=dtype, device=device)
    encoder_hidden_states = torch.randn(1, 22, 4096, generator=generator, dtype=dtype, device=device)
    num_vace_layers = len(WanVACE1_3B_Config().dit_config.arch_config.vace_layers)
    scale_tensor = torch.ones(num_vace_layers, device=device, dtype=dtype)
    timestep = torch.tensor([500], device=device, dtype=torch.long)

    with torch.inference_mode():
        official_out = official(hidden_states=hidden_states,
                                encoder_hidden_states=encoder_hidden_states,
                                timestep=timestep,
                                control_hidden_states=control_hidden_states,
                                control_hidden_states_scale=scale_tensor).sample
        with set_forward_context(current_timestep=int(timestep[0].item()),
                                 attn_metadata=None,
                                 forward_batch=ForwardBatch(data_type="dummy")):
            fastvideo_out = fastvideo(hidden_states=hidden_states,
                                      encoder_hidden_states=encoder_hidden_states,
                                      timestep=timestep,
                                      control_hidden_states=control_hidden_states,
                                      control_hidden_states_scale=scale_tensor)

    assert_dit_parity(fastvideo_out.detach().cpu(), official_out.detach().cpu(), label=f"vace_forward_{dtype}")
