# SPDX-License-Identifier: Apache-2.0
import os
from pathlib import Path

import pytest
import torch
from torch.testing import assert_close

from diffusers import Kandinsky5Transformer3DModel as DiffusersKandinsky5

from fastvideo.configs.models.dits import Kandinsky5VideoConfig
from fastvideo.configs.pipelines import PipelineConfig
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.models.loader.component_loader import TransformerLoader

os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", "29515")
os.environ.setdefault("FASTVIDEO_ATTENTION_BACKEND", "TORCH_SDPA")


def _resolve_transformer_path() -> Path:
    root = Path(
        os.getenv(
            "KANDINSKY5_DIFFUSERS_PATH",
            "official_weights/kandinskylab/Kandinsky-5.0-T2V-Lite-sft-5s-Diffusers",
        )
    )
    transformer_path = Path(
        os.getenv("KANDINSKY5_TRANSFORMER_PATH", str(root / "transformer"))
    )
    return transformer_path


def test_kandinsky5_lite_transformer_parity():
    transformer_path = _resolve_transformer_path()
    if not transformer_path.exists():
        pytest.skip(
            f"Kandinsky5 transformer weights not found at {transformer_path}"
        )

    if not torch.cuda.is_available():
        pytest.skip(
            "Kandinsky5 transformer parity test requires CUDA for practical runtime."
        )

    torch.manual_seed(42)
    device = torch.device("cuda:0")
    precision = torch.bfloat16
    precision_str = "bf16"

    reference_model = DiffusersKandinsky5.from_pretrained(
        transformer_path).to(device=device, dtype=precision)
    reference_model.eval()

    config = Kandinsky5VideoConfig()
    args = FastVideoArgs(
        model_path=str(transformer_path),
        dit_cpu_offload=True,
        use_fsdp_inference=False,
        pipeline_config=PipelineConfig(
            dit_config=config,
            dit_precision=precision_str,
        ),
    )
    args.device = device
    fastvideo_model = TransformerLoader().load(
        str(transformer_path), args).to(device=device, dtype=precision)
    fastvideo_model.eval()

    in_visual_dim = reference_model.config.in_visual_dim
    visual_cond = bool(getattr(reference_model.config, "visual_cond", False))
    in_text_dim = reference_model.config.in_text_dim
    in_text_dim2 = reference_model.config.in_text_dim2
    patch_size = reference_model.config.patch_size

    batch_size = 1
    grid_t, grid_h, grid_w = 2, 4, 4
    latent_t = grid_t * patch_size[0]
    latent_h = grid_h * patch_size[1]
    latent_w = grid_w * patch_size[2]

    base_latents = torch.randn(
        batch_size,
        latent_t,
        latent_h,
        latent_w,
        in_visual_dim,
        device=device,
        dtype=precision,
    )
    if visual_cond:
        visual_cond_latents = torch.zeros_like(base_latents)
        visual_cond_mask = torch.zeros(
            batch_size,
            latent_t,
            latent_h,
            latent_w,
            1,
            device=device,
            dtype=precision,
        )
        hidden_states = torch.cat(
            [base_latents, visual_cond_latents, visual_cond_mask], dim=-1)
    else:
        hidden_states = base_latents
    encoder_hidden_states = torch.randn(
        batch_size,
        8,
        in_text_dim,
        device=device,
        dtype=precision,
    )
    pooled_projections = torch.randn(
        batch_size,
        in_text_dim2,
        device=device,
        dtype=precision,
    )
    timestep = torch.tensor([500], device=device, dtype=precision)

    visual_rope_pos = [
        torch.arange(grid_t, device=device),
        torch.arange(grid_h, device=device),
        torch.arange(grid_w, device=device),
    ]
    text_rope_pos = torch.arange(
        encoder_hidden_states.shape[1], device=device)

    with torch.no_grad():
        ref_out = reference_model(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            pooled_projections=pooled_projections,
            timestep=timestep,
            visual_rope_pos=visual_rope_pos,
            text_rope_pos=text_rope_pos,
            scale_factor=(1.0, 1.0, 1.0),
            sparse_params=None,
            return_dict=False,
        )

        fv_out = fastvideo_model(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            pooled_projections=pooled_projections,
            timestep=timestep,
            visual_rope_pos=visual_rope_pos,
            text_rope_pos=text_rope_pos,
            scale_factor=(1.0, 1.0, 1.0),
            sparse_params=None,
            return_dict=False,
        )

    assert ref_out.shape == fv_out.shape
    assert ref_out.dtype == fv_out.dtype
    assert_close(ref_out, fv_out, atol=1e-4, rtol=1e-4)
