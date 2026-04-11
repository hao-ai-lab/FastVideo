# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import glob
import os

import pytest
import torch
from diffusers import FluxTransformer2DModel as HFFluxTransformer2DModel
from torch.testing import assert_close

from fastvideo.configs.models.dits.flux import FluxDiTConfig
from fastvideo.configs.pipelines.base import PipelineConfig
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.forward_context import set_forward_context
from fastvideo.models.loader.component_loader import TransformerLoader
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch

os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", "29517")

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_DEFAULT_FLUX_TRANSFORMER = os.path.join(
    _REPO_ROOT,
    "official_weights",
    "FLUX.1-dev",
    "transformer",
)


def _flux_transformer_path() -> str:
    return os.environ.get("FLUX_TRANSFORMER_PATH", _DEFAULT_FLUX_TRANSFORMER)


def _prepare_latent_image_ids(
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype = torch.long,
) -> torch.Tensor:
    """Match Diffusers ``FluxPipeline._prepare_latent_image_ids`` (batch omitted)."""
    latent_image_ids = torch.zeros(height, width, 3)
    latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height)[:, None]
    latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width)[None, :]
    h, w, c = latent_image_ids.shape
    latent_image_ids = latent_image_ids.reshape(h * w, c)
    return latent_image_ids.to(device=device, dtype=dtype)


@pytest.fixture
def torch_sdpa_attention_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FASTVIDEO_ATTENTION_BACKEND", "TORCH_SDPA")


requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="FLUX DiT parity test requires CUDA",
)

requires_weights = pytest.mark.skipif(
    not glob.glob(os.path.join(_flux_transformer_path(), "*.safetensors")),
    reason=(
        f"No safetensors under {_flux_transformer_path()} — download FLUX.1-dev "
        "transformer or set FLUX_TRANSFORMER_PATH"
    ),
)


@requires_cuda
@requires_weights
@pytest.mark.usefixtures("distributed_setup", "torch_sdpa_attention_backend")
def test_flux_transformer_parity_vs_diffusers() -> None:
    """Single forward: FastVideo DiT vs Diffusers ``FluxTransformer2DModel``."""
    device = torch.device("cuda:0")
    precision = torch.bfloat16
    transformer_path = _flux_transformer_path()

    args = FastVideoArgs(
        model_path=transformer_path,
        dit_cpu_offload=False,
        dit_layerwise_offload=False,
        pipeline_config=PipelineConfig(dit_config=FluxDiTConfig(), dit_precision="bf16"),
    )
    args.device = device

    generator = torch.Generator(device=device).manual_seed(0)
    torch.manual_seed(0)

    batch_size = 1
    latent_h, latent_w = 4, 4
    img_seq = latent_h * latent_w
    text_len = 32

    hidden_states = torch.randn(
        batch_size,
        img_seq,
        64,
        device=device,
        dtype=precision,
        generator=generator,
    )
    encoder_hidden_states = torch.randn(
        batch_size,
        text_len,
        4096,
        device=device,
        dtype=precision,
        generator=generator,
    )
    pooled_projections = torch.randn(
        batch_size,
        768,
        device=device,
        dtype=precision,
        generator=generator,
    )

    # Diffusers pipeline passes scheduler timesteps / 1000 (float, same dtype as latents).
    timestep = torch.tensor([512.0], device=device, dtype=precision) / 1000.0
    guidance = torch.full((batch_size,), 3.5, device=device, dtype=torch.float32)

    txt_ids = torch.zeros(text_len, 3, device=device, dtype=torch.long)
    img_ids = _prepare_latent_image_ids(latent_h, latent_w, device, dtype=torch.long)

    forward_batch = ForwardBatch(data_type="dummy")

    # One ~12B model at a time avoids peak VRAM from holding both checkpoints.
    loader = TransformerLoader()
    fv_model = loader.load(transformer_path, args).to(device=device, dtype=precision)
    fv_model.eval()
    with (
        torch.no_grad(),
        torch.amp.autocast("cuda", dtype=precision),
        set_forward_context(
            current_timestep=512,
            attn_metadata=None,
            forward_batch=forward_batch,
        ),
    ):
        fv_out = fv_model(
            hidden_states=hidden_states.clone(),
            encoder_hidden_states=encoder_hidden_states.clone(),
            pooled_projections=pooled_projections.clone(),
            timestep=timestep.clone(),
            guidance=guidance.clone(),
            txt_ids=txt_ids,
            img_ids=img_ids,
            return_dict=False,
        )[0]
    fv_out_cpu = fv_out.detach().float().cpu()
    del fv_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    hf_model = (
        HFFluxTransformer2DModel.from_pretrained(
            transformer_path,
            torch_dtype=precision,
        )
        .to(device)
        .eval()
    )
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=precision):
        hf_out = hf_model(
            hidden_states=hidden_states.clone(),
            encoder_hidden_states=encoder_hidden_states.clone(),
            pooled_projections=pooled_projections.clone(),
            timestep=timestep.clone(),
            guidance=guidance.clone(),
            txt_ids=txt_ids,
            img_ids=img_ids,
            return_dict=False,
        )[0]

    assert hf_out.shape == fv_out_cpu.shape
    assert_close(hf_out.float().cpu(), fv_out_cpu, atol=1e-4, rtol=1e-4)
