# SPDX-License-Identifier: Apache-2.0
"""Cross-impl parity for the FastVideo ``OvisImageTransformer2DModel`` (production
``TransformerLoader``) vs the official Diffusers transformer, comparing denoised
latents. The reference is fed ``timestep/1000`` because Diffusers rescales
internally; tolerance is the bf16 cross-kernel floor (atol/rtol 1e-1, drift <5%)."""

import inspect
import os

import pytest
import torch

import fastvideo  # noqa: F401  # ensure full package init before deep submodule imports
from fastvideo.configs.models.dits import OvisImageTransformer2DModelConfig
from fastvideo.configs.pipelines.base import PipelineConfig
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.forward_context import set_forward_context
from fastvideo.logger import init_logger
from fastvideo.models.dits.ovisimage import (_pack_latents, _prepare_img_ids,
                                             _prepare_txt_ids, _unpack_latents)
from fastvideo.models.loader.component_loader import TransformerLoader

logger = init_logger(__name__)

os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", "29508")

LOCAL_WEIGHTS = os.getenv("OVIS_WEIGHTS", "official_weights/ovis_image")
TRANSFORMER_PATH = os.path.join(LOCAL_WEIGHTS, "transformer")


def _call_diffusers(model, hidden_states, encoder_hidden_states, timestep,
                    img_ids, txt_ids):
    """Call the Diffusers transformer, passing only kwargs it accepts.

    Diffusers' Ovis transformer expects packed image tokens plus separate
    position-id tensors and returns a ``Transformer2DModelOutput`` (``.sample``)
    or a bare tensor. Signature details vary across diffusers releases, so we
    introspect rather than hard-code the kwarg set.
    """
    sig = inspect.signature(model.forward)
    accepted = set(sig.parameters)
    kwargs = {
        "hidden_states": hidden_states,
        "encoder_hidden_states": encoder_hidden_states,
        "timestep": timestep,
        "img_ids": img_ids,
        "txt_ids": txt_ids,
    }
    kwargs = {k: v for k, v in kwargs.items() if k in accepted}
    if "return_dict" in accepted:
        kwargs["return_dict"] = False
    if "guidance" in accepted and sig.parameters["guidance"].default is inspect.Parameter.empty:
        kwargs["guidance"] = None
    out = model(**kwargs)
    return out[0] if isinstance(out, (tuple, list)) else getattr(out, "sample", out)


@pytest.mark.skipif(
    not os.path.exists(TRANSFORMER_PATH),
    reason=(f"Ovis-Image transformer weights not found at {TRANSFORMER_PATH}. "
            f"Set OVIS_WEIGHTS or download from AIDC-AI/Ovis-Image-7B."))
@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason="Ovis-Image transformer parity requires CUDA.")
@pytest.mark.usefixtures("distributed_setup")
def test_ovis_transformer_parity():
    try:
        from diffusers import OvisImageTransformer2DModel as RefTransformer
    except Exception as exc:  # noqa: BLE001 - diffusers may lack Ovis support
        pytest.skip(f"Diffusers OvisImageTransformer2DModel unavailable: {exc!r}")

    torch.manual_seed(42)
    device = torch.device("cuda:0")
    precision = torch.bfloat16

    # ---- FastVideo side (production loader path) ----
    args = FastVideoArgs(
        model_path=TRANSFORMER_PATH,
        dit_cpu_offload=False,
        use_fsdp_inference=False,
        pipeline_config=PipelineConfig(
            dit_config=OvisImageTransformer2DModelConfig(),
            dit_precision="bf16",
        ),
    )
    args.device = device
    fv_model = TransformerLoader().load(TRANSFORMER_PATH, args)
    fv_model = fv_model.to(device=device, dtype=precision).eval()

    # ---- Reference side (official Diffusers) ----
    ref_model = RefTransformer.from_pretrained(
        TRANSFORMER_PATH, torch_dtype=precision).to(device).eval()

    # ---- Deterministic inputs (small 32x32 latent for speed) ----
    B, C_vae, H, W = 1, 16, 32, 32   # in_channels=64 == C_vae(16) * pack_factor(4)
    txt_seq, joint_dim = 32, 2048
    hidden_states = torch.randn(B, C_vae, H, W, device=device, dtype=precision)
    encoder_hidden_states = torch.randn(B, txt_seq, joint_dim,
                                        device=device, dtype=precision)
    # Timestep-scale convention differs between the two implementations:
    # the FastVideo port embeds the timestep directly (expects the [0, 1000]
    # scale), while Diffusers' forward does `timestep = timestep * 1000`
    # internally. Feed each side the value that yields the SAME effective
    # timestep (500) so this compares the transformer, not the scaling choice.
    timestep_fv = torch.tensor([500.0], device=device, dtype=precision)
    timestep_ref = timestep_fv / 1000.0

    # Reference receives the pre-packed sequence + position ids that FastVideo
    # builds internally — reuse the production helpers so they cannot drift.
    packed = _pack_latents(hidden_states)
    img_ids = _prepare_img_ids(H // 2, W // 2, device)
    txt_ids = _prepare_txt_ids(txt_seq, device)

    with torch.no_grad():
        with torch.amp.autocast(device_type="cuda", dtype=precision):
            ref_packed = _call_diffusers(ref_model, packed,
                                         encoder_hidden_states, timestep_ref,
                                         img_ids, txt_ids)
            ref_out = _unpack_latents(ref_packed, H, W)

            with set_forward_context(current_timestep=0, attn_metadata=None,
                                     forward_batch=None):
                fv_out = fv_model(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    timestep=timestep_fv,
                )

    assert ref_out.shape == fv_out.shape, \
        f"Shape mismatch: ref={ref_out.shape}, fv={fv_out.shape}"
    assert torch.isfinite(fv_out).all(), "FastVideo output has NaN/Inf"
    assert torch.isfinite(ref_out).all(), "Reference output has NaN/Inf"

    diff = (ref_out.float() - fv_out.float()).abs()
    abs_mean_drift = diff.mean().item() / (ref_out.float().abs().mean().item() + 1e-8)
    logger.info(f"max_diff={diff.max().item():.3e} "
                f"mean_diff={diff.mean().item():.3e} "
                f"abs_mean_drift={abs_mean_drift:.2%}")

    assert abs_mean_drift < 0.05, \
        f"Abs-mean drift {abs_mean_drift:.2%} exceeds 5%"
    torch.testing.assert_close(ref_out, fv_out, atol=1e-1, rtol=1e-1)
