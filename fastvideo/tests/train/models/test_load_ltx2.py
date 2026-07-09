# SPDX-License-Identifier: Apache-2.0
"""GPU loading + forward smoke test for ``LTX2Model``.

Loads the real FastVideo/LTX2-Distilled-Diffusers checkpoint (18.9B at
bf16, ~38GB — skips on GPUs with less than 60GB memory) via
``LTX2Model.__init__`` and runs one transformer forward pass on
synthetic inputs. Catches loader or forward-signature regressions in
``fastvideo.train.models.ltx2.LTX2Model`` and the underlying
``LTX2Transformer3DModel``.

LTX-2's transformer takes per-token sigma timesteps in [0, 1] shaped
[B, tokens] and a post-connector Gemma text embedding ([1024, 3840]);
it returns the denoised x0 prediction. This mirrors the kwargs in
``LTX2Model._build_distill_input_kwargs``.
"""

from __future__ import annotations

import os

# Required by the ``distributed_setup`` fixture pulled from
# ``fastvideo/tests/conftest.py``.  Set before any fastvideo import.
os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", "29519")

from pathlib import Path

import pytest
import torch

from fastvideo.forward_context import set_forward_context
from fastvideo.pipelines import ForwardBatch
from fastvideo.train.models.ltx2 import LTX2Model
from fastvideo.train.utils.config import load_run_config

_FIXTURE = str(
    Path(__file__).resolve().parent.parent / "fixtures"
    / "ltx2_t2v_finetune_min.yaml")

# Post-connector Gemma embedding width / length.
_LTX2_TEXT_DIM = 3840
_LTX2_TEXT_LEN = 1024

_MIN_GPU_MEMORY_GB = 60


def _gpu_too_small() -> bool:
    if not torch.cuda.is_available():
        return True
    total = torch.cuda.get_device_properties(0).total_memory
    return total < _MIN_GPU_MEMORY_GB * 1024**3


@pytest.mark.usefixtures("distributed_setup")
def test_ltx2_model_loads_and_forwards():
    if _gpu_too_small():
        pytest.skip(f"requires a CUDA GPU with >= {_MIN_GPU_MEMORY_GB}GB "
                    "memory (LTX-2 DiT is 18.9B params)")

    cfg = load_run_config(_FIXTURE)
    model = LTX2Model(
        init_from=cfg.models["student"]["init_from"],
        training_config=cfg.training,
        trainable=False,
    )

    transformer = model.transformer
    assert isinstance(transformer, torch.nn.Module)
    assert sum(p.numel() for p in transformer.parameters()) > 0

    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    transformer = transformer.to(device=device, dtype=dtype).eval()

    # LTX-2 transformer takes [B, 128, T, H, W] latents, a post-connector
    # Gemma embedding, and PER-TOKEN sigmas in [0, 1] ([B, T*H*W] with
    # patch size 1x1x1). Small spatial + few frames so this fits next to
    # the 18.9B model.
    b, c, t, h, w = 1, 128, 3, 8, 8
    tokens = t * h * w
    hidden_states = torch.randn(b, c, t, h, w, device=device, dtype=dtype)
    encoder_hidden_states = torch.randn(b,
                                        _LTX2_TEXT_LEN,
                                        _LTX2_TEXT_DIM,
                                        device=device,
                                        dtype=dtype)
    timestep = torch.full((b, tokens), 0.5, device=device, dtype=torch.float32)

    with torch.no_grad(), torch.autocast(device.type, dtype=dtype), \
            set_forward_context(
                current_timestep=timestep * 1000.0,
                attn_metadata=None,
                forward_batch=ForwardBatch(data_type="video", fps=24.0),
            ):
        out = transformer(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep,
            encoder_attention_mask=None,
            return_dict=False,
        )

    assert torch.is_tensor(out), f"expected a tensor output, got {type(out)}"
    assert out.shape == hidden_states.shape, (
        f"denoised output shape {tuple(out.shape)} != input latent shape "
        f"{tuple(hidden_states.shape)}")
    assert torch.isfinite(out).all().item(), "forward output contains NaN/Inf"
