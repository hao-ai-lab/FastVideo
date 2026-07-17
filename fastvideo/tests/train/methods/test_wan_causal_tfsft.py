# SPDX-License-Identifier: Apache-2.0
"""Per-method GPU smoke test: ``WanCausalModel`` + ``TeacherForcingSFTMethod``.

Mirrors ``test_wan_causal_dfsft.py``. Teacher forcing concatenates a clean
context copy of every frame inside the causal transformer (``clean_x``) and
denoises the current block while attending to *clean* history. This test
exercises the ``clean_x`` path end-to-end: forward, finite loss, and nonzero
gradients reaching the first transformer block.
"""

from __future__ import annotations

import os

os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", "29518")

from pathlib import Path

import pytest
import torch

from fastvideo.models.dits.causal_wanvideo import (
    CausalWanSelfAttention, CausalWanTransformer3DModel)
from fastvideo.train.methods.fine_tuning.tfsft import (
    TeacherForcingSFTMethod, )
from fastvideo.train.models.wan import WanCausalModel
from fastvideo.train.utils.config import load_run_config


_FIXTURE = str(
    Path(__file__).resolve().parent.parent / "fixtures"
    / "wan_causal_t2v_tfsft_min.yaml")


class _BatchShapeConditionEmbedder(torch.nn.Module):

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.encoder_batch_sizes: list[int] = []

    def forward(self, timestep, encoder_hidden_states,
                encoder_hidden_states_image):
        del encoder_hidden_states_image
        self.encoder_batch_sizes.append(int(encoder_hidden_states.shape[0]))
        count = int(timestep.numel())
        temb = timestep.new_zeros((count, self.hidden_size),
                                  dtype=torch.float32)
        timestep_proj = timestep.new_zeros(
            (count, 6 * self.hidden_size), dtype=torch.float32)
        return temb, timestep_proj, encoder_hidden_states, None


class _IdentityNormOut(torch.nn.Module):

    def forward(self, hidden_states, shift, scale):
        del shift, scale
        return hidden_states


@pytest.mark.parametrize("forward_name", ["_forward_train", "_forward_inference"])
def test_causal_wan_forward_preserves_batch_dimension(
        monkeypatch: pytest.MonkeyPatch, forward_name: str) -> None:
    """Both causal forward paths must pad and unpatchify every sample."""
    batch_size = 2
    hidden_size = 6
    model = CausalWanTransformer3DModel.__new__(
        CausalWanTransformer3DModel)
    torch.nn.Module.__init__(model)
    model.patch_size = (1, 1, 1)
    model.hidden_size = hidden_size
    model.num_attention_heads = 1
    model.rope_cache_policy = "absolute"
    model.text_len = 5
    model.patch_embedding = torch.nn.Identity()
    model.condition_embedder = _BatchShapeConditionEmbedder(hidden_size)
    model.blocks = torch.nn.ModuleList()
    model.gradient_checkpointing = False
    model.scale_shift_table = torch.nn.Parameter(
        torch.zeros(1, 2, hidden_size), requires_grad=False)
    model.norm_out = _IdentityNormOut()
    model.proj_out = torch.nn.Identity()
    monkeypatch.setattr(model, "_get_train_attention_spec",
                        lambda **_kwargs: None)
    monkeypatch.setattr(
        "fastvideo.models.dits.causal_wanvideo.get_sp_world_size",
        lambda: 1,
    )

    seen_grid_sizes: list[torch.Tensor] = []

    def _unpatchify(tokens: torch.Tensor,
                    grid_sizes: torch.Tensor) -> list[torch.Tensor]:
        seen_grid_sizes.append(grid_sizes.detach().clone())
        return [tokens[i] for i in range(grid_sizes.shape[0])]

    monkeypatch.setattr(model, "unpatchify", _unpatchify)

    hidden_states = torch.randn(batch_size, hidden_size, 2, 2, 2)
    encoder_hidden_states = torch.randn(batch_size, 3, 4)
    timestep = torch.ones(batch_size, 2)

    output = getattr(model, forward_name)(
        hidden_states=hidden_states,
        encoder_hidden_states=encoder_hidden_states,
        timestep=timestep,
    )

    assert model.condition_embedder.encoder_batch_sizes == [batch_size]
    assert len(seen_grid_sizes) == 1
    assert tuple(seen_grid_sizes[0].shape) == (batch_size, 3)
    assert output.shape[0] == batch_size


@pytest.mark.parametrize("sequence_length", [127, 128])
def test_causal_wan_flex_attention_preserves_unpadded_length(
        monkeypatch: pytest.MonkeyPatch, sequence_length: int) -> None:
    """FlexAttention must preserve lengths with zero or nonzero padding."""
    attention = CausalWanSelfAttention.__new__(CausalWanSelfAttention)
    torch.nn.Module.__init__(attention)
    attention.rope_cache_policy = "absolute"

    monkeypatch.setattr(
        "fastvideo.models.dits.causal_wanvideo._apply_rotary_emb",
        lambda tensor, *_args, **_kwargs: tensor,
    )
    monkeypatch.setattr(
        "fastvideo.models.dits.causal_wanvideo.flex_attention",
        lambda query, key, value, block_mask: query,
    )

    query = torch.randn(2, sequence_length, 1, 4)
    output = attention(
        q=query,
        k=query,
        v=query,
        freqs_cis=(torch.empty(0), torch.empty(0)),
        block_mask=object(),
    )

    assert output.shape == query.shape
    torch.testing.assert_close(output, query)


def _build_synthetic_batch(
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    batch_size = 1
    return {
        "text_embedding":
        torch.randn(batch_size, 16, 4096, device=device, dtype=dtype),
        "text_attention_mask":
        torch.ones(batch_size, 16, device=device),
        "vae_latent":
        torch.randn(batch_size, 16, 6, 8, 8, device=device, dtype=dtype),
    }


@pytest.mark.usefixtures("distributed_setup")
def test_wan_causal_tfsft_single_train_step(
        monkeypatch: pytest.MonkeyPatch) -> None:
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")

    cfg = load_run_config(_FIXTURE)

    device = torch.device("cuda:0")
    dtype = torch.bfloat16

    monkeypatch.setattr(
        "fastvideo.train.utils.dataloader."
        "build_parquet_t2v_train_dataloader",
        lambda *args, **kwargs: None,
    )

    model = WanCausalModel(
        init_from=cfg.models["student"]["init_from"],
        training_config=cfg.training,
        trainable=True,
    )
    model.transformer = model.transformer.to(device=device, dtype=dtype)

    method = TeacherForcingSFTMethod(
        cfg=cfg,
        role_models={"student": model},
    )
    method.on_train_start()

    batch = _build_synthetic_batch(device, dtype)
    loss_map, outputs, _metrics = method.single_train_step(batch, iteration=0)

    loss = loss_map["total_loss"]
    assert torch.is_tensor(loss), "total_loss must be a torch.Tensor"
    assert torch.isfinite(loss).item(), (
        f"total_loss is not finite: {loss.item()}")

    method.backward(loss_map, outputs, grad_accum_rounds=1)

    blocks = getattr(model.transformer, "blocks", None)
    assert blocks is not None and len(blocks) > 0, (
        "CausalWanTransformer is expected to expose ``.blocks``")
    layer0 = blocks[0]

    trainable = [p for p in layer0.parameters() if p.requires_grad]
    assert len(trainable) > 0, "layer 0 has no trainable parameters"

    for i, p in enumerate(trainable):
        assert p.grad is not None, f"layer 0 param[{i}] has None grad"
        assert torch.isfinite(p.grad).all().item(), (
            f"layer 0 param[{i}] grad contains NaN/Inf")

    any_nonzero = any(
        p.grad.detach().float().norm().item() > 0.0 for p in trainable)
    assert any_nonzero, (
        "all layer-0 grads are exactly zero; backward did not "
        "reach the first transformer block")

    # Teacher forcing must build its own (concatenated) attention mask and
    # must not have constructed the diffusion-forcing mask.
    assert model.transformer.teacher_forcing_block_mask is not None, (
        "teacher-forcing mask was not constructed")
    assert model.transformer.block_mask is None, (
        "diffusion-forcing mask should not be built on the TF path")
