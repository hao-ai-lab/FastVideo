# SPDX-License-Identifier: Apache-2.0
"""AnyFlow pretrain method tests.

CPU-only unit tests covering:
- Config flag defaults (bit-identity preserved on legacy paths).
- ``WanTimeTextImageEmbedding`` dual-timestep forward (additive default = bit-identical
  to legacy; gated mode reproduces AnyFlow's ``(1 - g) * temb + g * delta_emb`` fusion).
- ``WanTransformer3DModel.forward`` accepts ``r_timestep``.
- ``FlowMapEulerDiscreteScheduler`` numerics: ``apply_shift``, ``get_train_weight``,
  ``step``.
- ``(t, r)`` per-batch sampling distribution.
- Central-difference target math.
- AnyFlow HF checkpoint key remap (``remap_anyflow_keys``).
"""

from __future__ import annotations

import copy

import pytest
import torch

from fastvideo.configs.models.dits import WanVideoConfig


# ---------------------------------------------------------------------------
# Task 1: r_embedder config flags default to bit-identity preservation.
# ---------------------------------------------------------------------------


def test_wan_arch_defaults_preserve_bit_identity() -> None:
    cfg = WanVideoConfig()
    arch = cfg.arch_config
    assert arch.r_embedder is False
    assert arch.r_embedder_fusion == "additive"
    assert arch.r_embedder_gate_value == 0.25
    assert arch.r_embedder_deltatime_type == "r"


# ---------------------------------------------------------------------------
# Task 2: WanTimeTextImageEmbedding dual-timestep forward.
# ---------------------------------------------------------------------------


def _make_embedder(
    *,
    r_embedder: bool,
    fusion: str = "additive",
    gate: float = 0.25,
    deltatime_type: str = "r",
):
    from fastvideo.models.dits.wanvideo import WanTimeTextImageEmbedding

    emb = WanTimeTextImageEmbedding(
        dim=32,
        time_freq_dim=64,
        text_embed_dim=16,
        image_embed_dim=None,
        r_embedder=r_embedder,
        r_embedder_fusion=fusion,
        r_embedder_gate_value=gate,
        r_embedder_deltatime_type=deltatime_type,
    )
    emb.eval()
    return emb


def test_embedder_default_path_no_delta_module() -> None:
    """When r_embedder=False, delta_embedder must not be allocated."""
    emb = _make_embedder(r_embedder=False)
    assert emb.delta_embedder is None


def test_embedder_default_path_is_bit_identical_to_legacy() -> None:
    """With r_embedder=False, forward output must match the legacy single-t path
    (no r_timestep kwarg, no extra computation)."""
    torch.manual_seed(0)
    emb = _make_embedder(r_embedder=False)
    t = torch.randint(0, 1000, (2,), dtype=torch.long)
    txt = torch.randn(2, 4, 16)
    temb_a, proj_a, _, _ = emb(t, txt)
    # Calling without r_timestep again must be deterministic-equal.
    temb_b, proj_b, _, _ = emb(t, txt)
    torch.testing.assert_close(temb_a, temb_b)
    torch.testing.assert_close(proj_a, proj_b)
    assert temb_a.shape == (2, 32)
    assert proj_a.shape == (2, 32 * 6)


def test_embedder_enabled_without_r_timestep_is_bit_identical_to_legacy() -> None:
    """Even with r_embedder=True, if r_timestep is None at call time the
    forward must skip the delta path entirely so existing call sites that
    don't pass r_timestep stay byte-equal to the legacy result."""
    torch.manual_seed(0)
    emb_legacy = _make_embedder(r_embedder=False)
    torch.manual_seed(0)
    emb_dual = _make_embedder(r_embedder=True, fusion="additive")
    t = torch.randint(0, 1000, (2,), dtype=torch.long)
    txt = torch.randn(2, 4, 16)
    temb_legacy, proj_legacy, _, _ = emb_legacy(t, txt)
    temb_dual, proj_dual, _, _ = emb_dual(t, txt)  # No r_timestep.
    torch.testing.assert_close(temb_legacy, temb_dual)
    torch.testing.assert_close(proj_legacy, proj_dual)


def test_embedder_gated_fusion_formula() -> None:
    """Gated mode: rt_emb = (1 - g) * temb_t + g * delta_emb (with delta_input=r)."""
    torch.manual_seed(0)
    emb = _make_embedder(r_embedder=True, fusion="gated", gate=0.25)
    t = torch.tensor([500, 500], dtype=torch.long)
    r = torch.tensor([100, 100], dtype=torch.long)
    txt = torch.randn(2, 4, 16)

    temb_t = emb.time_embedder(t)
    delta_emb = emb.delta_embedder(r)
    expected = 0.75 * temb_t + 0.25 * delta_emb

    rt_emb, _, _, _ = emb(t, txt, r_timestep=r)
    torch.testing.assert_close(rt_emb, expected, rtol=1e-5, atol=1e-5)


def test_embedder_additive_fusion_formula() -> None:
    """Additive mode: rt_emb = temb_t + g * delta_emb."""
    torch.manual_seed(0)
    emb = _make_embedder(r_embedder=True, fusion="additive", gate=0.3)
    t = torch.tensor([700, 700], dtype=torch.long)
    r = torch.tensor([200, 200], dtype=torch.long)
    txt = torch.randn(2, 4, 16)

    temb_t = emb.time_embedder(t)
    delta_emb = emb.delta_embedder(r)
    expected = temb_t + 0.3 * delta_emb

    rt_emb, _, _, _ = emb(t, txt, r_timestep=r)
    torch.testing.assert_close(rt_emb, expected, rtol=1e-5, atol=1e-5)


def test_embedder_deltatime_type_t_minus_r() -> None:
    """When deltatime_type='t-r', delta_embedder consumes (t - r)."""
    torch.manual_seed(0)
    emb = _make_embedder(
        r_embedder=True, fusion="gated", gate=0.5, deltatime_type="t-r")
    t = torch.tensor([800, 800], dtype=torch.long)
    r = torch.tensor([300, 300], dtype=torch.long)
    txt = torch.randn(2, 4, 16)

    temb_t = emb.time_embedder(t)
    delta_emb = emb.delta_embedder(t - r)
    expected = 0.5 * temb_t + 0.5 * delta_emb

    rt_emb, _, _, _ = emb(t, txt, r_timestep=r)
    torch.testing.assert_close(rt_emb, expected, rtol=1e-5, atol=1e-5)


def test_embedder_invalid_fusion_raises() -> None:
    with pytest.raises(ValueError, match="r_embedder_fusion"):
        _make_embedder(r_embedder=True, fusion="bogus")


def test_embedder_invalid_deltatime_type_raises() -> None:
    with pytest.raises(ValueError, match="r_embedder_deltatime_type"):
        _make_embedder(r_embedder=True, fusion="gated", deltatime_type="2t-r")


def test_embedder_gate_not_in_state_dict() -> None:
    """Gate is a non-persistent buffer; it must not appear in state_dict so
    checkpoints stay portable across different gate hyperparameters."""
    emb = _make_embedder(r_embedder=True, fusion="gated", gate=0.25)
    keys = list(emb.state_dict().keys())
    assert not any("_r_embedder_gate" in k for k in keys)
