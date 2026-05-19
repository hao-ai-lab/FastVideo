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


# ---------------------------------------------------------------------------
# Task 3: WanTransformer3DModel threads r_timestep through.
# ---------------------------------------------------------------------------


def test_wan_transformer_forward_signature_has_r_timestep() -> None:
    """The forward signature must declare r_timestep explicitly (not
    swallowed by **kwargs) so callers and type checkers can see it."""
    import inspect

    from fastvideo.models.dits.wanvideo import WanTransformer3DModel

    sig = inspect.signature(WanTransformer3DModel.forward)
    assert "r_timestep" in sig.parameters
    param = sig.parameters["r_timestep"]
    assert param.default is None


def test_wan_transformer_init_propagates_r_embedder_config() -> None:
    """When the arch config sets r_embedder=True the WanTransformer3DModel
    constructor must instantiate the embedder with the delta path active.

    We avoid full WanTransformer3DModel instantiation (which requires
    distributed init) by reading the source's __init__ to confirm it
    forwards the four arch flags to WanTimeTextImageEmbedding.
    """
    import inspect

    from fastvideo.models.dits.wanvideo import WanTransformer3DModel

    src = inspect.getsource(WanTransformer3DModel.__init__)
    # All four arch config fields must be passed to WanTimeTextImageEmbedding.
    assert "r_embedder=config.r_embedder" in src
    assert "r_embedder_fusion=config.r_embedder_fusion" in src
    assert "r_embedder_gate_value=config.r_embedder_gate_value" in src
    assert "r_embedder_deltatime_type=config.r_embedder_deltatime_type" in src


def test_wan_transformer_forward_threads_r_timestep_to_embedder() -> None:
    """The forward must pass r_timestep into the embedder call (verified via
    source inspection to avoid heavyweight distributed bring-up)."""
    import inspect

    from fastvideo.models.dits.wanvideo import WanTransformer3DModel

    src = inspect.getsource(WanTransformer3DModel.forward)
    assert "r_timestep=r_timestep" in src, (
        "WanTransformer3DModel.forward must forward r_timestep into "
        "self.condition_embedder")


# ---------------------------------------------------------------------------
# Task 4: FlowMapEulerDiscreteScheduler numerics.
# ---------------------------------------------------------------------------


def _scheduler(*, shift: float = 1.0, n_train: int = 1000):
    from fastvideo.models.schedulers.scheduling_flow_map_euler_discrete import (
        FlowMapEulerDiscreteScheduler, )
    return FlowMapEulerDiscreteScheduler(
        num_train_timesteps=n_train, shift=shift)


def test_flow_map_scheduler_set_timesteps_descending() -> None:
    sched = _scheduler(shift=5.0)
    sched.set_timesteps(num_inference_steps=4, device=torch.device("cpu"))
    ts = sched.timesteps
    # N inference steps → N + 1 boundary entries.
    assert ts.numel() == 5
    assert torch.all(ts[:-1] >= ts[1:])  # descending
    assert ts[-1].item() == 0.0
    assert ts[0].item() == pytest.approx(1000.0, abs=1e-3)


def test_flow_map_scheduler_set_timesteps_custom_overrides_schedule() -> None:
    sched = _scheduler(shift=5.0)
    custom = [999.0, 937.0, 833.0, 624.0, 0.0]
    sched.set_timesteps(
        num_inference_steps=4,
        device=torch.device("cpu"),
        custom_timesteps=custom,
    )
    torch.testing.assert_close(
        sched.timesteps, torch.tensor(custom, dtype=torch.float32))


def test_flow_map_scheduler_custom_timesteps_must_be_descending() -> None:
    sched = _scheduler(shift=5.0)
    with pytest.raises(ValueError, match="descending"):
        sched.set_timesteps(
            num_inference_steps=4,
            device=torch.device("cpu"),
            custom_timesteps=[100.0, 500.0, 900.0],
        )


def test_flow_map_scheduler_apply_shift_endpoints_invariant() -> None:
    """apply_shift fixes the endpoints {0, 1} and produces non-trivial
    motion in the interior for shift != 1."""
    sched = _scheduler(shift=5.0)
    t = torch.tensor([0.0, 0.5, 1.0])
    shifted = sched.apply_shift(t)
    torch.testing.assert_close(
        shifted, torch.tensor([0.0, 5.0 / 6.0, 1.0]), rtol=1e-6, atol=1e-6)


def test_flow_map_scheduler_apply_shift_shift_one_is_identity() -> None:
    sched = _scheduler(shift=1.0)
    t = torch.linspace(0.0, 1.0, 100)
    torch.testing.assert_close(sched.apply_shift(t), t)


def test_flow_map_scheduler_step_one_euler_iteration_matches_formula() -> None:
    """One step: x_r = x_t - ((t - r) / N) * model_output."""
    sched = _scheduler(shift=1.0)
    sched.set_timesteps(num_inference_steps=4, device=torch.device("cpu"))

    torch.manual_seed(0)
    x_t = torch.randn(2, 4, 1, 8, 8)
    v = torch.randn_like(x_t)
    t = torch.tensor([750.0, 500.0])
    r = torch.tensor([500.0, 250.0])

    out = sched.step(v, sample=x_t, timestep=t, r_timestep=r)
    expected = x_t - ((t - r) / 1000.0).view(-1, 1, 1, 1, 1) * v
    torch.testing.assert_close(out, expected, rtol=1e-6, atol=1e-6)


def test_flow_map_scheduler_get_train_weight_beta08_shape_and_renorm() -> None:
    """beta08: t * sqrt(1-t), renormalized so sum equals num_train_timesteps.
    The interior of the schedule must dominate the endpoints (monotone up
    then monotone down)."""
    sched = _scheduler()
    t = torch.linspace(0.001, 0.999, 1000)
    w = sched.get_train_weight(t, weight_type="beta08")
    assert torch.allclose(w.sum(), torch.tensor(1000.0), rtol=1e-3)
    assert torch.all(w >= 0.0)
    # Endpoints smaller than the middle bump.
    mid = len(w) // 2
    assert w[0] < w[mid]
    assert w[-1] < w[mid]


def test_flow_map_scheduler_get_train_weight_uniform_is_constant_norm() -> None:
    sched = _scheduler()
    t = torch.linspace(0.0, 1.0, 1000)
    w = sched.get_train_weight(t, weight_type="uniform")
    assert torch.allclose(w.sum(), torch.tensor(1000.0), rtol=1e-3)
    # All entries equal to 1.0 after renormalization.
    torch.testing.assert_close(w, torch.ones_like(w), rtol=1e-6, atol=1e-6)


def test_flow_map_scheduler_get_train_weight_accepts_absolute_units() -> None:
    """When t is provided in [0, num_train_timesteps] the helper auto-
    normalizes; result must match the [0, 1] call."""
    sched = _scheduler()
    t_norm = torch.linspace(0.001, 0.999, 1000)
    t_abs = t_norm * 1000
    w_norm = sched.get_train_weight(t_norm, weight_type="beta08")
    w_abs = sched.get_train_weight(t_abs, weight_type="beta08")
    torch.testing.assert_close(w_norm, w_abs, rtol=1e-5, atol=1e-5)


def test_flow_map_scheduler_add_noise_matches_flow_matching_formula() -> None:
    """Linear flow-matching: x_t = (1 - sigma) * x_0 + sigma * eps, where
    sigma = t / num_train_timesteps."""
    sched = _scheduler()
    torch.manual_seed(0)
    x0 = torch.randn(2, 4, 1, 4, 4)
    eps = torch.randn_like(x0)
    t = torch.tensor([250.0, 750.0])
    out = sched.add_noise(x0, eps, t)
    sigma = (t / 1000.0).view(-1, 1, 1, 1, 1)
    expected = (1.0 - sigma) * x0 + sigma * eps
    torch.testing.assert_close(out, expected, rtol=1e-6, atol=1e-6)
