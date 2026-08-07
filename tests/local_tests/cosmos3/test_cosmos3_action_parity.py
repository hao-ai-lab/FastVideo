# SPDX-License-Identifier: Apache-2.0
"""Numerical-parity test: FastVideo Cosmos3 action pathway vs the framework.

Covers the action (multi-embodiment world-model) modality at the DiT level:

  * **action packing** — native ``pack_cosmos3_video_sequence`` with a
    ``Cosmos3ActionItem`` vs framework ``pack_input_sequence`` with
    ``has_action``: action tokens share the vision "full" split, with ``(T,)``
    shapes, a ``(T,1)`` condition mask, and 3D-MRoPE temporal positions at the
    vision offset with ``start_frame_offset=1`` (parallel to vision); and
  * **DiT action forward** — the dormant domain-aware ``action_proj_in`` /
    ``action_proj_out`` (``DomainAwareLinear``) + ``action_modality_embed`` heads,
    now activated, with a per-token embodiment ``domain_id``.

Framework model + pack is the parity ORACLE (CPU/float32 via SDPA monkey-patch).
We assert the native packer matches the framework field-by-field, then that
``preds_vision`` AND ``preds_action`` match the framework forward.

Run:
    cd <worktree> && <fv-cosmos3 python> -m pytest \
        tests/local_tests/cosmos3/test_cosmos3_action_parity.py -q -s
"""
from __future__ import annotations

import pytest
import torch

cosmos_framework = pytest.importorskip(
    "cosmos_framework",
    reason="cosmos_framework not installed; run in fv-cosmos3 env.",
)

from .test_cosmos3_dit_parity import (  # noqa: E402
    _fastvideo_inputs_from_packed_seq,
    _framework_to_fastvideo_state_dict,
)
from .test_cosmos3_dit_parity_mrope import (  # noqa: E402
    _ACTION_DIM,
    _LATENT_CHANNEL,
    _LATENT_PATCH_SIZE,
    _RESET_SPATIAL_IDS,
    _TCF,
    _TEMPORAL_MODALITY_MARGIN,
    _build_tiny_cosmos3_mrope,
    _build_tiny_fastvideo_dit_mrope,
)
from .test_cosmos3_reference_forward import _apply_sdpa_patches  # noqa: E402

pytestmark = [pytest.mark.local]
_apply_sdpa_patches()

_SPECIAL_TOKENS = {"start_of_generation": 60, "end_of_generation": 61, "eos_token_id": 62}


def _copy_weights_with_action(vfm, dit) -> None:
    """Copy backbone + vision weights AND the domain-aware action heads."""
    mapped = _framework_to_fastvideo_state_dict(vfm, num_layers=dit.num_hidden_layers)
    src = dict(vfm.named_parameters())
    mapped["action_proj_in.fc.weight"] = src["action2llm.fc.weight"].detach().clone()
    mapped["action_proj_in.bias.weight"] = src["action2llm.bias.weight"].detach().clone()
    mapped["action_proj_out.fc.weight"] = src["llm2action.fc.weight"].detach().clone()
    mapped["action_proj_out.bias.weight"] = src["llm2action.bias.weight"].detach().clone()
    mapped["action_modality_embed"] = src["action_modality_embed"].detach().clone()
    dst = dict(dit.named_parameters())
    with torch.no_grad():
        for name, tensor in mapped.items():
            assert name in dst, f"DiT missing param {name!r}"
            assert dst[name].shape == tensor.shape, f"shape mismatch {name}"
            dst[name].copy_(tensor.to(dst[name].dtype))


def _framework_pack_action(*, text_ids, vision, action, cond_vision, cond_action, domain_id, timestep,
                           is_image_batch):
    from cosmos_framework.data.vfm.sequence_packing import (
        GenerationDataClean,
        SequencePlan,
        pack_input_sequence,
    )

    gen = GenerationDataClean(
        batch_size=1,
        is_image_batch=is_image_batch,
        x0_tokens_vision=[vision],
        fps_vision=None,
        num_vision_items_per_sample=[1],
        x0_tokens_action=[action],
        fps_action=None,
        action_domain_id=[torch.tensor([domain_id], dtype=torch.long)],
    )
    plans = [SequencePlan(
        has_text=True, has_vision=True, has_action=True,
        condition_frame_indexes_vision=list(cond_vision),
        condition_frame_indexes_action=list(cond_action),
    )]
    ps = pack_input_sequence(
        sequence_plans=plans,
        input_text_indexes=[list(text_ids)],
        gen_data_clean=gen,
        input_timesteps=torch.tensor([timestep], dtype=torch.float32),
        special_tokens=_SPECIAL_TOKENS,
        latent_patch_size=_LATENT_PATCH_SIZE,
        include_end_of_generation_token=False,
        position_embedding_type="unified_3d_mrope",
        unified_3d_mrope_reset_spatial_ids=_RESET_SPATIAL_IDS,
        unified_3d_mrope_temporal_modality_margin=_TEMPORAL_MODALITY_MARGIN,
        enable_fps_modulation=False,
        base_fps=24.0,
        temporal_compression_factor=_TCF,
    )
    # The framework sets action.domain_id on the packed sequence from
    # gen_data_clean inside the model (_get_velocity); mirror that for the oracle.
    if ps.action is not None:
        ps.action.domain_id = [torch.tensor([domain_id], dtype=torch.long)]
    return ps


def _fastvideo_pack_action(*, text_ids, vision, action, cond_vision, cond_action, domain_id, timestep):
    from fastvideo.pipelines.basic.cosmos3.sequence_packing import (
        Cosmos3ActionItem,
        Cosmos3SampleInputs,
        Cosmos3VisionItem,
        pack_cosmos3_video_sequence,
    )

    samples = [Cosmos3SampleInputs(
        text_ids=list(text_ids),
        vision=Cosmos3VisionItem(latent=vision, condition_frame_indexes=list(cond_vision)),
        action=Cosmos3ActionItem(latent=action, condition_frame_indexes=list(cond_action), domain_id=domain_id),
        timestep=float(timestep),
    )]
    return pack_cosmos3_video_sequence(
        samples, _SPECIAL_TOKENS,
        latent_patch_size=_LATENT_PATCH_SIZE, include_end_of_generation_token=False,
        temporal_modality_margin=_TEMPORAL_MODALITY_MARGIN, reset_spatial_ids=_RESET_SPATIAL_IDS,
        enable_fps_modulation=False, base_fps=24.0, temporal_compression_factor=_TCF,
    )


def _fv_inputs_with_action(ps) -> dict:
    kw = _fastvideo_inputs_from_packed_seq(ps)
    a = ps.action
    kw.update(
        action_tokens=list(a.tokens),
        action_token_shapes=[tuple(x) for x in a.token_shapes],
        action_sequence_indexes=a.sequence_indexes,
        action_timesteps=a.timesteps,
        action_mse_loss_indexes=a.mse_loss_indexes,
        action_noisy_frame_indexes=list(a.noisy_frame_indexes),
        action_domain_id=list(a.domain_id),
    )
    return kw


def _diffs(a, b):
    d = (a - b).abs()
    return d.max().item(), d.mean().item()


# (grid_t, lh, lw, action_t, n_text, cond_vision, cond_action, domain_id)
_CASES = [
    pytest.param(2, 4, 4, 6, 4, [], [], 0, id="a2v_2x2x2_act6_dom0"),
    pytest.param(3, 8, 4, 9, 5, [], [], 7, id="a2v_3x4x2_act9_dom7"),
    pytest.param(2, 4, 4, 5, 5, [0], [0], 3, id="ai2v_cond_act5_dom3"),
]


class TestCosmos3ActionParity:

    def _build(self, num_layers=2, seed_model=42):
        vfm = _build_tiny_cosmos3_mrope(seed=seed_model, num_layers=num_layers, action_gen=True)
        dit = _build_tiny_fastvideo_dit_mrope(num_layers=num_layers)
        _copy_weights_with_action(vfm, dit)
        return vfm, dit

    def _make_inputs(self, grid_t, lh, lw, act_t, n_text, cond_v, cond_a, dom, seed=7):
        torch.manual_seed(seed)
        return dict(
            text_ids=torch.randint(0, 60, (n_text,)).tolist(),
            vision=torch.randn(1, _LATENT_CHANNEL, grid_t, lh, lw),
            action=torch.randn(act_t, _ACTION_DIM),  # [T, D]
            cond_vision=cond_v, cond_action=cond_a, domain_id=dom, timestep=500.0,
        )

    @pytest.mark.parametrize(("grid_t", "lh", "lw", "act_t", "n_text", "cond_v", "cond_a", "dom"), _CASES)
    def test_action_packing_matches_framework(self, grid_t, lh, lw, act_t, n_text, cond_v, cond_a, dom):
        ins = self._make_inputs(grid_t, lh, lw, act_t, n_text, cond_v, cond_a, dom)
        fw = _framework_pack_action(is_image_batch=(grid_t == 1), **ins)
        fv = _fastvideo_pack_action(**ins)
        assert fv.split_lens == list(fw.split_lens), f"split_lens fv={fv.split_lens} fw={list(fw.split_lens)}"
        assert fv.attn_modes == list(fw.attn_modes)
        assert int(fv.sequence_length) == int(fw.sequence_length)
        torch.testing.assert_close(fv.position_ids, fw.position_ids, rtol=0, atol=0)
        a = fw.action
        torch.testing.assert_close(fv.action_sequence_indexes, a.sequence_indexes.to(torch.long), rtol=0, atol=0)
        assert fv.action_token_shapes == [tuple(x) for x in a.token_shapes]
        torch.testing.assert_close(fv.action_timesteps.to(torch.float32), a.timesteps.to(torch.float32))
        torch.testing.assert_close(fv.action_mse_loss_indexes, a.mse_loss_indexes.to(torch.long), rtol=0, atol=0)
        for x, y in zip(fv.action_noisy_frame_indexes, a.noisy_frame_indexes):
            torch.testing.assert_close(x.to(torch.long), y.to(torch.long), rtol=0, atol=0)
        print(f"\n[action_packing {grid_t}x{lh}x{lw} act={act_t} dom={dom}] position_ids + action fields exact")

    @pytest.mark.parametrize(("grid_t", "lh", "lw", "act_t", "n_text", "cond_v", "cond_a", "dom"), _CASES)
    def test_action_dit_forward_matches_framework(self, grid_t, lh, lw, act_t, n_text, cond_v, cond_a, dom):
        vfm, dit = self._build()
        ins = self._make_inputs(grid_t, lh, lw, act_t, n_text, cond_v, cond_a, dom)
        fw_pack = _framework_pack_action(is_image_batch=(grid_t == 1), **ins)
        fv_pack = _fastvideo_pack_action(**ins)
        with torch.no_grad():
            fw_out = vfm(packed_seq=fw_pack)
            fv_out = dit(**fv_pack.to_dit_kwargs())
            fv_on_fw = dit(**_fv_inputs_with_action(fw_pack))
        pv_mx, pv_mn = _diffs(fv_out["preds_vision"][0], fw_out["preds_vision"][0])
        pa_mx, pa_mn = _diffs(fv_out["preds_action"][0], fw_out["preds_action"][0])
        paf_mx, paf_mn = _diffs(fv_on_fw["preds_action"][0], fw_out["preds_action"][0])
        print(f"\n[action_dit {grid_t}x{lh}x{lw} act={act_t} dom={dom}] "
              f"preds_vision max={pv_mx:.3e} mean={pv_mn:.3e} | "
              f"preds_action max={pa_mx:.3e} mean={pa_mn:.3e} | "
              f"preds_action(fwpack) max={paf_mx:.3e} mean={paf_mn:.3e}")
        assert fv_out["preds_action"][0].shape == fw_out["preds_action"][0].shape
        torch.testing.assert_close(fv_out["preds_vision"][0], fw_out["preds_vision"][0], atol=1e-4, rtol=1e-3)
        torch.testing.assert_close(fv_out["preds_action"][0], fw_out["preds_action"][0], atol=1e-4, rtol=1e-3)
        torch.testing.assert_close(fv_on_fw["preds_action"][0], fw_out["preds_action"][0], atol=1e-4, rtol=1e-3)

    @pytest.mark.parametrize(("grid_t", "lh", "lw", "act_t", "n_text", "cond_v", "cond_a", "dom"), _CASES)
    def test_action_cfg_velocity_matches_framework(self, grid_t, lh, lw, act_t, n_text, cond_v, cond_a, dom):
        """Combined [vision|action] sequential-CFG velocity (action pipeline glue)
        matches a framework-DiT oracle."""
        from fastvideo.pipelines.basic.cosmos3.cosmos3_pipeline import (
            Cosmos3ActionSpec,
            Cosmos3VisionSpec,
            cosmos3_get_cfg_velocity,
        )

        vfm, dit = self._build()
        vlat_shape = (_LATENT_CHANNEL, grid_t, lh, lw)
        action_shape = (act_t, _ACTION_DIM)
        torch.manual_seed(3)
        cond_ids = torch.randint(0, 60, (n_text,)).tolist()
        uncond_ids = torch.randint(0, 60, (max(1, n_text - 1),)).tolist()
        vis_numel = int(torch.tensor(vlat_shape).prod())
        act_numel = int(torch.tensor(action_shape).prod())
        flat = torch.randn(vis_numel + act_numel)
        guidance, ts = 6.0, 500.0

        def _fw_velocity(ids):
            vision = flat[:vis_numel].reshape(vlat_shape).unsqueeze(0)
            action = flat[vis_numel:].reshape(action_shape)
            ps = _framework_pack_action(text_ids=ids, vision=vision, action=action, cond_vision=cond_v,
                                        cond_action=cond_a, domain_id=dom, timestep=ts,
                                        is_image_batch=(grid_t == 1))
            with torch.no_grad():
                out = vfm(packed_seq=ps)
            pv = out["preds_vision"][0].squeeze(0)  # [C,T,H,W] (zero on clean)
            pa = out["preds_action"][0]  # [T,D] (zero on clean)
            return torch.cat([pv.reshape(-1), pa.reshape(-1)])

        fw_cond, fw_uncond = _fw_velocity(cond_ids), _fw_velocity(uncond_ids)
        fw_v = fw_uncond + guidance * (fw_cond - fw_uncond)
        fv_v = cosmos3_get_cfg_velocity(
            transformer=dit, flat_latent=flat, timestep=torch.tensor([ts]), guidance=guidance,
            specs=[Cosmos3VisionSpec(shape=vlat_shape, condition_frame_indexes=list(cond_v))],
            action_specs=[Cosmos3ActionSpec(shape=action_shape, condition_frame_indexes=list(cond_a), domain_id=dom)],
            cond_token_ids=cond_ids, uncond_token_ids=uncond_ids,
            special_tokens=_SPECIAL_TOKENS, latent_patch_size=_LATENT_PATCH_SIZE,
            temporal_modality_margin=_TEMPORAL_MODALITY_MARGIN, reset_spatial_ids=_RESET_SPATIAL_IDS,
            enable_fps_modulation=False, base_fps=24.0, temporal_compression_factor=_TCF,
        )
        assert fv_v.shape == fw_v.shape, f"shape fv={fv_v.shape} fw={fw_v.shape}"
        mx, mn = _diffs(fv_v, fw_v)
        print(f"\n[action_cfg_velocity {grid_t}x{lh}x{lw} act={act_t} dom={dom}] max={mx:.3e} mean={mn:.3e}")
        torch.testing.assert_close(fv_v, fw_v, atol=1e-4, rtol=1e-3)
