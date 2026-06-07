# SPDX-License-Identifier: Apache-2.0
"""Numerical-parity test: FastVideo Cosmos3 denoise/CFG glue vs the framework.

The DiT forward and the sequence-packing are already framework-parity-verified
(``test_cosmos3_dit_parity*`` / ``test_cosmos3_packing_parity``). This test pins
the remaining glue that the native pipeline adds — the SEQUENTIAL classifier-free
guidance velocity and one UniPC scheduler step — against the framework math
(``diffusers_cosmos3.pipeline.Cosmos3OmniDiffusersPipeline.get_cfg_velocity`` /
``__call__``):

  * for one denoise step, replicate the framework's ``get_cfg_velocity`` exactly
    on top of the OFFICIAL ``Cosmos3VFMNetwork`` forward (oracle): a conditional
    pass (prompt tokens) and an unconditional pass (negative-prompt tokens),
    each masking the prediction on conditioning frames
    (``pred * (1 - condition_mask)``), then ``v = uncond + g*(cond - uncond)``;
  * run FastVideo's :func:`cosmos3_get_cfg_velocity` with the native DiT (the
    framework weights copied in) + the native packer, and assert the velocity
    matches the oracle;
  * take one ``UniPCMultistepScheduler.step`` on each (the actual checkpoint
    scheduler) and assert the stepped latent matches;
  * drive :meth:`Cosmos3DenoiseEngine.denoise` for >= 2 steps and assert it
    equals the manual framework step-by-step loop.

CPU / float32, via the reference SDPA monkey-patch. The official model is the
parity ORACLE.

Run:
    cd <worktree> && <fv-cosmos3 python> -m pytest \
        tests/local_tests/cosmos3/test_cosmos3_denoise_cfg_parity.py -q
"""
from __future__ import annotations

import pytest
import torch

# The official framework provides the parity oracle.
cosmos_framework = pytest.importorskip(
    "cosmos_framework",
    reason="cosmos_framework not installed; run in fv-cosmos3 env.",
)

from diffusers import UniPCMultistepScheduler  # noqa: E402

from .test_cosmos3_dit_parity import _copy_weights  # noqa: E402
from .test_cosmos3_dit_parity_mrope import (  # noqa: E402
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

# Tiny special tokens (< tiny vocab_size=64), video path appends eos + sog.
_SPECIAL_TOKENS = {
    "start_of_generation": 60,
    "end_of_generation": 61,
    "eos_token_id": 62,
}


def _make_scheduler() -> UniPCMultistepScheduler:
    """UniPC scheduler matching the Cosmos3 checkpoint flow config."""
    return UniPCMultistepScheduler(
        num_train_timesteps=1000,
        solver_order=2,
        prediction_type="flow_prediction",
        use_flow_sigmas=True,
        flow_shift=10.0,
    )


# ---------------------------------------------------------------------------
# Framework-oracle CFG velocity (replicates pipeline.get_cfg_velocity math).
# ---------------------------------------------------------------------------
def _framework_pack(*, text_ids, vision_latent, cond_frames, timestep):
    from cosmos_framework.data.vfm.sequence_packing import (
        GenerationDataClean,
        SequencePlan,
        pack_input_sequence,
    )

    # vision_latent is [1, C, T, H, W]; temporal dim is axis 2.
    gen_data_clean = GenerationDataClean(
        batch_size=1,
        is_image_batch=(vision_latent.shape[2] == 1),
        x0_tokens_vision=[vision_latent],
        fps_vision=None,
        num_vision_items_per_sample=[1],
    )
    plans = [SequencePlan(has_text=True, has_vision=True, condition_frame_indexes_vision=list(cond_frames))]
    return pack_input_sequence(
        sequence_plans=plans,
        input_text_indexes=[list(text_ids)],
        gen_data_clean=gen_data_clean,
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


def _framework_inputs(ps):
    """Framework PackedSequence -> framework Cosmos3VFMNetwork forward kwargs."""
    return dict(packed_seq=ps)


def _framework_cfg_velocity(
    *,
    vfm,
    flat_latent: torch.Tensor,
    timestep: torch.Tensor,
    guidance: float,
    vision_shape: tuple[int, int, int, int],
    cond_frames: list[int],
    cond_ids: list[int],
    uncond_ids: list[int],
) -> torch.Tensor:
    """Replicate the framework ``get_cfg_velocity`` on the oracle model.

    Single vision item; sequential cond then uncond pass; mask condition
    frames; ``v = uncond + g*(cond - uncond)``.
    """
    timestep_value = float(timestep.reshape(()).item())
    vision_latent = flat_latent.reshape(vision_shape)  # [C, T, H, W]

    def _run(text_ids: list[int]) -> torch.Tensor:
        ps = _framework_pack(
            text_ids=text_ids,
            # The framework packer expects a 5D [1, C, T, H, W] latent.
            vision_latent=vision_latent.unsqueeze(0),
            cond_frames=cond_frames,
            timestep=timestep_value,
        )
        out = vfm(**_framework_inputs(ps))
        preds = out.get("preds_vision")
        cond_mask = ps.vision.condition_mask[0]  # [T] or [T,1,1]
        if preds is None:
            return torch.zeros_like(flat_latent)
        pred = preds[0].squeeze(0)  # [C, T, H, W]
        keep = (1.0 - cond_mask.reshape(-1, 1, 1)).to(dtype=pred.dtype, device=pred.device)
        velocity = pred * keep if keep.sum() > 0 else torch.zeros_like(pred)
        return velocity.reshape(-1)

    cond_v = _run(cond_ids)
    uncond_v = _run(uncond_ids)
    return uncond_v + guidance * (cond_v - uncond_v)


# ---------------------------------------------------------------------------
# Cases: T2V (no cond), I2V (cond frame 0), single-frame T2I.
# ---------------------------------------------------------------------------
_CASES = [
    pytest.param(2, 4, 4, 6, [], id="t2v_2x2x2"),
    pytest.param(3, 8, 4, 5, [0], id="i2v_3x4x2_cond0"),
    pytest.param(1, 8, 8, 4, [], id="t2i_1x4x4"),
]


def _build_models(num_layers: int = 2, seed_model: int = 42):
    vfm = _build_tiny_cosmos3_mrope(seed=seed_model, num_layers=num_layers)
    dit = _build_tiny_fastvideo_dit_mrope(num_layers=num_layers)
    _copy_weights(vfm, dit)
    return vfm, dit


def _fastvideo_velocity(dit, *, flat_latent, timestep, guidance, vision_shape, cond_frames, cond_ids, uncond_ids):
    from fastvideo.pipelines.basic.cosmos3.cosmos3_pipeline import (
        Cosmos3VisionSpec,
        cosmos3_get_cfg_velocity,
    )

    spec = Cosmos3VisionSpec(shape=vision_shape, condition_frame_indexes=list(cond_frames))
    return cosmos3_get_cfg_velocity(
        transformer=dit,
        flat_latent=flat_latent,
        timestep=timestep,
        guidance=guidance,
        specs=[spec],
        cond_token_ids=cond_ids,
        uncond_token_ids=uncond_ids,
        special_tokens=_SPECIAL_TOKENS,
        latent_patch_size=_LATENT_PATCH_SIZE,
        temporal_modality_margin=_TEMPORAL_MODALITY_MARGIN,
        reset_spatial_ids=_RESET_SPATIAL_IDS,
        enable_fps_modulation=False,
        base_fps=24.0,
        temporal_compression_factor=_TCF,
    )


class TestCosmos3DenoiseCFGParity:

    @pytest.mark.parametrize(("grid_t", "latent_h", "latent_w", "n_text", "cond"), _CASES)
    def test_cfg_velocity_matches_framework(self, grid_t, latent_h, latent_w, n_text, cond):
        vfm, dit = _build_models()
        torch.manual_seed(0)
        cond_ids = torch.randint(0, 60, (n_text,)).tolist()
        uncond_ids = torch.randint(0, 60, (max(1, n_text - 1),)).tolist()
        vision_shape = (_LATENT_CHANNEL, grid_t, latent_h, latent_w)
        flat_latent = torch.randn(int(torch.tensor(vision_shape).prod()))
        timestep = torch.tensor([[500.0]])  # framework expects [1,1]; we reshape to scalar
        guidance = 6.0

        fw_v = _framework_cfg_velocity(
            vfm=vfm,
            flat_latent=flat_latent,
            timestep=timestep,
            guidance=guidance,
            vision_shape=vision_shape,
            cond_frames=cond,
            cond_ids=cond_ids,
            uncond_ids=uncond_ids,
        )
        fv_v = _fastvideo_velocity(
            dit,
            flat_latent=flat_latent,
            timestep=timestep,
            guidance=guidance,
            vision_shape=vision_shape,
            cond_frames=cond,
            cond_ids=cond_ids,
            uncond_ids=uncond_ids,
        )
        assert fw_v.shape == fv_v.shape, f"shape: fw={fw_v.shape} fv={fv_v.shape}"
        max_abs = (fw_v - fv_v).abs().max().item()
        print(f"\n[cfg_velocity {grid_t}x{latent_h}x{latent_w} cond={cond}] max abs diff = {max_abs:.3e}")
        torch.testing.assert_close(fv_v, fw_v, atol=1e-4, rtol=1e-3)

    def test_one_unipc_step_matches_framework(self):
        """CFG velocity + one UniPC step: FastVideo == framework math."""
        vfm, dit = _build_models()
        grid_t, latent_h, latent_w = 2, 4, 4
        vision_shape = (_LATENT_CHANNEL, grid_t, latent_h, latent_w)
        torch.manual_seed(3)
        cond_ids = torch.randint(0, 60, (5,)).tolist()
        uncond_ids = torch.randint(0, 60, (4,)).tolist()
        flat_latent = torch.randn(int(torch.tensor(vision_shape).prod()))
        guidance = 6.0

        fw_sched = _make_scheduler()
        fv_sched = _make_scheduler()
        fw_sched.set_timesteps(4, device=torch.device("cpu"))
        fv_sched.set_timesteps(4, device=torch.device("cpu"))
        t = fw_sched.timesteps[0]

        fw_v = _framework_cfg_velocity(
            vfm=vfm,
            flat_latent=flat_latent,
            timestep=t.reshape(1, 1),
            guidance=guidance,
            vision_shape=vision_shape,
            cond_frames=[],
            cond_ids=cond_ids,
            uncond_ids=uncond_ids,
        )
        fw_stepped = fw_sched.step(model_output=fw_v, timestep=t, sample=flat_latent.unsqueeze(0),
                                   return_dict=False)[0].squeeze(0)

        fv_v = _fastvideo_velocity(
            dit,
            flat_latent=flat_latent,
            timestep=t.reshape(1),
            guidance=guidance,
            vision_shape=vision_shape,
            cond_frames=[],
            cond_ids=cond_ids,
            uncond_ids=uncond_ids,
        )
        fv_stepped = fv_sched.step(model_output=fv_v, timestep=t, sample=flat_latent.unsqueeze(0),
                                   return_dict=False)[0].squeeze(0)

        max_abs = (fw_stepped - fv_stepped).abs().max().item()
        print(f"\n[unipc_step] max abs diff = {max_abs:.3e}")
        torch.testing.assert_close(fv_stepped, fw_stepped, atol=1e-4, rtol=1e-3)

    def test_full_denoise_loop_matches_framework(self):
        """Cosmos3DenoiseEngine.denoise (>= 2 steps) == framework step-by-step."""
        from fastvideo.pipelines.basic.cosmos3.cosmos3_pipeline import (
            Cosmos3DenoiseEngine,
            Cosmos3VisionSpec,
        )

        vfm, dit = _build_models()
        grid_t, latent_h, latent_w = 2, 4, 4
        vision_shape = (_LATENT_CHANNEL, grid_t, latent_h, latent_w)
        torch.manual_seed(5)
        cond_ids = torch.randint(0, 60, (5,)).tolist()
        uncond_ids = torch.randint(0, 60, (4,)).tolist()
        flat_latent = torch.randn(int(torch.tensor(vision_shape).prod()))
        guidance = 6.0
        num_steps = 3

        # Manual framework loop (oracle).
        fw_sched = _make_scheduler()
        fw_sched.set_timesteps(num_steps, device=torch.device("cpu"))
        fw_latent = flat_latent.clone()
        for t in fw_sched.timesteps:
            v = _framework_cfg_velocity(
                vfm=vfm,
                flat_latent=fw_latent,
                timestep=t.reshape(1, 1),
                guidance=guidance,
                vision_shape=vision_shape,
                cond_frames=[],
                cond_ids=cond_ids,
                uncond_ids=uncond_ids,
            )
            fw_latent = fw_sched.step(model_output=v, timestep=t, sample=fw_latent.unsqueeze(0),
                                      return_dict=False)[0].squeeze(0)

        # FastVideo engine loop.
        fv_sched = _make_scheduler()
        fv_sched.set_timesteps(num_steps, device=torch.device("cpu"))
        engine = Cosmos3DenoiseEngine(
            transformer=dit,
            scheduler=fv_sched,
            special_tokens=_SPECIAL_TOKENS,
            latent_patch_size=_LATENT_PATCH_SIZE,
            temporal_modality_margin=_TEMPORAL_MODALITY_MARGIN,
            reset_spatial_ids=_RESET_SPATIAL_IDS,
            enable_fps_modulation=False,
            base_fps=24.0,
            temporal_compression_factor=_TCF,
        )
        spec = Cosmos3VisionSpec(shape=vision_shape, condition_frame_indexes=[])
        fv_latent = engine.denoise(
            flat_latent=flat_latent.clone(),
            timesteps=fv_sched.timesteps,
            guidance=guidance,
            specs=[spec],
            cond_token_ids=cond_ids,
            uncond_token_ids=uncond_ids,
        )

        assert fv_latent.shape == fw_latent.shape
        max_abs = (fw_latent - fv_latent).abs().max().item()
        print(f"\n[full_denoise {num_steps} steps] final latent max abs diff = {max_abs:.3e}")
        torch.testing.assert_close(fv_latent, fw_latent, atol=1e-4, rtol=1e-3)
