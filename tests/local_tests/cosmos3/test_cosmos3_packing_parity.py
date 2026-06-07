# SPDX-License-Identifier: Apache-2.0
"""Numerical-parity test: FastVideo Cosmos3 sequence packing vs the OFFICIAL framework.

FastVideo's native packer
(``fastvideo.pipelines.basic.cosmos3.sequence_packing.pack_cosmos3_video_sequence``)
builds the packed-sequence inputs the ``Cosmos3VFMTransformer`` consumes. This
test asserts, for the SAME logical inputs (prompt token ids, vision latents,
condition-frame indices, diffusion timestep, fps), that FastVideo's packing
matches the official ``cosmos_framework.data.vfm.sequence_packing.pack_input_sequence``
oracle field-by-field:

  * ``position_ids`` (exact, ``[3, seq]``),
  * ``text_ids`` / ``text_indexes``,
  * ``split_lens`` / ``attn_modes`` / ``sample_lens`` / ``sequence_length``,
  * vision ``sequence_indexes`` / ``token_shapes`` / ``timesteps`` /
    ``mse_loss_indexes`` / ``noisy_frame_indexes`` / ``condition_mask``.

Coverage spans T2V (no condition frames), I2V (condition frame 0), and T2I
(single conditioned frame), across multiple grids, plus a multi-sample batch.

Then BOTH the framework-packed and FastVideo-packed inputs are fed through the
SAME tiny FastVideo DiT (framework weights copied in as in the existing DiT
parity tests). Asserting bit-identical DiT output confirms FastVideo's own
packing drives the DiT to the same result as the framework's packing.

The official framework is the parity ORACLE; it runs on CPU / float32 via the
SDPA monkey-patch from ``test_cosmos3_reference_forward``.

Run:
    cd <worktree> && <fv-cosmos3 python> -m pytest \
        tests/local_tests/cosmos3/test_cosmos3_packing_parity.py -q
"""
from __future__ import annotations

import pytest
import torch

# The official framework provides the parity oracle.
cosmos_framework = pytest.importorskip(
    "cosmos_framework",
    reason="cosmos_framework not installed; run in fv-cosmos3 env.",
)

# Reuse the DiT parity helpers (weight copy + framework->DiT kwarg builder) and
# the mRoPE tiny-model builders (real-checkpoint rope constants).
from .test_cosmos3_dit_parity import (  # noqa: E402
    _copy_weights,
    _fastvideo_inputs_from_packed_seq,
)
from .test_cosmos3_dit_parity_mrope import (  # noqa: E402
    _LATENT_CHANNEL,
    _LATENT_PATCH_SIZE,
    _MROPE_SECTION,
    _RESET_SPATIAL_IDS,
    _ROPE_THETA,
    _TCF,
    _TEMPORAL_MODALITY_MARGIN,
    _build_tiny_cosmos3_mrope,
    _build_tiny_fastvideo_dit_mrope,
)
from .test_cosmos3_reference_forward import _apply_sdpa_patches  # noqa: E402

pytestmark = [pytest.mark.local]

# Ensure the CPU/float32 SDPA patches are installed (idempotent).
_apply_sdpa_patches()

# Tiny special-token ids (kept < tiny vocab_size=64). The video path appends
# eos + start_of_generation after the prompt tokens.
_SPECIAL_TOKENS = {
    "start_of_generation": 60,
    "end_of_generation": 61,
    "eos_token_id": 62,
}


# ---------------------------------------------------------------------------
# Builders for the two packers from the SAME logical sample inputs.
# ---------------------------------------------------------------------------
def _make_vision(grid_t: int, latent_h: int, latent_w: int, seed: int) -> torch.Tensor:
    """Deterministic VAE latent ``[1, C, T, H, W]``."""
    torch.manual_seed(seed)
    return torch.randn(1, _LATENT_CHANNEL, grid_t, latent_h, latent_w)


def _framework_pack(
    *,
    text_ids_per_sample: list[list[int]],
    visions: list[torch.Tensor],
    cond_frames_per_sample: list[list[int]],
    timesteps: list[float],
    is_image_batch: bool,
):
    from cosmos_framework.data.vfm.sequence_packing import (
        GenerationDataClean,
        SequencePlan,
        pack_input_sequence,
    )

    gen_data_clean = GenerationDataClean(
        batch_size=len(visions),
        is_image_batch=is_image_batch,
        x0_tokens_vision=list(visions),
        fps_vision=None,
        num_vision_items_per_sample=[1] * len(visions),
    )
    plans = [
        SequencePlan(has_text=True, has_vision=True, condition_frame_indexes_vision=list(cf))
        for cf in cond_frames_per_sample
    ]
    return pack_input_sequence(
        sequence_plans=plans,
        input_text_indexes=[list(t) for t in text_ids_per_sample],
        gen_data_clean=gen_data_clean,
        input_timesteps=torch.tensor(timesteps, dtype=torch.float32),
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


def _fastvideo_pack(
    *,
    text_ids_per_sample: list[list[int]],
    visions: list[torch.Tensor],
    cond_frames_per_sample: list[list[int]],
    timesteps: list[float],
):
    from fastvideo.pipelines.basic.cosmos3.sequence_packing import (
        Cosmos3SampleInputs,
        Cosmos3VisionItem,
        pack_cosmos3_video_sequence,
    )

    samples = [
        Cosmos3SampleInputs(
            text_ids=list(t),
            vision=Cosmos3VisionItem(latent=v, condition_frame_indexes=list(cf)),
            timestep=float(ts),
        )
        for t, v, cf, ts in zip(text_ids_per_sample, visions, cond_frames_per_sample, timesteps)
    ]
    return pack_cosmos3_video_sequence(
        samples,
        _SPECIAL_TOKENS,
        latent_patch_size=_LATENT_PATCH_SIZE,
        include_end_of_generation_token=False,
        temporal_modality_margin=_TEMPORAL_MODALITY_MARGIN,
        reset_spatial_ids=_RESET_SPATIAL_IDS,
        enable_fps_modulation=False,
        base_fps=24.0,
        temporal_compression_factor=_TCF,
    )


# ---------------------------------------------------------------------------
# Field-by-field comparison.
# ---------------------------------------------------------------------------
def _assert_packs_match(fw, fv) -> None:
    """Assert the framework PackedSequence and FastVideo pack agree field-by-field."""
    # Structure.
    assert fv.split_lens == list(fw.split_lens), f"split_lens: fv={fv.split_lens} fw={list(fw.split_lens)}"
    assert fv.attn_modes == list(fw.attn_modes), f"attn_modes: fv={fv.attn_modes} fw={list(fw.attn_modes)}"
    assert fv.sample_lens == list(fw.sample_lens), f"sample_lens: fv={fv.sample_lens} fw={list(fw.sample_lens)}"
    assert int(fv.sequence_length) == int(fw.sequence_length)

    # Text.
    torch.testing.assert_close(fv.text_ids, fw.text_ids.to(torch.long), rtol=0, atol=0)
    torch.testing.assert_close(fv.text_indexes, fw.text_indexes.to(torch.long), rtol=0, atol=0)

    # position_ids: exact, [3, seq], same dtype.
    assert fv.position_ids.shape == fw.position_ids.shape, (
        f"position_ids shape: fv={tuple(fv.position_ids.shape)} fw={tuple(fw.position_ids.shape)}")
    assert fv.position_ids.dtype == fw.position_ids.dtype, (
        f"position_ids dtype: fv={fv.position_ids.dtype} fw={fw.position_ids.dtype}")
    torch.testing.assert_close(fv.position_ids, fw.position_ids, rtol=0, atol=0)

    # Vision.
    fwv = fw.vision
    torch.testing.assert_close(fv.vision_sequence_indexes, fwv.sequence_indexes.to(torch.long), rtol=0, atol=0)
    assert fv.vision_token_shapes == [tuple(s) for s in fwv.token_shapes], (
        f"token_shapes: fv={fv.vision_token_shapes} fw={[tuple(s) for s in fwv.token_shapes]}")
    torch.testing.assert_close(fv.vision_timesteps.to(torch.float32), fwv.timesteps.to(torch.float32))
    torch.testing.assert_close(fv.vision_mse_loss_indexes, fwv.mse_loss_indexes.to(torch.long), rtol=0, atol=0)
    assert len(fv.vision_noisy_frame_indexes) == len(fwv.noisy_frame_indexes)
    for a, b in zip(fv.vision_noisy_frame_indexes, fwv.noisy_frame_indexes):
        torch.testing.assert_close(a.to(torch.long), b.to(torch.long), rtol=0, atol=0)
    assert len(fv.vision_condition_mask) == len(fwv.condition_mask)
    for a, b in zip(fv.vision_condition_mask, fwv.condition_mask):
        torch.testing.assert_close(a.flatten().to(torch.float32), b.flatten().to(torch.float32))


# (grid_t, latent_h, latent_w, n_text, cond_frames, id) — single-sample cases.
_CASES = [
    pytest.param(1, 8, 8, 4, [], id="t2i_1x4x4"),
    pytest.param(1, 4, 4, 5, [0], id="t2i_cond_1x2x2"),
    pytest.param(2, 4, 4, 4, [], id="t2v_2x2x2"),
    pytest.param(3, 8, 4, 6, [], id="t2v_3x4x2"),
    pytest.param(2, 4, 4, 5, [0], id="i2v_2x2x2"),
    pytest.param(3, 4, 4, 4, [0], id="i2v_3x2x2"),
]


class TestCosmos3PackingParity:

    # -- Field-by-field packing parity -------------------------------------
    @pytest.mark.parametrize(("grid_t", "latent_h", "latent_w", "n_text", "cond"), _CASES)
    def test_packing_fields_match_framework(self, grid_t, latent_h, latent_w, n_text, cond):
        torch.manual_seed(0)
        text_ids = torch.randint(0, 60, (n_text,)).tolist()
        vision = _make_vision(grid_t, latent_h, latent_w, seed=123)
        timestep = 500.0

        fw = _framework_pack(
            text_ids_per_sample=[text_ids],
            visions=[vision],
            cond_frames_per_sample=[cond],
            timesteps=[timestep],
            is_image_batch=(grid_t == 1),
        )
        fv = _fastvideo_pack(
            text_ids_per_sample=[text_ids],
            visions=[vision],
            cond_frames_per_sample=[cond],
            timesteps=[timestep],
        )
        _assert_packs_match(fw, fv)

    def test_packing_fields_match_framework_multi_sample(self):
        """A batch of two samples (T2V + I2V) packs identically to the framework."""
        torch.manual_seed(1)
        t0 = torch.randint(0, 60, (3,)).tolist()
        t1 = torch.randint(0, 60, (5,)).tolist()
        v0 = _make_vision(2, 4, 4, seed=11)
        v1 = _make_vision(2, 4, 4, seed=22)
        kwargs = dict(
            text_ids_per_sample=[t0, t1],
            visions=[v0, v1],
            cond_frames_per_sample=[[], [0]],
            timesteps=[500.0, 250.0],
        )
        fw = _framework_pack(is_image_batch=False, **kwargs)
        fv = _fastvideo_pack(**kwargs)
        _assert_packs_match(fw, fv)

    # -- End-to-end: FastVideo packing drives the DiT identically ----------
    @pytest.mark.parametrize(("grid_t", "latent_h", "latent_w", "n_text", "cond"), _CASES)
    def test_fastvideo_packing_drives_dit_like_framework(self, grid_t, latent_h, latent_w, n_text, cond):
        """Feed BOTH the framework-packed and FastVideo-packed inputs through the
        SAME FastVideo DiT (framework weights copied in); assert identical output.
        """
        num_layers = 2
        torch.manual_seed(0)
        text_ids = torch.randint(0, 60, (n_text,)).tolist()
        vision = _make_vision(grid_t, latent_h, latent_w, seed=123)
        timestep = 500.0

        fw_pack = _framework_pack(
            text_ids_per_sample=[text_ids],
            visions=[vision],
            cond_frames_per_sample=[cond],
            timesteps=[timestep],
            is_image_batch=(grid_t == 1),
        )
        fv_pack = _fastvideo_pack(
            text_ids_per_sample=[text_ids],
            visions=[vision],
            cond_frames_per_sample=[cond],
            timesteps=[timestep],
        )
        # Guard: the two packs must agree before we trust the DiT comparison.
        _assert_packs_match(fw_pack, fv_pack)

        # One DiT instance, framework weights copied in (parity oracle weights).
        vfm = _build_tiny_cosmos3_mrope(seed=42, num_layers=num_layers)
        dit = _build_tiny_fastvideo_dit_mrope(num_layers=num_layers)
        _copy_weights(vfm, dit)

        with torch.no_grad():
            out_fw = dit(**_fastvideo_inputs_from_packed_seq(fw_pack))
            out_fv = dit(**fv_pack.to_dit_kwargs())

        # last_hidden_state must be bit-identical.
        lhs_fw = out_fw["last_hidden_state"]
        lhs_fv = out_fv["last_hidden_state"]
        assert lhs_fw.shape == lhs_fv.shape
        max_abs_lhs = (lhs_fw - lhs_fv).abs().max().item()
        print(f"\n[packing->dit {grid_t}x{latent_h}x{latent_w} cond={cond}] "
              f"last_hidden_state max abs diff = {max_abs_lhs:.3e}")
        torch.testing.assert_close(lhs_fv, lhs_fw, rtol=0, atol=0)

        # preds_vision must be bit-identical when there are noisy frames to
        # predict. (A fully-conditioned clip has no noisy patches, so the DiT
        # emits no "preds_vision" — both packs agree the mse-loss set is empty,
        # already asserted by the field-parity guard above.)
        has_preds = fv_pack.vision_mse_loss_indexes.numel() > 0
        assert ("preds_vision" in out_fw) == has_preds
        assert ("preds_vision" in out_fv) == has_preds
        if has_preds:
            pv_fw = out_fw["preds_vision"][0]
            pv_fv = out_fv["preds_vision"][0]
            assert pv_fw.shape == pv_fv.shape
            max_abs_pv = (pv_fw - pv_fv).abs().max().item()
            print(f"[packing->dit {grid_t}x{latent_h}x{latent_w} cond={cond}] "
                  f"preds_vision max abs diff = {max_abs_pv:.3e}")
            torch.testing.assert_close(pv_fv, pv_fw, rtol=0, atol=0)
