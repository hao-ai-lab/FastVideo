# SPDX-License-Identifier: Apache-2.0
"""Numerical-parity test: FastVideo Cosmos3 sound (t2vs) pathway vs the framework.

Covers the audio generation pathway end-to-end at the DiT level:

  * **sound packing** — FastVideo's native packer
    (``pack_cosmos3_video_sequence`` with a ``Cosmos3SoundItem``) vs the
    framework ``pack_input_sequence`` with ``has_sound``: sound tokens share the
    vision "full" split, with ``(T,1,1)`` shapes, a ``(T,1)`` condition mask,
    and 3D-MRoPE temporal positions starting at the vision temporal offset
    (parallel to vision); and
  * **DiT sound forward** — the dormant ``audio_proj_in`` / ``audio_proj_out`` /
    ``audio_modality_embed`` heads, now activated (framework ``sound2llm`` /
    ``llm2sound`` / ``sound_modality_embed``).

Both tiny models are built sound-enabled from the SAME config, framework weights
(incl. the sound heads) are copied into the FastVideo DiT, and the FRAMEWORK
model + framework pack is the parity ORACLE (CPU/float32 via the SDPA
monkey-patch). We assert the native packer matches the framework field-by-field,
then that ``preds_vision`` AND ``preds_sound`` match the framework forward.

Run:
    cd <worktree> && <fv-cosmos3 python> -m pytest \
        tests/local_tests/cosmos3/test_cosmos3_sound_parity.py -q -s
"""
from __future__ import annotations

import pytest
import torch

# The official framework provides the parity oracle.
cosmos_framework = pytest.importorskip(
    "cosmos_framework",
    reason="cosmos_framework not installed; run in fv-cosmos3 env.",
)

from .test_cosmos3_dit_parity import (  # noqa: E402
    _fastvideo_inputs_from_packed_seq,
    _framework_to_fastvideo_state_dict,
)
from .test_cosmos3_dit_parity_mrope import (  # noqa: E402
    _LATENT_CHANNEL,
    _LATENT_PATCH_SIZE,
    _RESET_SPATIAL_IDS,
    _SOUND_DIM,
    _TCF,
    _TEMPORAL_MODALITY_MARGIN,
    _build_tiny_cosmos3_mrope,
    _build_tiny_fastvideo_dit_mrope,
)
from .test_cosmos3_reference_forward import _apply_sdpa_patches  # noqa: E402

pytestmark = [pytest.mark.local]
_apply_sdpa_patches()

_SPECIAL_TOKENS = {"start_of_generation": 60, "end_of_generation": 61, "eos_token_id": 62}


def _copy_weights_with_sound(vfm, dit) -> None:
    """Copy backbone + vision weights AND the sound MoT heads into the DiT."""
    mapped = _framework_to_fastvideo_state_dict(vfm, num_layers=dit.num_hidden_layers)
    src = dict(vfm.named_parameters())
    mapped["audio_proj_in.weight"] = src["sound2llm.weight"].detach().clone()
    mapped["audio_proj_in.bias"] = src["sound2llm.bias"].detach().clone()
    mapped["audio_proj_out.weight"] = src["llm2sound.weight"].detach().clone()
    mapped["audio_proj_out.bias"] = src["llm2sound.bias"].detach().clone()
    mapped["audio_modality_embed"] = src["sound_modality_embed"].detach().clone()
    dst = dict(dit.named_parameters())
    with torch.no_grad():
        for name, tensor in mapped.items():
            assert name in dst, f"DiT missing param {name!r}"
            assert dst[name].shape == tensor.shape, f"shape mismatch {name}"
            dst[name].copy_(tensor.to(dst[name].dtype))


def _framework_pack_sound(*, text_ids, vision, sound, cond_vision, cond_sound, timestep, is_image_batch):
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
        x0_tokens_sound=[sound],
        fps_sound=None,
    )
    plans = [SequencePlan(
        has_text=True, has_vision=True, has_sound=True,
        condition_frame_indexes_vision=list(cond_vision),
        condition_frame_indexes_sound=list(cond_sound),
    )]
    return pack_input_sequence(
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


def _fastvideo_pack_sound(*, text_ids, vision, sound, cond_vision, cond_sound, timestep):
    from fastvideo.pipelines.basic.cosmos3.sequence_packing import (
        Cosmos3SampleInputs,
        Cosmos3SoundItem,
        Cosmos3VisionItem,
        pack_cosmos3_video_sequence,
    )

    samples = [Cosmos3SampleInputs(
        text_ids=list(text_ids),
        vision=Cosmos3VisionItem(latent=vision, condition_frame_indexes=list(cond_vision)),
        sound=Cosmos3SoundItem(latent=sound, condition_frame_indexes=list(cond_sound)),
        timestep=float(timestep),
    )]
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


def _fv_inputs_with_sound(ps) -> dict:
    """Framework PackedSequence (with sound) -> native DiT forward kwargs."""
    kw = _fastvideo_inputs_from_packed_seq(ps)
    s = ps.sound
    kw.update(
        sound_tokens=list(s.tokens),
        sound_token_shapes=[tuple(x) for x in s.token_shapes],
        sound_sequence_indexes=s.sequence_indexes,
        sound_timesteps=s.timesteps,
        sound_mse_loss_indexes=s.mse_loss_indexes,
        sound_noisy_frame_indexes=list(s.noisy_frame_indexes),
        fps_sound=None,
    )
    return kw


def _diffs(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float]:
    d = (a - b).abs()
    return d.max().item(), d.mean().item()


# (grid_t, latent_h, latent_w, sound_t, n_text, cond_vision, cond_sound)
_CASES = [
    pytest.param(2, 4, 4, 5, 4, [], [], id="t2vs_2x2x2_snd5"),
    pytest.param(3, 8, 4, 8, 5, [], [], id="t2vs_3x4x2_snd8"),
    pytest.param(2, 4, 4, 6, 5, [0], [], id="i2vs_cond_snd6"),
]


class TestCosmos3SoundParity:

    def _build(self, num_layers=2, seed_model=42):
        vfm = _build_tiny_cosmos3_mrope(seed=seed_model, num_layers=num_layers, sound_gen=True)
        dit = _build_tiny_fastvideo_dit_mrope(num_layers=num_layers)
        _copy_weights_with_sound(vfm, dit)
        return vfm, dit

    def _make_inputs(self, grid_t, latent_h, latent_w, sound_t, n_text, cond_v, cond_s, seed=7):
        torch.manual_seed(seed)
        vision = torch.randn(1, _LATENT_CHANNEL, grid_t, latent_h, latent_w)
        sound = torch.randn(_SOUND_DIM, sound_t)  # [C, T]
        text_ids = torch.randint(0, 60, (n_text,)).tolist()
        return dict(text_ids=text_ids, vision=vision, sound=sound,
                    cond_vision=cond_v, cond_sound=cond_s, timestep=500.0)

    @pytest.mark.parametrize(("grid_t", "lh", "lw", "snd_t", "n_text", "cond_v", "cond_s"), _CASES)
    def test_sound_packing_matches_framework(self, grid_t, lh, lw, snd_t, n_text, cond_v, cond_s):
        ins = self._make_inputs(grid_t, lh, lw, snd_t, n_text, cond_v, cond_s)
        fw = _framework_pack_sound(is_image_batch=(grid_t == 1), **ins)
        fv = _fastvideo_pack_sound(**ins)

        assert fv.split_lens == list(fw.split_lens), f"split_lens fv={fv.split_lens} fw={list(fw.split_lens)}"
        assert fv.attn_modes == list(fw.attn_modes)
        assert int(fv.sequence_length) == int(fw.sequence_length)
        torch.testing.assert_close(fv.position_ids, fw.position_ids, rtol=0, atol=0)  # [3, seq], exact
        # Sound fields.
        s = fw.sound
        torch.testing.assert_close(fv.sound_sequence_indexes, s.sequence_indexes.to(torch.long), rtol=0, atol=0)
        assert fv.sound_token_shapes == [tuple(x) for x in s.token_shapes]
        torch.testing.assert_close(fv.sound_timesteps.to(torch.float32), s.timesteps.to(torch.float32))
        torch.testing.assert_close(fv.sound_mse_loss_indexes, s.mse_loss_indexes.to(torch.long), rtol=0, atol=0)
        for a, b in zip(fv.sound_noisy_frame_indexes, s.noisy_frame_indexes):
            torch.testing.assert_close(a.to(torch.long), b.to(torch.long), rtol=0, atol=0)
        for a, b in zip(fv.sound_condition_mask, s.condition_mask):
            torch.testing.assert_close(a.flatten().to(torch.float32), b.flatten().to(torch.float32))
        print(f"\n[sound_packing {grid_t}x{lh}x{lw} snd={snd_t}] position_ids + sound fields exact")

    @pytest.mark.parametrize(("grid_t", "lh", "lw", "snd_t", "n_text", "cond_v", "cond_s"), _CASES)
    def test_sound_dit_forward_matches_framework(self, grid_t, lh, lw, snd_t, n_text, cond_v, cond_s):
        vfm, dit = self._build()
        ins = self._make_inputs(grid_t, lh, lw, snd_t, n_text, cond_v, cond_s)
        fw_pack = _framework_pack_sound(is_image_batch=(grid_t == 1), **ins)
        fv_pack = _fastvideo_pack_sound(**ins)

        with torch.no_grad():
            fw_out = vfm(packed_seq=fw_pack)  # framework model + framework pack (oracle)
            fv_out = dit(**fv_pack.to_dit_kwargs())  # native model + native pack
            # Also run the native DiT on the framework pack to isolate the forward.
            fv_on_fw = dit(**_fv_inputs_with_sound(fw_pack))

        # preds_vision parity.
        pv_max, pv_mean = _diffs(fv_out["preds_vision"][0], fw_out["preds_vision"][0])
        # preds_sound parity.
        ps_max, ps_mean = _diffs(fv_out["preds_sound"][0], fw_out["preds_sound"][0])
        # native-DiT-on-framework-pack (forward only) parity.
        psf_max, psf_mean = _diffs(fv_on_fw["preds_sound"][0], fw_out["preds_sound"][0])
        print(f"\n[sound_dit {grid_t}x{lh}x{lw} snd={snd_t}] "
              f"preds_vision max={pv_max:.3e} mean={pv_mean:.3e} | "
              f"preds_sound max={ps_max:.3e} mean={ps_mean:.3e} | "
              f"preds_sound(fwpack) max={psf_max:.3e} mean={psf_mean:.3e}")

        assert fv_out["preds_sound"][0].shape == fw_out["preds_sound"][0].shape
        torch.testing.assert_close(fv_out["preds_vision"][0], fw_out["preds_vision"][0], atol=1e-4, rtol=1e-3)
        torch.testing.assert_close(fv_out["preds_sound"][0], fw_out["preds_sound"][0], atol=1e-4, rtol=1e-3)
        torch.testing.assert_close(fv_on_fw["preds_sound"][0], fw_out["preds_sound"][0], atol=1e-4, rtol=1e-3)
