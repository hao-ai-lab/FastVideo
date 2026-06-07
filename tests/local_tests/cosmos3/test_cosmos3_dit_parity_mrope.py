# SPDX-License-Identifier: Apache-2.0
"""Numerical-parity test: FastVideo Cosmos3 DiT vs ``Cosmos3VFMNetwork`` (mRoPE).

Companion to ``test_cosmos3_dit_parity.py`` (which covers ``3d_rope``). This
module exercises the rotary mode the REAL ``nvidia/Cosmos3-Nano`` checkpoint
uses: ``position_embedding_type="unified_3d_mrope"`` with the real-checkpoint
settings (``mrope_section=[24,20,20]``, ``mrope_interleaved=True``,
``rope_theta=5e6``, ``unified_3d_mrope_reset_spatial_ids=True``,
``temporal_modality_margin=15000``).

Under unified 3D mRoPE there is NO additive latent position embedding
(``latent_pos_embed is None``); all positional information rides on the
per-token 3D (T, H, W) rotary embedding. The packed-sequence ``position_ids``
are therefore shape ``[3, seq_len]``, built exactly like the framework data
packer (``cosmos_framework.data.vfm.sequence_packing``):

  * text tokens broadcast one monotone id across all three axes
    (``get_3d_mrope_ids_text_tokens``),
  * the temporal offset is bumped by ``temporal_modality_margin`` at the
    text->vision boundary,
  * vision tokens lay out a (T, H, W) grid with spatial ids reset per segment
    (``get_3d_mrope_ids_vae_tokens`` with ``reset_spatial_indices=True``).

Both models are built tiny from the SAME config, framework weights are copied
into the FastVideo DiT (reusing the ``3d_rope`` test's weight map — the
transformer key surface is identical across rotary modes), and BOTH forwards
run on identical deterministic CPU / float32 inputs. The official model is the
parity ORACLE (run on CPU via the SDPA monkey-patch in
``test_cosmos3_reference_forward``).

Run:
    cd <worktree> && <fv-cosmos3 python> -m pytest \
        tests/local_tests/cosmos3/test_cosmos3_dit_parity_mrope.py -q
"""
from __future__ import annotations

import pytest
import torch

# The official framework provides the parity oracle.
cosmos_framework = pytest.importorskip(
    "cosmos_framework",
    reason="cosmos_framework not installed; run in fv-cosmos3 env.",
)

# Reuse the reference harness's SDPA monkey-patch and the 3d_rope parity
# test's weight-copy + input-builder helpers (key surface is rotary-agnostic).
from .test_cosmos3_dit_parity import (  # noqa: E402
    _copy_weights,
    _fastvideo_inputs_from_packed_seq,
    _framework_to_fastvideo_state_dict,
)
from .test_cosmos3_reference_forward import _apply_sdpa_patches  # noqa: E402

pytestmark = [pytest.mark.local]

# Ensure the CPU/float32 SDPA patches are installed (idempotent).
_apply_sdpa_patches()

# Real-checkpoint unified_3d_mrope settings (tiny model, real rope constants).
_ROPE_THETA = 5_000_000.0
_MROPE_SECTION = [24, 20, 20]
_MROPE_INTERLEAVED = True
_RESET_SPATIAL_IDS = True
_TEMPORAL_MODALITY_MARGIN = 15_000
_LATENT_PATCH_SIZE = 2
_LATENT_CHANNEL = 16
_TCF = 4  # temporal compression factor


# ---------------------------------------------------------------------------
# Tiny model builders (framework + FastVideo) with unified_3d_mrope.
# ---------------------------------------------------------------------------
def _build_tiny_cosmos3_mrope(seed: int = 42, num_layers: int = 2):
    """Tiny framework ``Cosmos3VFMNetwork`` with ``unified_3d_mrope``.

    ``rope_theta`` / ``rope_scaling`` (carrying ``mrope_section`` +
    ``mrope_interleaved``) are threaded through the materialized text config;
    ``position_embedding_type="unified_3d_mrope"`` leaves ``latent_pos_embed``
    as ``None`` so positions ride solely on the 3D rotary embedding.
    """
    from cosmos_framework.model.vfm.mot.cosmos3_vfm_network import (
        Cosmos3VFMNetwork,
        Cosmos3VFMNetworkConfig,
    )
    from cosmos_framework.model.vfm.mot.unified_mot import (
        Qwen3MoTConfig,
        Qwen3VLTextForCausalLM,
    )
    from cosmos_framework.model.vfm.vlm.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig

    tiny_text_dict = dict(
        model_type="qwen3_vl_text",
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=num_layers,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
        rms_norm_eps=1e-6,
        attention_bias=False,
        attention_dropout=0.0,
        rope_theta=_ROPE_THETA,
        rope_scaling={
            "rope_type": "default",
            "mrope_section": _MROPE_SECTION,
            "mrope_interleaved": _MROPE_INTERLEAVED,
        },
        max_position_embeddings=262144,
    )
    mot_cfg = Qwen3MoTConfig(
        config_dict=tiny_text_dict,
        qk_norm_for_text=True,
        qk_norm_for_diffusion=True,
        include_visual=False,
    )
    tiny_vlm_cfg = Qwen3VLConfig(text_config=tiny_text_dict)
    vfm_cfg = Cosmos3VFMNetworkConfig(
        vision_gen=True,
        vlm_config=tiny_vlm_cfg,
        latent_patch_size=_LATENT_PATCH_SIZE,
        latent_downsample_factor=8,
        latent_channel_size=_LATENT_CHANNEL,
        position_embedding_type="unified_3d_mrope",
        max_latent_h=16,
        max_latent_w=16,
        max_latent_t=8,
        temporal_compression_factor_vision=_TCF,
    )
    torch.manual_seed(seed)
    lm = Qwen3VLTextForCausalLM(config=mot_cfg)
    vfm = Cosmos3VFMNetwork(language_model=lm, config=vfm_cfg)
    # inv_freq is a non-persistent buffer; init it on CPU (mirrors from_pretrained).
    vfm.language_model.model.rotary_emb.init_weights(buffer_device=None)
    vfm.eval()
    return vfm


def _build_tiny_fastvideo_dit_mrope(num_layers: int = 2):
    from fastvideo.configs.models.dits.cosmos3 import (
        Cosmos3ArchConfig,
        Cosmos3VideoConfig,
    )
    from fastvideo.models.dits.cosmos3 import Cosmos3VFMTransformer

    arch = Cosmos3ArchConfig(
        hidden_size=16,
        num_hidden_layers=num_layers,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
        intermediate_size=32,
        vocab_size=64,
        rms_norm_eps=1e-6,
        attention_bias=False,
        latent_patch_size=_LATENT_PATCH_SIZE,
        latent_channel=_LATENT_CHANNEL,
        rope_theta=_ROPE_THETA,
        mrope_section=_MROPE_SECTION,
        mrope_interleaved=_MROPE_INTERLEAVED,
        unified_3d_mrope_reset_spatial_ids=_RESET_SPATIAL_IDS,
        temporal_modality_margin=_TEMPORAL_MODALITY_MARGIN,
        position_embedding_type="unified_3d_mrope",
        base_fps=24.0,
        temporal_compression_factor=_TCF,
        enable_fps_modulation=False,
        # Dormant heads present in the checkpoint surface (constructed for
        # strict-load parity; not exercised by this video-path forward).
        action_gen=True,
        action_dim=64,
        max_action_dim=64,
        num_embodiment_domains=32,
        sound_gen=True,
        sound_dim=64,
    )
    cfg = Cosmos3VideoConfig(arch_config=arch)
    model = Cosmos3VFMTransformer(cfg, hf_config={})
    return model.to(torch.float32).eval()


# ---------------------------------------------------------------------------
# [3, seq_len] mRoPE position-id builder (mirrors the framework data packer).
# ---------------------------------------------------------------------------
def _build_mrope_position_ids(n_text: int, grid_t: int, patch_h: int, patch_w: int) -> torch.Tensor:
    """Build ``[3, seq_len]`` (T, H, W) mRoPE ids for one text+vision sample.

    Reproduces ``pack_input_sequence`` for a single causal-text + full-vision
    sample: monotone text ids on all axes, ``+temporal_modality_margin`` at the
    text->vision boundary, then a reset-spatial (T, H, W) vision grid.
    """
    from cosmos_framework.data.vfm.sequence_packing import (
        get_3d_mrope_ids_text_tokens,
        get_3d_mrope_ids_vae_tokens,
    )

    offset: int | float = 0
    text_ids, offset = get_3d_mrope_ids_text_tokens(num_tokens=n_text, temporal_offset=offset)
    # End of text modality: add the boundary margin before vision.
    offset += _TEMPORAL_MODALITY_MARGIN
    vision_ids, offset = get_3d_mrope_ids_vae_tokens(
        grid_t=grid_t,
        grid_h=patch_h,
        grid_w=patch_w,
        temporal_offset=offset,
        reset_spatial_indices=_RESET_SPATIAL_IDS,
        fps=None,  # integer positions (enable_fps_modulation=False)
        temporal_compression_factor=_TCF,
    )
    return torch.cat([text_ids, vision_ids], dim=1)  # [3, seq_len]


def _build_tiny_packed_seq_mrope(
    *,
    n_text: int = 6,
    grid_t: int = 2,
    latent_h: int = 4,
    latent_w: int = 4,
    seed: int = 7,
):
    """Minimal PackedSequence with ``[3, seq]`` mRoPE position ids.

    Vision latent ``[C, grid_t, latent_h, latent_w]`` patchifies (patch=2) to a
    ``(grid_t, latent_h/2, latent_w/2)`` token grid; all frames are noisy.
    """
    from cosmos_framework.data.vfm.sequence_packing import ModalityData, PackedSequence

    patch_h = latent_h // _LATENT_PATCH_SIZE
    patch_w = latent_w // _LATENT_PATCH_SIZE
    n_vision = grid_t * patch_h * patch_w
    total_len = n_text + n_vision

    torch.manual_seed(seed)
    vision_tensor = torch.randn(_LATENT_CHANNEL, grid_t, latent_h, latent_w)
    text_ids = torch.randint(0, 64, (n_text,))
    position_ids = _build_mrope_position_ids(n_text, grid_t, patch_h, patch_w)  # [3, total_len]

    noisy_frame_indexes = torch.arange(grid_t, dtype=torch.long)  # all frames noisy
    vision_mod = ModalityData(
        sequence_indexes=torch.arange(n_text, total_len, dtype=torch.long),
        timesteps=torch.full((n_vision,), 500.0),
        mse_loss_indexes=torch.arange(n_text, total_len, dtype=torch.long),
        token_shapes=[(grid_t, patch_h, patch_w)],
        tokens=[vision_tensor],
        condition_mask=[torch.zeros(grid_t, dtype=torch.long)],  # 0 = noisy
        noisy_frame_indexes=[noisy_frame_indexes],
    )
    packed_seq = PackedSequence(
        sample_lens=[total_len],
        split_lens=[n_text, n_vision],
        attn_modes=["causal", "full"],
        is_image_batch=(grid_t == 1),
        sequence_length=total_len,
        text_ids=text_ids,
        text_indexes=torch.arange(n_text, dtype=torch.long),
        position_ids=position_ids,
        vision=vision_mod,
    )
    return packed_seq


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
# (grid_t, latent_h, latent_w): a single image, a small video, and a taller
# video, to exercise the spatial mRoPE overwrite + gen<->gen full attention.
_GRIDS = [
    pytest.param(1, 8, 8, id="image_1x4x4"),
    pytest.param(2, 4, 4, id="video_2x2x2"),
    pytest.param(3, 8, 4, id="video_3x4x2"),
]


class TestCosmos3DiTParityMRoPE:

    def _run_both(
        self,
        *,
        grid_t: int,
        latent_h: int,
        latent_w: int,
        seed_model: int = 42,
        seed_data: int = 7,
        num_layers: int = 2,
    ):
        vfm = _build_tiny_cosmos3_mrope(seed=seed_model, num_layers=num_layers)
        dit = _build_tiny_fastvideo_dit_mrope(num_layers=num_layers)
        _copy_weights(vfm, dit)
        ps = _build_tiny_packed_seq_mrope(
            n_text=6, grid_t=grid_t, latent_h=latent_h, latent_w=latent_w, seed=seed_data
        )
        with torch.no_grad():
            fw_out = vfm(packed_seq=ps)
            fv_out = dit(**_fastvideo_inputs_from_packed_seq(ps))
        return fw_out, fv_out

    def test_position_ids_are_3xN_mrope(self):
        """The packed mRoPE ids are ``[3, seq_len]`` with the text->vision margin."""
        ps = _build_tiny_packed_seq_mrope(n_text=6, grid_t=2, latent_h=4, latent_w=4)
        pos = ps.position_ids
        assert pos.ndim == 2 and pos.shape[0] == 3, f"expected [3, N], got {tuple(pos.shape)}"
        assert pos.shape[1] == int(ps.sequence_length)
        # Text axis is monotone 0..5 on all 3 rows; vision temporal jumps by the margin.
        assert pos[0, :6].tolist() == [0, 1, 2, 3, 4, 5]
        assert pos[1, :6].tolist() == [0, 1, 2, 3, 4, 5]
        assert pos[2, :6].tolist() == [0, 1, 2, 3, 4, 5]
        # First vision token temporal id == last_text_id (5) + margin + 1.
        assert pos[0, 6].item() == 5 + _TEMPORAL_MODALITY_MARGIN + 1
        # Reset spatial: first vision token H/W ids are 0.
        assert pos[1, 6].item() == 0 and pos[2, 6].item() == 0

    def test_no_additive_latent_pos_embed(self):
        """unified_3d_mrope must NOT build an additive latent position embedding."""
        dit = _build_tiny_fastvideo_dit_mrope()
        assert dit.position_embedding_type == "unified_3d_mrope"
        assert dit.latent_pos_embed is None

    @pytest.mark.parametrize(("grid_t", "latent_h", "latent_w"), _GRIDS)
    def test_preds_vision_parity(self, grid_t, latent_h, latent_w):
        fw_out, fv_out = self._run_both(grid_t=grid_t, latent_h=latent_h, latent_w=latent_w)
        fw_pv = fw_out["preds_vision"][0]  # [1, C, T, H, W]
        fv_pv = fv_out["preds_vision"][0]
        assert fw_pv.shape == fv_pv.shape, f"shape mismatch: fw={fw_pv.shape} fv={fv_pv.shape}"
        max_abs = (fw_pv - fv_pv).abs().max().item()
        print(f"\n[preds_vision mrope {grid_t}x{latent_h}x{latent_w}] max abs diff = {max_abs:.3e}")
        torch.testing.assert_close(fv_pv, fw_pv, atol=1e-4, rtol=1e-3)

    @pytest.mark.parametrize(("grid_t", "latent_h", "latent_w"), _GRIDS)
    def test_last_hidden_state_parity(self, grid_t, latent_h, latent_w):
        fw_out, fv_out = self._run_both(grid_t=grid_t, latent_h=latent_h, latent_w=latent_w)
        fw_lhs = fw_out["last_hidden_state"]  # [N, hidden]
        fv_lhs = fv_out["last_hidden_state"]
        assert fw_lhs.shape == fv_lhs.shape, f"shape mismatch: fw={fw_lhs.shape} fv={fv_lhs.shape}"
        max_abs = (fw_lhs - fv_lhs).abs().max().item()
        print(f"\n[last_hidden_state mrope {grid_t}x{latent_h}x{latent_w}] max abs diff = {max_abs:.3e}")
        torch.testing.assert_close(fv_lhs, fw_lhs, atol=1e-4, rtol=1e-3)

    def test_parity_holds_across_seeds(self):
        """A different random init still matches bit-for-bit (not a fluke)."""
        fw_out, fv_out = self._run_both(
            grid_t=2, latent_h=4, latent_w=4, seed_model=99, seed_data=13
        )
        torch.testing.assert_close(
            fv_out["preds_vision"][0], fw_out["preds_vision"][0], atol=1e-4, rtol=1e-3
        )
        torch.testing.assert_close(
            fv_out["last_hidden_state"], fw_out["last_hidden_state"], atol=1e-4, rtol=1e-3
        )
