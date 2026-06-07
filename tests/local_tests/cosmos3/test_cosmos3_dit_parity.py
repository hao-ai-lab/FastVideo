# SPDX-License-Identifier: Apache-2.0
"""Numerical-parity test: FastVideo Cosmos3 DiT vs official ``Cosmos3VFMNetwork``.

Builds a tiny official-framework ``Cosmos3VFMNetwork`` AND a tiny FastVideo
``Cosmos3VFMTransformer`` from the SAME tiny config, copies the framework
weights into the FastVideo DiT via an explicit framework->fastvideo name map,
runs BOTH forwards on identical deterministic inputs (CPU / float32), and
asserts ``torch.allclose`` on the vision prediction output (``preds_vision``)
and the per-token ``last_hidden_state``.

The official model is the parity ORACLE. It runs on CPU / float32 via the SDPA
monkey-patch in ``test_cosmos3_reference_forward`` (flash2/flash3/natten are
CUDA-only). The FastVideo DiT runs natively on CPU with plain SDPA.

Run:
    cd <worktree> && <fv-cosmos3 python> -m pytest \
        tests/local_tests/cosmos3/test_cosmos3_dit_parity.py -q
"""
from __future__ import annotations

import pytest
import torch

# The official framework provides the parity oracle.
cosmos_framework = pytest.importorskip(
    "cosmos_framework",
    reason="cosmos_framework not installed; run in fv-cosmos3 env.",
)

# Reuse the reference harness's tiny-model builder + SDPA monkey-patch.
from .test_cosmos3_reference_forward import (  # noqa: E402
    _apply_sdpa_patches,
    _build_tiny_cosmos3,
    _build_tiny_packed_seq,
)

pytestmark = [pytest.mark.local]

# Ensure the CPU/float32 SDPA patches are installed (idempotent).
_apply_sdpa_patches()


# ---------------------------------------------------------------------------
# Tiny config shared by both models (must match _build_tiny_cosmos3).
# ---------------------------------------------------------------------------
def _build_tiny_fastvideo_dit() -> "Cosmos3VFMTransformer":  # noqa: F821
    from fastvideo.configs.models.dits.cosmos3 import (
        Cosmos3ArchConfig,
        Cosmos3VideoConfig,
    )
    from fastvideo.models.dits.cosmos3 import Cosmos3VFMTransformer

    arch = Cosmos3ArchConfig(
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
        intermediate_size=32,
        vocab_size=64,
        rms_norm_eps=1e-6,
        attention_bias=False,
        latent_patch_size=2,
        latent_channel=16,
        rope_theta=5_000_000.0,
        mrope_section=[24, 20, 20],
        position_embedding_type="3d_rope",
        base_fps=24.0,
        temporal_compression_factor=4,
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
# Framework -> FastVideo weight name map.
# ---------------------------------------------------------------------------
def _framework_to_fastvideo_state_dict(vfm, num_layers: int) -> dict[str, torch.Tensor]:
    """Translate framework param names into the FastVideo DiT param names.

    Framework (Cosmos3VFMNetwork):
      language_model.model.{embed_tokens,norm,norm_moe_gen}
      language_model.lm_head
      language_model.model.layers.{i}.self_attn.{q,k,v,o}_proj(+ _moe_gen)
      language_model.model.layers.{i}.self_attn.{q,k}_norm(+ _moe_gen)
      language_model.model.layers.{i}.{mlp,mlp_moe_gen}.{gate,up,down}_proj
      language_model.model.layers.{i}.{input,post_attention}_layernorm(+ _moe_gen)
      vae2llm / llm2vae / time_embedder.mlp.{0,2}

    FastVideo (Cosmos3VFMTransformer):
      embed_tokens / norm / norm_moe_gen / lm_head
      layers.{i}.self_attn.{to_q,to_k,to_v,to_out} (und)
      layers.{i}.self_attn.{add_q,add_k,add_v}_proj / to_add_out (gen)
      layers.{i}.self_attn.{norm_q,norm_k,norm_added_q,norm_added_k}
      layers.{i}.{mlp,mlp_moe_gen}.{gate,up,down}_proj
      layers.{i}.{input,post_attention}_layernorm(+ _moe_gen)
      proj_in / proj_out / time_embedder.linear_{1,2}
    """
    src = dict(vfm.named_parameters())
    out: dict[str, torch.Tensor] = {}

    def take(name: str) -> torch.Tensor:
        return src[name].detach().clone()

    # ---- Top-level backbone ----
    out["embed_tokens.weight"] = take("language_model.model.embed_tokens.weight")
    out["norm.weight"] = take("language_model.model.norm.weight")
    out["norm_moe_gen.weight"] = take("language_model.model.norm_moe_gen.weight")
    out["lm_head.weight"] = take("language_model.lm_head.weight")

    # ---- Vision adapters ----
    out["proj_in.weight"] = take("vae2llm.weight")
    out["proj_in.bias"] = take("vae2llm.bias")
    out["proj_out.weight"] = take("llm2vae.weight")
    out["proj_out.bias"] = take("llm2vae.bias")

    # ---- Timestep embedder (mlp.0/mlp.2 -> linear_1/linear_2) ----
    out["time_embedder.linear_1.weight"] = take("time_embedder.mlp.0.weight")
    out["time_embedder.linear_1.bias"] = take("time_embedder.mlp.0.bias")
    out["time_embedder.linear_2.weight"] = take("time_embedder.mlp.2.weight")
    out["time_embedder.linear_2.bias"] = take("time_embedder.mlp.2.bias")

    # ---- Per layer ----
    und_attn = {"q_proj": "to_q", "k_proj": "to_k", "v_proj": "to_v", "o_proj": "to_out"}
    gen_attn = {
        "q_proj_moe_gen": "add_q_proj",
        "k_proj_moe_gen": "add_k_proj",
        "v_proj_moe_gen": "add_v_proj",
        "o_proj_moe_gen": "to_add_out",
    }
    und_norm = {"q_norm": "norm_q", "k_norm": "norm_k"}
    gen_norm = {"q_norm_moe_gen": "norm_added_q", "k_norm_moe_gen": "norm_added_k"}

    for i in range(num_layers):
        fw = f"language_model.model.layers.{i}"
        fv = f"layers.{i}"
        for s, d in und_attn.items():
            out[f"{fv}.self_attn.{d}.weight"] = take(f"{fw}.self_attn.{s}.weight")
        for s, d in gen_attn.items():
            out[f"{fv}.self_attn.{d}.weight"] = take(f"{fw}.self_attn.{s}.weight")
        for s, d in und_norm.items():
            out[f"{fv}.self_attn.{d}.weight"] = take(f"{fw}.self_attn.{s}.weight")
        for s, d in gen_norm.items():
            out[f"{fv}.self_attn.{d}.weight"] = take(f"{fw}.self_attn.{s}.weight")
        for mlp in ("mlp", "mlp_moe_gen"):
            for proj in ("gate_proj", "up_proj", "down_proj"):
                out[f"{fv}.{mlp}.{proj}.weight"] = take(f"{fw}.{mlp}.{proj}.weight")
        for ln in ("input_layernorm", "input_layernorm_moe_gen", "post_attention_layernorm",
                   "post_attention_layernorm_moe_gen"):
            out[f"{fv}.{ln}.weight"] = take(f"{fw}.{ln}.weight")

    return out


def _copy_weights(vfm, dit) -> None:
    """Copy framework weights into the FastVideo DiT (shape-checked)."""
    mapped = _framework_to_fastvideo_state_dict(vfm, num_layers=dit.num_hidden_layers)
    dst = dict(dit.named_parameters())
    # Every mapped tensor must land on an existing FastVideo param with a matching shape.
    for name, tensor in mapped.items():
        assert name in dst, f"FastVideo DiT missing param for mapped key {name!r}"
        assert dst[name].shape == tensor.shape, (f"shape mismatch for {name}: "
                                                 f"dit={tuple(dst[name].shape)} fw={tuple(tensor.shape)}")
    with torch.no_grad():
        for name, tensor in mapped.items():
            dst[name].copy_(tensor.to(dst[name].dtype))


def _fastvideo_inputs_from_packed_seq(ps) -> dict:
    """Build the FastVideo DiT forward kwargs from a framework PackedSequence."""
    v = ps.vision
    return dict(
        text_ids=ps.text_ids,
        text_indexes=ps.text_indexes,
        position_ids=ps.position_ids,
        sequence_length=int(ps.sequence_length),
        split_lens=list(ps.split_lens),
        attn_modes=list(ps.attn_modes),
        vision_tokens=list(v.tokens),
        vision_token_shapes=list(v.token_shapes),
        vision_sequence_indexes=v.sequence_indexes,
        vision_timesteps=v.timesteps,
        vision_mse_loss_indexes=v.mse_loss_indexes,
        vision_noisy_frame_indexes=list(v.noisy_frame_indexes),
        fps_vision=None,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestCosmos3DiTParity:

    def _run_both(self, seed_model: int = 42, seed_data: int = 7):
        vfm = _build_tiny_cosmos3(seed=seed_model)
        dit = _build_tiny_fastvideo_dit()
        _copy_weights(vfm, dit)
        ps = _build_tiny_packed_seq(n_text=4, seed=seed_data)

        with torch.no_grad():
            fw_out = vfm(packed_seq=ps)
            fv_out = dit(**_fastvideo_inputs_from_packed_seq(ps))
        return fw_out, fv_out

    def test_weight_map_is_complete(self):
        """The framework->fastvideo map must cover EVERY FastVideo DiT parameter
        that is exercised by the video path (i.e. all non-dormant params).

        Dormant action/audio heads have no framework counterpart in this tiny
        vision-only setup, so they are excluded from the copy; everything else
        must be covered.
        """
        vfm = _build_tiny_cosmos3(seed=42)
        dit = _build_tiny_fastvideo_dit()
        mapped = set(_framework_to_fastvideo_state_dict(vfm, num_layers=dit.num_hidden_layers))
        dit_params = set(n for n, _ in dit.named_parameters())
        dormant = {
            n
            for n in dit_params
            if n.startswith(("action_", "audio_"))
        }
        uncovered = dit_params - mapped - dormant
        assert not uncovered, f"FastVideo DiT params not covered by weight map: {sorted(uncovered)}"

    def test_preds_vision_parity(self):
        fw_out, fv_out = self._run_both()
        fw_pv = fw_out["preds_vision"][0]  # [1, C, T, H, W]
        fv_pv = fv_out["preds_vision"][0]
        assert fw_pv.shape == fv_pv.shape, f"shape mismatch: fw={fw_pv.shape} fv={fv_pv.shape}"
        max_abs = (fw_pv - fv_pv).abs().max().item()
        print(f"\n[preds_vision] max abs diff = {max_abs:.3e}")
        torch.testing.assert_close(fv_pv, fw_pv, atol=1e-4, rtol=1e-3)

    def test_last_hidden_state_parity(self):
        fw_out, fv_out = self._run_both()
        fw_lhs = fw_out["last_hidden_state"]  # [N, hidden]
        fv_lhs = fv_out["last_hidden_state"]
        assert fw_lhs.shape == fv_lhs.shape, f"shape mismatch: fw={fw_lhs.shape} fv={fv_lhs.shape}"
        max_abs = (fw_lhs - fv_lhs).abs().max().item()
        print(f"\n[last_hidden_state] max abs diff = {max_abs:.3e}")
        torch.testing.assert_close(fv_lhs, fw_lhs, atol=1e-4, rtol=1e-3)

    def test_parity_holds_across_seeds(self):
        """Re-running with a different random init still matches (not a fluke)."""
        fw_out, fv_out = self._run_both(seed_model=99, seed_data=13)
        torch.testing.assert_close(fv_out["preds_vision"][0], fw_out["preds_vision"][0], atol=1e-4, rtol=1e-3)
