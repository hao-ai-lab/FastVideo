# SPDX-License-Identifier: Apache-2.0
"""
Numerical-parity reference test for the official Cosmos3 DiT (Cosmos3VFMNetwork).

Runs a tiny deterministic forward of the OFFICIAL framework model on CPU / float32
using a torch SDPA monkey-patch (flash2/flash3/natten are CUDA-only; SDPA works on CPU).
The test exercises the full forward contract:
  packed_seq -> vfm(packed_seq) -> {last_hidden_state, preds_vision}
and is used as the "ground truth" side of any FastVideo parity check.

Environment requirements
------------------------
- cosmos_framework must be installed (editable) in the active interpreter.
  The canonical env is: /home/william5lin/miniconda3/envs/fv-cosmos3/bin/python
- No transformer_engine / natten / GPU required.
- PYTHONSAFEPATH=1 is recommended to avoid cwd import shadowing.

Run:
    PYTHONSAFEPATH=1 pytest fastvideo/tests/layers/test_cosmos3_reference_forward.py -v
"""

import math
import sys

import pytest
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Skip guard: cosmos_framework may not be installed in the default dev env.
# ---------------------------------------------------------------------------
cosmos_framework = pytest.importorskip(
    "cosmos_framework",
    reason="cosmos_framework not installed; run in fv-cosmos3 env.",
)

pytestmark = [pytest.mark.local]

# ---------------------------------------------------------------------------
# SDPA attention monkey-patch
# ---------------------------------------------------------------------------
# The imaginaire attention backend (flash2/flash3) requires CUDA *and*
# float16/bfloat16. For CPU/float32 parity testing we replace it with a
# simple SDPA implementation that handles both standard and varlen packed
# formats (cumulative_seqlen_{Q,KV}).
# ---------------------------------------------------------------------------


def _sdpa_attention(
    query,
    key,
    value,
    is_causal=False,
    causal_type=None,
    scale=None,
    seqlens_Q=None,
    seqlens_KV=None,
    cumulative_seqlen_Q=None,
    cumulative_seqlen_KV=None,
    max_seqlen_Q=None,
    max_seqlen_KV=None,
    backend=None,
    return_lse=False,
    backend_kwargs=None,
    deterministic=False,
):
    """Minimal SDPA wrapper that mirrors the imaginaire attention signature."""
    B, Sq, H, D = query.shape
    Hkv = key.shape[2]
    attn_scale = scale if scale is not None else D**-0.5

    if cumulative_seqlen_Q is not None:
        # Varlen packed layout: B==1, tokens from different samples are concatenated.
        oq = cumulative_seqlen_Q.cpu().tolist()
        okv = cumulative_seqlen_KV.cpu().tolist()
        outs = []
        for i in range(len(oq) - 1):
            qi = query[0, oq[i] : oq[i + 1]].unsqueeze(0).permute(0, 2, 1, 3)  # [1,H,S,D]
            ki = key[0, okv[i] : okv[i + 1]].unsqueeze(0).permute(0, 2, 1, 3)
            vi = value[0, okv[i] : okv[i + 1]].unsqueeze(0).permute(0, 2, 1, 3)
            if Hkv != H:
                ki = ki.repeat_interleave(H // Hkv, dim=1)
                vi = vi.repeat_interleave(H // Hkv, dim=1)
            oi = F.scaled_dot_product_attention(qi, ki, vi, scale=attn_scale, is_causal=is_causal)
            outs.append(oi.permute(0, 2, 1, 3))  # [1,S,H,D]
        out = torch.cat(outs, dim=1)  # [1,S_total,H,D]
    else:
        q = query.permute(0, 2, 1, 3)
        k = key.permute(0, 2, 1, 3)
        v = value.permute(0, 2, 1, 3)
        if Hkv != H:
            k = k.repeat_interleave(H // Hkv, dim=1)
            v = v.repeat_interleave(H // Hkv, dim=1)
        out = F.scaled_dot_product_attention(q, k, v, scale=attn_scale, is_causal=is_causal)
        out = out.permute(0, 2, 1, 3)  # [B,S,H,D]

    if return_lse:
        lse = torch.zeros(B, Sq, H, 1, dtype=query.dtype, device=query.device)
        return out, lse
    return out


def _sdpa_merge_attentions(outputs, lse_tensors, torch_compile=False):
    """Log-sum-exp weighted merge of two attention outputs."""
    if len(outputs) == 1:
        return outputs[0], lse_tensors[0]
    o1, lse1 = outputs[0], lse_tensors[0]
    o2, lse2 = outputs[1], lse_tensors[1]
    m = torch.maximum(lse1, lse2)
    w1 = torch.exp(lse1 - m)
    w2 = torch.exp(lse2 - m)
    ws = w1 + w2
    return (o1 * w1 + o2 * w2) / ws, m + torch.log(ws)


def _apply_sdpa_patches():
    """Monkey-patch every attention reference in cosmos_framework to use SDPA."""
    import cosmos_framework.model.attention as attn_pkg
    import cosmos_framework.model.attention.frontend as attn_frontend
    import cosmos_framework.model.vfm.mot.attention as vfm_attn
    import cosmos_framework.model.vfm.mot.unified_mot as mot_module

    attn_frontend.attention = _sdpa_attention
    attn_pkg.attention = _sdpa_attention
    mot_module.imaginaire_attention = _sdpa_attention
    vfm_attn.attention = _sdpa_attention

    attn_frontend.merge_attentions = _sdpa_merge_attentions
    attn_pkg.merge_attentions = _sdpa_merge_attentions
    vfm_attn.merge_attentions = _sdpa_merge_attentions


# Apply patches at import time (before any model is constructed).
_apply_sdpa_patches()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_tiny_cosmos3(seed: int = 42):
    """Construct a tiny Cosmos3VFMNetwork on CPU / float32.

    Architecture:
        hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=2, head_dim=8,
        vocab_size=64, latent_channel_size=16, latent_patch_size=2,
        max_latent_{h,w,t}=8,8,4
    """
    from cosmos_framework.model.vfm.mot.cosmos3_vfm_network import (
        Cosmos3VFMNetwork,
        Cosmos3VFMNetworkConfig,
    )
    from cosmos_framework.model.vfm.mot.unified_mot import Qwen3MoTConfig, Qwen3VLTextForCausalLM
    from cosmos_framework.model.vfm.vlm.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig

    TINY_TEXT_DICT = dict(
        model_type="qwen3_vl_text",
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
        rms_norm_eps=1e-6,
        attention_bias=False,
        attention_dropout=0.0,
    )
    mot_cfg = Qwen3MoTConfig(
        config_dict=TINY_TEXT_DICT,
        qk_norm_for_text=True,
        qk_norm_for_diffusion=True,
        include_visual=False,
    )
    tiny_vlm_cfg = Qwen3VLConfig(text_config=TINY_TEXT_DICT)
    vfm_cfg = Cosmos3VFMNetworkConfig(
        vision_gen=True,
        vlm_config=tiny_vlm_cfg,
        latent_patch_size=2,
        latent_downsample_factor=8,
        latent_channel_size=16,
        position_embedding_type="3d_rope",
        max_latent_h=8,
        max_latent_w=8,
        max_latent_t=4,
    )
    torch.manual_seed(seed)
    lm = Qwen3VLTextForCausalLM(config=mot_cfg)
    vfm = Cosmos3VFMNetwork(language_model=lm, config=vfm_cfg)
    vfm.eval()
    return vfm


def _build_tiny_packed_seq(*, n_text: int = 4, seed: int = 7):
    """Build a minimal PackedSequence: 4 text tokens + 1 vision patch.

    Vision: C=16, T=1, H=2, W=2 → after patch_size=2: 1*1*1 = 1 patch.
    All vision frames are noisy (timestep=500).
    """
    from cosmos_framework.data.vfm.sequence_packing import ModalityData, PackedSequence

    torch.manual_seed(seed)
    vision_tensor = torch.randn(16, 1, 2, 2)  # [C=16, T=1, H=2, W=2]
    text_ids = torch.randint(0, 64, (n_text,))
    n_vision = 1  # 1 patch after patchify
    total_len = n_text + n_vision

    vision_mod = ModalityData(
        sequence_indexes=torch.arange(n_text, total_len, dtype=torch.long),
        timesteps=torch.tensor([500.0]),  # one noisy frame
        mse_loss_indexes=torch.arange(n_text, total_len, dtype=torch.long),
        token_shapes=[(1, 1, 1)],  # (t_patches, h_patches, w_patches) = (1,1,1)
        tokens=[vision_tensor],
        condition_mask=[torch.zeros(1, dtype=torch.long)],  # 0=noisy
        noisy_frame_indexes=[torch.tensor([0])],
    )
    packed_seq = PackedSequence(
        sample_lens=[total_len],
        split_lens=[n_text, n_vision],
        attn_modes=["causal", "full"],
        is_image_batch=True,
        sequence_length=total_len,
        text_ids=text_ids,
        text_indexes=torch.arange(n_text, dtype=torch.long),
        position_ids=torch.arange(total_len),
        vision=vision_mod,
    )
    return packed_seq


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCosmos3ReferenceConfig:
    """Step 1: verify config construction and field enumeration."""

    def test_qwen3vl_text_config_fields(self):
        from cosmos_framework.model.vfm.vlm.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig

        cfg = Qwen3VLTextConfig(
            vocab_size=64,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=8,
        )
        assert cfg.vocab_size == 64
        assert cfg.hidden_size == 16
        assert cfg.num_attention_heads == 2
        assert cfg.num_key_value_heads == 2
        assert cfg.head_dim == 8

    def test_cosmos3_vfm_network_config(self):
        from cosmos_framework.model.vfm.mot.cosmos3_vfm_network import Cosmos3VFMNetworkConfig
        from cosmos_framework.model.vfm.vlm.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig

        tiny_vlm = Qwen3VLConfig(text_config=dict(vocab_size=64, hidden_size=16))
        cfg = Cosmos3VFMNetworkConfig(
            vision_gen=True,
            vlm_config=tiny_vlm,
            latent_patch_size=2,
            latent_downsample_factor=8,
            latent_channel_size=16,
            position_embedding_type="3d_rope",
            max_latent_h=8,
            max_latent_w=8,
            max_latent_t=4,
        )
        assert cfg.vision_gen is True
        assert cfg.latent_patch_size == 2
        assert cfg.latent_channel_size == 16


class TestCosmos3ReferenceInstantiation:
    """Step 2: verify parameter key pattern and module tree."""

    @pytest.fixture(scope="class")
    def tiny_vfm(self):
        return _build_tiny_cosmos3(seed=42)

    def test_instantiation_succeeds(self, tiny_vfm):
        assert tiny_vfm is not None

    def test_param_key_pattern_attention(self, tiny_vfm):
        """Verify understanding and generation attention projections exist."""
        param_keys = {n for n, _ in tiny_vfm.named_parameters()}
        layer = "language_model.model.layers.0.self_attn"
        # Understanding pathway
        for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
            assert f"{layer}.{proj}.weight" in param_keys, f"Missing {layer}.{proj}.weight"
        # Generation pathway (moe_gen suffix)
        for proj in ("q_proj_moe_gen", "k_proj_moe_gen", "v_proj_moe_gen", "o_proj_moe_gen"):
            assert f"{layer}.{proj}.weight" in param_keys, f"Missing {layer}.{proj}.weight"
        # QK norms
        for norm in ("q_norm", "k_norm", "q_norm_moe_gen", "k_norm_moe_gen"):
            assert f"{layer}.{norm}.weight" in param_keys, f"Missing {layer}.{norm}.weight"

    def test_param_key_pattern_mlp(self, tiny_vfm):
        param_keys = {n for n, _ in tiny_vfm.named_parameters()}
        for pathway in ("mlp", "mlp_moe_gen"):
            base = f"language_model.model.layers.0.{pathway}"
            for proj in ("gate_proj", "up_proj", "down_proj"):
                assert f"{base}.{proj}.weight" in param_keys

    def test_param_key_pattern_layernorms(self, tiny_vfm):
        param_keys = {n for n, _ in tiny_vfm.named_parameters()}
        layer = "language_model.model.layers.0"
        for ln in (
            "input_layernorm",
            "input_layernorm_moe_gen",
            "post_attention_layernorm",
            "post_attention_layernorm_moe_gen",
        ):
            assert f"{layer}.{ln}.weight" in param_keys

    def test_param_key_pattern_toplevel(self, tiny_vfm):
        param_keys = {n for n, _ in tiny_vfm.named_parameters()}
        # LM submodules
        assert "language_model.model.embed_tokens.weight" in param_keys
        assert "language_model.model.norm.weight" in param_keys
        assert "language_model.model.norm_moe_gen.weight" in param_keys
        assert "language_model.lm_head.weight" in param_keys
        # VFM vision head
        assert "vae2llm.weight" in param_keys
        assert "vae2llm.bias" in param_keys
        assert "llm2vae.weight" in param_keys
        assert "llm2vae.bias" in param_keys
        # Timestep embedder
        assert "time_embedder.mlp.0.weight" in param_keys
        assert "time_embedder.mlp.2.weight" in param_keys

    def test_expected_param_count(self, tiny_vfm):
        """Sanity-check total param count for the tiny model."""
        n_params = sum(p.numel() for p in tiny_vfm.parameters())
        # Rough bound: tiny model should be under 50 K params
        assert n_params < 50_000, f"Unexpected param count: {n_params}"

    def test_dtype_is_float32(self, tiny_vfm):
        for name, p in tiny_vfm.named_parameters():
            if "inv_freq" in name:
                continue  # inv_freq stays float32 always
            assert p.dtype == torch.float32, f"{name} has dtype {p.dtype}"


class TestCosmos3ReferenceForward:
    """Step 3 + 4: verify forward contract and determinism."""

    @pytest.fixture(scope="class")
    def tiny_vfm(self):
        return _build_tiny_cosmos3(seed=42)

    @pytest.fixture(scope="class")
    def packed_seq(self):
        return _build_tiny_packed_seq(n_text=4, seed=7)

    @pytest.fixture(scope="class")
    def fwd_output(self, tiny_vfm, packed_seq):
        with torch.no_grad():
            return tiny_vfm(packed_seq=packed_seq)

    def test_forward_returns_dict(self, fwd_output):
        assert isinstance(fwd_output, dict)

    def test_last_hidden_state_shape(self, fwd_output):
        # 4 text + 1 vision patch = 5 total tokens; hidden_size=16
        lhs = fwd_output["last_hidden_state"]
        assert lhs.shape == torch.Size([5, 16]), f"Got {lhs.shape}"

    def test_last_hidden_state_finite(self, fwd_output):
        assert torch.isfinite(fwd_output["last_hidden_state"]).all()

    def test_preds_vision_present(self, fwd_output):
        assert "preds_vision" in fwd_output

    def test_preds_vision_shape(self, fwd_output):
        # latent_channel=16, T=1, H=2, W=2 → [1, 16, 1, 2, 2]
        pv = fwd_output["preds_vision"][0]
        assert pv.shape == torch.Size([1, 16, 1, 2, 2]), f"Got {pv.shape}"

    def test_preds_vision_finite(self, fwd_output):
        assert torch.isfinite(fwd_output["preds_vision"][0]).all()

    def test_forward_deterministic_same_seed(self):
        """Two models with identical seed should produce identical output."""
        ps = _build_tiny_packed_seq(n_text=4, seed=7)
        vfm1 = _build_tiny_cosmos3(seed=42)
        vfm2 = _build_tiny_cosmos3(seed=42)
        with torch.no_grad():
            out1 = vfm1(packed_seq=ps)
            out2 = vfm2(packed_seq=ps)
        assert torch.allclose(out1["last_hidden_state"], out2["last_hidden_state"])
        assert torch.allclose(out1["preds_vision"][0], out2["preds_vision"][0])

    def test_forward_repeatable_same_model(self, tiny_vfm, packed_seq):
        """Same model, same input → identical output on two calls."""
        with torch.no_grad():
            out1 = tiny_vfm(packed_seq=packed_seq)
            out2 = tiny_vfm(packed_seq=packed_seq)
        assert torch.allclose(out1["last_hidden_state"], out2["last_hidden_state"])

    def test_different_seed_gives_different_output(self):
        """Different model seeds should give different outputs."""
        ps = _build_tiny_packed_seq(n_text=4, seed=7)
        vfm1 = _build_tiny_cosmos3(seed=42)
        vfm2 = _build_tiny_cosmos3(seed=99)
        with torch.no_grad():
            out1 = vfm1(packed_seq=ps)
            out2 = vfm2(packed_seq=ps)
        # With very high probability random init → different outputs
        assert not torch.allclose(out1["last_hidden_state"], out2["last_hidden_state"])

    def test_float32_dtype_preserved(self, fwd_output):
        assert fwd_output["last_hidden_state"].dtype == torch.float32

    def test_cpu_device(self, fwd_output):
        assert fwd_output["last_hidden_state"].device.type == "cpu"


class TestCosmos3ReasonerForward:
    """Optional: reasoner (und-only) pathway via standard [B,T,H] layout."""

    def test_reasoner_forward_shape(self):
        """reasoner_forward runs the und tower only; no PackedSequence needed."""
        vfm = _build_tiny_cosmos3(seed=42)
        lm = vfm.language_model
        input_ids = torch.randint(0, 64, (1, 6))
        with torch.no_grad():
            out = lm.model.reasoner_forward(input_ids=input_ids, cache=None)
        # [B=1, T=6, hidden_size=16]
        assert out.shape == torch.Size([1, 6, 16])

    def test_reasoner_forward_finite(self):
        vfm = _build_tiny_cosmos3(seed=42)
        lm = vfm.language_model
        input_ids = torch.randint(0, 64, (1, 6))
        with torch.no_grad():
            out = lm.model.reasoner_forward(input_ids=input_ids, cache=None)
        assert torch.isfinite(out).all()

    def test_reasoner_forward_deterministic(self):
        vfm = _build_tiny_cosmos3(seed=42)
        lm = vfm.language_model
        input_ids = torch.randint(0, 64, (1, 6))
        with torch.no_grad():
            out1 = lm.model.reasoner_forward(input_ids=input_ids, cache=None)
            out2 = lm.model.reasoner_forward(input_ids=input_ids, cache=None)
        assert torch.allclose(out1, out2)


# ---------------------------------------------------------------------------
# Convenience: print a concise summary when run directly.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Building tiny Cosmos3VFMNetwork...")
    vfm = _build_tiny_cosmos3(seed=42)

    print("\nParameter keys:")
    for name, p in sorted(vfm.named_parameters()):
        print(f"  {name}: {tuple(p.shape)}")

    print("\nRunning forward...")
    ps = _build_tiny_packed_seq(n_text=4, seed=7)
    with torch.no_grad():
        out = vfm(packed_seq=ps)

    lhs = out["last_hidden_state"]
    pv = out["preds_vision"][0]
    print(f"\nlast_hidden_state: {lhs.shape}, finite={torch.isfinite(lhs).all().item()}, mean={lhs.mean():.4f}")
    print(f"preds_vision[0]:  {pv.shape},  finite={torch.isfinite(pv).all().item()}, mean={pv.mean():.4f}")

    # Determinism
    with torch.no_grad():
        out2 = vfm(packed_seq=ps)
    print(f"\nDeterministic repeat: {torch.allclose(lhs, out2['last_hidden_state'])}")
