# SPDX-License-Identifier: Apache-2.0
"""Numerical-parity test: FastVideo Cosmos3 text reasoning vs the framework.

The omni model's reasoning (VLM text generation) uses ONLY the und (causal)
pathway weights (no ``_moe_gen``) + ``embed_tokens`` / ``norm`` / ``lm_head``;
the generation pathway and all multimodal embedders are bypassed. FastVideo's
native ``Cosmos3VFMTransformer`` already contains exactly those (the und branch
of the dual-pathway forward + ``lm_head``), so a text-only forward + ``lm_head``
reproduces the framework reasoner.

This pins:
  * **prefill logits** — native text-only forward + ``lm_head`` vs the framework
    ``language_model.model.reasoner_forward`` + ``lm_head`` (per-position); and
  * **greedy generation** — ``cosmos3_generate_reasoner_text`` vs the framework
    ``generate_reasoner_text(do_sample=False)`` (token-for-token).

Framework is the parity ORACLE (CPU/float32 via SDPA monkey-patch). Text-only
(image-conditioned reasoning additionally needs the Qwen3-VL ``vision_encoder``,
tracked separately).

Run:
    cd <worktree> && <fv-cosmos3 python> -m pytest \
        tests/local_tests/cosmos3/test_cosmos3_reasoning_parity.py -q -s
"""
from __future__ import annotations

import pytest
import torch

cosmos_framework = pytest.importorskip(
    "cosmos_framework",
    reason="cosmos_framework not installed; run in fv-cosmos3 env.",
)

from fastvideo.pipelines.basic.cosmos3.cosmos3_pipeline import (  # noqa: E402
    cosmos3_generate_reasoner_text,
)

from .test_cosmos3_dit_parity import _copy_weights  # noqa: E402
from .test_cosmos3_dit_parity_mrope import (  # noqa: E402
    _build_tiny_cosmos3_mrope,
    _build_tiny_fastvideo_dit_mrope,
)
from .test_cosmos3_reference_forward import _apply_sdpa_patches  # noqa: E402

pytestmark = [pytest.mark.local]
_apply_sdpa_patches()


def _framework_prefill_logits(vfm, input_ids: torch.Tensor) -> torch.Tensor:
    """Framework reasoner text-only prefill logits ``[1, T, vocab]`` (the oracle).

    Mirrors ``_impl_generate_reasoner_text`` prefill: ``model.reasoner_forward``
    then ``lm_head`` (here over ALL positions, not just the last)."""
    from cosmos_framework.model.vfm.mot.unified_mot import ReasonerKVCache

    causal_lm = vfm.language_model
    model = causal_lm.model
    cache = ReasonerKVCache.empty(num_layers=len(model.layers))
    hidden = model.reasoner_forward(input_ids.unsqueeze(0), cache=cache)  # [1,T,hidden]
    return causal_lm.lm_head(hidden)  # [1,T,vocab]


def _native_prefill_logits(dit, input_ids: torch.Tensor) -> torch.Tensor:
    """Native text-only forward + ``lm_head`` -> ``[T, vocab]``."""
    n = input_ids.numel()
    pos = torch.arange(n).unsqueeze(0).expand(3, -1).contiguous()
    out = dit(
        text_ids=input_ids.to(torch.long),
        text_indexes=torch.arange(n),
        position_ids=pos,
        sequence_length=n,
        split_lens=[n],
        attn_modes=["causal"],
        vision_tokens=[],
        vision_token_shapes=[],
        vision_sequence_indexes=torch.empty(0, dtype=torch.long),
        vision_timesteps=torch.empty(0),
        vision_mse_loss_indexes=torch.empty(0, dtype=torch.long),
        vision_noisy_frame_indexes=[],
    )
    return dit.lm_head(out["last_hidden_state"])  # [T, vocab]


def _diffs(a, b):
    d = (a - b).abs()
    return d.max().item(), d.mean().item()


class TestCosmos3ReasoningParity:

    def _build(self, seed_model=42, num_layers=2):
        vfm = _build_tiny_cosmos3_mrope(seed=seed_model, num_layers=num_layers)
        dit = _build_tiny_fastvideo_dit_mrope(num_layers=num_layers)
        _copy_weights(vfm, dit)
        return vfm, dit

    @pytest.mark.parametrize("n_text", [4, 8, 12])
    def test_reasoner_prefill_logits_match_framework(self, n_text):
        vfm, dit = self._build()
        torch.manual_seed(n_text)
        input_ids = torch.randint(0, 60, (n_text,))
        with torch.no_grad():
            fw = _framework_prefill_logits(vfm, input_ids)[0]  # [T, vocab]
            fv = _native_prefill_logits(dit, input_ids)  # [T, vocab]
        assert fw.shape == fv.shape, f"shape fw={fw.shape} fv={fv.shape}"
        mx, mn = _diffs(fv, fw)
        print(f"\n[reasoner_prefill n={n_text}] logits max abs diff = {mx:.3e} mean abs diff = {mn:.3e}")
        torch.testing.assert_close(fv, fw, atol=1e-4, rtol=1e-3)
        # The greedy decisions (argmax per position) must agree exactly.
        assert torch.equal(fv.argmax(-1), fw.argmax(-1))

    @pytest.mark.parametrize("seed", [0, 1, 2, 3])
    def test_greedy_reasoning_matches_framework(self, seed):
        vfm, dit = self._build()
        torch.manual_seed(seed)
        input_ids = torch.randint(0, 60, (6,))
        fw = vfm.generate_reasoner_text(
            input_ids.unsqueeze(0), max_new_tokens=8, do_sample=False, return_only_new_tokens=True,
        )[0].tolist()
        fv = cosmos3_generate_reasoner_text(dit, input_ids.tolist(), max_new_tokens=8)
        print(f"\n[greedy_reason seed={seed}] framework={fw} native={fv}")
        assert fv == fw, f"token mismatch: native={fv} framework={fw}"

    def test_deepstack_reasoner_forward_matches_framework(self):
        """The deepstack reasoner backbone (image-conditioned reasoning's one new
        native piece) matches the framework ``reasoner_forward`` given identical
        prefill inputs (inputs_embeds + positions + per-layer deepstack embeds +
        visual mask)."""
        from cosmos_framework.model.vfm.mot.unified_mot import ReasonerKVCache

        vfm, dit = self._build()
        hidden = dit.hidden_size
        n_layers = dit.num_hidden_layers
        seq = 10
        torch.manual_seed(5)
        inputs_embeds = torch.randn(seq, hidden)
        position_ids = torch.arange(seq).unsqueeze(0).expand(3, -1).contiguous()  # [3, seq]
        visual_pos_mask = torch.zeros(seq, dtype=torch.bool)
        visual_pos_mask[2:6] = True  # 4 "image" tokens
        k = int(visual_pos_mask.sum())
        deepstack = [torch.randn(k, hidden) for _ in range(n_layers)]  # one per layer

        model = vfm.language_model.model
        cache = ReasonerKVCache.empty(num_layers=len(model.layers))
        with torch.no_grad():
            hid_fw = model.reasoner_forward(
                input_ids=None,
                inputs_embeds=inputs_embeds.unsqueeze(0),
                position_ids=position_ids.unsqueeze(1),  # [3, B=1, seq]
                visual_pos_masks=visual_pos_mask.unsqueeze(0),
                deepstack_visual_embeds=deepstack,
                cache=cache,
            )[0]  # [seq, hidden]
            hid_fv = dit.reason_forward(inputs_embeds, position_ids, deepstack, visual_pos_mask)  # [seq, hidden]
        assert hid_fw.shape == hid_fv.shape, f"shape fw={hid_fw.shape} fv={hid_fv.shape}"
        mx, mn = _diffs(hid_fv, hid_fw)
        print(f"\n[deepstack_reasoner] hidden max abs diff = {mx:.3e} mean abs diff = {mn:.3e}")
        torch.testing.assert_close(hid_fv, hid_fw, atol=1e-4, rtol=1e-3)
