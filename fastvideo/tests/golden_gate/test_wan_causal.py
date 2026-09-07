# SPDX-License-Identifier: Apache-2.0
"""Real Wan block weights through causal append, rewrite, and sink eviction.

This is a cache/attention fingerprint, not a quality test of distilled weights.
Only block 0 is loaded. Five tiny forwards replace a full causal video render.
"""

from dataclasses import replace

import torch

from fastvideo.tests.golden_gate._harness import DEFAULT_SEED, _component_dir, distributed_runtime, load_layer_state
from fastvideo.tests.golden_gate._tensor_golden import assert_tensor_golden, deterministic_forward
from fastvideo.tests.golden_gate._wan_checkpoint import WAN_REVISION
from fastvideo.tests.golden_gate.test_wan_t2v import INNER_DIM, SPEC

__all__ = ["distributed_runtime"]


def causal_outputs(device):
    from fastvideo.forward_context import set_forward_context
    from fastvideo.layers.rotary_embedding import get_rotary_pos_embed
    from fastvideo.models.loader.utils import set_default_torch_dtype
    from fastvideo.models.wan.causal_transformer import CausalWanTransformerBlock
    from fastvideo.platforms import AttentionBackendEnum

    spec = replace(SPEC, revision=WAN_REVISION)
    # Attention selects its backend during construction, not .to(bfloat16).
    with set_default_torch_dtype(torch.bfloat16):
        block = CausalWanTransformerBlock(
            INNER_DIM, 8960, 12, local_attn_size=3, sink_size=1, cross_attn_norm=True,
            supported_attention_backends=(AttentionBackendEnum.FLASH_ATTN,),
        )
    assert block.attn1.attn.backend == AttentionBackendEnum.FLASH_ATTN
    assert block.attn2.attn.backend == AttentionBackendEnum.FLASH_ATTN
    block.load_state_dict(load_layer_state(spec, _component_dir(spec)), strict=True)
    block = block.to(device=device, dtype=torch.bfloat16).eval()
    generator = torch.Generator(device="cpu").manual_seed(DEFAULT_SEED)

    def randn(*shape):
        return torch.randn(*shape, generator=generator).to(device=device, dtype=torch.bfloat16)

    encoder = randn(1, 16, INNER_DIM)
    kv = {
        "k": torch.zeros(1, 12, 12, 128, device=device, dtype=torch.bfloat16),
        "v": torch.zeros(1, 12, 12, 128, device=device, dtype=torch.bfloat16),
        "global_end_index": torch.zeros(1, device=device, dtype=torch.long),
        "local_end_index": torch.zeros(1, device=device, dtype=torch.long),
    }
    cross = {"k": torch.zeros_like(encoder).reshape(1, 16, 12, 128),
             "v": torch.zeros_like(encoder).reshape(1, 16, 12, 128), "is_init": False}
    cos, sin = get_rotary_pos_embed((4, 2, 2), INNER_DIM, 12, [44, 42, 42],
                                    rope_theta=10000, dtype=torch.float64)
    outputs = {}
    starts = [0, 0, 4, 8, 12]  # Rewrite denoised context, append, then evict while preserving the sink.
    with torch.inference_mode(), set_forward_context(current_timestep=0, attn_metadata=None):
        for index, start in enumerate(starts):
            outputs[f"block_{index}"] = block(
                hidden_states=randn(1, 4, INNER_DIM), encoder_hidden_states=encoder,
                temb=randn(1, 1, 6, INNER_DIM),
                freqs_cis=(cos[start:start + 4].to(device).float(), sin[start:start + 4].to(device).float()),
                block_mask=None, kv_cache=kv, crossattn_cache=cross, current_start=start, frame_seqlen=4,
            )
            outputs[f"cache_k_{index}"] = kv["k"].clone()
            outputs[f"cache_v_{index}"] = kv["v"].clone()
            outputs[f"indices_{index}"] = torch.cat([kv["global_end_index"], kv["local_end_index"]]).clone()
    assert cross["is_init"]
    return outputs, {"repo": spec.repo_id, "revision": spec.revision, "layer": 0,
                     "starts": starts, "local_attn_size": 3, "sink_size": 1,
                     "frame_seqlen": 4, "text_length": 16, "rope_cache_policy": "absolute"}


def test_wan_causal_golden_gate(distributed_runtime):
    with deterministic_forward("FLASH_ATTN") as device:
        outputs, identity = causal_outputs(device)
        assert_tensor_golden("wan_causal", outputs, identity=identity, attention_backend="FLASH_ATTN")
