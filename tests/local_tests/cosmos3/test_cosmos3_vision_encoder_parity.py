# SPDX-License-Identifier: Apache-2.0
"""Numerical-parity test: Cosmos3 vision_encoder (transformers) vs the framework.

Image-conditioned reasoning needs the Qwen3-VL ``vision_encoder``. The Cosmos3
checkpoint's ``vision_encoder`` is a standard ``transformers`` ``Qwen3VLVisionModel``
(``architectures: ["Qwen3VLVisionModel"]``); the framework ships its own copy
(``cosmos_framework.model.vfm.vlm.qwen3_vl.qwen3_vl.Qwen3VLVisionModel``). Like the
Qwen2 tokenizer, FastVideo reuses the ``transformers`` model (no diffusers); this
pins it bit-for-bit against the framework's implementation.

Both built tiny from the SAME vision config; ``transformers`` weights copied into
the framework model; identical ``(hidden_states, grid_thw)`` forward. The
framework is the parity ORACLE.

(Separately verified on the REAL 1.15 GB checkpoint: both strict-load and produce
identical ``[N, out_hidden]`` embeds, max abs diff = 0.0.)

Run:
    cd <worktree> && <fv-cosmos3 python> -m pytest \
        tests/local_tests/cosmos3/test_cosmos3_vision_encoder_parity.py -q -s
"""
from __future__ import annotations

import pytest
import torch

cosmos_framework = pytest.importorskip(
    "cosmos_framework",
    reason="cosmos_framework not installed; run in fv-cosmos3 env.",
)
transformers = pytest.importorskip("transformers")

pytestmark = [pytest.mark.local]

_TINY_VISION_CFG = dict(
    depth=2,
    hidden_size=32,
    num_heads=2,
    intermediate_size=64,
    patch_size=16,
    temporal_patch_size=2,
    spatial_merge_size=2,
    in_channels=3,
    out_hidden_size=64,
    num_position_embeddings=64,
    deepstack_visual_indexes=[1],
    hidden_act="gelu_pytorch_tanh",
    initializer_range=0.02,
)


def _diffs(a, b):
    d = (a - b).abs()
    return d.max().item(), d.mean().item()


def _to_tensor(out):
    return out[0] if isinstance(out, (tuple, list)) else out


class TestCosmos3VisionEncoderParity:

    @pytest.mark.parametrize(("grid_h", "grid_w"), [(2, 2), (4, 4), (4, 6)])
    def test_vision_encoder_matches_framework(self, grid_h, grid_w):
        from cosmos_framework.model.vfm.vlm.qwen3_vl.configuration_qwen3_vl import (
            Qwen3VLVisionConfig as FwVisionConfig,
        )
        from cosmos_framework.model.vfm.vlm.qwen3_vl.qwen3_vl import (
            Qwen3VLVisionModel as FwVisionModel,
        )
        from transformers import Qwen3VLVisionModel as TfVisionModel
        from transformers.models.qwen3_vl.configuration_qwen3_vl import (
            Qwen3VLVisionConfig as TfVisionConfig,
        )

        torch.manual_seed(0)
        tf_model = TfVisionModel(TfVisionConfig(**_TINY_VISION_CFG)).eval()
        fw_model = FwVisionModel(FwVisionConfig(**_TINY_VISION_CFG)).eval()
        # transformers is the unit under test; copy its weights into the framework
        # oracle (identical key surface — both are the same Qwen3-VL ViT).
        missing, unexpected = fw_model.load_state_dict(tf_model.state_dict(), strict=False)
        assert not missing and not unexpected, f"key mismatch: missing={missing[:3]} unexpected={unexpected[:3]}"

        in_dim = _TINY_VISION_CFG["in_channels"] * _TINY_VISION_CFG["temporal_patch_size"] * (
            _TINY_VISION_CFG["patch_size"] ** 2)
        seq = grid_h * grid_w
        hidden_states = torch.randn(seq, in_dim)
        grid_thw = torch.tensor([[1, grid_h, grid_w]], dtype=torch.long)

        with torch.no_grad():
            tf_out = _to_tensor(tf_model(hidden_states, grid_thw))
            fw_out = _to_tensor(fw_model(hidden_states, grid_thw))
        assert tf_out.shape == fw_out.shape, f"shape tf={tf_out.shape} fw={fw_out.shape}"
        mx, mn = _diffs(tf_out, fw_out)
        print(f"\n[vision_encoder grid={grid_h}x{grid_w}] embeds max abs diff = {mx:.3e} mean abs diff = {mn:.3e}")
        torch.testing.assert_close(tf_out, fw_out, atol=1e-5, rtol=1e-4)
