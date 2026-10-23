# SPDX-License-Identifier: Apache-2.0
"""Small Bernini contracts; official-checkpoint parity lives in local_tests."""
import pytest
import torch
from einops import rearrange

from fastvideo.models.dits.bernini import pack_latents, source_rotary_embedding, unpack_latents
from fastvideo.configs.models.dits.bernini import BerniniRArchConfig, BerniniRConfig
from fastvideo.models.dits.bernini import BerniniRTransformer3DModel
from fastvideo.forward_context import set_forward_context


@pytest.mark.parametrize("shape", [(1, 16, 1, 4, 6), (1, 16, 3, 8, 4), (2, 3, 2, 4, 6)])
def test_bernini_patch_order_matches_official(shape):
    x = torch.arange(torch.tensor(shape).prod()).reshape(shape)
    expected = rearrange(x, "b c t (h ph) (w pw) -> b (t h w) (ph pw c)", ph=2, pw=2)
    assert torch.equal(pack_latents(x), expected)
    assert torch.equal(unpack_latents(expected, shape), x)


def test_source_rope_is_not_a_temporal_offset():
    base = source_rotary_embedding((2, 3, 4), 128, 0)
    reference = source_rotary_embedding((2, 3, 4), 128, 1)
    torch.testing.assert_close(base[0][0], torch.ones(128, dtype=torch.float64))
    torch.testing.assert_close(base[1][0], torch.zeros(128, dtype=torch.float64))
    phase = 1 / (10000 ** (torch.arange(0, 128, 2, dtype=torch.float64) / 128))
    torch.testing.assert_close(reference[0][0], phase.cos().repeat_interleave(2))
    torch.testing.assert_close(reference[1][0], phase.sin().repeat_interleave(2))
    assert not torch.equal(base[0], reference[0])


@pytest.mark.parametrize("name,expected", [
    ("condition_embedder.time_embedder.mlp.fc_in.weight", torch.float32),
    ("blocks.0.scale_shift_table", torch.float32),
    ("scale_shift_table", torch.float32),
    ("blocks.0.self_attn_residual_norm.norm.weight", torch.float32),
    ("condition_embedder.time_modulation.linear.weight", torch.bfloat16),
    ("blocks.0.norm_q.weight", torch.bfloat16),
    ("blocks.0.attn2.to_q.weight", torch.bfloat16),
])
def test_bernini_checkpoint_dtype_policy(name, expected):
    assert BerniniRTransformer3DModel._get_parameter_dtype(None, name, torch.bfloat16) == expected


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_bernini_condition_changes_target_and_output_shape(distributed_setup):
    arch = BerniniRArchConfig(num_attention_heads=2, attention_head_dim=16, num_layers=2,
                              ffn_dim=64, text_dim=24, freq_dim=8)
    model = BerniniRTransformer3DModel(BerniniRConfig(arch_config=arch), {}).cuda().eval()
    # Native parallel linears allocate empty checkpoint destinations; unlike
    # nn.Linear they intentionally do not initialize random training weights.
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name.endswith("bias"):
                parameter.zero_()
            elif parameter.ndim == 1:
                parameter.fill_(1)
            else:
                parameter.normal_(std=0.02)
    x = torch.randn(1, 16, 2, 4, 6, device="cuda")
    source = torch.randn(1, 16, 3, 6, 4, device="cuda")
    text = torch.randn(1, 7, 24, device="cuda")
    with torch.no_grad(), set_forward_context(current_timestep=500, attn_metadata=None):
        actual = model(x, text, torch.tensor([500.0], device="cuda"), source_latents=source)
        changed = model(x, text, torch.tensor([500.0], device="cuda"), source_latents=source * 0)
    assert actual.shape == x.shape and torch.isfinite(actual).all()
    assert not torch.allclose(actual, changed)
