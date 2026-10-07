# SPDX-License-Identifier: Apache-2.0
"""Kandinsky6 SR DiT: batched forward semantics on tiny random weights (CPU)."""
from __future__ import annotations

import k6_sr_tiny
import pytest
import torch

B, T, H, W, C = 3, 3, 4, 6, 4
IN = 2 * C + 1  # noised latent | anchor | anchor mask


def _rope_pos():
    return [torch.arange(T), torch.arange(H // 2), torch.arange(W // 2)]


def _inputs(seed=1, batch=B):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(batch, T, H, W, IN, generator=generator), torch.linspace(100.0, 900.0, batch)


@pytest.mark.parametrize("piflow", [k6_sr_tiny.TINY_PIFLOW, None])
def test_forward_shape_and_head_width(piflow):
    dit = k6_sr_tiny.build_dit(piflow)
    x, t = _inputs()
    with torch.no_grad():
        y = dit(x, t, _rope_pos(), (1.0, 1.0, 1.0))
    n_grid = piflow["n_grid"] if piflow else 1
    assert y.shape == (B, T, H, W, C * n_grid)
    assert torch.isfinite(y).all()


def test_samples_of_a_batch_do_not_interact():
    # The reference packs samples into one ragged sequence; the batched layout must keep them independent
    # (attention over the wrong tokens or a shared time embedding would leak between tiles).
    dit = k6_sr_tiny.build_dit()
    x, t = _inputs()
    with torch.no_grad():
        batched = dit(x, t, _rope_pos(), (1.0, 1.0, 1.0))
        single = torch.cat([dit(x[i:i + 1], t[i:i + 1], _rope_pos(), (1.0, 1.0, 1.0)) for i in range(B)])
    torch.testing.assert_close(batched, single, rtol=1e-5, atol=1e-5)


def test_output_depends_on_the_time_embedding_and_pooled_bias():
    dit = k6_sr_tiny.build_dit()
    x, t = _inputs()
    with torch.no_grad():
        base = dit(x, t, _rope_pos(), (1.0, 1.0, 1.0))
        other_t = dit(x, t * 0.5, _rope_pos(), (1.0, 1.0, 1.0))
        dit.pooled_bias.add_(1.0)  # dropping the bias corrupts the time conditioning (see DiffusionTransformer3D)
        shifted = dit(x, t, _rope_pos(), (1.0, 1.0, 1.0))
    assert not torch.allclose(base, other_t)
    assert not torch.allclose(base, shifted)


def test_rope_scale_factor_changes_the_output():
    dit = k6_sr_tiny.build_dit()
    x, t = _inputs()
    with torch.no_grad():
        a = dit(x, t, _rope_pos(), (1.0, 1.0, 1.0))
        b = dit(x, t, _rope_pos(), (1.0, 2.0, 2.0))
    assert not torch.allclose(a, b)


def test_wrong_channel_count_raises_a_clear_error():
    dit = k6_sr_tiny.build_dit()
    narrow = torch.randn(1, T, H, W, C)
    with pytest.raises(ValueError, match=f"expects {IN}"):
        dit(narrow, torch.tensor([500.0]), _rope_pos())


def test_unsupported_forward_arguments_are_rejected():
    dit = k6_sr_tiny.build_dit()
    x, t = _inputs(batch=1)
    with pytest.raises(TypeError, match="sparse_params"):
        dit(x, t, _rope_pos(), sparse_params={"to_fractal": True})


def test_state_dict_layout_matches_the_bundle_contract():
    dit = k6_sr_tiny.build_dit()
    keys = set(dit.state_dict())
    assert "pooled_bias" in keys
    assert "time_embeddings.in_layer.weight" in keys and "time_embeddings.out_layer.bias" in keys
    assert "visual_embeddings.in_layer.weight" in keys
    for name in ("to_query", "to_key", "to_value", "out_layer"):
        assert f"visual_transformer_blocks.0.self_attention.{name}.weight" in keys
    assert "visual_transformer_blocks.1.self_attention.query_norm.weight" in keys
    assert "visual_transformer_blocks.0.feed_forward.mlp.fc_in.weight" in keys
    assert "visual_transformer_blocks.0.visual_modulation.out_layer.weight" in keys
    assert "out_layer.modulation.out_layer.weight" in keys and "out_layer.out_layer.weight" in keys
    # text-free: no text tower, no cross-attention
    assert not any("text" in key or "cross_attention" in key for key in keys)
    # 6 modulation params (self-attention + ffn) per block; pi-Flow head width C * n_grid * prod(patch)
    assert dit.state_dict()["visual_transformer_blocks.0.visual_modulation.out_layer.weight"].shape[0] == 6 * 32
    assert dit.state_dict()["out_layer.out_layer.weight"].shape[0] == 4 * 3 * 4


def test_official_keys_round_trip_through_the_mapping():
    from fastvideo.models.loader.utils import get_param_names_mapping, hf_to_custom_state_dict

    dit = k6_sr_tiny.build_dit()
    official_keys = k6_sr_tiny.to_official_dit_keys(dit.state_dict())
    assert not any(key.startswith("model.") for key in official_keys)  # the official checkpoint has no prefix
    custom, _ = hf_to_custom_state_dict(official_keys, get_param_names_mapping(dit.param_names_mapping))
    assert set(custom) == set(dit.state_dict())  # every official key lands on a model key, none left over
    dit.load_state_dict(custom, strict=True)


def test_forward_survives_fp8_quantization():
    # With transformer_quant=get_quantization_config("FP8")() (per docs/inference/optimizations.md),
    # convert_model_to_fp8 pops `.weight` off every FP8-tagged ReplicatedLinear (e.g.
    # self_attention.to_query), so the blocks must take their compute dtype from the model's
    # never-quantized `compute_dtype`, not from a quantized layer's `.weight.dtype`.
    from fastvideo.layers.quantization import get_quantization_config
    from fastvideo.layers.quantization.fp8_config import convert_model_to_fp8

    cfg = k6_sr_tiny.dit_config_dict(k6_sr_tiny.TINY_PIFLOW)
    from fastvideo.configs.models.dits.kandinsky6_sr import Kandinsky6SRConfig
    from fastvideo.models.dits.kandinsky6_sr import Kandinsky6SRTransformer3DModel

    config = Kandinsky6SRConfig(quant_config=get_quantization_config("FP8")())
    config.update_model_arch(cfg)
    dit = k6_sr_tiny.randomize(Kandinsky6SRTransformer3DModel(config, cfg), seed=2).eval()

    convert_model_to_fp8(dit)
    # Sanity: quantization actually removed `.weight` from at least one attention projection, so this
    # test would have caught the original bug (it wasn't a no-op).
    assert not hasattr(dit.visual_transformer_blocks[0].self_attention.to_query, "weight")

    x, t = _inputs(batch=1)
    with torch.no_grad():
        y = dit(x, t, _rope_pos(), (1.0, 1.0, 1.0))
    assert torch.isfinite(y).all()
