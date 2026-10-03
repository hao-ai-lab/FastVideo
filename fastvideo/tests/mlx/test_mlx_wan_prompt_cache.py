# SPDX-License-Identifier: Apache-2.0
"""Prompt cache reuse and invalidation without loading UMT5 weights."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from fastvideo.mlx_runtime import wan_helpers


@pytest.fixture
def prompt_encoder(tmp_path, monkeypatch):
    import transformers

    calls = []
    root = tmp_path / "model"
    (root / "tokenizer").mkdir(parents=True)
    (root / "text_encoder").mkdir()
    weights = root / "text_encoder" / "weights"
    weights.write_bytes(b"version one")

    class Tokenizer:
        def __call__(self, prompts, *, max_length, **kwargs):
            mask = torch.zeros((1, max_length), dtype=torch.long)
            mask[:, :2] = 1
            return SimpleNamespace(input_ids=torch.ones_like(mask), attention_mask=mask)

    class Encoder:
        def __init__(self, dtype):
            self.dtype = dtype

        def to(self, device):
            return self

        def eval(self):
            return self

        def __call__(self, ids, mask):
            # Use values that expose recipe precision changes.
            return SimpleNamespace(last_hidden_state=torch.full((*ids.shape, 4), 1.001, dtype=self.dtype))

    def load_encoder(path, *, torch_dtype, **kwargs):
        calls.append(torch_dtype)
        return Encoder(torch_dtype)

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: Tokenizer())
    monkeypatch.setattr(transformers.UMT5EncoderModel, "from_pretrained", load_encoder)
    monkeypatch.setattr(wan_helpers, "cleanup_torch_mps", lambda: None)
    return root, weights, calls


@pytest.mark.parametrize("dtype,output_dtype", [("fp16", torch.float16), ("bf16", torch.float32)])
def test_repeat_prompt_uses_cache_with_exact_recipe_values(prompt_encoder, tmp_path, dtype, output_dtype):
    root, _, calls = prompt_encoder
    kwargs = dict(model_root=root, prompt="a fox", max_sequence_length=4, device_arg="cpu",
                  dtype_arg=dtype, cache_dir=tmp_path / "cache")
    first = wan_helpers.encode_wan_prompt(**kwargs)
    second = wan_helpers.encode_wan_prompt(**kwargs)
    assert calls == [torch.float16 if dtype == "fp16" else torch.bfloat16]
    assert first.dtype == second.dtype == output_dtype
    assert torch.equal(first, second)
    assert torch.count_nonzero(second[:, 2:]) == 0


def test_cache_key_covers_prompt_length_precision_weights_and_device(prompt_encoder, tmp_path):
    root, weights, calls = prompt_encoder
    kwargs = dict(model_root=root, prompt="a fox", max_sequence_length=4, device_arg="cpu",
                  dtype_arg="fp16", cache_dir=tmp_path / "cache")
    wan_helpers.encode_wan_prompt(**kwargs)
    for update in ({"prompt": "a cat"}, {"max_sequence_length": 8}, {"dtype_arg": "bf16"},
                   {"device_arg": "cpu:0"}):
        wan_helpers.encode_wan_prompt(**{**kwargs, **update})
    weights.write_bytes(b"version two changed")
    wan_helpers.encode_wan_prompt(**kwargs)
    assert len(calls) == 6


def test_cache_can_be_disabled(prompt_encoder):
    root, _, calls = prompt_encoder
    for _ in range(2):
        wan_helpers.encode_wan_prompt(model_root=root, prompt="a fox", max_sequence_length=4,
                                      device_arg="cpu", dtype_arg="fp16", cache_dir=None)
    assert len(calls) == 2


@pytest.mark.parametrize("cached", [
    np.zeros((1, 3, 4), dtype=np.float16),
    np.zeros((1, 4, 4), dtype=np.float32),
    np.full((1, 4, 4), np.nan, dtype=np.float16),
])
def test_invalid_cached_embeddings_are_reencoded(prompt_encoder, tmp_path, monkeypatch, cached):
    root, _, calls = prompt_encoder
    monkeypatch.setattr(wan_helpers, "load_prompt_cache", lambda *args: cached)
    embeds = wan_helpers.encode_wan_prompt(model_root=root, prompt="a fox", max_sequence_length=4,
                                          device_arg="cpu", dtype_arg="fp16", cache_dir=tmp_path / "cache")
    assert len(calls) == 1
    assert embeds.shape == (1, 4, 4)
    assert torch.isfinite(embeds).all()


@pytest.mark.parametrize("config,match", [
    ({"num_attention_heads": 1, "attention_head_dim": 12}, "patch_size"),
    ({"num_attention_heads": 1, "attention_head_dim": 12, "patch_size": [1, 0, 2]}, "positive"),
    ({"num_attention_heads": 1, "attention_head_dim": 12, "patch_size": [1, 2, 2]}, "align"),
])
def test_incomplete_or_invalid_rotary_config_has_a_clear_error(config, match):
    with pytest.raises(ValueError, match=match):
        wan_helpers.make_wan_rotary_embeddings(config, latent_frames=2, latent_height=3, latent_width=4)


def test_cli_and_pipeline_use_the_same_prompt_and_rotary_helpers():
    from examples.inference.basic import mlx_wan_prompt_to_video as cli
    from fastvideo.mlx_runtime import wan_pipeline

    assert cli.encode_prompt is wan_pipeline._encode_wan_prompt
    assert cli.make_rotary_embeddings is wan_pipeline._make_wan_rotary_embeddings
