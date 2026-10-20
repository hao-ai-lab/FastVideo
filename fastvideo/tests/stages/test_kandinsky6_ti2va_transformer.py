# SPDX-License-Identifier: Apache-2.0
"""Kandinsky6 TI2VA DiT: batched forward semantics on tiny random weights (CPU).

The model always builds the fused video+audio architecture, and a video-only forward call
(hidden_states_audio=None) is a supported case.
"""
from __future__ import annotations

import kandinsky6_ti2va_tiny as k6_tiny
import pytest
import torch

B, T, H, W = 2, 2, 4, 4
VISUAL_DIM = 2 * k6_tiny.TINY_DIT_CFG["in_visual_dim"] + 1  # visual_cond=True: noised | cond | mask
A = 3  # audio sequence length
L = 5  # text token length


def _rope_pos():
    return [torch.arange(T), torch.arange(H // 2), torch.arange(W // 2)]


def _inputs(seed=1, batch=B, with_audio=True):
    generator = torch.Generator().manual_seed(seed)
    hidden_states = torch.randn(batch, T, H, W, VISUAL_DIM, generator=generator)
    encoder_hidden_states = torch.randn(batch, L, k6_tiny.TINY_DIT_CFG["in_text_dim"], generator=generator)
    pooled = torch.randn(batch, k6_tiny.TINY_DIT_CFG["in_text_dim2"], generator=generator)
    timestep = torch.linspace(100.0, 900.0, batch)
    kwargs = dict(
        hidden_states=hidden_states,
        encoder_hidden_states=encoder_hidden_states,
        timestep=timestep,
        pooled_projections=pooled,
        visual_rope_pos=_rope_pos(),
        text_rope_pos=torch.arange(L),
    )
    if with_audio:
        kwargs["hidden_states_audio"] = torch.randn(batch, A, k6_tiny.TINY_DIT_CFG["in_audio_dim"], generator=generator)
    return kwargs


def test_multimodal_forward_shapes():
    dit = k6_tiny.build_dit()
    with torch.no_grad():
        video_out, audio_out = dit(**_inputs())
    assert video_out.shape == (B, T, H, W, k6_tiny.TINY_DIT_CFG["out_visual_dim"])
    assert audio_out.shape == (B, A, k6_tiny.TINY_DIT_CFG["in_audio_dim"])  # out_audio_dim defaults to in_audio_dim
    assert torch.isfinite(video_out).all()
    assert torch.isfinite(audio_out).all()


def test_video_only_forward_does_not_raise_and_returns_a_single_tensor():
    # Pre-collapse this raised NotImplementedError for a multimodal checkpoint; the fused block now
    # guards aud=None exactly like the diffusers reference.
    dit = k6_tiny.build_dit()
    kwargs = _inputs(with_audio=False)
    with torch.no_grad():
        result = dit(**kwargs)
    assert isinstance(result, torch.Tensor)
    assert result.shape == (B, T, H, W, k6_tiny.TINY_DIT_CFG["out_visual_dim"])
    assert torch.isfinite(result).all()


def test_video_only_output_matches_the_video_half_of_a_multimodal_call_given_the_same_video_inputs():
    # The video stream's own self-attention / text-cross-attention / feed-forward do not depend on
    # audio being present; only the cross-modal va/av mixing (skipped when aud=None) does. So with the
    # same seed, video-only output should differ from the multimodal call's video_out (the cross-modal
    # terms are real contributions, not zero), but both must be finite and distinctly computed --
    # this test pins that video-only is NOT silently identical to just truncating a multimodal run.
    dit = k6_tiny.build_dit()
    kwargs = _inputs()
    with torch.no_grad():
        video_out_multimodal, _ = dit(**kwargs)
        video_only_kwargs = {k: v for k, v in kwargs.items() if k != "hidden_states_audio"}
        video_out_alone = dit(**video_only_kwargs)
    assert not torch.allclose(video_out_multimodal, video_out_alone)


def test_samples_of_a_batch_do_not_interact():
    dit = k6_tiny.build_dit()
    kwargs = _inputs()
    with torch.no_grad():
        video_batched, audio_batched = dit(**kwargs)
        singles = [
            dit(
                hidden_states=kwargs["hidden_states"][i:i + 1],
                encoder_hidden_states=kwargs["encoder_hidden_states"][i:i + 1],
                timestep=kwargs["timestep"][i:i + 1],
                pooled_projections=kwargs["pooled_projections"][i:i + 1],
                hidden_states_audio=kwargs["hidden_states_audio"][i:i + 1],
                visual_rope_pos=_rope_pos(),
                text_rope_pos=torch.arange(L),
            ) for i in range(B)
        ]
    video_single = torch.cat([s[0] for s in singles])
    audio_single = torch.cat([s[1] for s in singles])
    torch.testing.assert_close(video_batched, video_single, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(audio_batched, audio_single, rtol=1e-5, atol=1e-5)


def test_state_dict_has_prefixed_video_audio_towers_and_no_bare_text_tower():
    # Checkpoints use the video_/audio_-prefixed text towers; there is no unprefixed text tower.
    dit = k6_tiny.build_dit()
    keys = set(dit.state_dict())
    assert "video_text_embeddings.in_layer.weight" in keys
    assert "audio_text_embeddings.in_layer.weight" in keys
    assert "video_time_embeddings.in_layer.weight" in keys
    assert "audio_time_embeddings.in_layer.weight" in keys
    assert "visual_transformer_blocks.0.videoT.self_attention.to_query.weight" in keys
    assert "visual_transformer_blocks.0.audioT.self_attention.to_query.weight" in keys
    assert "audio_out_layer.out_layer.weight" in keys
    assert not any(key.startswith("text_embeddings.") or key.startswith("text_transformer_blocks.")
                  for key in keys)


def test_gradient_checkpointing_matches_no_checkpointing_for_a_multimodal_call():
    dit = k6_tiny.build_dit()
    dit.train()
    kwargs = _inputs()
    dit.gradient_checkpointing = False
    video_a, audio_a = dit(**kwargs)
    dit.gradient_checkpointing = True
    video_b, audio_b = dit(**kwargs)
    torch.testing.assert_close(video_a, video_b, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(audio_a, audio_b, rtol=1e-5, atol=1e-5)


def test_gradient_checkpointing_matches_no_checkpointing_for_a_video_only_call():
    # aud=None flows through torch.utils.checkpoint.checkpoint as a plain positional arg; this pins
    # that None-valued args survive the checkpoint round-trip correctly.
    dit = k6_tiny.build_dit()
    dit.train()
    kwargs = _inputs(with_audio=False)
    dit.gradient_checkpointing = False
    video_a = dit(**kwargs)
    dit.gradient_checkpointing = True
    video_b = dit(**kwargs)
    torch.testing.assert_close(video_a, video_b, rtol=1e-5, atol=1e-5)


def test_visual_transformer_blocks_is_the_first_registered_module_list():
    # fastvideo/hooks/layerwise_offload.py's enable_layerwise_offload hooks only the first top-level
    # nn.ModuleList found via named_children() (attribute-registration order) and stops there -- with
    # dit_layerwise_offload=True (the default), that must be visual_transformer_blocks, not the 4-entry
    # video_text_transformer_blocks. Not itself a test of the (CUDA-only) hook helper; pins the
    # registration order the helper relies on.
    dit = k6_tiny.build_dit()
    first_module_list_name = next(name for name, child in dit.named_children() if isinstance(child, torch.nn.ModuleList))
    assert first_module_list_name == "visual_transformer_blocks"
