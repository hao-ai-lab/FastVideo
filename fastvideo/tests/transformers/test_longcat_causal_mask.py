# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from fastvideo.configs.models.dits.longcat import (
    LongCatVideoArchConfig,
    LongCatVideoConfig,
)
from fastvideo.models.dits import longcat as longcat_module
from fastvideo.models.dits.longcat import (
    LongCatTransformer3DModel,
    build_longcat_causal_block_ranges,
    build_longcat_block_causal_mask,
)
from fastvideo.train.methods.distribution_matching.self_forcing import (
    SelfForcingMethod, )


def _frame_visibility(
    mask: torch.Tensor,
    tokens_per_frame: int,
) -> list[list[bool]]:
    token_mask = mask[0, 0]
    frames = []
    for row in token_mask[::tokens_per_frame]:
        frames.append([
            bool(row[frame * tokens_per_frame])
            for frame in range(token_mask.shape[1] // tokens_per_frame)
        ])
    return frames


def test_longcat_block_causal_mask_blocks_future_chunks():
    mask = build_longcat_block_causal_mask(
        query_frames=4,
        key_frames=4,
        tokens_per_frame=2,
        causal_block_size=2,
        device="cpu",
    )

    assert mask.shape == (1, 1, 8, 8)
    assert _frame_visibility(mask, tokens_per_frame=2) == [
        [True, True, False, False],
        [True, True, False, False],
        [True, True, True, True],
        [True, True, True, True],
    ]
    assert mask[0, 0, 1].tolist() == mask[0, 0, 0].tolist()


def test_longcat_block_causal_mask_handles_cached_prefix():
    mask = build_longcat_block_causal_mask(
        query_frames=2,
        key_frames=5,
        tokens_per_frame=1,
        causal_block_size=2,
        device="cpu",
    )

    assert _frame_visibility(mask, tokens_per_frame=1) == [
        [True, True, True, True, False],
        [True, True, True, True, True],
    ]


def test_longcat_block_causal_mask_allows_first_single_block_only():
    mask = build_longcat_block_causal_mask(
        query_frames=1,
        key_frames=1,
        tokens_per_frame=2,
        causal_block_size=2,
        device="cpu",
    )

    assert _frame_visibility(mask, tokens_per_frame=2) == [
        [True],
    ]
    assert mask[0, 0, 1].tolist() == mask[0, 0, 0].tolist()


def test_longcat_block_causal_mask_rejects_invalid_shape():
    with pytest.raises(ValueError, match="query_frames must be <= key_frames"):
        build_longcat_block_causal_mask(
            query_frames=3,
            key_frames=2,
            tokens_per_frame=1,
            causal_block_size=1,
            device="cpu",
        )


def test_longcat_self_forcing_rejects_chunk_block_mismatch():
    class Student:
        _causal_block_size = 4

    with pytest.raises(ValueError, match="causal_block_size.*chunk_size"):
        SelfForcingMethod._validate_causal_block_size(
            student=Student(),  # type: ignore[arg-type]
            chunk_size=3,
        )


def test_longcat_causal_stage_remainder_goes_in_first_block():
    assert build_longcat_causal_block_ranges(
        num_frames=20,
        chunk_size=3,
    ) == [(0, 5), (5, 8), (8, 11), (11, 14), (14, 17), (17, 20)]
    assert build_longcat_causal_block_ranges(
        num_frames=21,
        chunk_size=3,
    ) == [
        (0, 3),
        (3, 6),
        (6, 9),
        (9, 12),
        (12, 15),
        (15, 18),
        (18, 21),
    ]


class _UnexpectedAttention(torch.nn.Module):
    """Constructor-only stand-in for backend-selected attention layers."""

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, *args, **kwargs):
        raise AssertionError("causal LongCat test should use direct SDPA")


def _merge_kv_cache(
    cache: dict[int, tuple[torch.Tensor, torch.Tensor]] | None,
    new_chunk: dict[int, tuple[torch.Tensor, torch.Tensor]],
) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
    if cache is None:
        return new_chunk
    return {
        idx: (
            torch.cat((cache[idx][0], k), dim=2),
            torch.cat((cache[idx][1], v), dim=2),
        )
        for idx, (k, v) in new_chunk.items()
    }


def _tiny_longcat_transformer(monkeypatch) -> LongCatTransformer3DModel:
    monkeypatch.setattr(
        longcat_module,
        "DistributedAttention",
        _UnexpectedAttention,
    )
    monkeypatch.setattr(
        longcat_module,
        "LocalAttention",
        _UnexpectedAttention,
    )

    arch_config = LongCatVideoArchConfig(
        hidden_size=16,
        depth=2,
        num_attention_heads=2,
        in_channels=4,
        out_channels=4,
        num_channels_latents=4,
        patch_size=(1, 1, 1),
        caption_channels=8,
        adaln_tembed_dim=16,
        frequency_embedding_size=8,
        mlp_ratio=1,
        enable_bsa=False,
    )
    model = LongCatTransformer3DModel(
        config=LongCatVideoConfig(arch_config=arch_config),
        hf_config={},
    )
    for param in model.parameters():
        if param.ndim > 1:
            torch.nn.init.normal_(param, mean=0.0, std=0.02)
        else:
            torch.nn.init.uniform_(param, a=-0.02, b=0.02)
    return model.eval()


def test_longcat_chunked_causal_rollout_matches_full_prefix(monkeypatch):
    torch.manual_seed(0)
    model = _tiny_longcat_transformer(monkeypatch)

    batch_size = 1
    channels = 4
    num_frames = 6
    height = 2
    width = 2
    chunk_size = 2
    text_len = 3

    hidden_states = torch.randn(batch_size, channels, num_frames, height,
                                width)
    encoder_hidden_states = torch.randn(batch_size, text_len, 8)
    encoder_attention_mask = torch.ones(batch_size, text_len)
    timesteps = torch.full((batch_size, num_frames), 123.0)

    chunked_outputs = []
    kv_cache = None
    with torch.no_grad():
        for start in range(0, num_frames, chunk_size):
            end = start + chunk_size
            chunk = hidden_states[:, :, start:end]
            timestep_chunk = timesteps[:, start:end]
            cached_frames = start

            chunked = model(
                hidden_states=chunk,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                timestep=timestep_chunk,
                num_cond_latents=cached_frames,
                kv_cache_dict=kv_cache,
                kv_cache_start_frame=0,
                causal_block_size=chunk_size,
                skip_crs_attn=True,
            )

            prefix = model(
                hidden_states=hidden_states[:, :, :end],
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                timestep=timesteps[:, :end],
                causal_block_size=chunk_size,
                skip_crs_attn=True,
            )[:, :, start:end]
            torch.testing.assert_close(chunked, prefix, atol=1e-5, rtol=1e-5)
            chunked_outputs.append(chunked)

            _, new_cache = model(
                hidden_states=chunk,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                timestep=timestep_chunk,
                num_cond_latents=cached_frames,
                kv_cache_dict=kv_cache,
                kv_cache_start_frame=0,
                return_kv=True,
                causal_block_size=chunk_size,
                skip_crs_attn=True,
            )
            kv_cache = _merge_kv_cache(kv_cache, new_cache)

        full_reference = model(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            timestep=timesteps,
            causal_block_size=chunk_size,
            skip_crs_attn=True,
        )

    torch.testing.assert_close(
        torch.cat(chunked_outputs, dim=2),
        full_reference,
        atol=1e-5,
        rtol=1e-5,
    )


def test_longcat_forward_with_kv_cache_matches_rope_offset(monkeypatch):
    torch.manual_seed(1)
    model = _tiny_longcat_transformer(monkeypatch)
    attn = model.blocks[0].self_attn

    batch_size = 1
    num_frames = 4
    height = 2
    width = 2
    hidden_size = 16
    chunk_size = 2
    tokens_per_frame = height * width
    split = chunk_size * tokens_per_frame

    x = torch.randn(batch_size, num_frames * tokens_per_frame, hidden_size)

    with torch.no_grad():
        full = attn(
            x,
            latent_shape=(num_frames, height, width),
            causal_block_size=chunk_size,
        )
        _, kv_cache = attn(
            x[:, :split],
            latent_shape=(chunk_size, height, width),
            return_kv=True,
            causal_block_size=chunk_size,
        )
        cached = attn.forward_with_kv_cache(
            x[:, split:],
            latent_shape=(chunk_size, height, width),
            num_cond_latents=chunk_size,
            kv_cache=kv_cache,
            kv_cache_start_frame=0,
            causal_block_size=chunk_size,
        )

    torch.testing.assert_close(cached, full[:, split:], atol=1e-5, rtol=1e-5)
