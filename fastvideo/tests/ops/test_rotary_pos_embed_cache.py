# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the get_rotary_pos_embed memoization cache."""

import pytest
import torch

from fastvideo.layers.rotary_embedding import (
    _ROTARY_POS_EMBED_CACHE,
    _ROTARY_POS_EMBED_CACHE_MAXSIZE,
    get_rotary_pos_embed,
)


def _rope_dim_list(hidden_size: int, heads_num: int) -> list[int]:
    """Return the default 3-axis rope_dim_list used by the video DiTs."""
    d = hidden_size // heads_num
    return [d - 4 * (d // 6), 2 * (d // 6), 2 * (d // 6)]


def _call(
    rope_sizes=(21, 30, 52),
    hidden_size=1536,
    heads_num=12,
    rope_dim_list="default",
    rope_theta=10000.0,
    dtype=torch.float64,
    start_frame=0,
    use_real=True,
    **kwargs,
):
    """Thin wrapper around get_rotary_pos_embed with DiT-like defaults."""
    if rope_dim_list == "default":
        rope_dim_list = _rope_dim_list(hidden_size, heads_num)
    return get_rotary_pos_embed(
        rope_sizes,
        hidden_size,
        heads_num,
        rope_dim_list,
        rope_theta,
        dtype=dtype,
        start_frame=start_frame,
        use_real=use_real,
        **kwargs,
    )


@pytest.fixture(autouse=True)
def _clear_cache():
    """Isolate every test by clearing the module-level cache around it."""
    _ROTARY_POS_EMBED_CACHE.clear()
    yield
    _ROTARY_POS_EMBED_CACHE.clear()


def test_repeated_call_hits_cache():
    """A second identical call returns the exact same tensor objects."""
    cos1, sin1 = _call()
    cos2, sin2 = _call()
    assert cos1 is cos2 and sin1 is sin2
    assert len(_ROTARY_POS_EMBED_CACHE) == 1


def test_many_identical_calls_keep_single_entry():
    """Many identical calls never grow the cache beyond one entry."""
    for _ in range(10):
        _call()
    assert len(_ROTARY_POS_EMBED_CACHE) == 1


@pytest.mark.parametrize(
    "rope_sizes,hidden_size,heads_num,dtype,use_real",
    [
        ((21, 30, 52), 1536, 12, torch.float64, True),
        ((21, 45, 80), 5120, 40, torch.float64, True),
        ((1, 16, 16), 1536, 12, torch.float32, True),
        ((4, 8, 8), 1536, 12, torch.float64, False),
    ],
)
def test_cached_matches_fresh_recompute(rope_sizes, hidden_size, heads_num,
                                        dtype, use_real):
    """Cached tensors are bitwise-equal to a fresh uncached recompute."""
    cos_cached, sin_cached = _call(rope_sizes=rope_sizes,
                                   hidden_size=hidden_size,
                                   heads_num=heads_num,
                                   dtype=dtype,
                                   use_real=use_real)
    _ROTARY_POS_EMBED_CACHE.clear()
    cos_fresh, sin_fresh = _call(rope_sizes=rope_sizes,
                                 hidden_size=hidden_size,
                                 heads_num=heads_num,
                                 dtype=dtype,
                                 use_real=use_real)
    assert torch.equal(cos_cached, cos_fresh)
    assert torch.equal(sin_cached, sin_fresh)


@pytest.mark.parametrize(
    "kwargs_a,kwargs_b",
    [
        ({"rope_sizes": (21, 30, 52)}, {"rope_sizes": (21, 45, 80)}),
        ({"dtype": torch.float64}, {"dtype": torch.float32}),
        ({"use_real": True}, {"use_real": False}),
        ({"start_frame": 0}, {"start_frame": 3}),
        ({"rope_theta": 10000.0}, {"rope_theta": 5000.0}),
        ({"shard_dim": 0}, {"shard_dim": 1}),
    ],
)
def test_distinct_args_create_distinct_entries(kwargs_a, kwargs_b):
    """Any output-affecting argument difference yields a separate cache entry."""
    _call(**kwargs_a)
    _call(**kwargs_b)
    assert len(_ROTARY_POS_EMBED_CACHE) == 2


def test_none_rope_dim_list_shares_key_with_equivalent_list():
    """None rope_dim_list and its derived explicit list map to one entry."""
    # head_dim must be divisible by 3 for the None branch to stay valid.
    hidden_size, heads_num = 1536, 16  # head_dim == 96 -> [32, 32, 32]
    _call(rope_dim_list=None, hidden_size=hidden_size, heads_num=heads_num)
    before = len(_ROTARY_POS_EMBED_CACHE)
    _call(rope_dim_list=[32, 32, 32], hidden_size=hidden_size, heads_num=heads_num)
    assert len(_ROTARY_POS_EMBED_CACHE) == before == 1


def test_use_real_controls_last_dim():
    """use_real=True spans full head_dim; use_real=False spans half."""
    cos_full, _ = _call(use_real=True)
    cos_half, _ = _call(use_real=False)
    assert cos_full.shape[-1] == 128
    assert cos_half.shape[-1] == 64


@pytest.mark.parametrize("rope_sizes", [(1, 1, 1), (1, 30, 52), (21, 1, 1)])
def test_degenerate_grid_shapes(rope_sizes):
    """Degenerate single-element axes still produce a correctly sized table."""
    cos, sin = _call(rope_sizes=rope_sizes)
    expected = rope_sizes[0] * rope_sizes[1] * rope_sizes[2]
    assert cos.shape[0] == expected
    assert sin.shape[0] == expected


def test_scalar_and_list_factors_are_hashable_and_distinct():
    """List-valued rescale factors are hashable and keyed apart from scalars."""
    _call(theta_rescale_factor=1.0)
    _call(theta_rescale_factor=[1.0, 1.0, 1.0])
    assert len(_ROTARY_POS_EMBED_CACHE) == 2


def test_caller_device_copy_does_not_corrupt_cache():
    """The .to()/.float() copy callers perform must not mutate cached tensors."""
    cos, _ = _call()
    snapshot = cos.clone()
    _ = cos.to("cpu").float()
    cos_again, _ = _call()
    assert torch.equal(cos_again, snapshot)


def test_start_frame_offsets_values():
    """A non-zero start_frame shifts the temporal positions, changing output."""
    cos0, _ = _call(start_frame=0)
    cos3, _ = _call(start_frame=3)
    assert not torch.equal(cos0, cos3)
    assert len(_ROTARY_POS_EMBED_CACHE) == 2


def test_cache_is_bounded_and_evicts_oldest():
    """The cache caps at the max size and evicts the oldest entry first."""
    # Tiny grids keep this lightweight; each start_frame is a distinct key.
    overshoot = _ROTARY_POS_EMBED_CACHE_MAXSIZE + 4
    for frame in range(overshoot):
        _call(rope_sizes=(2, 2, 2), start_frame=frame)
        assert len(_ROTARY_POS_EMBED_CACHE) <= _ROTARY_POS_EMBED_CACHE_MAXSIZE
    assert len(_ROTARY_POS_EMBED_CACHE) == _ROTARY_POS_EMBED_CACHE_MAXSIZE
    # The earliest-inserted frames must have been evicted; the latest survive.
    surviving = {key[-2] for key in _ROTARY_POS_EMBED_CACHE}  # start_frame slot
    assert overshoot - 1 in surviving
    assert 0 not in surviving


def test_cache_hit_refreshes_recency():
    """Re-accessing an entry protects it from eviction over an untouched one."""
    _call(rope_sizes=(2, 2, 2), start_frame=0)  # entry we will keep hot
    for frame in range(1, _ROTARY_POS_EMBED_CACHE_MAXSIZE):
        _call(rope_sizes=(2, 2, 2), start_frame=frame)
    assert len(_ROTARY_POS_EMBED_CACHE) == _ROTARY_POS_EMBED_CACHE_MAXSIZE
    _call(rope_sizes=(2, 2, 2), start_frame=0)   # hit -> frame 0 becomes most recent
    _call(rope_sizes=(2, 2, 2), start_frame=99)  # miss -> evicts now-oldest (frame 1)
    surviving = {key[-2] for key in _ROTARY_POS_EMBED_CACHE}
    assert 0 in surviving
    assert 1 not in surviving
