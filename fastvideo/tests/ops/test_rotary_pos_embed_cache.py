# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the get_rotary_pos_embed memoization cache."""

import pytest
import torch

from fastvideo.layers.rotary_embedding import (
    _ROTARY_POS_EMBED_CACHE,
    _ROTARY_POS_EMBED_CACHE_MAXSIZE,
    _ROTARY_POS_EMBED_DEVICE_CACHE,
    _ROTARY_POS_EMBED_DEVICE_CACHE_MAXSIZE,
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
    """Isolate every test by clearing the module-level caches around it."""
    _ROTARY_POS_EMBED_CACHE.clear()
    _ROTARY_POS_EMBED_DEVICE_CACHE.clear()
    yield
    _ROTARY_POS_EMBED_CACHE.clear()
    _ROTARY_POS_EMBED_DEVICE_CACHE.clear()


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
def test_cached_matches_fresh_recompute(rope_sizes, hidden_size, heads_num, dtype, use_real):
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
        ({
            "rope_sizes": (21, 30, 52)
        }, {
            "rope_sizes": (21, 45, 80)
        }),
        ({
            "dtype": torch.float64
        }, {
            "dtype": torch.float32
        }),
        ({
            "use_real": True
        }, {
            "use_real": False
        }),
        ({
            "start_frame": 0
        }, {
            "start_frame": 3
        }),
        ({
            "rope_theta": 10000.0
        }, {
            "rope_theta": 5000.0
        }),
        ({
            "shard_dim": 0
        }, {
            "shard_dim": 1
        }),
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
    _call(rope_sizes=(2, 2, 2), start_frame=0)  # hit -> frame 0 becomes most recent
    _call(rope_sizes=(2, 2, 2), start_frame=99)  # miss -> evicts now-oldest (frame 1)
    surviving = {key[-2] for key in _ROTARY_POS_EMBED_CACHE}
    assert 0 in surviving
    assert 1 not in surviving


_DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("device", _DEVICES)
def test_device_tables_match_caller_side_move_and_cast(device):
    """device/output_dtype output is bitwise-equal to the old caller-side .to(device).float()."""
    host_cos, host_sin = _call(rope_sizes=(21, 45, 80), hidden_size=5120, heads_num=40)
    cos, sin = _call(rope_sizes=(21, 45, 80),
                     hidden_size=5120,
                     heads_num=40,
                     device=device,
                     output_dtype=torch.float32)
    assert cos.device.type == device and cos.dtype == torch.float32
    assert torch.equal(cos.cpu(), host_cos.to(device).float().cpu())
    assert torch.equal(sin.cpu(), host_sin.to(device).float().cpu())


@pytest.mark.parametrize("device", _DEVICES)
def test_device_repeated_call_hits_cache(device):
    """A second identical device call returns the same tensors without re-uploading."""
    cos1, sin1 = _call(device=device, output_dtype=torch.float32)
    cos2, sin2 = _call(device=device, output_dtype=torch.float32)
    assert cos1 is cos2 and sin1 is sin2
    assert len(_ROTARY_POS_EMBED_DEVICE_CACHE) == 1


def test_device_miss_reuses_host_entry():
    """Device lookups share the host table instead of recomputing it."""
    host_cos, _ = _call()
    _call(device="cpu", output_dtype=torch.float32)
    assert len(_ROTARY_POS_EMBED_CACHE) == 1
    assert _call()[0] is host_cos


def test_device_without_output_dtype_keeps_dtype():
    """output_dtype=None leaves the tables in the computed dtype."""
    cos, _ = _call(dtype=torch.float64, device="cpu")
    assert cos.dtype == torch.float64


def test_output_dtype_without_device_raises():
    """output_dtype only applies to device tables."""
    with pytest.raises(ValueError):
        _call(output_dtype=torch.float32)


def test_distinct_output_dtypes_create_distinct_device_entries():
    """Each output dtype is cached separately."""
    _call(device="cpu", output_dtype=torch.float32)
    _call(device="cpu", output_dtype=torch.bfloat16)
    assert len(_ROTARY_POS_EMBED_DEVICE_CACHE) == 2


def test_device_cache_is_bounded_and_evicts_oldest():
    """The device cache caps at its own, smaller, max size."""
    assert _ROTARY_POS_EMBED_DEVICE_CACHE_MAXSIZE <= _ROTARY_POS_EMBED_CACHE_MAXSIZE
    overshoot = _ROTARY_POS_EMBED_DEVICE_CACHE_MAXSIZE + 3
    for frame in range(overshoot):
        _call(rope_sizes=(2, 2, 2), start_frame=frame, device="cpu", output_dtype=torch.float32)
        assert len(_ROTARY_POS_EMBED_DEVICE_CACHE) <= _ROTARY_POS_EMBED_DEVICE_CACHE_MAXSIZE
    surviving = {key[0][-2] for key in _ROTARY_POS_EMBED_DEVICE_CACHE}  # start_frame slot
    assert overshoot - 1 in surviving
    assert 0 not in surviving


def test_inference_mode_tables_are_not_reused_for_autograd():
    """A table cached under inference_mode must not leak into a training forward."""
    with torch.inference_mode():
        inference_cos, _ = _call(device="cpu", output_dtype=torch.float32)
    assert inference_cos.is_inference()

    cos, _ = _call(device="cpu", output_dtype=torch.float32)
    assert cos is not inference_cos and not cos.is_inference()
    assert len(_ROTARY_POS_EMBED_DEVICE_CACHE) == 2
    # Mirrors _apply_rotary_emb: multiplying a grad-requiring input by the table
    # saves the table for backward, which fails for inference tensors.
    x = torch.randn(cos.shape, requires_grad=True)
    (x * cos).sum().backward()
    assert x.grad is not None
