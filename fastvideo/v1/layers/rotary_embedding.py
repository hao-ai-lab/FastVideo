# SPDX-License-Identifier: Apache-2.0
# Adapted from vllm: https://github.com/vllm-project/vllm/blob/v0.7.3/vllm/model_executor/layers/rotary_embedding.py

# Adapted from
# https://github.com/huggingface/transformers/blob/v4.33.2/src/transformers/models/llama/modeling_llama.py
# Copyright 2023 The vLLM team.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Rotary Positional Embeddings."""
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

from fastvideo.v1.distributed.parallel_state import get_sp_group
from fastvideo.v1.layers.custom_op import CustomOp


def _rotate_neox(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def _rotate_gptj(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    x = torch.stack((-x2, x1), dim=-1)
    return x.flatten(-2)


def _apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    is_neox_style: bool,
) -> torch.Tensor:
    """
    Args:
        x: [num_tokens, num_heads, head_size]
        cos: [num_tokens, head_size // 2]
        sin: [num_tokens, head_size // 2]
        is_neox_style: Whether to use the Neox-style or GPT-J-style rotary
            positional embeddings.
    """
    # cos = cos.unsqueeze(-2).to(x.dtype)
    # sin = sin.unsqueeze(-2).to(x.dtype)
    cos = cos.unsqueeze(-2)
    sin = sin.unsqueeze(-2)
    if is_neox_style:
        x1, x2 = torch.chunk(x, 2, dim=-1)
    else:
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
    o1 = (x1.float() * cos - x2.float() * sin).type_as(x)
    o2 = (x2.float() * cos + x1.float() * sin).type_as(x)
    if is_neox_style:
        return torch.cat((o1, o2), dim=-1)
    else:
        return torch.stack((o1, o2), dim=-1).flatten(-2)


@CustomOp.register("rotary_embedding")
class RotaryEmbedding(CustomOp):
    """Original rotary positional embedding."""

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: Union[int, float],
        is_neox_style: bool,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.head_size = head_size
        self.rotary_dim = rotary_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.is_neox_style = is_neox_style
        self.dtype = dtype

        cache = self._compute_cos_sin_cache()
        cache = cache.to(dtype)
        self.cos_sin_cache: torch.Tensor
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    def _compute_inv_freq(self, base: Union[int, float]) -> torch.Tensor:
        """Compute the inverse frequency."""
        # NOTE(woosuk): To exactly match the HF implementation, we need to
        # use CPU to compute the cache and then move it to GPU. However, we
        # create the cache on GPU for faster initialization. This may cause
        # a slight numerical difference between the HF implementation and ours.
        inv_freq = 1.0 / (base**(torch.arange(
            0, self.rotary_dim, 2, dtype=torch.float) / self.rotary_dim))
        return inv_freq

    def _compute_cos_sin_cache(self) -> torch.Tensor:
        """Compute the cos and sin cache."""
        inv_freq = self._compute_inv_freq(self.base)
        t = torch.arange(self.max_position_embeddings, dtype=torch.float)

        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        cache = torch.cat((cos, sin), dim=-1)
        return cache

    def forward_native(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        offsets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """A PyTorch-native implementation of forward()."""
        if offsets is not None:
            positions = positions + offsets
        positions = positions.flatten()
        num_tokens = positions.shape[0]
        cos_sin = self.cos_sin_cache.index_select(0, positions)
        cos, sin = cos_sin.chunk(2, dim=-1)

        query_shape = query.shape
        query = query.view(num_tokens, -1, self.head_size)
        query_rot = query[..., :self.rotary_dim]
        query_pass = query[..., self.rotary_dim:]
        query_rot = _apply_rotary_emb(query_rot, cos, sin, self.is_neox_style)
        query = torch.cat((query_rot, query_pass), dim=-1).reshape(query_shape)

        key_shape = key.shape
        key = key.view(num_tokens, -1, self.head_size)
        key_rot = key[..., :self.rotary_dim]
        key_pass = key[..., self.rotary_dim:]
        key_rot = _apply_rotary_emb(key_rot, cos, sin, self.is_neox_style)
        key = torch.cat((key_rot, key_pass), dim=-1).reshape(key_shape)
        return query, key

    def extra_repr(self) -> str:
        s = f"head_size={self.head_size}, rotary_dim={self.rotary_dim}"
        s += f", max_position_embeddings={self.max_position_embeddings}"
        s += f", base={self.base}, is_neox_style={self.is_neox_style}"
        return s


def _to_tuple(x: Union[int, Tuple[int, ...]], dim: int = 2) -> Tuple[int, ...]:
    if isinstance(x, int):
        return (x, ) * dim
    elif len(x) == dim:
        return x
    else:
        raise ValueError(f"Expected length {dim} or int, but got {x}")


def get_meshgrid_nd(start: Union[int, Tuple[int, ...]],
                    *args: Union[int, Tuple[int, ...]],
                    dim: int = 2) -> torch.Tensor:
    """
    Get n-D meshgrid with start, stop and num.

    Args:
        start (int or tuple): If len(args) == 0, start is num; If len(args) == 1, start is start, args[0] is stop,
            step is 1; If len(args) == 2, start is start, args[0] is stop, args[1] is num. For n-dim, start/stop/num
            should be int or n-tuple. If n-tuple is provided, the meshgrid will be stacked following the dim order in
            n-tuples.
        *args: See above.
        dim (int): Dimension of the meshgrid. Defaults to 2.

    Returns:
        grid (np.ndarray): [dim, ...]
    """
    if len(args) == 0:
        # start is grid_size
        num = _to_tuple(start, dim=dim)
        start = (0, ) * dim
        stop = num
    elif len(args) == 1:
        # start is start, args[0] is stop, step is 1
        start = _to_tuple(start, dim=dim)
        stop = _to_tuple(args[0], dim=dim)
        num = tuple(stop[i] - start[i] for i in range(dim))
    elif len(args) == 2:
        # start is start, args[0] is stop, args[1] is num
        start = _to_tuple(start, dim=dim)  # Left-Top       eg: 12,0
        stop = _to_tuple(args[0], dim=dim)  # Right-Bottom   eg: 20,32
        num = _to_tuple(args[1], dim=dim)  # Target Size    eg: 32,124
    else:
        raise ValueError(f"len(args) should be 0, 1 or 2, but got {len(args)}")

    # PyTorch implement of np.linspace(start[i], stop[i], num[i], endpoint=False)
    axis_grid = []
    for i in range(dim):
        a, b, n = start[i], stop[i], num[i]
        g = torch.linspace(a, b, n + 1, dtype=torch.float32)[:n]
        axis_grid.append(g)
    grid = torch.meshgrid(*axis_grid, indexing="ij")  # dim x [W, H, D]
    grid = torch.stack(grid, dim=0)  # [dim, W, H, D]

    return grid


def get_1d_rotary_pos_embed(
    dim: int,
    pos: Union[torch.FloatTensor, int],
    theta: float = 10000.0,
    theta_rescale_factor: float = 1.0,
    interpolation_factor: float = 1.0,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Precompute the frequency tensor for complex exponential (cis) with given dimensions.
    (Note: `cis` means `cos + i * sin`, where i is the imaginary unit.)

    This function calculates a frequency tensor with complex exponential using the given dimension 'dim'
    and the end index 'end'. The 'theta' parameter scales the frequencies.

    Args:
        dim (int): Dimension of the frequency tensor.
        pos (int or torch.FloatTensor): Position indices for the frequency tensor. [S] or scalar
        theta (float, optional): Scaling factor for frequency computation. Defaults to 10000.0.
        theta_rescale_factor (float, optional): Rescale factor for theta. Defaults to 1.0.
        interpolation_factor (float, optional): Factor to scale positions. Defaults to 1.0.

    Returns:
        freqs_cos, freqs_sin: Precomputed frequency tensor with real and imaginary parts separately. [S, D]
    """
    if isinstance(pos, int):
        pos = torch.arange(pos).float()

    # proposed by reddit user bloc97, to rescale rotary embeddings to longer sequence length without fine-tuning
    # has some connection to NTK literature
    if theta_rescale_factor != 1.0:
        theta *= theta_rescale_factor**(dim / (dim - 2))

    freqs = 1.0 / (theta**(torch.arange(0, dim, 2)[:(dim // 2)].to(dtype) / dim)
                   )  # [D/2]
    freqs = torch.outer(pos * interpolation_factor, freqs)  # [S, D/2]
    freqs_cos = freqs.cos()  # [S, D/2]
    freqs_sin = freqs.sin()  # [S, D/2]
    return freqs_cos, freqs_sin


def get_nd_rotary_pos_embed(
    rope_dim_list,
    start,
    *args,
    theta=10000.0,
    theta_rescale_factor: Union[float, List[float]] = 1.0,
    interpolation_factor: Union[float, List[float]] = 1.0,
    shard_dim: int = 0,
    sp_rank: int = 0,
    sp_world_size: int = 1,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    This is a n-d version of precompute_freqs_cis, which is a RoPE for tokens with n-d structure.
    Supports sequence parallelism by allowing sharding of a specific dimension.

    Args:
        rope_dim_list (list of int): Dimension of each rope. len(rope_dim_list) should equal to n.
            sum(rope_dim_list) should equal to head_dim of attention layer.
        start (int | tuple of int | list of int): If len(args) == 0, start is num; If len(args) == 1, start is start,
            args[0] is stop, step is 1; If len(args) == 2, start is start, args[0] is stop, args[1] is num.
        *args: See above.
        theta (float): Scaling factor for frequency computation. Defaults to 10000.0.
        theta_rescale_factor (float): Rescale factor for theta. Defaults to 1.0.
        interpolation_factor (float): Factor to scale positions. Defaults to 1.0.
        shard_dim (int): Which dimension to shard for sequence parallelism. Defaults to 0.
        sp_rank (int): Rank in the sequence parallel group. Defaults to 0.
        sp_world_size (int): World size of the sequence parallel group. Defaults to 1.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: (cos, sin) tensors of shape [HW, D/2]
    """
    # Get the full grid
    full_grid = get_meshgrid_nd(
        start, *args, dim=len(rope_dim_list))  # [3, W, H, D] / [2, W, H]

    # Shard the grid if using sequence parallelism (sp_world_size > 1)
    assert shard_dim < len(
        rope_dim_list
    ), f"shard_dim {shard_dim} must be less than number of dimensions {len(rope_dim_list)}"
    if sp_world_size > 1:
        # Get the shape of the full grid
        grid_shape = list(full_grid.shape[1:])

        # Ensure the dimension to shard is divisible by sp_world_size
        assert grid_shape[shard_dim] % sp_world_size == 0, (
            f"Dimension {shard_dim} with size {grid_shape[shard_dim]} is not divisible "
            f"by sequence parallel world size {sp_world_size}")

        # Compute the start and end indices for this rank's shard
        shard_size = grid_shape[shard_dim] // sp_world_size
        start_idx = sp_rank * shard_size
        end_idx = (sp_rank + 1) * shard_size

        # Create slicing indices for each dimension
        slice_indices = [slice(None) for _ in range(len(grid_shape))]
        slice_indices[shard_dim] = slice(start_idx, end_idx)

        # Shard the grid
        # Update grid shape for the sharded dimension
        grid_shape[shard_dim] = grid_shape[shard_dim] // sp_world_size
        grid = torch.empty((len(rope_dim_list), ) + tuple(grid_shape),
                           dtype=full_grid.dtype)
        for i in range(len(rope_dim_list)):
            grid[i] = full_grid[i][tuple(slice_indices)]
    else:
        grid = full_grid

    if isinstance(theta_rescale_factor, (int, float)):
        theta_rescale_factor = [theta_rescale_factor] * len(rope_dim_list)
    elif isinstance(theta_rescale_factor,
                    list) and len(theta_rescale_factor) == 1:
        theta_rescale_factor = [theta_rescale_factor[0]] * len(rope_dim_list)
    assert len(theta_rescale_factor) == len(
        rope_dim_list
    ), "len(theta_rescale_factor) should equal to len(rope_dim_list)"

    if isinstance(interpolation_factor, (int, float)):
        interpolation_factor = [interpolation_factor] * len(rope_dim_list)
    elif isinstance(interpolation_factor,
                    list) and len(interpolation_factor) == 1:
        interpolation_factor = [interpolation_factor[0]] * len(rope_dim_list)
    assert len(interpolation_factor) == len(
        rope_dim_list
    ), "len(interpolation_factor) should equal to len(rope_dim_list)"

    # use 1/ndim of dimensions to encode grid_axis
    embs = []
    for i in range(len(rope_dim_list)):
        emb = get_1d_rotary_pos_embed(
            rope_dim_list[i],
            grid[i].reshape(-1),
            theta,
            theta_rescale_factor=theta_rescale_factor[i],
            interpolation_factor=interpolation_factor[i],
            dtype=dtype,
        )  # 2 x [WHD, rope_dim_list[i]]
        embs.append(emb)

    cos = torch.cat([emb[0] for emb in embs], dim=1)  # (WHD, D/2)
    sin = torch.cat([emb[1] for emb in embs], dim=1)  # (WHD, D/2)
    return cos, sin


def get_rotary_pos_embed(
    rope_sizes,
    hidden_size,
    heads_num,
    rope_dim_list,
    rope_theta,
    theta_rescale_factor=1.0,
    interpolation_factor=1.0,
    shard_dim: int = 0,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate rotary positional embeddings for the given sizes.
    
    Args:
        rope_sizes: Tuple of dimensions (t, h, w)
        hidden_size: Hidden dimension size
        heads_num: Number of attention heads
        rope_dim_list: List of dimensions for each axis, or None
        rope_theta: Base for frequency calculations
        theta_rescale_factor: Rescale factor for theta. Defaults to 1.0
        interpolation_factor: Factor to scale positions. Defaults to 1.0
        shard_dim: Which dimension to shard for sequence parallelism. Defaults to 0.
        
    Returns:
        Tuple of (cos, sin) tensors for rotary embeddings
    """

    target_ndim = 3
    head_dim = hidden_size // heads_num

    if rope_dim_list is None:
        rope_dim_list = [head_dim // target_ndim for _ in range(target_ndim)]

    assert sum(
        rope_dim_list
    ) == head_dim, "sum(rope_dim_list) should equal to head_dim of attention layer"

    # Get SP info
    sp_group = get_sp_group()
    sp_rank = sp_group.rank_in_group
    sp_world_size = sp_group.world_size

    freqs_cos, freqs_sin = get_nd_rotary_pos_embed(
        rope_dim_list,
        rope_sizes,
        theta=rope_theta,
        theta_rescale_factor=theta_rescale_factor,
        interpolation_factor=interpolation_factor,
        shard_dim=shard_dim,
        sp_rank=sp_rank,
        sp_world_size=sp_world_size,
        dtype=dtype,
    )
    return freqs_cos, freqs_sin


_ROPE_DICT: Dict[Tuple, RotaryEmbedding] = {}


def get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: Union[int, float],
    is_neox_style: bool = True,
    rope_scaling: Optional[Dict[str, Any]] = None,
    dtype: Optional[torch.dtype] = None,
    partial_rotary_factor: float = 1.0,
) -> RotaryEmbedding:
    if dtype is None:
        dtype = torch.get_default_dtype()
    if rope_scaling is not None:
        # Transforms every value that is a list into a tuple for caching calls
        rope_scaling_tuple = {
            k: tuple(v) if isinstance(v, list) else v
            for k, v in rope_scaling.items()
        }
        rope_scaling_args = tuple(rope_scaling_tuple.items())
    else:
        rope_scaling_args = None
    if partial_rotary_factor < 1.0:
        rotary_dim = int(rotary_dim * partial_rotary_factor)
    key = (head_size, rotary_dim, max_position, base, is_neox_style,
           rope_scaling_args, dtype)
    if key in _ROPE_DICT:
        return _ROPE_DICT[key]

    if rope_scaling is None:
        rotary_emb = RotaryEmbedding(head_size, rotary_dim, max_position, base,
                                     is_neox_style, dtype)
    else:
        raise ValueError(f"Unknown RoPE scaling {rope_scaling}")
    _ROPE_DICT[key] = rotary_emb
    return rotary_emb
