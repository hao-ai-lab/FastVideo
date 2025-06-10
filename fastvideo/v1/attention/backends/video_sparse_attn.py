# SPDX-License-Identifier: Apache-2.0
import math
from dataclasses import dataclass
from typing import Any, List, Optional, Type, cast

import torch
import triton
import triton.language as tl
from einops import rearrange

try:
    from vsa import block_sparse_attn
except ImportError:  # noqa: E722
    block_sparse_attn = None
from typing import Tuple

from fastvideo.v1.attention.backends.abstract import (AttentionBackend,
                                                      AttentionImpl,
                                                      AttentionMetadata,
                                                      AttentionMetadataBuilder)
from fastvideo.v1.distributed import get_sp_group
from fastvideo.v1.fastvideo_args import FastVideoArgs
from fastvideo.v1.logger import init_logger
from fastvideo.v1.pipelines.pipeline_batch_info import ForwardBatch

logger = init_logger(__name__)


class VideoSparseAttentionBackend(AttentionBackend):

    accept_output_buffer: bool = True

    @staticmethod
    def get_supported_head_sizes() -> List[int]:
        return [64, 128]

    @staticmethod
    def get_name() -> str:
        return "VIDEO_SPARSE_ATTN"

    @staticmethod
    def get_impl_cls() -> Type["VideoSparseAttentionImpl"]:
        return VideoSparseAttentionImpl

    @staticmethod
    def get_metadata_cls() -> Type["VideoSparseAttentionMetadata"]:
        return VideoSparseAttentionMetadata

    @staticmethod
    def get_builder_cls() -> Type["VideoSparseAttentionMetadataBuilder"]:
        return VideoSparseAttentionMetadataBuilder


@dataclass
class VideoSparseAttentionMetadata(AttentionMetadata):
    current_timestep: int
    dit_seq_shape: List[int]
    VSA_sparsity: float


class VideoSparseAttentionMetadataBuilder(AttentionMetadataBuilder):

    def __init__(self):
        pass

    def prepare(self):
        pass

    def build(
        self,
        current_timestep: int,
        forward_batch: ForwardBatch,
        fastvideo_args: FastVideoArgs,
    ) -> VideoSparseAttentionMetadata:
        if forward_batch.latents is None:
            raise ValueError("latents cannot be None")

        raw_latent_shape = forward_batch.latents.shape
        patch_size = fastvideo_args.dit_config.patch_size
        dit_seq_shape = [
            raw_latent_shape[2] // patch_size[0],
            raw_latent_shape[3] // patch_size[1],
            raw_latent_shape[4] // patch_size[2]
        ]
        VSA_sparsity = forward_batch.VSA_sparsity
        return VideoSparseAttentionMetadata(current_timestep=current_timestep,
                                            dit_seq_shape=dit_seq_shape,
                                            VSA_sparsity=VSA_sparsity)


class VideoSparseAttentionImpl(AttentionImpl):

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        causal: bool,
        softmax_scale: float,
        num_kv_heads: Optional[int] = None,
        prefix: str = "",
        **extra_impl_args,
    ) -> None:
        self.prefix = prefix
        sp_group = get_sp_group()
        self.sp_size = sp_group.world_size
        self.VSA_base_tile_size = [4, 4, 4]
        self.dit_seq_shape: List[int]
        self.full_window_size: List[int]
        self.img_seq_length: int

    def tile(self, x: torch.Tensor) -> torch.Tensor:
        x = rearrange(x,
                      "b (sp t h w) head d -> b (t sp h w) head d",
                      sp=self.sp_size,
                      t=self.dit_seq_shape[0] // self.sp_size,
                      h=self.dit_seq_shape[1],
                      w=self.dit_seq_shape[2])

        return rearrange(
            x,
            "b (n_t ts_t n_h ts_h n_w ts_w) h d -> b (n_t n_h n_w ts_t ts_h ts_w) h d",
            n_t=self.full_window_size[0],
            n_h=self.full_window_size[1],
            n_w=self.full_window_size[2],
            ts_t=self.VSA_base_tile_size[0],
            ts_h=self.VSA_base_tile_size[1],
            ts_w=self.VSA_base_tile_size[2])

    def untile(self, x: torch.Tensor) -> torch.Tensor:
        x = rearrange(
            x,
            "b (n_t n_h n_w ts_t ts_h ts_w) h d -> b (n_t ts_t n_h ts_h n_w ts_w) h d",
            n_t=self.full_window_size[0],
            n_h=self.full_window_size[1],
            n_w=self.full_window_size[2],
            ts_t=self.VSA_base_tile_size[0],
            ts_h=self.VSA_base_tile_size[1],
            ts_w=self.VSA_base_tile_size[2])
        return rearrange(x,
                         "b (t sp h w) head d -> b (sp t h w) head d",
                         sp=self.sp_size,
                         t=self.dit_seq_shape[0] // self.sp_size,
                         h=self.dit_seq_shape[1],
                         w=self.dit_seq_shape[2])

    def preprocess_qkv(
        self,
        qkv: torch.Tensor,
        attn_metadata: VideoSparseAttentionMetadata,
    ) -> torch.Tensor:
        self.dit_seq_shape = attn_metadata.dit_seq_shape
        self.full_window_size = [
            self.dit_seq_shape[0] // self.VSA_base_tile_size[0],
            self.dit_seq_shape[1] // self.VSA_base_tile_size[1],
            self.dit_seq_shape[2] // self.VSA_base_tile_size[2]
        ]
        self.img_seq_length = math.prod(self.dit_seq_shape)
        return self.tile(qkv)

    def postprocess_output(
        self,
        output: torch.Tensor,
        attn_metadata: VideoSparseAttentionMetadata,
    ) -> torch.Tensor:
        return self.untile(output)

    def forward(  # type: ignore[override]
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        gate_compress: torch.Tensor,
        attn_metadata: VideoSparseAttentionMetadata,
    ) -> torch.Tensor:
        query = query.transpose(1, 2).contiguous()
        key = key.transpose(1, 2).contiguous()
        value = value.transpose(1, 2).contiguous()
        gate_compress = gate_compress.transpose(1, 2).contiguous()

        cur_topk = math.ceil(
            (1 - attn_metadata.VSA_sparsity) *
            (self.img_seq_length / math.prod(self.VSA_base_tile_size)))

        # Cast to Any to bypass type checking for untyped function
        hidden_states = cast(Any, sparse_attn_c_s_p)(
            query,
            key,
            value,
            topk=cur_topk,
            block_size=(4, 4, 4),
            compress_attn_weight=gate_compress).transpose(1, 2)

        return hidden_states


def torch_attention(q, k, v) -> Tuple[torch.Tensor, torch.Tensor]:
    QK = torch.matmul(q, k.transpose(-2, -1))
    QK /= (q.size(-1)**0.5)

    # Causal mask removed since causal is always false

    QK = torch.nn.functional.softmax(QK, dim=-1)
    output = torch.matmul(QK, v)
    return output, QK


def sparse_attn_c_s_p(q, k, v, topk, block_size, compress_attn_weight=None):
    """
    q: [batch_size, num_heads, seq_len, head_dim]
    k: [batch_size, num_heads, seq_len, head_dim]
    v: [batch_size, num_heads, seq_len, head_dim]
    topk: int
    block_size: int or tuple of 3 ints
    video_shape: tuple of (T, H, W)
    compress_attn_weight: [batch_size, num_heads, seq_len, head_dim]
    select_attn_weight: [batch_size, num_heads, seq_len, head_dim]
    
    V1 of sparse attention. Include compress attn and sparse attn branch, use average pooling to compress. 
    Assume q, k, v is flattened in this way: [batch_size, num_heads, T//block_size[0], H//block_size[1], W//block_size[2], block_size[0], block_size[1], block_size[2]]
    """

    if isinstance(block_size, int):
        block_size = (block_size, block_size, block_size)

    block_elements = block_size[0] * block_size[1] * block_size[2]
    assert block_elements % 64 == 0 and block_elements >= 64
    assert q.shape[2] % block_elements == 0
    batch_size, num_heads, seq_len, head_dim = q.shape
    # compress attn
    q_compress = q.view(batch_size, num_heads, seq_len // block_elements,
                        block_elements, head_dim).mean(dim=3)
    k_compress = k.view(batch_size, num_heads, seq_len // block_elements,
                        block_elements, head_dim).mean(dim=3)
    v_compress = v.view(batch_size, num_heads, seq_len // block_elements,
                        block_elements, head_dim).mean(dim=3)

    output_compress, block_attn_score = torch_attention(q_compress, k_compress,
                                                        v_compress)

    output_compress = output_compress.view(batch_size, num_heads,
                                           seq_len // block_elements, 1,
                                           head_dim)
    output_compress = output_compress.repeat(1, 1, 1, block_elements,
                                             1).view(batch_size, num_heads,
                                                     seq_len, head_dim)

    q2k_block_sparse_index, q2k_block_sparse_num, k2q_block_sparse_index, k2q_block_sparse_num = generate_topk_block_sparse_pattern(
        block_attn_score, topk)

    output_select = block_sparse_attn(q, k, v, q2k_block_sparse_index,
                                      q2k_block_sparse_num,
                                      k2q_block_sparse_index,
                                      k2q_block_sparse_num)

    if compress_attn_weight is not None:
        final_output = output_compress * compress_attn_weight + output_select
    else:
        final_output = output_compress + output_select
    return final_output


@triton.jit
def topk_index_to_map_kernel(
    map_ptr,
    index_ptr,
    map_bs_stride,
    map_h_stride,
    map_q_stride,
    map_kv_stride,
    index_bs_stride,
    index_h_stride,
    index_q_stride,
    index_kv_stride,
    topk: tl.constexpr,
):
    b, h, q = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    index_ptr_base = index_ptr + b * index_bs_stride + h * index_h_stride + q * index_q_stride
    map_ptr_base = map_ptr + b * map_bs_stride + h * map_h_stride + q * map_q_stride

    for i in tl.static_range(topk):
        index = tl.load(index_ptr_base + i * index_kv_stride)
        tl.store(map_ptr_base + index * map_kv_stride, 1.0)


@triton.jit
def map_to_index_kernel(
    map_ptr,
    index_ptr,
    index_num_ptr,
    map_bs_stride,
    map_h_stride,
    map_q_stride,
    map_kv_stride,
    index_bs_stride,
    index_h_stride,
    index_q_stride,
    index_kv_stride,
    index_num_bs_stride,
    index_num_h_stride,
    index_num_q_stride,
    num_kv_blocks: tl.constexpr,
):
    b, h, q = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    index_ptr_base = index_ptr + b * index_bs_stride + h * index_h_stride + q * index_q_stride
    map_ptr_base = map_ptr + b * map_bs_stride + h * map_h_stride + q * map_q_stride

    num = 0
    for i in tl.static_range(num_kv_blocks):
        map_entry = tl.load(map_ptr_base + i * map_kv_stride)
        if map_entry:
            tl.store(index_ptr_base + num * index_kv_stride, i)
            num += 1

    tl.store(
        index_num_ptr + b * index_num_bs_stride + h * index_num_h_stride +
        q * index_num_q_stride, num)


def topk_index_to_map(index: torch.Tensor,
                      num_kv_blocks: int,
                      transpose_map: bool = False):
    """
    Convert topk indices to a map.
    
    Args:
        index: [bs, h, num_q_blocks, topk]
            The topk indices tensor.
        num_kv_blocks: int
            The number of key-value blocks in the block_map returned
        transpose_map: bool
            If True, the block_map will be transposed on the final two dimensions.
    
    Returns:
        block_map: [bs, h, num_q_blocks, num_kv_blocks]
            A binary map where 1 indicates that the q block attends to the kv block.
    """
    bs, h, num_q_blocks, topk = index.shape

    if transpose_map is False:
        block_map = torch.zeros((bs, h, num_q_blocks, num_kv_blocks),
                                dtype=torch.bool,
                                device=index.device)
    else:
        block_map = torch.zeros((bs, h, num_kv_blocks, num_q_blocks),
                                dtype=torch.bool,
                                device=index.device)
        block_map = block_map.transpose(2, 3)

    grid = (bs, h, num_q_blocks)
    topk_index_to_map_kernel[grid](
        block_map,
        index,
        block_map.stride(0),
        block_map.stride(1),
        block_map.stride(2),
        block_map.stride(3),
        index.stride(0),
        index.stride(1),
        index.stride(2),
        index.stride(3),
        topk=topk,
    )

    return block_map


def map_to_index(block_map: torch.Tensor):
    """
    Convert a block map to indices and counts.
    
    Args:
        block_map: [bs, h, num_q_blocks, num_kv_blocks]
            The block map tensor.
    
    Returns:
        index: [bs, h, num_q_blocks, num_kv_blocks]
            The indices of the blocks.
        index_num: [bs, h, num_q_blocks]
            The number of blocks for each q block.
    """
    bs, h, num_q_blocks, num_kv_blocks = block_map.shape

    index = torch.full((block_map.shape),
                       -1,
                       dtype=torch.int32,
                       device=block_map.device)
    index_num = torch.empty((bs, h, num_q_blocks),
                            dtype=torch.int32,
                            device=block_map.device)

    grid = (bs, h, num_q_blocks)
    map_to_index_kernel[grid](
        block_map,
        index,
        index_num,
        block_map.stride(0),
        block_map.stride(1),
        block_map.stride(2),
        block_map.stride(3),
        index.stride(0),
        index.stride(1),
        index.stride(2),
        index.stride(3),
        index_num.stride(0),
        index_num.stride(1),
        index_num.stride(2),
        num_kv_blocks=num_kv_blocks,
    )

    return index, index_num


def generate_topk_block_sparse_pattern(block_attn_score: torch.Tensor,
                                       topk: int):
    """
    Generate a block sparse pattern where each q block attends to exactly topk kv blocks,
    based on the provided attention scores.
    
    Args:
        block_attn_score: [bs, h, num_q_blocks, num_kv_blocks]
            Attention scores between query and key blocks
        topk: int
            Number of kv blocks each q block attends to
        
    Returns:
        q2k_block_sparse_index: [bs, h, num_q_blocks, topk]
            Contains the indices of kv blocks that each q block attends to.
        q2k_block_sparse_num: [bs, h, num_q_blocks]
            Contains the number of kv blocks that each q block attends to (all equal to topk).
        k2q_block_sparse_index: [bs, h, num_kv_blocks, max_q_per_kv]
            Contains the indices of q blocks that attend to each kv block.
        k2q_block_sparse_num: [bs, h, num_kv_blocks]
            Contains the number of q blocks that attend to each kv block.
    """
    device = block_attn_score.device
    # Extract dimensions from block_attn_score
    bs, h, num_q_blocks, num_kv_blocks = block_attn_score.shape

    sorted_result = torch.sort(block_attn_score, dim=-1, descending=True)

    sorted_indice = sorted_result.indices

    q2k_block_sparse_index, _ = torch.sort(sorted_indice[:, :, :, :topk],
                                           dim=-1)
    q2k_block_sparse_index = q2k_block_sparse_index.to(dtype=torch.int32)
    q2k_block_sparse_num = torch.full((bs, h, num_q_blocks),
                                      topk,
                                      device=device,
                                      dtype=torch.int32)

    block_map = topk_index_to_map(q2k_block_sparse_index,
                                  num_kv_blocks,
                                  transpose_map=True)
    k2q_block_sparse_index, k2q_block_sparse_num = map_to_index(
        block_map.transpose(2, 3))

    return q2k_block_sparse_index, q2k_block_sparse_num, k2q_block_sparse_index, k2q_block_sparse_num
