# -*- coding: utf-8 -*-

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Optional

import torch
from einops import rearrange
from magi_attention.common import AttnRange, AttnRanges
from magi_attention.functional import flex_flash_attn_func
from odin.common.parallel_state import ParallelState

from odin.functional.block_pooling import block_pooling_triton


@dataclass
class SparseAttnKey:

    q_ranges: AttnRanges = None
    k_ranges: AttnRanges = None
    block_ranges: AttnRanges = AttnRanges()

    block_size: int = 1
    q_block_size: int = -1  # -1 means equal to block_size
    block_topk: int = 16
    win_size: int = 1

    cmp_method: str = "mean"
    mask_overlap: bool = False

    use_slt_attn: bool = True
    use_cmp_attn: bool = False
    use_win_attn: bool = False
    use_nsa: bool = False
    use_gating: bool = False
    gating_method: str = "sigmoid"

    # FIXME: for debug, remove it asap
    parallel_state: ParallelState = None
    cp_pad_size: int = 0
    deterministic: bool = False

    def _reduce_range(self, attn_range: AttnRange):
        ranges_tensors = self.block_ranges.to_tensor()
        indices_start = torch.where(ranges_tensors[:, 0] == attn_range.start)[0]
        indices_end = torch.where(ranges_tensors[:, 1] == attn_range.end)[0]
        assert len(indices_start) == 1, "len(indices_start) != 1"
        assert len(indices_end) == 1, "len(indices_end) != 1"
        return indices_start[0].item(), indices_end[0].item() + 1

    def __post_init__(self):

        self.cp_process_group = self.parallel_state.get_process_group("cp")
        self.device = torch.cuda.current_device()
        self.q_block_size = (
            self.block_size if self.q_block_size == -1 else self.q_block_size
        )
        assert self.block_ranges.max_seqlen <= min(
            self.q_block_size, self.block_size
        ), f"{self.block_ranges.max_seqlen=} should be no larger than min({self.q_block_size=}, {self.block_size=})"
        assert (
            self.q_block_size == self.block_size
        ), "only support q_block_size == block_size"
        self.max_seqlen_q = self.q_ranges.max_seqlen
        self.max_seqlen_k = self.k_ranges.max_seqlen

        # init win_q_ranges, win_k_ranges
        self.win_q_ranges = AttnRanges()
        self.win_k_ranges = AttnRanges()
        cmp_q_ranges_list = []
        cmp_k_ranges_list = []
        # init
        self.map_cmp_q_range_to_cmp_k_ranges: defaultdict[
            AttnRange, AttnRanges
        ] = defaultdict(AttnRanges)
        self.map_cmp_q_range_to_q_range: defaultdict[
            AttnRange, AttnRange
        ] = defaultdict(AttnRange)
        for q_range, k_range in zip(self.q_ranges, self.k_ranges):
            # aggregate cmp_k_ranges by q_range
            q_start, q_end = self._reduce_range(q_range)
            k_start, k_end = self._reduce_range(k_range)
            cmp_q_range = AttnRange(start=q_start, end=q_end)

            self.map_cmp_q_range_to_cmp_k_ranges[cmp_q_range].append(
                AttnRange(start=k_start, end=k_end)
            )
            self.map_cmp_q_range_to_q_range[cmp_q_range] = q_range

            cmp_q_ranges_list.append([q_start, q_end])
            cmp_k_ranges_list.append([k_start, k_end])

            # construct win_q_ranges, win_k_ranges, invalid for wanx
            if q_range.end == k_range.end:
                chunk_seq_lens = q_range.end - q_range.start
                win_k_range = AttnRange(
                    start=max(
                        k_range.end - self.win_size * chunk_seq_lens, k_range.start
                    ),
                    end=k_range.end,
                )
                self.win_q_ranges.append(q_range)
                self.win_k_ranges.append(win_k_range)

        assert len(self.map_cmp_q_range_to_cmp_k_ranges) == 1, (
            "FIXME: For now, nsa_v2 only supports one cmp_q_range_to_cmp_k_ranges, "
            "due to for-loop cmp_q block pooling is not supported yet"
        )

        # init args for win attn
        self.win_q_ranges_tensor = self.win_q_ranges.to_tensor(device=self.device)
        self.win_k_ranges_tensor = self.win_k_ranges.to_tensor(device=self.device)
        self.max_seqlen_win_q = self.win_q_ranges.max_seqlen
        self.max_seqlen_win_k = self.win_k_ranges.max_seqlen

        # init args for cmp attn
        self.q_ranges_tensor = self.q_ranges.to_tensor(self.device)
        self.cmp_q_ranges = AttnRanges.from_ranges(cmp_q_ranges_list)
        self.max_seqlen_cmp_q = self.cmp_q_ranges.max_seqlen
        self.cmp_q_ranges_tensor = self.cmp_q_ranges.to_tensor(self.device)
        self.cmp_k_ranges = AttnRanges.from_ranges(cmp_k_ranges_list)
        self.max_seqlen_cmp_k = self.cmp_k_ranges.max_seqlen
        self.cmp_k_ranges_tensor = self.cmp_k_ranges.to_tensor(self.device)
        self.block_ranges_tensor = self.block_ranges.to_tensor(self.device)
        self.block_sizes_tensor = (
            self.block_ranges_tensor[:, 1] - self.block_ranges_tensor[:, 0]
        )
        self.cu_block_sizes_list_tensor = torch.cat(
            [self.block_ranges_tensor[0:1, 0], self.block_ranges_tensor[:, 1]]
        )
        self.block_ranges_start_to_idx_map = {
            start.item(): i for i, start in enumerate(self.block_ranges_tensor[:, 0])
        }

    def calc_attn_meta(
        self,
        total_seqlen_k: int,
        total_seqlen_cmp_k: int,
        q: torch.Tensor | None = None,
        cmp_k: torch.Tensor | None = None,
        full_cmp_attn_score: torch.Tensor | None = None,
        softmax_scale: float | None = None,
    ):
        has_full_cmp_attn_score = full_cmp_attn_score is not None
        if not has_full_cmp_attn_score:
            assert q is not None and cmp_k is not None and softmax_scale is not None, (
                "If the full_cmp_attn_score is not provided, "
                "q, cmp_k, and softmax_scale must be given to compute attn scores."
            )

        # init max_seqlen_slt
        # NOTE: this should be the max block size in block_ranges
        max_seqlen_slt_q = self.q_block_size
        max_seqlen_slt_k = self.block_size

        # init cmp_mask buffer
        cmp_mask_buffer = torch.zeros(
            (self.max_seqlen_q, total_seqlen_cmp_k),
            dtype=torch.bool,
            device=self.device,
        )
        # since the whole attn score with shape [nhq, total_seqlen_q, total_seqlen_cmp_k] is too large
        # we fall back to for-loop each q_range, and calculate the attn score
        # with shape [nhq, seqlen_q, total_seqlen_cmp_k]
        flat_q_indices_list: list[torch.Tensor] = []
        flat_k_indices_list: list[torch.Tensor] = []
        for idx, (cmp_q_range, cmp_k_ranges) in enumerate(
            self.map_cmp_q_range_to_cmp_k_ranges.items()
        ):
            q_range = self.map_cmp_q_range_to_q_range[cmp_q_range]
            if has_full_cmp_attn_score:
                cmp_attn_score = full_cmp_attn_score[  # type: ignore[index]
                    :, cmp_q_range.start : cmp_q_range.end
                ]
            else:
                # get original unnormalized attn score
                # with shape: [nhq, seqlen_q, total_seqlen_cmp_k]
                q_chunk = q[q_range.start : q_range.end]  # type: ignore[index]
                # FIXME: this pooling is only accurate when fop-loop iters == 1
                # which had better optimize to an non for-loop version,
                # instead of computing the pooling args for each cmp_q_range
                q_cmp = block_pooling_triton(q_chunk, self.cu_block_sizes_list_tensor)
                cmp_attn_score = (
                    torch.einsum("qhd,khd->hqk", q_cmp, cmp_k) * softmax_scale
                )
            # get cmp mask of this q_range, broadcastable to each head
            # with shape: [1, seqlen_q, total_seqlen_cmp_k]

            cmp_mask = self._set_cmp_mask(
                k_ranges=cmp_k_ranges,
                total_seqlen_q=cmp_q_range.seqlen,
                total_seqlen_k=total_seqlen_cmp_k,
                device=self.device,
                mask=cmp_mask_buffer,
                need_zero=idx > 0,
            ).unsqueeze(0)

            # mask the attn score and normalize with softmax
            # with shape: [nhq, seqlen_q, total_seqlen_cmp_k]
            full_block_nums = cmp_k_ranges.total_seqlen
            if full_block_nums == 0:
                cmp_attn_score.fill_(0.0)
            else:
                cmp_attn_score = cmp_attn_score.masked_fill(
                    cmp_mask.logical_not(), float("-inf")
                )
            slt_k_num = min(full_block_nums, self.block_topk)

            (
                flat_q_indices,
                flat_k_indices,
            ) = self._gen_qk_indice(
                cmp_attn_score,
                total_seqlen_k=total_seqlen_k,
                q_offset=q_range.start,
                slt_k=slt_k_num,
            )

            flat_q_indices_list.append(flat_q_indices)
            flat_k_indices_list.append(flat_k_indices)

        # get the final q_starts and k_starts
        slt_q_ranges_tensor = torch.cat(flat_q_indices_list, dim=0)
        slt_k_ranges_tensor = torch.cat(flat_k_indices_list, dim=0)

        # get the final attn_type_map_slt
        attn_type_map_slt = torch.zeros(
            len(slt_q_ranges_tensor), dtype=torch.int32, device=self.device
        )

        return (
            slt_q_ranges_tensor,
            slt_k_ranges_tensor,
            attn_type_map_slt,
            max_seqlen_slt_q,
            max_seqlen_slt_k,
        )

    def _gen_qk_indice(
        self,
        cmp_attn_score,
        total_seqlen_k,
        q_offset,
        slt_k,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        nhq, seqlen_block_q, _ = cmp_attn_score.shape
        # 优化1：直接生成扁平化索引网格，避免后续扩展
        h_idx = torch.arange(nhq, device=self.device, dtype=torch.int32).view(
            nhq, 1, 1
        )  # [h, 1, 1]
        q_idx = torch.arange(
            seqlen_block_q, device=self.device, dtype=torch.int32
        ).view(
            1, seqlen_block_q, 1
        )  # [1, q, 1]

        # 2. 广播到TopK维度 [h, q, slt_k]（仍为视图，零拷贝）
        h_broadcast = h_idx.expand(nhq, seqlen_block_q, slt_k)
        q_broadcast = q_idx.expand(nhq, seqlen_block_q, slt_k)

        # 非两阶段逻辑优化：直接生成扁平化索引
        value, valid_index = cmp_attn_score.softmax(dim=-1).topk(slt_k, dim=-1)
        valid_index = valid_index.to(torch.int32)
        # meta_info["k_accum_prob"] = value.detach().sum() / (nhq * seqlen_block_q)

        head_idx = h_broadcast.flatten()  # 等价于原h网格展平+重复
        q_block_idx = q_broadcast.flatten()  # 等价于原q网格展平+重复
        flat_slt_cmp_k_indices_filtered = valid_index.flatten()

        # 优化5：使用缓存的rank和条件判断替代assert
        q_block_offset_scalar = self.block_ranges_start_to_idx_map.get(q_offset)
        if q_block_offset_scalar is None:
            raise ValueError("q_offset not found in pre-computed map")

        # 向量化计算索引，减少维度操作
        flat_head_offsets = head_idx * total_seqlen_k
        flat_q_indices = (
            flat_head_offsets[:, None]
            + self.block_ranges_tensor[q_block_idx + q_block_offset_scalar]
        )
        flat_k_indice = (
            flat_head_offsets[:, None]
            + self.block_ranges_tensor[flat_slt_cmp_k_indices_filtered]
        )

        return flat_q_indices, flat_k_indice

    def _set_cmp_mask(
        self,
        k_ranges: AttnRanges,
        total_seqlen_q: int,
        total_seqlen_k: int,
        device,
        mask: Optional[torch.Tensor] = None,
        need_zero: bool = True,
    ) -> torch.Tensor:
        if mask is None:
            mask = torch.zeros(
                (total_seqlen_q, total_seqlen_k),
                dtype=torch.bool,
                device=device,
            )
        else:
            mask = mask[:total_seqlen_q, :total_seqlen_k]
            if need_zero:
                mask.zero_()

        for k_range in k_ranges:
            mask[:, k_range.start : k_range.end] = True

        return mask


def _out_weighted_average(
    tensor_list: list[torch.Tensor], bool_list: list[bool]
) -> torch.Tensor:
    """
    Compute the average of tensors based on the boolean flags.

    Parameters:
    - tensor_list: List of tensors.
    - bool_list: List of boolean values indicating which tensors to include in the average calculation.

    Returns:
    - The average tensor.
    """
    if len(tensor_list) != len(bool_list):
        raise ValueError("The lengths of tensor_list and bool_list must be the same.")

    # Filter out the tensors that need to be included in the average calculation
    selected_tensors = [tensor for tensor, use in zip(tensor_list, bool_list) if use]

    # Compute the average
    if selected_tensors:
        average_tensor = sum(selected_tensors) / len(selected_tensors)
    else:
        average_tensor = torch.zeros_like(
            tensor_list[0]
        )  # Return a zero tensor if no tensors are selected

    return average_tensor


def _out_weighted_sum(
    tensor_list: list[torch.Tensor], bool_list: list[bool]
) -> torch.Tensor:
    """
    Compute the average of tensors based on the boolean flags.

    Parameters:
    - tensor_list: List of tensors.
    - bool_list: List of boolean values indicating which tensors to include in the average calculation.

    Returns:
    - The average tensor.
    """
    if len(tensor_list) != len(bool_list):
        raise ValueError("The lengths of tensor_list and bool_list must be the same.")

    # Filter out the tensors that need to be included in the average calculation
    selected_tensors = [tensor for tensor, use in zip(tensor_list, bool_list) if use]

    # Compute the average
    if selected_tensors:
        sum_tensor = sum(selected_tensors)
    else:
        sum_tensor = torch.zeros_like(
            tensor_list[0]
        )  # Return a zero tensor if no tensors are selected

    return sum_tensor


# ---   User-defined funcs for DistNSA    --- #


def _apply_win_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    key: SparseAttnKey,
    **kwargs,
) -> torch.Tensor | None:
    o_win, _ = flex_flash_attn_func(
        q=q,  # [sq//r, nhq//u, hd]
        k=k,  # [sk, nhq//u, hd]
        v=v,  # [sk, nhq//u, hd]
        q_ranges=key.win_q_ranges_tensor,  # remaked for ring
        k_ranges=key.win_k_ranges_tensor,
        deterministic=key.deterministic,
    )

    return o_win  # [sq//r, nhq//u, hd]


def _apply_cmp_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    key: SparseAttnKey,
    **kwargs,
) -> torch.Tensor | None:
    o_cmp, _ = flex_flash_attn_func(  # [sq//r*b, nhq//u, hd]
        q=q,  # [sq//r*b, nhq//u, hd]
        k=k,  # [sk//b, nhq//u, hd]
        v=v,  # [sk//b, nhq//u, hd]
        q_ranges=key.cmp_q_ranges_tensor,
        k_ranges=key.cmp_k_ranges_tensor,
        deterministic=key.deterministic,
    )

    return o_cmp  # [sq//r, nhq//u, hd]


def _cmp_qkv(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    key: SparseAttnKey,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # shape: [s, nhq//u, hd//r] -> [s//b, nhq//u, hd//r]
    cmp_q, cmp_k, cmp_v = [
        block_pooling_triton(x, cu_block_sizes=key.cu_block_sizes_list_tensor)
        for x in (q, k, v)
    ]

    return cmp_q, cmp_k, cmp_v


def _dep_o(
    o_cmp: torch.Tensor | None,
    total_seqlen_q: int,
    key: SparseAttnKey,
    **kwargs,
) -> torch.Tensor | None:
    # [sq//b, nhq//u, hd//r] -> [sq, nhq//u, hd//r]
    o_cmp = torch.repeat_interleave(
        o_cmp,
        repeats=key.block_sizes_tensor,
        dim=0,
        output_size=total_seqlen_q,
    )

    return o_cmp


def _flatten_qkv_before_slt_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    key: SparseAttnKey,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # --- flatten heads to apply per-head selection --- #

    # for q: shape: [sq//r, nhq//u, hd] -> [nhq*sq//r*u, 1, hd]
    # for k, v: shape: [sk, nhq//u, hd] -> [nhq*sk//u, 1, hd]
    q, k, v = [rearrange(x, "s nh hd -> (nh s) 1 hd").contiguous() for x in (q, k, v)]

    return q, k, v


def _calc_cmp_attn_score(
    cmp_q: torch.Tensor,
    cmp_k: torch.Tensor,
    softmax_scale: float,
    key: SparseAttnKey,
    **kwargs,
) -> torch.Tensor:
    partial_cmp_attn_score = (  # [nhq//u, sq//b, sk//b]
        torch.einsum(
            "qhd,khd->hqk",
            cmp_q,  # [sq//b, nhq//u, hd//r]
            cmp_k,  # [sk//b, nhq//u, hd//r]
        )
        * softmax_scale
    )

    return partial_cmp_attn_score


def _calc_slt_attn_args(
    total_seqlen_k: int,
    total_seqlen_cmp_k: int,
    cmp_attn_score: torch.Tensor,
    key: SparseAttnKey,
    **kwargs,
):
    (
        slt_q_ranges_tensor,  # [nhq*topk*sq//u*b, 2]
        slt_k_ranges_tensor,  # [nhq*topk*sq//u*b, 2]
        attn_type_map_slt,  # [nhq*topk*sq//u*b,]
        max_seqlen_slt_q,
        max_seqlen_slt_k,
    ) = key.calc_attn_meta(
        total_seqlen_k=total_seqlen_k,
        total_seqlen_cmp_k=total_seqlen_cmp_k,
        full_cmp_attn_score=cmp_attn_score,
    )

    return (
        slt_q_ranges_tensor,  # [nhq*topk*sq//u*b, 2]
        slt_k_ranges_tensor,  # [nhq*topk*sq//u*b, 2]
        attn_type_map_slt,  # [nhq*topk*sq//u*b,]
        max_seqlen_slt_q,
        max_seqlen_slt_k,
    )


def _apply_slt_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_ranges: torch.Tensor,
    k_ranges: torch.Tensor,
    attn_type_map: torch.Tensor,
    key: SparseAttnKey,
    **kwargs,
) -> torch.Tensor | None:
    o_slt, _ = flex_flash_attn_func(  # [sq//r*b, nhq//u, hd]
        q=q,  # [nhq*sq//r*u, 1, hd]
        k=k,  # [nhq*sk//u, 1, hd]
        v=v,  # [nhq*sk//u, 1, hd]
        q_ranges=q_ranges,
        k_ranges=k_ranges,
        attn_type_map=attn_type_map,
        auto_range_merge=True,
        # deterministic=key.deterministic,
        deterministic=False,  # FIXME: for now, auto range merge can not be deterministic
    )

    return o_slt  # [sq//r, nhq//u, hd]


def _unflatten_o_after_slt_attn(
    o_slt: torch.Tensor,
    num_heads: int,
    key: SparseAttnKey,
    **kwargs,
) -> torch.Tensor | None:
    # [nhq*sq//r*u, 1, hd] -> [sq//r, nhq//u, hd]
    o_slt = rearrange(o_slt, "(nh s) 1 hd -> s nh hd", nh=num_heads).contiguous()

    return o_slt


def _apply_out_gating(
    o_win: torch.Tensor | None,
    o_cmp: torch.Tensor | None,
    o_slt: torch.Tensor | None,
    g_win: torch.Tensor | None,
    g_cmp: torch.Tensor | None,
    g_slt: torch.Tensor | None,
    key: SparseAttnKey,
    **kwargs,
) -> tuple[torch.Tensor | None, ...]:
    if key.use_win_attn:
        o_win = o_win * g_win  # type: ignore[operator]
    if key.use_cmp_attn:
        o_cmp = o_cmp * g_cmp  # type: ignore[operator]
    if key.use_slt_attn:
        o_slt = o_slt * g_slt  # type: ignore[operator]

    return o_win, o_cmp, o_slt  # [sq//r, nhq//u, hd]


def _apply_out_reduce(
    o_win: torch.Tensor | None,
    o_cmp: torch.Tensor | None,
    o_slt: torch.Tensor | None,
    key: SparseAttnKey,
    **kwargs,
) -> torch.Tensor:
    reduce_func = _out_weighted_sum if key.use_gating else _out_weighted_average
    o = reduce_func(
        [
            o_cmp,  # [sq//r, nhq//u, hd]
            o_slt,  # [sq//r, nhq//u, hd]
            o_win,  # [sq//r, nhq//u, hd]
        ],
        [key.use_cmp_attn, key.use_slt_attn, (key.use_win_attn and key.use_nsa)],
    )

    return o  # [sq//r, nhq//u, hd]



def native_sparse_attn_optim(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    key: SparseAttnKey,
    g_cmp: Optional[torch.Tensor] = None,
    g_slt: Optional[torch.Tensor] = None,
    g_win: Optional[torch.Tensor] = None,
    slt_k: Optional[torch.Tensor] = None,
    slt_v: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Args:
        q (torch.Tensor): [seqlen_q, num_heads_q, head_dim]
        k (torch.Tensor): [seqlen_k, num_heads_k, head_dim]
        v (torch.Tensor): [seqlen_v, num_heads_v, head_dim]
        key (SparseAttnKey): some meta info for native sparse attention
    Returns:
        torch.Tensor: output with shape: [seqlen_q, num_heads_q, head_dim]
    """

    _, _, hd = q.shape
    softmax_scale = hd**-0.5

    # compute cmp attn
    # cmp_k = reduce(k, "(bs b) nh hd -> bs nh hd", reduction="mean", b=key.block_size)
    # cmp_v = reduce(v, "(bs b) nh hd -> bs nh hd", reduction="mean", b=key.block_size)
    cmp_k = block_pooling_triton(k, cu_block_sizes=key.cu_block_sizes_list_tensor)
    if key.use_cmp_attn or key.full_simulater_on:
        cmp_v = block_pooling_triton(v, cu_block_sizes=key.cu_block_sizes_list_tensor)
    o_cmp = None
    o_win = None
    o_slt = None
    if key.use_cmp_attn:
        # cmp_q = reduce(
        #     q, "(bs b) nh hd -> bs nh hd", reduction="mean", b=key.q_block_size
        # )
        cmp_q = block_pooling_triton(
            q, cu_block_sizes=key.cu_block_sizes_list_tensor
        )
        o_cmp, _ = flex_flash_attn_func(
            q=cmp_q,
            k=cmp_k,
            v=cmp_v,
            q_ranges=key.cmp_q_ranges_tensor,
            k_ranges=key.cmp_k_ranges_tensor,
            deterministic=key.deterministic,
        )
        # o_cmp = o_cmp[:, None, :, :].repeat(1, key.q_block_size, 1, 1).flatten(0, 1)
        o_cmp = torch.repeat_interleave(
            o_cmp, repeats=key.block_sizes_tensor, dim=0
        )
        
        if key.use_gating:
            assert g_cmp is not None, "g_cmp is None!"
            if torch.distributed.get_rank() == 0:
                print(f"use_gating:{g_cmp.detach().mean(), g_cmp.requires_grad}")
            o_cmp = o_cmp * g_cmp

    if key.use_win_attn:
        # compute win attn
        o_win, _ = flex_flash_attn_func(
            q=q,
            k=k,
            v=v,
            q_ranges=key.win_q_ranges_tensor,
            k_ranges=key.win_k_ranges_tensor,
            deterministic=key.deterministic,
        )
        if key.use_gating:
            assert g_win is not None, "g_win is None!"
            o_win = o_win * g_win

    if key.use_slt_attn:
        # compute slt attn
        # NOTE: each head has varied selection, thus we flatten it to seqlen dim
        slt_q = rearrange(q, "s h d -> (h s) 1 d")
        slt_k = rearrange(k, "s h d -> (h s) 1 d")
        slt_v = rearrange(v, "s h d -> (h s) 1 d")
        (
            slt_q_ranges_tensor,
            slt_k_ranges_tensor,
            attn_type_map_slt,
            max_seqlen_slt_q,
            max_seqlen_slt_k,
        ) = key.calc_attn_meta(
            total_seqlen_k=k.shape[0],
            total_seqlen_cmp_k=cmp_k.shape[0],
            softmax_scale=softmax_scale,
            q=q,
            cmp_k=cmp_k,
        )

        o_slt, slt_attn_lse = flex_flash_attn_func(
            q=slt_q,
            k=slt_k,
            v=slt_v,
            q_ranges=slt_q_ranges_tensor,
            k_ranges=slt_k_ranges_tensor,
            attn_type_map=attn_type_map_slt,
            auto_range_merge=True,
            # deterministic=key.deterministic,
            deterministic=False,  # FIXME: for now, auto range merge can not be deterministic
        )
        o_slt = rearrange(o_slt, "(h s) 1 d -> s h d", h=q.shape[1])
        
        if key.use_gating and (g_slt is not None):
            o_slt = o_slt * g_slt

    # reduce
    reduce_func = _out_weighted_sum if key.use_gating else _out_weighted_average
    o = reduce_func(
        [o_cmp, o_slt, o_win],
        [key.use_cmp_attn, key.use_slt_attn, (key.use_win_attn and key.use_nsa)],
    )
    return o