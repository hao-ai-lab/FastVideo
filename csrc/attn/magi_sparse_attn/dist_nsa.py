import os
from collections import defaultdict
from dataclasses import dataclass
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from functools import partial
from typing import Callable

import torch
import torch.distributed as dist
import torch.nn as nn
from einops import rearrange, repeat
from magi_attention.common import AttnRange, AttnRanges
from magi_attention.api.functools import pad_at_dim
from magi_attention.common.ranges import AttnRanges

from odin.comm.functional import (
    all_gather_fwd_reduce_scatter_bwd,
    all_gather_fwd_scatter_bwd,
    all_reduce_fwd_scale_bwd,
    all_to_all,
    scatter_fwd_all_gather_bwd,
)
from odin.utils.numerical import ensure_divisibility, ensure_divisibility_and_divide
from torch.distributed.device_mesh import DeviceMesh

from nsa_v2 import (
    SparseAttnKey,
    _apply_cmp_attn,
    _apply_out_gating,
    _apply_out_reduce,
    _apply_slt_attn,
    _apply_win_attn,
    _calc_cmp_attn_score,
    _calc_slt_attn_args,
    _cmp_qkv,
    _dep_o,
    _flatten_qkv_before_slt_attn,
    _unflatten_o_after_slt_attn,
)

@dataclass
class DistSparseAttnKey(SparseAttnKey):
    cp_ring_mesh_idx: int = 0  # default inter
    cp_ulysses_mesh_idx: int = 1  # default intra
    cp_ring_pad_size: int = 0

    # if enabling `save_ring_ag_kv_for_xxx_attn`,
    # we will directly save kv after all-gather in ring mesh
    # for backward during _apply_xxx_attn
    # which trade off more activation memory overhead
    # for avoid all-gather kv in ring mesh before ffa backward
    save_ring_ag_kv_for_win_attn: bool = False
    save_ring_ag_kv_for_cmp_attn: bool = True
    save_ring_ag_kv_for_slt_attn: bool = False

    def __post_init__(self):
        super().__post_init__()

        self._init_cp_mesh()

        self._remake_meta_for_cp_ring()

    def _init_cp_mesh(self):
        self.cp_mesh = self.parallel_state[("cp_inter", "cp_intra")]

        assert self.cp_mesh is not None, "cp_mesh is required"
        assert self.cp_mesh.ndim == 2, (
            f"cp_mesh must be 2D mesh with shape [cp_ring_dim, cp_ulysses_dim], "
            f"but got {self.cp_mesh.ndim=} | {self.cp_mesh=}"
        )

        self.cp_ring_group = self.cp_mesh.get_group(self.cp_ring_mesh_idx)
        self.cp_ring_size, self.cp_ring_rank = (
            self.cp_ring_group.size(),
            self.cp_ring_group.rank(),
        )
        self.is_last_cp_ring_rank = self.cp_ring_rank == self.cp_ring_size - 1

        self.cp_ulysses_group = self.cp_mesh.get_group(self.cp_ulysses_mesh_idx)
        self.cp_ulysses_size, self.cp_ulysses_rank = (
            self.cp_ulysses_group.size(),
            self.cp_ulysses_group.rank(),
        )

        # FIXME: for now, the original cp_pad_size only applies to ulysses mesh
        # and there's no pad size for ring mesh
        self.cp_ulysses_pad_size = self.cp_pad_size
        self.cp_ring_last_pad_size = (
            self.cp_ring_pad_size if self.is_last_cp_ring_rank else 0
        )

    def _remake_meta_for_cp_ring(self):
        # for win_attn
        if self.use_win_attn:
            self._remake_meta_for_win_attn()
        # for cmp_attn
        if self.use_cmp_attn:
            self._remake_meta_for_cmp_attn()
        # for slt_attn
        if self.use_slt_attn:
            self._remake_meta_for_slt_attn()

    def _remake_meta_for_slt_attn(self):
        assert (
            self.q_ranges.is_non_overlap()
        ), "For now, dist_nsa only supports non-overlapping q ranges"
        # we assume q_ranges and k_ranges are global ranges w/o padding
        ring_seqlen_q = ensure_divisibility_and_divide(
            self.q_ranges.total_seqlen + self.cp_ring_pad_size, self.cp_ring_size
        )
        assert ring_seqlen_q > self.cp_ring_pad_size, (
            f"{ring_seqlen_q=} must be larger than {self.cp_ring_pad_size=}, "
            "otherwise, the padding tokens does not belong to the last rank only"
        )

        ring_local_q_start = ring_seqlen_q * self.cp_ring_rank
        ring_local_q_end = min(
            ring_seqlen_q * (self.cp_ring_rank + 1), self.q_ranges.total_seqlen
        )
        (
            ring_local_q_ranges,
            ring_local_k_ranges,
        ) = self._truncate_and_localize_attn_slices(
            q_ranges=self.q_ranges,
            k_ranges=self.k_ranges,
            q_trunc_start=ring_local_q_start,
            q_truck_end=ring_local_q_end,
        )

        # re-make ring-local attn args
        self.q_ranges_tensor = ring_local_q_ranges.to_tensor(self.device)
        self.k_ranges_tensor = ring_local_k_ranges.to_tensor(self.device)
        self.max_seqlen_q = ring_local_q_ranges.max_seqlen
        self.max_seqlen_k = ring_local_k_ranges.max_seqlen

        # add some new info for ring-local slt attn
        self.ring_local_q_start = ring_local_q_start
        self.ring_local_q_end = ring_local_q_end
        self.ring_local_seqlen_q = ring_seqlen_q - self.cp_ring_last_pad_size

    def _remake_meta_for_cmp_attn(self):
        assert (
            self.cmp_q_ranges.is_non_overlap()
        ), "For now, dist_nsa only supports non-overlapping cmp q ranges"

        # we assume cmp_q_ranges and cmp_k_ranges are global ranges w/o padding
        # so we have to pad cmp_q to shard along ring mesh for local cmp_attn
        self.cp_ring_cmp_pad_size = (
            self.cp_ring_size - remainder
            if (remainder := self.cmp_q_ranges.total_seqlen % self.cp_ring_size) > 0
            else 0
        )
        ring_seqlen_cmp_q = ensure_divisibility_and_divide(
            self.cmp_q_ranges.total_seqlen + self.cp_ring_cmp_pad_size,
            self.cp_ring_size,
        )
        assert ring_seqlen_cmp_q > self.cp_ring_cmp_pad_size, (
            f"{ring_seqlen_cmp_q=} must be larger than {self.cp_ring_cmp_pad_size=}, "
            "otherwise, the padding tokens does not belong to the last rank only"
        )

        ring_local_cmp_q_start = ring_seqlen_cmp_q * self.cp_ring_rank
        ring_local_cmp_q_end = min(
            ring_seqlen_cmp_q * (self.cp_ring_rank + 1), self.cmp_q_ranges.total_seqlen
        )
        (
            ring_local_cmp_q_ranges,
            ring_local_cmp_k_ranges,
        ) = self._truncate_and_localize_attn_slices(
            q_ranges=self.cmp_q_ranges,
            k_ranges=self.cmp_k_ranges,
            q_trunc_start=ring_local_cmp_q_start,
            q_truck_end=ring_local_cmp_q_end,
        )

        # re-make ring-local cmp attn args
        self.cmp_q_ranges_tensor = ring_local_cmp_q_ranges.to_tensor(self.device)
        self.cmp_k_ranges_tensor = ring_local_cmp_k_ranges.to_tensor(self.device)
        self.max_seqlen_cmp_q = ring_local_cmp_q_ranges.max_seqlen
        self.max_seqlen_cmp_k = ring_local_cmp_k_ranges.max_seqlen

        # add some new info for ring-local cmp_attn
        self.cp_ring_last_cmp_pad_size = (
            self.cp_ring_cmp_pad_size if self.is_last_cp_ring_rank else 0
        )

    def _remake_meta_for_win_attn(self):
        assert (
            self.win_q_ranges.is_non_overlap()
        ), "For now, dist_nsa only supports non-overlapping win q ranges"
        # we assume win_q_ranges and win_k_ranges are global ranges w/o padding
        ring_seqlen_win_q = ensure_divisibility_and_divide(
            self.win_q_ranges.total_seqlen + self.cp_ring_pad_size, self.cp_ring_size
        )
        assert ring_seqlen_win_q > self.cp_ring_pad_size, (
            f"{ring_seqlen_win_q=} must be larger than {self.cp_ring_pad_size=}, "
            "otherwise, the padding tokens does not belong to the last rank only"
        )

        ring_local_win_q_start = ring_seqlen_win_q * self.cp_ring_rank
        ring_local_win_q_end = min(
            ring_seqlen_win_q * (self.cp_ring_rank + 1), self.win_q_ranges.total_seqlen
        )

        (
            ring_local_win_q_ranges,
            ring_local_win_k_ranges,
        ) = self._truncate_and_localize_attn_slices(
            q_ranges=self.win_q_ranges,
            k_ranges=self.win_k_ranges,
            q_trunc_start=ring_local_win_q_start,
            q_truck_end=ring_local_win_q_end,
        )

        # re-make ring-local win attn args
        self.win_q_ranges_tensor = ring_local_win_q_ranges.to_tensor(self.device)
        self.win_k_ranges_tensor = ring_local_win_k_ranges.to_tensor(self.device)
        self.max_seqlen_win_q = ring_local_win_q_ranges.max_seqlen
        self.max_seqlen_win_k = ring_local_win_k_ranges.max_seqlen

    def _truncate_and_localize_attn_slices(
        self,
        q_ranges: AttnRanges,
        k_ranges: AttnRanges,
        q_trunc_start: int,
        q_truck_end: int,
        q_offset: int = 0,
    ) -> tuple[AttnRanges, AttnRanges]:
        new_q_ranges = AttnRanges()
        new_k_ranges = AttnRanges()

        # truncate
        for q_range, k_range in zip(q_ranges, k_ranges):
            truct_q_range = q_range.truncate(q_trunc_start, q_truck_end)
            if truct_q_range.is_empty():
                continue
            new_q_ranges.append(truct_q_range)
            new_k_ranges.append(k_range)
        # localize
        new_q_ranges = new_q_ranges.make_ranges_local(new_q_ranges)
        # offset TODO: use AttnRanges.offset if supported
        if q_offset > 0:
            new_q_ranges_ = AttnRanges()
            for q_range in new_q_ranges:
                new_q_ranges_.append(q_range.offset(q_offset))
            new_q_ranges = new_q_ranges_

        return new_q_ranges, new_k_ranges


class DistNSA(nn.Module):
    """Distributed NSA Module"""

    def __init__(self):
        super().__init__()

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        key: DistSparseAttnKey,
        g_win: torch.Tensor | None = None,
        g_cmp: torch.Tensor | None = None,
        g_slt: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, dict]:
        """Distributed NSA forward

        Args:
            q (torch.Tensor): query tensor
            k (torch.Tensor): key tensor
            v (torch.Tensor): value tensor
            key (DistSparseAttnKey): dist sparse attn key
            kwargs: keyword arguments

        Returns:
            torch.Tensor: output tensor

        Shape:
            q: [s // cp_size, num_heads_q, head_dim]
            k: [s // cp_size, num_heads_k, head_dim]
            v: [s // cp_size, num_heads_k, head_dim]

            NOTE: cp_size = cp_ring_size * cp_ulysses_size
        """

        # ------    initialize and check     ------ #

        assert q.dim() == k.dim() == v.dim() == 3, (
            f"q, k, v must be 3D tensor with shape [s, nh, hd], "
            f"but got {q.dim()=}, {k.dim()=}, {v.dim()=}"
        )
        nhq, nhk, hd = q.size(1), k.size(1), q.size(-1)
        softmax_scale = hd ** (-0.5)
        # TODO: for GQA, we don't have to always repeat to num_heads_q
        # as long as repeated num_heads_kv can be divided by cp_ulysses_size
        kv_rep_times = ensure_divisibility_and_divide(nhq, nhk)
        nhq_shard = ensure_divisibility_and_divide(nhq, key.cp_ulysses_size)  # nhq//u
        ensure_divisibility(hd, key.cp_ring_size)  # hd//r
        meta_info_dict = {}
        o_win, o_cmp, o_slt = None, None, None

        # ------    ulysses pre-process     ------ #

        (
            q,  # [sq//r, nhq//u, hd]
            k,  # [sk//r, nhq//u, hd]
            v,  # [sk//r, nhq//u, hd]
            g_win,  # [sq//r, nhq//u, 1]
            g_cmp,  # [sq//r, nhq//u, 1]
            g_slt,  # [sq//r, nhq//u, 1]
        ) = self._ulysses_process_before_nsa(
            q=q,  # [sq//r*u, nhq, hd]
            k=k,  # [sk//r*u, nhk, hd]
            v=v,  # [sk//r*u, nhk, hd]
            g_win=g_win,  # [sq//r*u, nhq, 1]
            g_cmp=g_cmp,  # [sq//r*u, nhq, 1]
            g_slt=g_slt,  # [sq//r*u, nhq, 1]
            key=key,
            kv_rep_times=kv_rep_times,
            **kwargs,
        )

        # ------    ring pre-process     ------ #

        (
            q_shard_hd,  # [sq, nhq//u, hd//r]
            k_shard_hd,  # [sk, nhq//u, hd//r]
            v_shard_hd,  # [sk, nhq//u, hd//r]
            k_full,  # [sk, nhq//u, hd]
            v_full,  # [sk, nhq//u, hd]
        ) = self._ring_process_before_nsa(
            q=q,  # [sq//r, nhq//u, hd]
            k=k,  # [sk//r, nhq//u, hd]
            v=v,  # [sk//r, nhq//u, hd]
            key=key,
            **kwargs,
        )

        # ------    win-attn     ------ #

        if key.use_win_attn:
            (
                win_q,  # [sq//r, nhq//u, hd]
                win_k,  # [sk, nhq//u, hd]
                win_v,  # [sk, nhq//u, hd]
                save_tensors_ctx_for_win_attn,
            ) = self._prepare_save_tensors_hook_for_win_attn(
                q=q,  # [sq//r, nhq//u, hd]
                k=k_full,  # [sk, nhq//u, hd]
                v=v_full,  # [sk, nhq//u, hd]
                key=key,
                **kwargs,
            )

            with save_tensors_ctx_for_win_attn:
                o_win = self.apply_win_attn(  # [sq//r, nhq//u, hd]
                    q=win_q,  # [sq//r, nhq//u, hd]
                    k=win_k,  # [sk, nhq//u, hd]
                    v=win_v,  # [sk, nhq//u, hd]
                    key=key,
                    **kwargs,
                )

        # ------    cmp-attn     ------ #

        if key.use_cmp_attn or key.use_slt_attn:
            (
                cmp_q,  # [sq//b, nhq//u, hd//r]
                cmp_k,  # [sk//b, nhq//u, hd//r]
                cmp_v,  # [sk//b, nhq//u, hd//r]
            ) = self.cmp_qkv(
                q=q_shard_hd,  # [sq, nhq//u, hd//r]
                k=k_shard_hd,  # [sk, nhq//u, hd//r]
                v=v_shard_hd,  # [sk, nhq//u, hd//r]
                key=key,
                **kwargs,
            )

        if key.use_cmp_attn:
            (
                cmp_q_,  # [sq//r*b, nhq//u, hd]
                cmp_k_,  # [sk//b, nhq//u, hd]
                cmp_v_,  # [sk//b, nhq//u, hd]
            ) = self._ring_process_before_cmp_attn(
                q=cmp_q,  # [sq//b, nhq//u, hd//r]
                k=cmp_k,  # [sk//b, nhq//u, hd//r]
                v=cmp_v,  # [sk//b, nhq//u, hd//r]
                key=key,
                **kwargs,
            )

            (
                cmp_q_,  # [sq//r*b, nhq//u, hd]
                cmp_k_,  # [sk//b, nhq//u, hd]
                cmp_v_,  # [sk//b, nhq//u, hd]
                save_tensors_ctx_for_cmp_attn,
            ) = self._prepare_save_tensors_hook_for_cmp_attn(
                q=cmp_q_,  # [sq//r*b, nhq//u, hd]
                k=cmp_k_,  # [sk//b, nhq//u, hd]
                v=cmp_v_,  # [sk//b, nhq//u, hd]
                key=key,
                **kwargs,
            )

            with save_tensors_ctx_for_cmp_attn:
                # TODO: merge it with cmp_attn_score to avoid re-computation
                o_cmp = self.apply_cmp_attn(  # [sq//r*b, nhq//u, hd]
                    q=cmp_q_,  # [sq//r*b, nhq//u, hd]
                    k=cmp_k_,  # [sk//b, nhq//u, hd]
                    v=cmp_v_,  # [sk//b, nhq//u, hd]
                    key=key,
                    **kwargs,
                )

            # [sq//r*b, nhq//u, hd] -> [sq//b, nhq//u, hd//r]
            o_cmp = self._ring_process_after_cmp_attn(o=o_cmp, key=key, **kwargs)

            # [sq//b, nhq//u, hd//r] -> [sq, nhq//u, hd//r]
            o_cmp = self.dep_o(
                o_cmp=o_cmp,
                total_seqlen_q=q_shard_hd.shape[0],
                key=key,
                **kwargs,
            )

            # [sq, nhq//u, hd//r] -> [sq//r, nhq//u, hd]
            o_cmp = self._ring_process_after_dep_o(o=o_cmp, key=key, **kwargs)

        # ------    slt-attn     ------ #

        if key.use_slt_attn:
            (
                slt_q,  # [nhq*sq//r*u, 1, hd]
                slt_k,  # [nhq*sk//u, 1, hd]
                slt_v,  # [nhq*sk//u, 1, hd]
            ) = self.flatten_qkv_before_slt_attn(
                q=q,  # [sq//r, nhq//u, hd]
                k=k_full,  # [sk, nhq//u, hd]
                v=v_full,  # [sk, nhq//u, hd]
                key=key,
                **kwargs,
            )

            cmp_attn_score = self.calc_cmp_attn_score(  # [nhq//u, sq//b, sk//b]
                cmp_q=cmp_q,  # [sq//b, nhq//u, hd//r]
                cmp_k=cmp_k,  # [sk//b, nhq//u, hd//r]
                softmax_scale=softmax_scale,
                key=key,
                **kwargs,
            )

            cmp_attn_score = self._reduce_cmp_attn_score(  # [nhq//u, sq//b, sk//b]
                cmp_attn_score=cmp_attn_score,  # [nhq//u, sq//b, sk//b]
                key=key,
                **kwargs,
            )

            (
                slt_q_ranges,  # [nhq*topk*sq//u*b, 2]
                slt_k_ranges,  # [nhq*topk*sq//u*b, 2]
                attn_type_map_slt,  # [nhq*topk*sq//u*b,]
                max_seqlen_slt_q,
                max_seqlen_slt_k,
            ) = self.calc_slt_attn_args(  # TODO: find a better way to design this func
                total_seqlen_k=k_full.shape[0],  # sk
                total_seqlen_cmp_k=cmp_k.shape[0],  # sk//b
                cmp_attn_score=cmp_attn_score,  # [nhq//u, sq//b, sk//b]
                key=key,
                **kwargs,
            )

            (
                slt_q_ranges,  # [nhq*topk*sq//u*r*b, 2]
                slt_k_ranges,  # [nhq*topk*sq//u*r*b, 2]
                attn_type_map_slt,  # [nhq*topk*sq//u*r*b,]
                max_seqlen_slt_q,
                max_seqlen_slt_k,
            ) = self._remake_slt_attn_args(
                slt_q_ranges=slt_q_ranges,  # [nhq*topk*sq//u*b, 2]
                slt_k_ranges=slt_k_ranges,  # [nhq*topk*sq//u*b, 2]
                attn_type_map_slt=attn_type_map_slt,  # [nhq*topk*sq//u*b,]
                max_seqlen_slt_q=max_seqlen_slt_q,
                max_seqlen_slt_k=max_seqlen_slt_k,
                num_heads=nhq_shard,  # nhq//u
                total_seqlen_q=q_shard_hd.shape[0],  # sq
                key=key,
                **kwargs,
            )

            (
                slt_q,  # [nhq*sq//r*u, 1, hd]
                slt_k,  # [nhq*sk//u, 1, hd]
                slt_v,  # [nhq*sk//u, 1, hd]
                save_tensors_hook_for_slt_attn,
            ) = self._prepare_save_tensors_hook_for_slt_attn(
                q=slt_q,  # [nhq*sq//r*u, 1, hd]
                k=slt_k,  # [nhq*sk//u, 1, hd]
                v=slt_v,  # [nhq*sk//u, 1, hd]
                num_heads=nhq_shard,  # nhq//u
                key=key,
                **kwargs,
            )

            with save_tensors_hook_for_slt_attn:
                o_slt = self.apply_slt_attn(  # [nhq*sq//r*u, 1, hd]
                    q=slt_q,  # [nhq*sq//r*u, 1, hd]
                    k=slt_k,  # [nhq*sk//u, 1, hd]
                    v=slt_v,  # [nhq*sk//u, 1, hd]
                    q_ranges=slt_q_ranges,  # [nhq*topk*sq//u*r*b, 2]
                    k_ranges=slt_k_ranges,  # [nhq*topk*sq//u*r*b, 2]
                    attn_type_map=attn_type_map_slt,  # [nhq*topk*sq//u*r*b,]
                    key=key,
                    **kwargs,
                )

            # [nhq*sq//r*u, 1, hd] -> [sq//r, nhq//u, hd]
            o_slt = self.unflatten_o_after_slt_attn(
                o_slt=o_slt, num_heads=nhq_shard, key=key, **kwargs
            )

        # ------    out gating     ------ #

        if key.use_gating:
            o_win, o_cmp, o_slt = self.apply_out_gating(  # [sq//r, nhq//u, hd]
                o_win=o_win,  # [sq//r, nhq//u, hd]
                o_cmp=o_cmp,  # [sq//r, nhq//u, hd]
                o_slt=o_slt,  # [sq//r, nhq//u, hd]
                g_win=g_win,  # [sq//r, nhq//u, 1]
                g_cmp=g_cmp,  # [sq//r, nhq//u, 1]
                g_slt=g_slt,  # [sq//r, nhq//u, 1]
                key=key,
                **kwargs,
            )

        # ------    out reduce     ------ #

        o = self.apply_out_reduce(  # [sq//r, nhq//u, hd]
            o_win=o_win,  # [sq//r, nhq//u, hd]
            o_cmp=o_cmp,  # [sq//r, nhq//u, hd]
            o_slt=o_slt,  # [sq//r, nhq//u, hd]
            key=key,
            **kwargs,
        )

        # ------    ulysses post-process     ------ #

        o = self._ulysses_process_after_nsa(  # [sq//r*u, nhq, hd]
            o=o,  # [sq//r, nhq//u, hd]
            key=key,
        )

        return o, meta_info_dict

    # ---   API funcs    --- #

    @classmethod
    def compute_pad_size(
        cls,
        seqlen: int,
        cp_mesh: DeviceMesh,
        cp_ring_mesh_idx: int = 0,
        cp_ulysses_mesh_idx: int = 1,
    ) -> tuple[int, int]:
        """
        Compute pad size for both ring and ulysses mesh.

        Args:
            seqlen (int): total sequence length.
            cp_mesh (DeviceMesh): 2D cp device mesh.
            cp_ring_mesh_idx (int, optional): index of ring mesh. Defaults to 0.
            cp_ulysses_mesh_idx (int, optional): index of ulysses mesh. Defaults to 1.

        Returns:
            tuple[int, int]: (cp_ring_pad_size, cp_ulysses_pad_size)
        """
        cp_ring_size = cp_mesh.get_group(cp_ring_mesh_idx).size()
        cp_ulysses_size = cp_mesh.get_group(cp_ulysses_mesh_idx).size()

        (
            cp_ring_pad_size,
            cp_ring_seqlen,
            cp_ring_shard_seqlen,
        ) = cls.compute_pad_size_single(seqlen, cp_ring_size)
        (
            cp_ulysses_pad_size,
            cp_ulysses_seqlen,
            cp_ulysses_shard_seqlen,
        ) = cls.compute_pad_size_single(cp_ring_shard_seqlen, cp_ulysses_size)

        return cp_ring_pad_size, cp_ulysses_pad_size

    @classmethod
    def dispatch(
        cls,
        x: torch.Tensor,
        cp_mesh: DeviceMesh,
        cp_ring_pad_size: int,
        cp_ulysses_pad_size: int,
        cp_ring_mesh_idx: int = 0,
        cp_ulysses_mesh_idx: int = 1,
    ) -> torch.Tensor:
        """
        Dispatch tensor to both ring and ulysses mesh.

        Args:
            x (torch.Tensor): input tensor.
            cp_mesh (DeviceMesh): 2D cp device mesh.
            cp_ring_pad_size (int): pad size for ring mesh.
            cp_ulysses_pad_size (int): pad size for ulysses mesh.
            cp_ring_mesh_idx (int, optional): index of ring mesh. Defaults to 0.
            cp_ulysses_mesh_idx (int, optional): index of ulysses mesh. Defaults to 1.

        Returns:
            torch.Tensor: dispatched tensor.
        """
        cp_ring_group = cp_mesh.get_group(cp_ring_mesh_idx)
        cp_ulysses_group = cp_mesh.get_group(cp_ulysses_mesh_idx)

        x = cls.dispatch_single(x, group=cp_ring_group, cp_pad_size=cp_ring_pad_size)
        x = cls.dispatch_single(
            x, group=cp_ulysses_group, cp_pad_size=cp_ulysses_pad_size
        )

        return x

    @classmethod
    def undispatch(
        cls,
        x: torch.Tensor,
        cp_mesh: DeviceMesh,
        cp_ring_pad_size: int,
        cp_ulysses_pad_size: int,
        cp_ring_mesh_idx: int = 0,
        cp_ulysses_mesh_idx: int = 1,
    ) -> torch.Tensor:
        """
        Undispatch tensor from both ring and ulysses mesh.

        Args:
            x (torch.Tensor): input tensor.
            cp_mesh (DeviceMesh): 2D cp device mesh.
            cp_ring_pad_size (int): pad size for ring mesh.
            cp_ulysses_pad_size (int): pad size for ulysses mesh.
            cp_ring_mesh_idx (int, optional): index of ring mesh. Defaults to 0.
            cp_ulysses_mesh_idx (int, optional): index of ulysses mesh. Defaults to 1.

        Returns:
            torch.Tensor: undispatched tensor.
        """
        cp_ring_group = cp_mesh.get_group(cp_ring_mesh_idx)
        cp_ulysses_group = cp_mesh.get_group(cp_ulysses_mesh_idx)

        x = cls.undispatch_single(
            x, group=cp_ulysses_group, cp_pad_size=cp_ulysses_pad_size
        )
        x = cls.undispatch_single(x, group=cp_ring_group, cp_pad_size=cp_ring_pad_size)

        return x

    @classmethod
    def compute_pad_size_single(cls, seqlen: int, cp_size: int) -> tuple[int, int, int]:
        padded_shard_seqlen = (seqlen + cp_size - 1) // cp_size
        padded_seqlen = padded_shard_seqlen * cp_size
        pad_size = padded_seqlen - seqlen

        assert padded_shard_seqlen > pad_size, (
            f"To keep all padding tokens in the last cp rank, "
            f"{pad_size=} must be less than {padded_shard_seqlen=}."
        )

        return pad_size, padded_seqlen, padded_shard_seqlen

    @classmethod
    def dispatch_single(
        cls, x: torch.Tensor, group: dist.ProcessGroup, cp_pad_size: int
    ) -> torch.Tensor:
        x = cls.apply_pad(x, cp_pad_size=cp_pad_size)
        x_local = scatter_fwd_all_gather_bwd(x, group=group)
        return x_local

    @classmethod
    def undispatch_single(
        cls, x: torch.Tensor, group: dist.ProcessGroup, cp_pad_size: int
    ) -> torch.Tensor:
        x_global = all_gather_fwd_scatter_bwd(x, group=group)
        x_global = cls.apply_unpad(
            x_global,
            cp_pad_size=cp_pad_size,
        )

        return x_global

    @classmethod
    def apply_pad(
        cls,
        x: torch.Tensor,
        cp_pad_size: int,
        value: float = 0.0,
    ) -> torch.Tensor:
        """Right Pad to x at dim 0"""
        # most cases: x.shape: [seqlen, num_heads, head_dim]
        if cp_pad_size == 0:
            return x

        return pad_at_dim(x, dim=0, pad_size=cp_pad_size, value=value, side="right")

    @classmethod
    def apply_unpad(cls, x: torch.Tensor, cp_pad_size: int) -> torch.Tensor:
        """Right UnPad x at dim 0"""
        if cp_pad_size == 0:
            return x

        # most cases: x.shape: [seqlen, num_heads, head_dim]
        return x[:-cp_pad_size]

    # ---   User-defined funcs    --- #

    @staticmethod
    def apply_win_attn(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        key: DistSparseAttnKey,
        **kwargs,
    ) -> torch.Tensor | None:
        return _apply_win_attn(q, k, v, key, **kwargs)

    @staticmethod
    def apply_cmp_attn(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        key: DistSparseAttnKey,
        **kwargs,
    ) -> torch.Tensor | None:
        return _apply_cmp_attn(q, k, v, key, **kwargs)

    @staticmethod
    def cmp_qkv(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        key: DistSparseAttnKey,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return _cmp_qkv(q, k, v, key, **kwargs)

    @staticmethod
    def dep_o(
        o_cmp: torch.Tensor | None,
        total_seqlen_q: int,
        key: DistSparseAttnKey,
        **kwargs,
    ) -> torch.Tensor | None:
        return _dep_o(o_cmp, total_seqlen_q, key, **kwargs)

    @staticmethod
    def flatten_qkv_before_slt_attn(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        key: DistSparseAttnKey,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return _flatten_qkv_before_slt_attn(q, k, v, key, **kwargs)

    @staticmethod
    def calc_cmp_attn_score(
        cmp_q: torch.Tensor,
        cmp_k: torch.Tensor,
        softmax_scale: float,
        key: DistSparseAttnKey,
        **kwargs,
    ) -> torch.Tensor:
        return _calc_cmp_attn_score(cmp_q, cmp_k, softmax_scale, key, **kwargs)

    @staticmethod
    def calc_slt_attn_args(
        total_seqlen_k: int,
        total_seqlen_cmp_k: int,
        cmp_attn_score: torch.Tensor,
        key: DistSparseAttnKey,
        **kwargs,
    ):
        return _calc_slt_attn_args(
            total_seqlen_k=total_seqlen_k,
            total_seqlen_cmp_k=total_seqlen_cmp_k,
            cmp_attn_score=cmp_attn_score,
            key=key,
            **kwargs,
        )

    @staticmethod
    def apply_slt_attn(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        q_ranges: torch.Tensor,
        k_ranges: torch.Tensor,
        attn_type_map: torch.Tensor,
        key: DistSparseAttnKey,
        **kwargs,
    ) -> torch.Tensor | None:
        return _apply_slt_attn(
            q=q,
            k=k,
            v=v,
            q_ranges=q_ranges,
            k_ranges=k_ranges,
            attn_type_map=attn_type_map,
            key=key,
            **kwargs,
        )

    @staticmethod
    def unflatten_o_after_slt_attn(
        o_slt: torch.Tensor,
        num_heads: int,
        key: DistSparseAttnKey,
        **kwargs,
    ) -> torch.Tensor | None:
        return _unflatten_o_after_slt_attn(o_slt, num_heads, key, **kwargs)

    @staticmethod
    def apply_out_gating(
        o_win: torch.Tensor | None,
        o_cmp: torch.Tensor | None,
        o_slt: torch.Tensor | None,
        g_win: torch.Tensor | None,
        g_cmp: torch.Tensor | None,
        g_slt: torch.Tensor | None,
        key: DistSparseAttnKey,
        **kwargs,
    ) -> tuple[torch.Tensor | None, ...]:
        return _apply_out_gating(
            o_win, o_cmp, o_slt, g_win, g_cmp, g_slt, key, **kwargs
        )

    @staticmethod
    def apply_out_reduce(
        o_win: torch.Tensor | None,
        o_cmp: torch.Tensor | None,
        o_slt: torch.Tensor | None,
        key: DistSparseAttnKey,
        **kwargs,
    ) -> torch.Tensor:
        return _apply_out_reduce(o_win, o_cmp, o_slt, key, **kwargs)

    # ---   Distributed process funcs    --- #

    def _ulysses_process_before_nsa(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g_win: torch.Tensor | None,
        g_cmp: torch.Tensor | None,
        g_slt: torch.Tensor | None,
        key: DistSparseAttnKey,
        kv_rep_times: int,
        **kwargs,
    ) -> tuple[torch.Tensor, ...]:
        # --- a2a qkv to shard heads and make seqlen complete in ulysses mesh --- #

        # shape: [s//r*u, nh, hd] -> [s//r, nhq//u, hd]
        if key.cp_ulysses_size == 1:
            q, k, v = [
                self.apply_unpad(
                    self._repeat_head(x, rep_times),
                    cp_pad_size=key.cp_ring_last_pad_size,
                )  # TODO: we don't need to repeat kv if cp_ulysses_size == 1
                for x, rep_times in zip((q, k, v), (1, kv_rep_times, kv_rep_times))
            ]
        else:
            q, k, v = [
                self._qkv_rearrange_after_a2a(
                    self._all2all_func(
                        self._repeat_head(self._qkv_rearrange_before_a2a(x), rep_times),
                        group=key.cp_ulysses_group,
                    ),
                    cp_size=key.cp_ulysses_size,
                    cp_pad_size=key.cp_ulysses_pad_size + key.cp_ring_last_pad_size,
                )
                for x, rep_times in zip((q, k, v), (1, kv_rep_times, kv_rep_times))
            ]

        # --- a2a gating to shard heads and make seqlen complete in ulysses mesh --- #

        if key.use_gating:
            if key.use_win_attn:
                assert g_win is not None and g_win.dim() == 3 and g_win.size(-1) == 1
            if key.use_cmp_attn:
                assert g_cmp is not None and g_cmp.dim() == 3 and g_cmp.size(-1) == 1
            if key.use_slt_attn:
                assert g_slt is not None and g_slt.dim() == 3 and g_slt.size(-1) == 1

            # shape: [s//r*u, nhq, 1] -> [s//r, nhq//u, 1]
            if key.cp_ulysses_size == 1:
                g_win, g_cmp, g_slt = [
                    self.apply_unpad(
                        g,
                        cp_pad_size=key.cp_ring_last_pad_size,
                    )
                    if g is not None
                    else None
                    for g in (g_win, g_cmp, g_slt)
                ]
            else:
                g_win, g_cmp, g_slt = [
                    self._qkv_rearrange_after_a2a(
                        self._all2all_func(
                            self._qkv_rearrange_before_a2a(g),
                            group=key.cp_ulysses_group,
                        ),
                        cp_size=key.cp_ulysses_size,
                        cp_pad_size=key.cp_ulysses_pad_size + key.cp_ring_last_pad_size,
                    )
                    if g is not None
                    else None
                    for g in (g_win, g_cmp, g_slt)
                ]

        return q, k, v, g_win, g_cmp, g_slt

    def _ulysses_process_after_nsa(
        self,
        o: torch.Tensor,
        key: DistSparseAttnKey,
        **kwargs,
    ) -> torch.Tensor:
        # --- a2a o to shard seqlen and make heads complete in ulysses mesh --- #

        # shape: [s//r, nhq//u, hd] -> [s//r*u, nhq, hd]
        if key.cp_ulysses_size == 1:
            o = self.apply_pad(
                o,
                cp_pad_size=key.cp_ring_last_pad_size,
            )
        else:
            o = self._out_rearrange_after_a2a(
                self._all2all_func(
                    self._out_rearrange_before_a2a(
                        o,
                        cp_pad_size=key.cp_ulysses_pad_size + key.cp_ring_last_pad_size,
                    ),
                    group=key.cp_ulysses_group,
                ),
                cp_size=key.cp_ulysses_size,
            )

        return o

    def _ring_process_before_nsa(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        key: DistSparseAttnKey,
        **kwargs,
    ) -> tuple[torch.Tensor, ...]:
        if key.cp_ring_size == 1:
            return q, k, v, k, v

        # --- ag k,v to make seqlen complete in ring mesh --- #

        # shape: [sk//r, nhq//u, hd] -> [sk, nhq//u, hd]
        k_full, v_full = [
            self._full_kv_rearrange_after_ag(
                self._allgather_func(
                    self._full_kv_rearrange_before_ag(
                        x,
                        cp_pad_size=key.cp_ring_last_pad_size,
                    ),
                    group=key.cp_ring_group,
                    # NOTE: partial dk,dv for win_attn, slt_attn,
                    # except for cmp_attn, should be reduce-scatter here
                    reduce_bwd=True,
                ),
                cp_pad_size=key.cp_ring_pad_size,
            )
            for x in (k, v)
        ]

        # --- a2a qkv to shard head dim and make seqlen complete in ring mesh --- #

        # NOTE: for k,v, since we already got the k_full, v_full above
        # we can easily get k_shard_hd, v_shard_hd by scatter,
        # however, it will introduce one all-gather in bwd
        # thus here we still use a2a as q for two reasons:
        #   1. one a2a in fwd + one a2a in bwd usually costs less than one all-gather
        #   2. if getting k_shard_hd, v_shard_hd from k_full, v_full,
        #      we cannot apply reduce-scatter in bwd to get partial dk, dv for cmp attn
        #      because it has already been reduced when scattering head-dim
        # shape: [s//r, nhq//u, hd] -> [s, nhq//u, hd//r]
        q_shard_hd, k_shard_hd, v_shard_hd = [
            self._qkv_rearrange_after_a2a(
                self._all2all_func(
                    self._qkv_rearrange_before_a2a(
                        x,
                        shard_heads=False,
                        cp_pad_size=key.cp_ring_last_pad_size,
                    ),
                    group=key.cp_ring_group,
                ),
                shard_heads=False,
                cp_size=key.cp_ring_size,
                cp_pad_size=key.cp_ring_pad_size,
            )
            for x in (q, k, v)
        ]

        return (
            q_shard_hd,  # [sq, nhq//u, hd//r]
            k_shard_hd,  # [sk, nhq//u, hd//r]
            v_shard_hd,  # [sk, nhq//u, hd//r]
            k_full,  # [sk, nhq//u, hd]
            v_full,  # [sk, nhq//u, hd]
        )

    def _ring_process_before_cmp_attn(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        key: DistSparseAttnKey,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if key.cp_ring_size == 1:
            return q, k, v

        # --- a2a q to shard seqlen and make head dim complete in ring mesh --- #

        # shape: [sq//b, nhq//u, hd//r] -> [sq//r*b, nhq//u, hd]
        q = self._cmp_q_rearrange_after_a2a(
            self._all2all_func(
                self._cmp_q_rearrange_before_a2a(
                    q,
                    cp_pad_size=key.cp_ring_cmp_pad_size,
                ),
                group=key.cp_ring_group,
            ),
            cp_size=key.cp_ring_size,
            cp_pad_size=key.cp_ring_last_cmp_pad_size,
        )

        # --- ag k,v to make head dim complete in ring mesh --- #

        # shape: [sk//b, nhq//u, hd//r] -> [sk//b, nhq//u, hd]
        k, v = [
            self._cmp_kv_rearrange_after_ag(
                self._allgather_func(
                    self._cmp_kv_rearrange_before_ag(
                        x,
                    ),
                    group=key.cp_ring_group,
                    # NOTE: partial dk,dv for cmp_attn
                    # should be reduce-scatter here
                    reduce_bwd=True,
                ),
            )
            for x in (k, v)
        ]

        return (
            q,  # [sq//r*b, nhq//u, hd]
            k,  # [sk//b, nhq//u, hd]
            v,  # [sk//b, nhq//u, hd]
        )

    def _ring_process_after_cmp_attn(
        self,
        o: torch.Tensor | None,
        key: DistSparseAttnKey,
        **kwargs,
    ) -> torch.Tensor | None:
        if key.cp_ring_size == 1:
            return o

        # --- a2a o to shard head dim and make seqlen complete in ring mesh --- #

        # shape: [sq//r*b, nhq//u, hd] -> [sq//b, nhq//u, hd//r]
        o = self._qkv_rearrange_after_a2a(
            self._all2all_func(
                self._qkv_rearrange_before_a2a(
                    o,
                    shard_heads=False,
                    cp_pad_size=key.cp_ring_last_cmp_pad_size,
                ),
                group=key.cp_ring_group,
            ),
            shard_heads=False,
            cp_size=key.cp_ring_size,
            cp_pad_size=key.cp_ring_cmp_pad_size,
        )

        return o

    def _ring_process_after_dep_o(
        self,
        o: torch.Tensor | None,
        key: DistSparseAttnKey,
        **kwargs,
    ) -> torch.Tensor | None:
        if key.cp_ring_size == 1:
            return o

        # --- a2a o to shard seqlen and make head dim complete in ring mesh --- #

        # shape: [sq, nhq//u, hd//r] -> [sq//r, nhq//u, hd]
        o = self._out_rearrange_after_a2a(
            self._all2all_func(
                self._out_rearrange_before_a2a(
                    o,
                    shard_heads=False,
                    cp_pad_size=key.cp_ring_pad_size,
                ),
                group=key.cp_ring_group,
            ),
            cp_size=key.cp_ring_size,
            shard_heads=False,
            cp_pad_size=key.cp_ring_last_pad_size,
        )

        return o

    def _reduce_cmp_attn_score(
        self,
        cmp_attn_score: torch.Tensor,
        key: DistSparseAttnKey,
        **kwargs,
    ) -> torch.Tensor:
        if key.cp_ring_size == 1:
            return cmp_attn_score

        cmp_attn_score = self._allreduce_func(
            cmp_attn_score,
            group=key.cp_ring_group,
            reduce_op="sum",
        )

        return cmp_attn_score

    def _remake_slt_attn_args(
        self,
        slt_q_ranges: torch.Tensor,
        slt_k_ranges: torch.Tensor,
        attn_type_map_slt: torch.Tensor,
        max_seqlen_slt_q: int,
        max_seqlen_slt_k: int,
        num_heads: int,
        total_seqlen_q: int,
        key: DistSparseAttnKey,
        **kwargs,
    ):
        if key.cp_ring_size > 1:
            slt_q_ranges = AttnRanges.from_ranges(
                slt_q_ranges.tolist()
            )  # [nhq*topk*sq//u*b, 2]
            slt_k_ranges = AttnRanges.from_ranges(
                slt_k_ranges.tolist()
            )  # [nhq*topk*sq//u*b, 2]
            global_offset_per_head = total_seqlen_q
            local_offset_per_head = key.ring_local_seqlen_q

            ring_local_slt_q_ranges_per_head = AttnRanges()
            ring_local_slt_k_ranges_per_head = AttnRanges()
            for head_idx in range(num_heads):
                global_offset_this_head = global_offset_per_head * head_idx
                local_offset_this_head = local_offset_per_head * head_idx
                ring_local_q_start_this_head = (
                    global_offset_this_head + key.ring_local_q_start
                )
                ring_local_q_end_this_head = (
                    global_offset_this_head + key.ring_local_q_end
                )
                (
                    ring_local_slt_q_ranges_this_head,
                    ring_local_slt_k_ranges_this_head,
                ) = key._truncate_and_localize_attn_slices(
                    slt_q_ranges,
                    slt_k_ranges,
                    q_trunc_start=ring_local_q_start_this_head,
                    q_truck_end=ring_local_q_end_this_head,
                    q_offset=local_offset_this_head,
                )
                ring_local_slt_q_ranges_per_head.extend(
                    ring_local_slt_q_ranges_this_head
                )
                ring_local_slt_k_ranges_per_head.extend(
                    ring_local_slt_k_ranges_this_head
                )

            slt_q_ranges = ring_local_slt_q_ranges_per_head.to_tensor(
                key.device
            )  # [nhq*topk*sq//u*r*b, 2]
            slt_k_ranges = ring_local_slt_k_ranges_per_head.to_tensor(
                key.device
            )  # [nhq*topk*sq//u*r*b, 2]
            attn_type_map_slt = torch.zeros(  # [nhq*topk*sq//u*r*b,]
                len(slt_q_ranges), dtype=torch.int32, device=key.device
            )
            max_seqlen_slt_q = ring_local_slt_q_ranges_per_head.max_seqlen
            max_seqlen_slt_k = ring_local_slt_k_ranges_per_head.max_seqlen

        return (
            slt_q_ranges,
            slt_k_ranges,
            attn_type_map_slt,
            max_seqlen_slt_q,
            max_seqlen_slt_k,
        )

    def _prepare_save_tensors_hook_for_win_attn(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        key: DistSparseAttnKey,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, AbstractContextManager]:
        save_tensors_ctx = (
            self._get_save_tensors_ctx(
                tensors=[k, v],
                tag_attr="_trigger_hook_for_win_attn",
                group=key.cp_ring_group,
                cp_pad_size=key.cp_ring_pad_size,
            )
            if not key.save_ring_ag_kv_for_win_attn
            else nullcontext()
        )

        return q, k, v, save_tensors_ctx

    def _prepare_save_tensors_hook_for_cmp_attn(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        key: DistSparseAttnKey,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, AbstractContextManager]:
        save_tensors_ctx = (
            self._get_save_tensors_ctx(
                tensors=[k, v],
                tag_attr="_trigger_hook_for_cmp_attn",
                group=key.cp_ring_group,
                cp_pad_size=key.cp_ring_cmp_pad_size,
            )
            if not key.save_ring_ag_kv_for_cmp_attn
            else nullcontext()
        )

        return q, k, v, save_tensors_ctx

    def _prepare_save_tensors_hook_for_slt_attn(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        key: DistSparseAttnKey,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, AbstractContextManager]:
        cp_ring_flatten_heads_pad_size, *rest = self.compute_pad_size_single(
            seqlen=k.shape[0],  # nhq*sk//u
            cp_size=key.cp_ring_size,
        )

        save_tensors_ctx = (
            self._get_save_tensors_ctx(
                tensors=[k, v],
                tag_attr="_trigger_hook_for_slt_attn",
                group=key.cp_ring_group,
                cp_pad_size=cp_ring_flatten_heads_pad_size,
            )
            if not key.save_ring_ag_kv_for_slt_attn
            else nullcontext()
        )

        return q, k, v, save_tensors_ctx

    # ---   Distributed rearrange funcs    --- #

    def _qkv_rearrange_before_a2a(
        self,
        x: torch.Tensor,
        shard_heads: bool = True,
        cp_pad_size: int = 0,
    ) -> torch.Tensor:
        x = self.apply_pad(x, cp_pad_size)

        if shard_heads:
            return rearrange(
                x,
                "local_s nh hd -> nh local_s hd",
            ).contiguous()
        else:  # shard head dim
            return rearrange(
                x,
                "local_s nh hd -> hd local_s nh",
            ).contiguous()

    def _qkv_rearrange_after_a2a(
        self,
        x: torch.Tensor,
        cp_size: int,
        cp_pad_size: int = 0,
        shard_heads: bool = True,
    ) -> torch.Tensor:
        if shard_heads:
            x = rearrange(
                x,
                "(cp_size local_nh) local_s hd -> (cp_size local_s) local_nh hd",
                cp_size=cp_size,
            ).contiguous()
        else:  # shard head dim
            x = rearrange(
                x,
                "(cp_size local_hd) local_s nh -> (cp_size local_s) nh local_hd",
                cp_size=cp_size,
            ).contiguous()

        x = self.apply_unpad(x, cp_pad_size)

        return x

    def _out_rearrange_before_a2a(
        self,
        x: torch.Tensor,
        cp_pad_size: int = 0,
        shard_heads: bool = True,
    ) -> torch.Tensor:
        x = self.apply_pad(x, cp_pad_size)

        if not shard_heads:  # shard head dim
            x = rearrange(x, "s nh hd -> s hd nh").contiguous()

        return x

    def _out_rearrange_after_a2a(
        self,
        x: torch.Tensor,
        cp_size: int,
        cp_pad_size: int = 0,
        shard_heads: bool = True,
    ) -> torch.Tensor:
        if shard_heads:
            x = rearrange(
                x,
                "(cp_size local_s) local_nh hd -> local_s (cp_size local_nh) hd",
                cp_size=cp_size,
            ).contiguous()
        else:  # shard head dim
            x = rearrange(
                x,
                "(cp_size local_s) local_hd nh -> local_s nh (cp_size local_hd)",
                cp_size=cp_size,
            ).contiguous()

        x = self.apply_unpad(x, cp_pad_size)

        return x

    def _full_kv_rearrange_before_ag(
        self,
        x: torch.Tensor,
        cp_pad_size: int = 0,
    ) -> torch.Tensor:
        x = self.apply_pad(x, cp_pad_size)
        return x

    def _full_kv_rearrange_after_ag(
        self,
        x: torch.Tensor,
        cp_pad_size: int = 0,
    ) -> torch.Tensor:
        x = self.apply_unpad(x, cp_pad_size)
        return x

    def _cmp_q_rearrange_before_a2a(
        self,
        x: torch.Tensor,
        cp_pad_size: int = 0,
    ) -> torch.Tensor:
        x = self.apply_pad(x, cp_pad_size)
        return rearrange(x, "s nh local_hd -> s local_hd nh").contiguous()

    def _cmp_q_rearrange_after_a2a(
        self,
        x: torch.Tensor,
        cp_size: int,
        cp_pad_size: int = 0,
    ) -> torch.Tensor:
        x = rearrange(
            x,
            "(cp_size local_s) local_hd nh -> local_s nh (cp_size local_hd)",
            cp_size=cp_size,
        ).contiguous()

        x = self.apply_unpad(x, cp_pad_size)

        return x

    def _cmp_kv_rearrange_before_ag(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return rearrange(x, "s nh local_hd -> local_hd s nh").contiguous()

    def _cmp_kv_rearrange_after_ag(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return rearrange(x, "hd s nh -> s nh hd").contiguous()

    def _repeat_head(
        self,
        x: torch.Tensor,
        repeat_times: int,
    ) -> torch.Tensor:
        return repeat(x, "nh local_s hd -> (nh nr) local_s hd", nr=repeat_times)

    # ---   Distributed comm funcs    --- #

    def _allreduce_func(
        self,
        input: torch.Tensor,
        group: dist.ProcessGroup,
        reduce_op: str = "sum",
    ) -> torch.Tensor:
        if group.size() == 1:
            return input

        output = all_reduce_fwd_scale_bwd(input, group, reduce_op)
        return output

    def _all2all_func(
        self,
        input: torch.Tensor,
        group: dist.ProcessGroup,
    ) -> torch.Tensor:
        if group.size() == 1:
            return input

        output = all_to_all(input, group=group)
        return output

    def _allgather_func(
        self,
        input: torch.Tensor,
        group: dist.ProcessGroup,
        reduce_bwd: bool = True,
    ) -> torch.Tensor:
        if group.size() == 1:
            return input

        if reduce_bwd:
            output = all_gather_fwd_reduce_scatter_bwd(input, group)
        else:
            output = all_gather_fwd_scatter_bwd(input, group)
        return output

    def _scatter_func(
        self,
        input: torch.Tensor,
        group: dist.ProcessGroup,
    ) -> torch.Tensor:
        if group.size() == 1:
            return input

        output = scatter_fwd_all_gather_bwd(input, group)
        return output

    # ---   Distributed save-tensors-hook funcs    --- #

    def _get_save_tensors_ctx(
        self,
        tensors: list[torch.Tensor],
        tag_attr: str,
        group: dist.ProcessGroup,
        cp_pad_size: int = 0,
    ) -> AbstractContextManager:
        for tensor in tensors:
            setattr(tensor, tag_attr, True)

        ring_dispatch_func = partial(
            DistNSA.dispatch_single,
            group=group,
            cp_pad_size=cp_pad_size,
        )

        ring_undispatch_func = partial(
            DistNSA.undispatch_single,
            group=group,
            cp_pad_size=cp_pad_size,
        )

        def dispatch_pack_hook(
            x: torch.Tensor,
            dispatch_func: Callable,
            tag_attr: str,
        ) -> torch.Tensor:
            if hasattr(x, tag_attr):
                x = dispatch_func(x)
                setattr(x, tag_attr, True)
            return x

        def undispatch_unpack_hook(
            x: torch.Tensor,
            undispatch_func: Callable,
            tag_attr: str,
        ) -> torch.Tensor:
            if hasattr(x, tag_attr):
                delattr(x, tag_attr)
                x = undispatch_func(x)
            return x

        pack_hook = partial(
            dispatch_pack_hook,
            tag_attr=tag_attr,
            dispatch_func=ring_dispatch_func,
        )

        unpack_hook = partial(
            undispatch_unpack_hook,
            tag_attr=tag_attr,
            undispatch_func=ring_undispatch_func,
        )

        save_tensors_hook = torch.autograd.graph.saved_tensors_hooks(
            pack_hook=pack_hook,
            unpack_hook=unpack_hook,
        )

        return save_tensors_hook
