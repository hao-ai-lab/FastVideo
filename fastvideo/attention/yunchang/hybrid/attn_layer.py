from fastvideo.attention.yunchang.comm.all_to_all import SeqAllToAll4D, SeqAllToAll5D

import torch

from typing import Any
from torch import Tensor

import torch.distributed as dist
from .utils import RING_IMPL_DICT, RING_IMPL_QKVPACKED_DICT
from fastvideo.attention.yunchang.globals import PROCESS_GROUP, HAS_SPARSE_SAGE_ATTENTION
from fastvideo.attention.yunchang.kernels import AttnType


class LongContextAttention(torch.nn.Module):
    """Initialization.

    Arguments:
        ulysses_pg (ProcessGroup): ulysses process group
        ring_pg (ProcessGroup): ring process group
        scatter_idx (int): scatter_idx for all2all comm
        gather_idx (int): gather_idx for all2all comm
        use_sync (bool): whether to synchronize after all-to-all
    """

    def __init__(
        self,
        scatter_idx: int = 2,
        gather_idx: int = 1,
        ring_impl_type: str = "basic",
        use_pack_qkv: bool = False,
        use_sync: bool = False,
        attn_type: AttnType = AttnType.FA,
        attn_processor: torch.nn.Module = None,
    ) -> None:

        super(LongContextAttention, self).__init__()
        self.ring_pg = PROCESS_GROUP.RING_PG
        self.ulysses_pg = PROCESS_GROUP.ULYSSES_PG

        self.use_pack_qkv = use_pack_qkv
        self.use_sync = use_sync
        self.attn_type = attn_type
        assert (
            self.ulysses_pg is not None or self.ring_pg is not None
        ), f"use set_seq_parallel_pg() first. Now ulysses pg {self.ulysses_pg} and ring pg {self.ring_pg}"
        self.scatter_idx = scatter_idx
        self.gather_idx = gather_idx
        self.attn_processor = attn_processor
        self.ring_attn_fn = RING_IMPL_DICT[ring_impl_type]

        if HAS_SPARSE_SAGE_ATTENTION:
            from spas_sage_attn.autotune import SparseAttentionMeansim
            if isinstance(attn_processor,
                          SparseAttentionMeansim) and dist.get_world_size(
                              self.ring_pg) > 1:
                raise RuntimeError(
                    "Sparse Sage attention does not support ring degree > 1.")

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        dropout_p=0.0,
        softmax_scale=None,
        causal=False,
        window_size=(-1, -1),
        softcap=0.0,
        alibi_slopes=None,
        deterministic=False,
        return_attn_probs=False,
        *args: Any,
    ) -> Tensor:
        """forward

        Arguments:
            query (Tensor): query input to the layer
            key (Tensor): key input to the layer
            value (Tensor): value input to the layer
            args: other args

        Returns:
            * output (Tensor): context output
        """

        # 3 X (bs, seq_len/N, head_cnt, head_size) -> 3 X (bs, seq_len, head_cnt/N, head_size)
        # scatter 2, gather 1
        if self.use_pack_qkv:
            # (3*bs, seq_len/N, head_cnt, head_size)
            qkv = torch.cat([query, key, value]).continous()
            # (3*bs, seq_len, head_cnt/N, head_size)
            qkv = SeqAllToAll4D.apply(self.ulysses_pg,
                                      qkv,
                                      self.scatter_idx,
                                      self.gather_idx,
                                      use_sync=self.use_sync)
            qkv = torch.chunk(qkv, 3, dim=0)
            out = self.ring_attn_fn(
                qkv[0],
                qkv[1],
                qkv[2],
                dropout_p=dropout_p,
                softmax_scale=softmax_scale,
                causal=causal,
                window_size=window_size,
                softcap=softcap,
                alibi_slopes=alibi_slopes,
                deterministic=deterministic,
                return_attn_probs=return_attn_probs,
                group=self.ring_pg,
                attn_type=self.attn_type,
                attn_processor=self.attn_processor,
            )
        else:
            query_layer = SeqAllToAll4D.apply(self.ulysses_pg, query,
                                              self.scatter_idx, self.gather_idx,
                                              self.use_sync)
            key_layer = SeqAllToAll4D.apply(self.ulysses_pg, key,
                                            self.scatter_idx, self.gather_idx,
                                            self.use_sync)
            value_layer = SeqAllToAll4D.apply(self.ulysses_pg, value,
                                              self.scatter_idx, self.gather_idx,
                                              self.use_sync)
            if self.attn_type is AttnType.NPU:
                out = self.ring_attn_fn(
                    self.ring_pg,
                    query_layer,
                    key_layer,
                    value_layer,
                    softmax_scale=softmax_scale,
                    causal=causal,
                    attn_type=self.attn_type,
                    attn_processor=self.attn_processor,
                )
            else:
                out = self.ring_attn_fn(
                    query_layer,
                    key_layer,
                    value_layer,
                    dropout_p=dropout_p,
                    softmax_scale=softmax_scale,
                    causal=causal,
                    window_size=window_size,
                    softcap=softcap,
                    alibi_slopes=alibi_slopes,
                    deterministic=deterministic,
                    return_attn_probs=return_attn_probs,
                    group=self.ring_pg,
                    attn_type=self.attn_type,
                    attn_processor=self.attn_processor,
                )

        if type(out) == tuple:
            context_layer, _, _ = out
        else:
            context_layer = out

        # (bs, seq_len, head_cnt/N, head_size) -> (bs, seq_len/N, head_cnt, head_size)
        # scatter 1, gather 2
        output = SeqAllToAll4D.apply(self.ulysses_pg, context_layer,
                                     self.gather_idx, self.scatter_idx,
                                     self.use_sync)

        # out e.g., [s/p::h]
        return output


class LongContextAttentionQKVPacked(torch.nn.Module):
    """Initialization.

    Arguments:
        ulysses_pg (ProcessGroup): ulysses process group
        ring_pg (ProcessGroup): ring process group
        scatter_idx (int): scatter_idx for all2all comm
        gather_idx (int): gather_idx for all2all comm
        use_sync (bool): whether to synchronize after all-to-all
    """

    def __init__(
        self,
        scatter_idx: int = 3,
        gather_idx: int = 1,
        ring_impl_type: str = "basic",
        use_sync: bool = False,
        attn_type: AttnType = AttnType.FA,
    ) -> None:

        super(LongContextAttentionQKVPacked, self).__init__()

        self.ring_pg = PROCESS_GROUP.RING_PG
        self.ulysses_pg = PROCESS_GROUP.ULYSSES_PG

        assert (
            self.ulysses_pg is not None or self.ring_pg is not None
        ), f"use set_seq_parallel_pg() first. Now ulysses pg {self.ulysses_pg} and ring pg {self.ring_pg}"
        self.scatter_idx = scatter_idx
        self.gather_idx = gather_idx
        self.use_sync = use_sync
        self.ring_attn_fn = RING_IMPL_QKVPACKED_DICT[ring_impl_type]
        self.attn_type = attn_type

    def forward(
        self,
        qkv,
        dropout_p=0.0,
        softmax_scale=None,
        causal=False,
        window_size=(-1, -1),
        softcap=0.0,
        alibi_slopes=None,
        deterministic=False,
        return_attn_probs=False,
        *args: Any,
    ) -> Tensor:
        """forward

        Arguments:
            query (Tensor): query input to the layer
            key (Tensor): key input to the layer
            value (Tensor): value input to the layer
            args: other args

        Returns:
            * output (Tensor): context output
        """

        # scatter 3, gather 1

        world_size = dist.get_world_size(self.ulysses_pg)

        if world_size > 1:
            qkv = SeqAllToAll5D.apply(self.ulysses_pg, qkv, self.scatter_idx,
                                      self.gather_idx, self.use_sync)

        out = self.ring_attn_fn(
            qkv,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            softcap=softcap,
            alibi_slopes=alibi_slopes,
            deterministic=deterministic,
            return_attn_probs=return_attn_probs,
            group=self.ring_pg,
            attn_type=self.attn_type,
        )

        # print(f"out {out.shape}")

        if type(out) == tuple:
            out = out[0]

        # (bs, seq_len, head_cnt/N, head_size) -> (bs, seq_len/N, head_cnt, head_size)
        # scatter 1, gather 2

        if world_size > 1:
            out = SeqAllToAll4D.apply(self.ulysses_pg, out, self.gather_idx,
                                      self.scatter_idx - 1, self.use_sync)
        # out e.g., [s/p::h]
        return out
