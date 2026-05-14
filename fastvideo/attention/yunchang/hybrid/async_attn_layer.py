import torch

from typing import Any
from torch import Tensor

import torch.distributed as dist
from .utils import RING_IMPL_DICT
from fastvideo.attention.yunchang.globals import PROCESS_GROUP


class AsyncLongContextAttention(torch.nn.Module):
    """Initialization.

    Arguments:
        ulysses_pg (ProcessGroup): ulysses process group
        ring_pg (ProcessGroup): ring process group
        scatter_idx (int): scatter_idx for all2all comm
        gather_idx (int): gather_idx for all2all comm
    """

    def __init__(
        self,
        scatter_idx: int = 2,
        gather_idx: int = 1,
        ring_impl_type: str = "basic",
    ) -> None:

        super(AsyncLongContextAttention, self).__init__()
        self.ring_pg = PROCESS_GROUP.RING_PG
        self.ulysses_pg = PROCESS_GROUP.ULYSSES_PG

        self.stream = torch.cuda.Stream()
        self._async_op = True

        assert (
            self.ulysses_pg is not None or self.ring_pg is not None
        ), f"use set_seq_parallel_pg() first. Now ulysses pg {self.ulysses_pg} and ring pg {self.ring_pg}"
        self.scatter_idx = scatter_idx
        self.gather_idx = gather_idx
        self.ring_attn_fn = RING_IMPL_DICT[ring_impl_type]

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
            query (Tensor): query input to the layer (bs, seqlen/P, hc, hs)
            key (Tensor): key input to the layer (bs, seqlen/P, hc_kv, hs)
            value (Tensor): value input to the layer (bs, seqlen/P, hc_kv, hs)
            args: other args

        Returns:
            * output (Tensor): context output
        """

        # un*ud = hc

        ulysses_degree = dist.get_world_size(self.ulysses_pg)

        bs, shard_seqlen, hc, hs = query.shape
        bs, shard_seqlen, hc_kv, hs = key.shape
        seq_len = shard_seqlen * ulysses_degree
        un = hc // ulysses_degree
        un_kv = hc_kv // ulysses_degree

        assert un_kv == un, f"un_kv {un_kv} un {un}"

        qkv = torch.cat([query, key, value]).contiguous()
        # (3*bs, seqlen/P, hc, hs) -> (hc, seqlen/P, 3*bs, hs) -> (un, ud, seqlen/P, 3*bs, hs), where hc = un*ud
        qkv_list = torch.unbind(
            qkv.transpose(0, 2).contiguous().reshape(un, ulysses_degree,
                                                     shard_seqlen, 3 * bs, hs))
        # 3xall-to-all output buffer
        qkv_trans_list = [
            torch.zeros(
                ulysses_degree,
                1,
                shard_seqlen,
                3 * bs,
                hs,
                dtype=query.dtype,
                device=query.device,
            ) for i in range(len(qkv_list))
        ]
        # last all-to-all buffter
        context_layer_list = [
            torch.zeros(
                ulysses_degree,
                1,
                shard_seqlen,
                bs,
                hs,
                dtype=query.dtype,
                device=query.device,
            ) for i in range(len(qkv_list))
        ]

        comm_handle_list = []

        # un * (ud, shard_seqlen, 3*bs, hs)
        for i, qkv in enumerate(qkv_list):
            with torch.cuda.stream(self.stream):
                ret = dist.all_to_all_single(
                    qkv_trans_list[i],
                    qkv,
                    group=self.ulysses_pg,
                    async_op=self._async_op,
                )
            comm_handle_list.append(ret)

        last_comm_handle_list = []
        for i, qkv_trans in enumerate(qkv_trans_list):
            if comm_handle_list[i] is not None:
                comm_handle_list[i].wait()
            qkv_trans = (qkv_trans.reshape(seq_len, 3 * bs, 1, hs).transpose(
                0, 1).contiguous().reshape(3 * bs, seq_len, 1, hs))

            # qkv_trans = all_to_all_4D_async(qkv, qkv_trans_list[i], self.scatter_idx, self.gather_idx, self.ulysses_pg)
            qkv_trans = torch.chunk(qkv_trans, 3, dim=0)

            out = self.ring_attn_fn(
                qkv_trans[0],
                qkv_trans[1],
                qkv_trans[2],
                dropout_p=dropout_p,
                softmax_scale=softmax_scale,
                causal=causal,
                window_size=window_size,
                softcap=softcap,
                alibi_slopes=alibi_slopes,
                deterministic=deterministic,
                return_attn_probs=return_attn_probs,
                group=self.ring_pg,
            )

            if type(out) == tuple:
                context_layer, _, _ = out
            else:
                context_layer = out

            # (bs, seq_len, head_cnt/N, head_size) -> (bs, seq_len/N, head_cnt, head_size)
            # scatter 1, gather 2

            context_layer = (context_layer.reshape(
                bs, ulysses_degree, shard_seqlen, 1,
                hs).transpose(0, 3).transpose(0, 1).contiguous().reshape(
                    ulysses_degree, 1, shard_seqlen, bs, hs))
            with torch.cuda.stream(self.stream):
                ret = dist.all_to_all_single(
                    context_layer_list[i],
                    context_layer,
                    group=self.ulysses_pg,
                    async_op=self._async_op,
                )
            last_comm_handle_list.append(ret)

        # hc = un * P
        # un x (hc = P, seq_len/P, bs, hs) -> (bs, seq_len, hc = P, hs)
        for i, ret in enumerate(last_comm_handle_list):
            if ret is not None:
                ret.wait()
            context_layer_list[i] = (context_layer_list[i].reshape(
                ulysses_degree, shard_seqlen, bs,
                hs).transpose(0,
                              2).contiguous().reshape(bs, shard_seqlen,
                                                      ulysses_degree, hs))

        output = torch.cat(context_layer_list, dim=2)
        return output

    def backward(self, *args, **kwargs):
        raise RuntimeError(
            "Backward computation is not allowed for AsyncLongContextAttention."
        )
