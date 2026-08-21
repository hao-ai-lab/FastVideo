# SPDX-License-Identifier: Apache-2.0
#
# Ring Attention (Ascend NPU backend) forward, vendored and adapted for
# FastVideo from yunchang (long-context-attention):
#   https://github.com/feifeibear/long-context-attention
# which itself derives from ring-flash-attention:
#   https://github.com/zhuzilin/ring-flash-attention

import torch
from .utils import RingComm, update_npu_out
from .kernels import AttnType, select_flash_attn_impl
from .kernels.attention import (
    npu_fused_attn_forward, )


def ring_npu_flash_attn_forward(process_group,
                                q: torch.Tensor,
                                k: torch.Tensor,
                                v: torch.Tensor,
                                softmax_scale: float | None = None,
                                head_num: int | None = None,
                                input_layout: str = "BSND",
                                causal: bool = False,
                                attn_type: AttnType = AttnType.NPU,
                                attn_processor=None):
    comm = RingComm(process_group)
    # print(f"{datetime.now()} current device is: {torch.cuda.current_device()}, ring_npu_flash_attn_forward")
    # Single-GPU case: compute directly
    if comm.world_size == 1:
        return npu_fused_attn_forward(q, k, v, head_num, input_layout, softmax_scale)

    attention_out, softmax_max, softmax_sum = None, None, None
    global_attention_out, global_softmax_max, global_softmax_sum = None, None, None

    next_k, next_v = None, None

    for step in range(comm.world_size):
        # print(f"{datetime.now()} current device is: {torch.cuda.current_device()},ring_npu_flash_attn_forward step: {step}")
        # Not the last step: kick off the next kv communication (async)
        if step + 1 != comm.world_size:
            next_k = comm.send_recv(k)
            next_v = comm.send_recv(v)
            # print(f"{datetime.now()} current device is: {torch.cuda.current_device()},ring_npu_flash_attn_forward commit: {step}")
            comm.commit()

        # Compute for the current step (only process local kv when step <= current rank)
        if not causal or step <= comm.rank:
            # print(f"{datetime.now()} current device is: {torch.cuda.current_device()},ring_npu_flash_attn_forward calculation: {step}")
            fn = select_flash_attn_impl(attn_type, stage="fwd-only", attn_processor=attn_processor)
            attention_out, softmax_max, softmax_sum = fn(q, k, v, head_num, input_layout, softmax_scale)
            global_attention_out, global_softmax_max, global_softmax_sum = update_npu_out(
                attention_out, softmax_max, softmax_sum, global_attention_out, global_softmax_max, global_softmax_sum)

        # Not the last step: wait for communication to finish, update kv
        if step + 1 != comm.world_size:
            comm.wait()
            # print(f"{datetime.now()} current device is: {torch.cuda.current_device()},ring_npu_flash_attn_forward wait: {step}")
            k = next_k
            v = next_v
    return global_attention_out, global_softmax_max, global_softmax_sum


def ring_npu_flash_attn_backward(process_group,
                                 q: torch.Tensor,
                                 k: torch.Tensor,
                                 v: torch.Tensor,
                                 grad_attention_out: torch.Tensor,
                                 head_num: int | None = None,
                                 input_layout: str = "BSND",
                                 softmax_max: torch.Tensor | None = None,
                                 softmax_sum: torch.Tensor | None = None,
                                 attention_in: torch.Tensor | None = None,
                                 scale_value: float | None = None,
                                 causal: bool = False,
                                 attn_type: AttnType = AttnType.NPU) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    kv_comm = RingComm(process_group)
    d_kv_comm = RingComm(process_group)
    # print(f"{datetime.now()} current device is: {torch.cuda.current_device()}, ring_npu_flash_attn_backward")

    # Initialize gradient tensors (avoid None by using zero tensors)
    dq = torch.zeros_like(q, dtype=torch.float32)
    dk = torch.zeros_like(k, dtype=torch.float32)
    dv = torch.zeros_like(v, dtype=torch.float32)
    next_k, next_v = None, None
    next_dk, next_dv = None, None

    for step in range(kv_comm.world_size):
        # 1. Kick off kv communication (fetch kv for the next step)
        if step + 1 != kv_comm.world_size:
            next_k = kv_comm.send_recv(k)
            next_v = kv_comm.send_recv(v)
            # print(f"{datetime.now()} current device is: {torch.cuda.current_device()},ring_npu_flash_attn_backward commit: {step}")
            kv_comm.commit()

        # 2. Compute gradients for the current step
        if step <= kv_comm.rank or not causal:
            fn = select_flash_attn_impl(attn_type, stage="bwd-only")
            grad_query, grad_key, grad_value = fn(q,
                                                  k,
                                                  v,
                                                  grad_attention_out,
                                                  head_num,
                                                  input_layout,
                                                  softmax_max=softmax_max,
                                                  softmax_sum=softmax_sum,
                                                  attention_in=attention_in,
                                                  scale_value=scale_value)
            # print(f"{datetime.now()} current device is: {torch.cuda.current_device()},ring_npu_flash_attn_backward calculation: {step}")
            # Accumulate query gradient (each rank only computes its own q gradient)
            dq += grad_query.to(torch.float32)

            # Accumulate kv gradient: if not the first step, add the gradient received via communication
            if step > 0:
                d_kv_comm.wait()  # Wait for the previous round's dk/dv communication to finish
                # print(f"{datetime.now()} current device is: {torch.cuda.current_device()},ring_npu_flash_attn_backward d_kv_comm wait: {step}")
                dk = grad_key.to(torch.float32) + next_dk
                dv = grad_value.to(torch.float32) + next_dv
            else:
                # First step: assign directly
                # print(f"{datetime.now()} current device is: {torch.cuda.current_device()},ring_npu_flash_attn_backward dkdv: {step}")
                dk = grad_key.to(torch.float32)
                dv = grad_value.to(torch.float32)
        else:
            # step > current rank: only receive the previous round's dk/dv
            if step > 0:
                d_kv_comm.wait()
                # print(f"{datetime.now()} current device is: {torch.cuda.current_device()},ring_npu_flash_attn_backward d_kv_comm wait to next_dk: {step}")
                dk = next_dk
                dv = next_dv

        # 3. Wait for kv communication to finish, update kv
        if step + 1 != kv_comm.world_size:
            kv_comm.wait()
            # print(f"{datetime.now()} current device is: {torch.cuda.current_device()},ring_npu_flash_attn_backward kv_comm wait for update: {step}")
            k = next_k
            v = next_v

        next_dk = d_kv_comm.send_recv(dk)
        next_dv = d_kv_comm.send_recv(dv)
        d_kv_comm.commit()
        # print(f"{datetime.now()} current device is: {torch.cuda.current_device()},ring_npu_flash_attn_backward d_kv_comm commit: {step}")

    # Wait for the last round's dk/dv communication to finish
    d_kv_comm.wait()
    # print(f"{datetime.now()} current device is: {torch.cuda.current_device()},ring_npu_flash_attn_backward d_kv_comm wait for last: {step}")

    # Convert to the input dtype and return
    return dq.to(q.dtype), next_dk.to(q.dtype), next_dv.to(q.dtype)


class RingNpuFlashAttnFunc(torch.autograd.Function):

    @staticmethod
    def forward(ctx,
                group,
                q,
                k,
                v,
                head_num,
                input_layout="BSND",
                softmax_scale=None,
                causal=False,
                attn_type=AttnType.NPU,
                attn_processor=None):
        if softmax_scale is None:
            softmax_scale = q.shape[-1]**-0.5
        # Forward pass logic
        attention_out, softmax_max, softmax_sum = ring_npu_flash_attn_forward(group,
                                                                              q=q,
                                                                              k=k,
                                                                              v=v,
                                                                              head_num=head_num,
                                                                              softmax_scale=softmax_scale,
                                                                              input_layout=input_layout,
                                                                              causal=causal,
                                                                              attn_type=attn_type,
                                                                              attn_processor=attn_processor)
        # Save intermediate results for use in the backward pass
        ctx.save_for_backward(q, k, v, attention_out, softmax_max, softmax_sum)
        ctx.head_num = head_num
        ctx.input_layout = input_layout
        ctx.group = group
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.attn_type = attn_type
        ctx.attn_processor = attn_processor

        return attention_out

    @staticmethod
    def backward(ctx, grad_attention_out):
        # Retrieve the saved intermediate results
        q, k, v, attention_out, softmax_max, softmax_sum = ctx.saved_tensors
        # Backward pass logic
        # Assumes a function implementing the backward pass, e.g. `npu_fusion_attention_backward`
        grad_query, grad_key, grad_value = ring_npu_flash_attn_backward(ctx.group, q, k, v, grad_attention_out,
                                                                        ctx.head_num, ctx.input_layout, softmax_max,
                                                                        softmax_sum, attention_out, ctx.softmax_scale,
                                                                        ctx.causal, ctx.attn_type)
        return None, grad_query, grad_key, grad_value, None, None, None, None, None, None


def ring_npu_flash_attn_func(group,
                             q: torch.Tensor,
                             k: torch.Tensor,
                             v: torch.Tensor,
                             softmax_scale: float | None = None,
                             head_num: int | None = None,
                             input_layout: str = "BSND",
                             causal: bool = False,
                             attn_type: AttnType = AttnType.NPU,
                             attn_processor=None):
    head_num = q.shape[-2]
    return RingNpuFlashAttnFunc.apply(group, q, k, v, head_num, input_layout, softmax_scale, causal, attn_type,
                                      attn_processor)
