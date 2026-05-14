import torch
from .utils import RingComm, update_npu_out
from fastvideo.attention.yunchang.kernels.attention import (
    npu_fused_attn_forward, )
from fastvideo.attention.yunchang.kernels import select_flash_attn_impl, AttnType


def ring_npu_flash_attn_forward(process_group,
                                q: torch.Tensor,
                                k: torch.Tensor,
                                v: torch.Tensor,
                                softmax_scale: float = None,
                                head_num: int = None,
                                input_layout: str = "BSND",
                                causal: bool = False,
                                attn_type: AttnType = AttnType.NPU,
                                attn_processor=None):
    comm = RingComm(process_group)
    # print(f"{datetime.now()} current device is: {torch.cuda.current_device()}, ring_npu_flash_attn_forward")
    # 单卡场景直接计算
    if comm.world_size == 1:
        return npu_fused_attn_forward(q, k, v, head_num, input_layout,
                                      softmax_scale)

    attention_out, softmax_max, softmax_sum = None, None, None
    global_attention_out, global_softmax_max, global_softmax_sum = None, None, None

    next_k, next_v = None, None

    for step in range(comm.world_size):
        # print(f"{datetime.now()} current device is: {torch.cuda.current_device()},ring_npu_flash_attn_forward step: {step}")
        # 非最后一步：发起下一个kv的通信（异步）
        if step + 1 != comm.world_size:
            next_k = comm.send_recv(k)
            next_v = comm.send_recv(v)
            # print(f"{datetime.now()} current device is: {torch.cuda.current_device()},ring_npu_flash_attn_forward commit: {step}")
            comm.commit()

        # 当前step计算（仅当step <= 当前rank时处理本地kv）
        if not causal or step <= comm.rank:
            # print(f"{datetime.now()} current device is: {torch.cuda.current_device()},ring_npu_flash_attn_forward calculation: {step}")
            fn = select_flash_attn_impl(attn_type,
                                        stage="fwd-only",
                                        attn_processor=attn_processor)
            attention_out, softmax_max, softmax_sum = fn(
                q, k, v, head_num, input_layout, softmax_scale)
            global_attention_out, global_softmax_max, global_softmax_sum = update_npu_out(
                attention_out, softmax_max, softmax_sum, global_attention_out,
                global_softmax_max, global_softmax_sum)

        # 非最后一步：等待通信完成，更新kv
        if step + 1 != comm.world_size:
            comm.wait()
            # print(f"{datetime.now()} current device is: {torch.cuda.current_device()},ring_npu_flash_attn_forward wait: {step}")
            k = next_k
            v = next_v
    return global_attention_out, global_softmax_max, global_softmax_sum


def ring_npu_flash_attn_backward(process_group,
                                 q,
                                 k,
                                 v,
                                 grad_attention_out,
                                 head_num=None,
                                 input_layout="BSND",
                                 softmax_max=None,
                                 softmax_sum=None,
                                 attention_in=None,
                                 scale_value=None,
                                 causal=False,
                                 attn_type=AttnType.NPU):
    kv_comm = RingComm(process_group)
    d_kv_comm = RingComm(process_group)
    # print(f"{datetime.now()} current device is: {torch.cuda.current_device()}, ring_npu_flash_attn_backward")

    # 初始化梯度张量（避免None，用0张量初始化）
    dq = torch.zeros_like(q, dtype=torch.float32)
    dk = torch.zeros_like(k, dtype=torch.float32)
    dv = torch.zeros_like(v, dtype=torch.float32)
    next_k, next_v = None, None
    next_dk, next_dv = None, None

    for step in range(kv_comm.world_size):
        # 1. 发起kv通信（获取下一个step的kv）
        if step + 1 != kv_comm.world_size:
            next_k = kv_comm.send_recv(k)
            next_v = kv_comm.send_recv(v)
            # print(f"{datetime.now()} current device is: {torch.cuda.current_device()},ring_npu_flash_attn_backward commit: {step}")
            kv_comm.commit()

        # 2. 计算当前step的梯度
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
            # 累加query梯度（每个rank只计算自己的q梯度）
            dq += grad_query.to(torch.float32)

            # 累加kv梯度：如果不是第一步，需要加上通信过来的梯度
            if step > 0:
                d_kv_comm.wait()  # 等待上一轮dk/dv通信完成
                # print(f"{datetime.now()} current device is: {torch.cuda.current_device()},ring_npu_flash_attn_backward d_kv_comm wait: {step}")
                dk = grad_key.to(torch.float32) + next_dk
                dv = grad_value.to(torch.float32) + next_dv
            else:
                # 第一步直接赋值
                # print(f"{datetime.now()} current device is: {torch.cuda.current_device()},ring_npu_flash_attn_backward dkdv: {step}")
                dk = grad_key.to(torch.float32)
                dv = grad_value.to(torch.float32)
        else:
            # step > 当前rank：仅接收上一轮的dk/dv
            if step > 0:
                d_kv_comm.wait()
                # print(f"{datetime.now()} current device is: {torch.cuda.current_device()},ring_npu_flash_attn_backward d_kv_comm wait to next_dk: {step}")
                dk = next_dk
                dv = next_dv

        # 3. 等待kv通信完成，更新kv
        if step + 1 != kv_comm.world_size:
            kv_comm.wait()
            # print(f"{datetime.now()} current device is: {torch.cuda.current_device()},ring_npu_flash_attn_backward kv_comm wait for update: {step}")
            k = next_k
            v = next_v

        next_dk = d_kv_comm.send_recv(dk)
        next_dv = d_kv_comm.send_recv(dv)
        d_kv_comm.commit()
        # print(f"{datetime.now()} current device is: {torch.cuda.current_device()},ring_npu_flash_attn_backward d_kv_comm commit: {step}")

    # 等待最后一轮dk/dv通信完成
    d_kv_comm.wait()
    # print(f"{datetime.now()} current device is: {torch.cuda.current_device()},ring_npu_flash_attn_backward d_kv_comm wait for last: {step}")

    # 转换为输入 dtype 并返回
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
        # 前向传播逻辑
        attention_out, softmax_max, softmax_sum = ring_npu_flash_attn_forward(
            group,
            q=q,
            k=k,
            v=v,
            head_num=head_num,
            softmax_scale=softmax_scale,
            input_layout=input_layout,
            causal=causal,
            attn_type=attn_type,
            attn_processor=attn_processor)
        # 保存中间结果，以便在反向传播中使用
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
        # 获取保存的中间结果
        q, k, v, attention_out, softmax_max, softmax_sum = ctx.saved_tensors
        # 反向传播逻辑
        # 这里假设有一个实现反向传播的函数 `npu_fusion_attention_backward`
        grad_query, grad_key, grad_value = ring_npu_flash_attn_backward(
            ctx.group, q, k, v, grad_attention_out, ctx.head_num,
            ctx.input_layout, softmax_max, softmax_sum, attention_out,
            ctx.softmax_scale, ctx.causal, ctx.attn_type)
        return None, grad_query, grad_key, grad_value, None, None, None, None, None, None


def ring_npu_flash_attn_func(group,
                             q: torch.Tensor,
                             k: torch.Tensor,
                             v: torch.Tensor,
                             softmax_scale: float = None,
                             head_num: int = None,
                             input_layout: str = "BSND",
                             causal: bool = False,
                             attn_type: AttnType = AttnType.NPU,
                             attn_processor=None):
    head_num = q.shape[-2]
    return RingNpuFlashAttnFunc.apply(group, q, k, v, head_num, input_layout,
                                      softmax_scale, causal, attn_type,
                                      attn_processor)
