import math
import torch
import triton
import triton.language as tl

def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"


def is_hip():
    target = triton.runtime.driver.active.get_current_target()
    return target.backend == 'hip'


def is_cdna3_cdna4():
    target = triton.runtime.driver.active.get_current_target()
    return (target.arch == 'gfx1201' or target.arch == 'gfx1101' or target.arch == 'gfx1100' or target.arch == 'gfx1030')


def get_common_autotune_config():
    # cdna arch does not support a 4-stage software pipeline, see https://github.com/ROCm/triton/issues/916
    supported_num_staged = [1, 2] if is_cdna3_cdna4() else [1, 2, 3, 4]
    configs = [
        triton.Config({'BLOCK_Q': BLOCK_Q, 'BLOCK_KV': BLOCK_KV}, num_stages=s, num_warps=w) \
        for BLOCK_Q in [32, 64, 128]\
        for BLOCK_KV in [32, 64, 128]\
        for s in supported_num_staged\
        for w in [4, 8]\
    ]
    return configs


def get_cuda_autotune_config():
    # cuda and hip can use differnt autotune configs
    return get_common_autotune_config()


def get_hip_autotune_config():
    # cuda and hip can use differnt autotune configs
    return get_common_autotune_config()


def get_autotune_config():
    if is_cuda():
        return get_cuda_autotune_config()
    else:
        return get_hip_autotune_config()


@triton.jit
def clamp_int(value, min_val, max_val):
    ret = tl.where(value > max_val, max_val, value)
    ret = tl.where(ret < min_val, min_val, ret)
    return ret


@triton.jit
def _attn_fwd_loop(
    q, k, v, kv_mask, m, l, acc, sm_scale,
    MASK_KV: tl.constexpr,
):
    scores = tl.dot(q, k.T) #[BLOCK_Q, BLOCK_KV]
    scores = scores * sm_scale
    if MASK_KV:
        scores = tl.where(kv_mask[None, :], scores, -float('inf'))

    current_m = tl.max(scores, axis=1)
    new_m = tl.maximum(m, current_m)
    exp_scores = tl.math.exp2(scores - new_m[:, None])
    current_l = tl.sum(exp_scores, axis=1)

    # Update L <- L * exp(M - M') + L1, M <- M'
    alpha = tl.math.exp2(m - new_m)
    l = l * alpha + current_l
    m = new_m

    # Update O <- O * exp(M - M') + P @ V
    acc = (acc * alpha[:, None] + tl.dot(exp_scores.to(v.type.element_ty), v))

    return m, l, acc


@triton.autotune(
    configs=get_autotune_config(),
    key=['head_dim'],
)
@triton.jit
def triton_sta_kernel(
    Q, K, V, output,
    batch_size: int, num_heads: int, seq_len: int,  head_dim: int,
    img_seq_len: int,
    text_length: int,
    canvas_t: int, canvas_h: int, canvas_w: int,
    kernel_t: int, kernel_h: int, kernel_w: int,
    tile_t: int, tile_h: int, tile_w: int,
    scale: float,
    has_text: tl.constexpr,
    text_q: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
):
    total_tile_size = tile_t * tile_h * tile_w
    q_block_per_tile = (total_tile_size + BLOCK_Q - 1) // BLOCK_Q

    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    if text_q:
        q_block_idx = tl.program_id(2)
    else:
        q_tile_flat = tl.program_id(2) // q_block_per_tile
        q_block_idx = tl.program_id(2) % q_block_per_tile

    m = tl.full((BLOCK_Q,), -float('inf'), dtype=tl.float32)
    l = tl.zeros((BLOCK_Q,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_Q, BLOCK_DIM), dtype=tl.float32)

    q_offset = (batch_idx * num_heads + head_idx) * seq_len * head_dim
    if text_q:
        q_base_idx = img_seq_len + q_block_idx * BLOCK_Q
    else:
        q_base_idx = q_tile_flat * total_tile_size + q_block_idx * BLOCK_Q

    q_offset_in_tile = tl.arange(0, BLOCK_Q)
    q_idx = q_base_idx + q_offset_in_tile
    q_mask = (q_block_idx * BLOCK_Q + tl.arange(0, BLOCK_Q)) < total_tile_size

    q = tl.load(
        Q + q_offset + q_idx[:, None] * head_dim + tl.arange(0, BLOCK_DIM)[None, :],
        mask=q_mask[:, None],
        other=0.0
    )  # [BLOCK_Q, BLOCK_DIM]

    # Scale sm_scale by log_2(e) and use 2^x instead of exp
    sm_scale = scale * 1.4426950408889634

    num_tiles_t = canvas_t // tile_t
    num_tiles_h = canvas_h // tile_h
    num_tiles_w = canvas_w // tile_w
    tiles_per_hw = num_tiles_h * num_tiles_w

    if text_q:
        kv_tile_start_t = 0
        kv_tile_end_t = num_tiles_t

        kv_tile_start_h = 0
        kv_tile_end_h = num_tiles_h

        kv_tile_start_w = 0
        kv_tile_end_w = num_tiles_w

    else:
        q_tile_t = q_tile_flat // tiles_per_hw
        remaining = q_tile_flat % tiles_per_hw
        q_tile_h = remaining // num_tiles_w
        q_tile_w = remaining % num_tiles_w

        kernel_center_t = clamp_int(q_tile_t, kernel_t // 2, (num_tiles_t - 1) - kernel_t // 2)
        kernel_center_h = clamp_int(q_tile_h, kernel_h // 2, (num_tiles_h - 1) - kernel_h // 2)
        kernel_center_w = clamp_int(q_tile_w, kernel_w // 2, (num_tiles_w - 1) - kernel_w // 2)

        kv_tile_start_t = kernel_center_t - kernel_t // 2
        kv_tile_end_t = kernel_center_t + kernel_t // 2 + 1
        kv_tile_end_t = tl.where(kv_tile_end_t > num_tiles_t, num_tiles_t, kv_tile_end_t)

        kv_tile_start_h = kernel_center_h - kernel_h // 2
        kv_tile_end_h = kernel_center_h + kernel_h // 2 + 1
        kv_tile_end_h = tl.where(kv_tile_end_h > num_tiles_h, num_tiles_h, kv_tile_end_h)

        kv_tile_start_w = kernel_center_w - kernel_w // 2
        kv_tile_end_w = kernel_center_w + kernel_w // 2 + 1
        kv_tile_end_w = tl.where(kv_tile_end_w > num_tiles_w, num_tiles_w, kv_tile_end_w)

    # for kv_img
    for kv_tile_t in tl.range(kv_tile_start_t, kv_tile_end_t):
        for kv_tile_h in tl.range(kv_tile_start_h, kv_tile_end_h):
            for kv_tile_w in tl.range(kv_tile_start_w, kv_tile_end_w):
                kv_base_idx = (kv_tile_t * num_tiles_h * num_tiles_w + kv_tile_h * num_tiles_w + kv_tile_w) * total_tile_size

                for kv_block_idx in tl.range(0, total_tile_size, BLOCK_KV):
                    kv_offset_in_block = tl.arange(0, BLOCK_KV)
                    kv_idx = kv_base_idx + kv_block_idx + kv_offset_in_block
                    kv_mask = (kv_block_idx + tl.arange(0, BLOCK_KV)) < total_tile_size

                    kv_offset = (batch_idx * num_heads + head_idx) * seq_len * head_dim

                    k = tl.load(
                        K + kv_offset + kv_idx[:, None] * head_dim + tl.arange(0, BLOCK_DIM)[None, :],
                        mask=kv_mask[:, None],
                        other=0.0
                    )  # [BLOCK_KV, BLOCK_DIM]
                    v = tl.load(
                        V + kv_offset + kv_idx[:, None] * head_dim + tl.arange(0, BLOCK_DIM)[None, :],
                        mask=kv_mask[:, None],
                        other=0.0
                    )  # [BLOCK_KV, BLOCK_DIM]

                    m, l, acc = _attn_fwd_loop(q, k, v, kv_mask, m, l, acc, sm_scale, False)


    # for kv_text
    if has_text:
        kv_base_idx = img_seq_len
        for kv_block_idx in tl.range(0, total_tile_size, BLOCK_KV):
            kv_offset_in_block = tl.arange(0, BLOCK_KV)
            kv_idx = kv_base_idx + kv_block_idx + kv_offset_in_block
            kv_mask = (kv_block_idx + tl.arange(0, BLOCK_KV)) < text_length

            kv_offset = (batch_idx * num_heads + head_idx) * seq_len * head_dim

            k = tl.load(
                K + kv_offset + kv_idx[:, None] * head_dim + tl.arange(0, BLOCK_DIM)[None, :],
                mask=kv_mask[:, None],
                other=0.0
            )  # [BLOCK_KV, BLOCK_DIM]
            v = tl.load(
                V + kv_offset + kv_idx[:, None] * head_dim + tl.arange(0, BLOCK_DIM)[None, :],
                mask=kv_mask[:, None],
                other=0.0
            )  # [BLOCK_KV, BLOCK_DIM]

            m, l, acc = _attn_fwd_loop(q, k, v, kv_mask, m, l, acc, sm_scale, True)


    output_acc = acc / l[:, None]
    tl.store(
        output + q_offset + q_idx[:, None] * head_dim + tl.arange(0, BLOCK_DIM)[None, :],
        output_acc,
        mask=q_mask[:, None]
    ) # [BLOCK_Q, BLOCK_DIM]


def sliding_tile_attention_triton(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    window_size, text_length: int,
    has_text=True, dit_seq_shape='30x48x80') -> torch.Tensor:
    seq_length = q.shape[2]
    if has_text:
        assert q.shape[2] >= 115200 and q.shape[2] <= 115456, f"Unsupported {dit_seq_shape}, current shape is {q.shape}, only support '30x48x80' for HunyuanVideo"
        target_size = math.ceil(seq_length / 384) * 384
        pad_size = target_size - seq_length
        if pad_size > 0:
            q = torch.cat([q, q[:, :, -pad_size:]], dim=2)
            k = torch.cat([k, k[:, :, -pad_size:]], dim=2)
            v = torch.cat([v, v[:, :, -pad_size:]], dim=2)
    else:
        if dit_seq_shape == '36x48x48': # Stepvideo
            assert q.shape[2] == 82944
        elif dit_seq_shape == '18x48x80': # Wan
            assert q.shape[2] == 69120
        else:
            raise ValueError(f"Unsupported {dit_seq_shape}, current shape is {q.shape}, only support '36x48x48' for Stepvideo and '18x48x80' for Wan")
    assert q.shape[1] == len(window_size), "Number of heads must match the number of window sizes"

    batch_size, num_heads, seq_len, head_dim = q.shape
    if dit_seq_shape == '30x48x80': # Hunyuan
        canvas_t, canvas_h, canvas_w = 30, 48, 80
        tile_t, tile_h, tile_w = 6, 8, 8
    elif dit_seq_shape == '36x48x48': # Stepvideo
        canvas_t, canvas_h, canvas_w = 36, 48, 48
        tile_t, tile_h, tile_w = 6, 8, 8
    elif dit_seq_shape == '18x48x80': # Wan
        canvas_t, canvas_h, canvas_w = 18, 48, 80
        tile_t, tile_h, tile_w = 6, 8, 8

    img_seq_len = canvas_t * canvas_h * canvas_w

    num_tiles_t = canvas_t // tile_t
    num_tiles_h = canvas_h // tile_h
    num_tiles_w = canvas_w // tile_w
    num_tiles = num_tiles_t * num_tiles_h * num_tiles_w

    total_tile_size = tile_t * tile_h * tile_w

    # BLOCK_Q=128
    # BLOCK_KV=128
    BLOCK_DIM = head_dim

    output = torch.empty_like(q)

    # for q_img
    # kernel_size maybe different for different head
    # This for loop is ugly. but it is actually quite efficient. The sequence dimension alone can already oversubscribe SMs
    for head_index, (kernel_t, kernel_h, kernel_w) in enumerate(window_size):
        for batch in range(batch_size):
            q_head, k_head, v_head, o_head = (q[batch:batch + 1, head_index:head_index + 1],
                                              k[batch:batch + 1, head_index:head_index + 1],
                                              v[batch:batch + 1, head_index:head_index + 1],
                                              output[batch:batch + 1, head_index:head_index + 1])

            # triton_sta_kernel[(1, 1, num_tiles * triton.cdiv(total_tile_size, BLOCK_Q))](
            grid = lambda META: (1, 1, num_tiles * triton.cdiv(total_tile_size, META['BLOCK_Q']))
            triton_sta_kernel[grid](
                q_head, k_head, v_head, o_head,
                1, 1, seq_len, head_dim,
                img_seq_len,
                text_length,
                canvas_t, canvas_h, canvas_w,
                kernel_t, kernel_h, kernel_w,
                tile_t, tile_h, tile_w,
                scale=1.0 / (head_dim ** 0.5),
                has_text=has_text,
                text_q=False,
                # BLOCK_Q=BLOCK_Q,
                # BLOCK_KV=BLOCK_KV,
                BLOCK_DIM=BLOCK_DIM,
            )

    # for q_text
    # kernel_t, kernel_h, kernel_w is not used, set to (3, 3, 3)
    if has_text:
        # triton_sta_kernel[(batch_size, num_heads, triton.cdiv(total_tile_size, BLOCK_Q))](
        grid = lambda META: (batch_size, num_heads, triton.cdiv(total_tile_size, META['BLOCK_Q']))
        triton_sta_kernel[grid](
            q, k, v, output,
            batch_size, num_heads, seq_len, head_dim,
            img_seq_len,
            text_length,
            canvas_t, canvas_h, canvas_w,
            3, 3, 3,
            #kernel_t, kernel_h, kernel_w,
            tile_t, tile_h, tile_w,
            scale=1.0 / (head_dim ** 0.5),
            has_text=has_text,
            text_q=True,
            # BLOCK_Q=BLOCK_Q,
            # BLOCK_KV=BLOCK_KV,
            BLOCK_DIM=BLOCK_DIM,
        )

    if has_text:
        if pad_size > 0:
            output = output[:, :, :seq_length]

    return output