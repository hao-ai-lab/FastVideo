import triton
import triton.language as tl

CONFIG_LIST = [
    triton.Config({
        "BLOCK_M": 256,
        "BLOCK_N": 32
    }, num_stages=2, num_warps=4),
    triton.Config({
        "BLOCK_M": 128,
        "BLOCK_N": 64
    }, num_stages=2, num_warps=4),
    triton.Config({
        "BLOCK_M": 128,
        "BLOCK_N": 32
    }, num_stages=2, num_warps=4),
    triton.Config({
        "BLOCK_M": 64,
        "BLOCK_N": 128
    }, num_stages=2, num_warps=4),
    triton.Config({
        "BLOCK_M": 64,
        "BLOCK_N": 64
    }, num_stages=2, num_warps=4),
    triton.Config({
        "BLOCK_M": 64,
        "BLOCK_N": 32
    }, num_stages=2, num_warps=4),
    triton.Config({
        "BLOCK_M": 32,
        "BLOCK_N": 64
    }, num_stages=2, num_warps=4),
    triton.Config({
        "BLOCK_M": 32,
        "BLOCK_N": 128
    }, num_stages=2, num_warps=4),
    triton.Config({
        "BLOCK_M": 32,
        "BLOCK_N": 256
    }, num_stages=2, num_warps=4),
]


@triton.autotune(
    configs=CONFIG_LIST,
    key=["M", "N"],
)
@triton.jit
def _modulate_fwd(
    x_ptr,  # *Pointer* to first input vector.
    output_ptr,  # *Pointer* to output vector.
    scale_ptr,
    shift_ptr,
    m_stride,
    s_stride,
    M,
    N,
    seq_len,
    BLOCK_M: tl.constexpr,  # Number of elements each program should process.
    BLOCK_N: tl.constexpr,
    # NOTE: `constexpr` so it can be used as a shape value.
):
    row_id = tl.program_id(axis=0)  # We use a 1D launch grid so axis is 0.
    rows = row_id * BLOCK_M + tl.arange(0, BLOCK_M)
    batch_indices = rows // seq_len
    # s_rows = (row_id // seq_len) * BLOCK_M

    col_id = tl.program_id(axis=1)
    cols = col_id * BLOCK_N + tl.arange(0, BLOCK_N)

    x_ptrs = x_ptr + rows[:, None] * m_stride + cols[None, :]
    scale_ptrs = scale_ptr + batch_indices[:, None] * s_stride + cols[None, :]
    shift_ptrs = shift_ptr + batch_indices[:, None] * s_stride + cols[None, :]

    # scale_ptrs = scale_ptr + s_rows * s_stride + cols[None, :]
    # shift_ptrs = shift_ptr + s_rows * s_stride + cols[None, :]

    col_mask = cols[None, :] < N
    block_mask = (rows[:, None] < M) & col_mask
    x = tl.load(x_ptrs, mask=block_mask, other=0.0)
    scale = tl.load(scale_ptrs, mask=block_mask, other=0.0)
    shift = tl.load(shift_ptrs, mask=block_mask, other=0.0)

    output = x * (scale + 1) + shift
    # Write x + y back to DRAM.
    tl.store(output_ptr + rows[:, None] * m_stride + cols[None, :],
             output,
             mask=block_mask)


@triton.autotune(
    configs=CONFIG_LIST,
    key=["M", "N"],
)
@triton.jit
def _modulate_bwd(
    dx_ptr,  # *Pointer* to first input vector.
    x_ptr,
    dy_ptr,  # *Pointer* to output vector.
    scale_ptr,
    dscale_ptr,
    m_stride,
    s_stride,
    M,
    N,
    seq_len,
    BLOCK_M: tl.constexpr,  # Number of elements each program should process.
    BLOCK_N: tl.constexpr,
    # NOTE: `constexpr` so it can be used as a shape value.
):
    row_id = tl.program_id(axis=0)  # We use a 1D launch grid so axis is 0.
    rows = row_id * BLOCK_M + tl.arange(0, BLOCK_M)
    batch_indices = rows // seq_len
    col_id = tl.program_id(axis=1)
    cols = col_id * BLOCK_N + tl.arange(0, BLOCK_N)

    x_ptrs = x_ptr + rows[:, None] * m_stride + cols[None, :]
    dy_ptrs = dy_ptr + rows[:, None] * m_stride + cols[None, :]
    dx_ptrs = dx_ptr + rows[:, None] * m_stride + cols[None, :]
    dscale_ptrs = dscale_ptr + rows[:, None] * m_stride + cols[None, :]

    scale_ptrs = scale_ptr + batch_indices[:, None] * s_stride + cols[None, :]

    col_mask = cols[None, :] < N
    block_mask = (rows[:, None] < M) & col_mask
    x = tl.load(x_ptrs, mask=block_mask, other=0.0)
    dy = tl.load(dy_ptrs, mask=block_mask, other=0.0)
    scale = tl.load(scale_ptrs, mask=block_mask, other=0.0)

    dx = dy * (1 + scale)
    dscale = dy * x
    # Write x + y back to DRAM.
    tl.store(dx_ptrs, dx, mask=block_mask)
    tl.store(dscale_ptrs, dscale, mask=block_mask)
