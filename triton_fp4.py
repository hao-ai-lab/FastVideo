import argparse

import torch
import triton
import triton.language as tl
import triton.profiler as proton
from triton.tools.tensor_descriptor import TensorDescriptor
from triton.tools.mxfp import MXFP4Tensor, MXScaleTensor
from test_quantization_precision import generate_fake_quant

VEC_SIZE = 16
MAX_MXF4 = 6.0      # max magnitude for E2M1 (nvfp4) with M∈{0,1}, E∈{0..3}
EPS = 1e-8



def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"


def supports_block_scaling():
    return is_cuda() and torch.cuda.get_device_capability()[0] == 10


def _matmul_launch_metadata(grid, kernel, args):
    ret = {}
    M, N, K = args["M"], args["N"], args["K"]
    kernel_name = kernel.name
    if "ELEM_PER_BYTE_A" and "ELEM_PER_BYTE_B" and "VEC_SIZE" in args:
        kernel_name += "_nvfp4"
    ret["name"] = f"{kernel_name} [M={M}, N={N}, K={K}]"
    ret["flops"] = 2.0 * M * N * K
    return ret


@triton.jit(launch_metadata=_matmul_launch_metadata)
def block_scaled_matmul_kernel(  #
        a_desc,  #
        a_scale_desc,  #
        b_desc,  #
        b_scale_desc,  #
        c_desc,  #
        M: tl.constexpr,  #
        N: tl.constexpr,  #
        K: tl.constexpr,  #
        output_type: tl.constexpr,  #
        ELEM_PER_BYTE_A: tl.constexpr,  #
        ELEM_PER_BYTE_B: tl.constexpr,  #
        VEC_SIZE: tl.constexpr,  #
        BLOCK_M: tl.constexpr,  #
        BLOCK_N: tl.constexpr,  #
        BLOCK_K: tl.constexpr,  #
        rep_m: tl.constexpr,  #
        rep_n: tl.constexpr,  #
        rep_k: tl.constexpr,  #
        NUM_STAGES: tl.constexpr,  #
):  #
    if output_type == 0:
        output_dtype = tl.float32
    elif output_type == 1:
        output_dtype = tl.float16
    elif output_type == 2:
        output_dtype = tl.bfloat16
    elif output_type == 3:
        output_dtype = tl.float8e4nv

    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m
    offs_am = pid_m * BLOCK_M
    offs_bn = pid_n * BLOCK_N
    offs_k_a = 0
    offs_k_b = 0
    offs_scale_m = pid_m * rep_m
    offs_scale_n = pid_n * rep_n
    offs_scale_k = 0

    MIXED_PREC: tl.constexpr = ELEM_PER_BYTE_A == 1 and ELEM_PER_BYTE_B == 2

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in tl.range(0, tl.cdiv(K, BLOCK_K), num_stages=NUM_STAGES):
        a = a_desc.load([offs_am, offs_k_a])
        b = b_desc.load([offs_bn, offs_k_b])
        scale_a = a_scale_desc.load([0, offs_scale_m, offs_scale_k, 0, 0])
        scale_b = b_scale_desc.load([0, offs_scale_n, offs_scale_k, 0, 0])

        scale_a = scale_a.reshape(rep_m, rep_k, 32, 4, 4).trans(0, 3, 2, 1, 4).reshape(BLOCK_M, BLOCK_K // VEC_SIZE)
        scale_b = scale_b.reshape(rep_n, rep_k, 32, 4, 4).trans(0, 3, 2, 1, 4).reshape(BLOCK_N, BLOCK_K // VEC_SIZE)

        if MIXED_PREC:
            accumulator = tl.dot_scaled(a, scale_a, "e4m3", b.T, scale_b, "e2m1", accumulator)
        elif ELEM_PER_BYTE_A == 2 and ELEM_PER_BYTE_B == 2:
            accumulator = tl.dot_scaled(a, scale_a, "e2m1", b.T, scale_b, "e2m1", accumulator)
        else:
            accumulator = tl.dot_scaled(a, scale_a, "e4m3", b.T, scale_b, "e4m3", accumulator)

        offs_k_a += BLOCK_K // ELEM_PER_BYTE_A
        offs_k_b += BLOCK_K // ELEM_PER_BYTE_B
        offs_scale_k += rep_k

    c_desc.store([offs_am, offs_bn], accumulator.to(output_dtype))


def block_scaled_matmul(a_desc, a_scale_desc, b_desc, b_scale_desc, dtype_dst, M, N, K, rep_m, rep_n, rep_k, configs):
    output = torch.empty((M, N), dtype=dtype_dst, device="cuda")
    if dtype_dst == torch.float32:
        dtype_dst = 0
    elif dtype_dst == torch.float16:
        dtype_dst = 1
    elif dtype_dst == torch.bfloat16:
        dtype_dst = 2
    elif dtype_dst == torch.float8_e4m3fn:
        dtype_dst = 3
    else:
        raise ValueError(f"Unsupported dtype: {dtype_dst}")

    BLOCK_M = configs["BLOCK_SIZE_M"]
    BLOCK_N = configs["BLOCK_SIZE_N"]
    c_desc = TensorDescriptor.from_tensor(output, [BLOCK_M, BLOCK_N])

    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), 1)
    block_scaled_matmul_kernel[grid](
        a_desc,
        a_scale_desc,
        b_desc,
        b_scale_desc,
        c_desc,
        M,
        N,
        K,
        dtype_dst,
        configs["ELEM_PER_BYTE_A"],
        configs["ELEM_PER_BYTE_B"],
        configs["VEC_SIZE"],
        configs["BLOCK_SIZE_M"],
        configs["BLOCK_SIZE_N"],
        configs["BLOCK_SIZE_K"],
        rep_m,
        rep_n,
        rep_k,
        configs["num_stages"],
    )
    return output
def initialize_block_scaled(M, N, K, block_scale_type="nvfp4", compute_reference=False, input_output_dtype=torch.bfloat16):
    assert block_scale_type == "nvfp4", f"Only nvfp4 supported here (got {block_scale_type})"

    # --- constants ---
    BLOCK_M = 128
    BLOCK_N = 256
    BLOCK_K = 256
    VEC_SIZE = 16
    ELEM_PER_BYTE_A = 2
    ELEM_PER_BYTE_B = 2
    device = "cuda"
    MAX_MXF4 = 6.0
    EPS = 1e-8

    assert M % 128 == 0
    assert N % 128 == 0
    assert K % (VEC_SIZE * 4) == 0

    # --- helpers: tile scales (FP8 E4M3) ---
    def _tile_scales_A(a_fp32):
        m_chunks = M // 128
        k_chunks = K // (VEC_SIZE * 4)  # 64-wide tiles along K
        x = a_fp32.view(m_chunks, 128, k_chunks, 4, VEC_SIZE)
        tile_max = x.abs().amax(dim=(1, 3, 4))                     # [m_chunks, k_chunks]
        scale = torch.clamp(tile_max / MAX_MXF4, min=EPS)          # fp32
        s = scale[:, :, None, None].expand(m_chunks, k_chunks, 32, 16)
        return s.to(torch.float8_e4m3fn)                           # [m_chunks, k_chunks, 32, 16]

    def _tile_scales_B(b_fp32):  # b_fp32 is (K, N)
        k_chunks = K // (VEC_SIZE * 4)
        n_chunks = N // 128
        x = b_fp32.view(k_chunks, 4, VEC_SIZE, n_chunks, 128)
        tile_max = x.abs().amax(dim=(1, 2, 4))                     # [k_chunks, n_chunks]
        scale = torch.clamp(tile_max / MAX_MXF4, min=EPS).t()      # [n_chunks, k_chunks]
        s = scale[:, :, None, None].expand(n_chunks, k_chunks, 32, 16)
        return s.to(torch.float8_e4m3fn)                           # [n_chunks, k_chunks, 32, 16]

    # --- expand tile scales to per-element fp32 (for quant/dequant math) ---
    def _expand_A_scale_full(a_scale_fp8):
        m_chunks, k_chunks, _, _ = a_scale_fp8.shape
        base = a_scale_fp8[..., 0, 0].to(input_output_dtype)            # [m_chunks, k_chunks]
        tile = base[:, :, None, None].expand(m_chunks, k_chunks, 128, 64)
        return tile.permute(0, 2, 1, 3).reshape(M, k_chunks * 64).contiguous()  # [M, K]

    def _expand_B_scale_full(b_scale_fp8):
        n_chunks, k_chunks, _, _ = b_scale_fp8.shape
        base = b_scale_fp8[..., 0, 0].to(input_output_dtype)            # [n_chunks, k_chunks]
        tile = base[:, :, None, None].expand(n_chunks, k_chunks, 64, 128)
        return tile.permute(1, 2, 0, 3).reshape(k_chunks * 64, n_chunks * 128).contiguous()  # [K, N]

    # --- 1) fp32 sources (pre-quant) ---
    torch.manual_seed(0)
    a_f32_pre = torch.randn((M, K), device=device, dtype=input_output_dtype)
    b_f32_pre = torch.randn((K, N), device=device, dtype=input_output_dtype)

    # --- 2) tile scales (FP8 E4M3, finite-only) ---
    a_scale_fp8 = _tile_scales_A(a_f32_pre)    # [M//128, K//64, 32, 16]
    b_scale_fp8 = _tile_scales_B(b_f32_pre)    # [N//128, K//64, 32, 16]

    # --- 3) expand scales to full for quantization ---
    a_scale_full = _expand_A_scale_full(a_scale_fp8)  # [M, K] fp32
    b_scale_full = _expand_B_scale_full(b_scale_fp8)  # [K, N] fp32

    # --- 4) quantize to nvfp4 codes (E2M1) ---
    a_codes = MXFP4Tensor(data=(a_f32_pre / a_scale_full)).data             # [M, K] uint8 (nibble used)
    b_codes_KN = MXFP4Tensor(data=(b_f32_pre / b_scale_full)).data          # [K, N] uint8

    # IMPORTANT: the kernel expects B laid out as (N, K_packed) (like original)
    # so create a view for descriptors with shape (N, K) and pack along dim=1.
    b_codes_NK = b_codes_KN.transpose(0, 1).contiguous()                    # [N, K] uint8

    # --- 5) pack to 2-per-byte along K and build descriptors ---
    a_fp4 = MXFP4Tensor(size=(M, K), device=device); a_fp4.data = a_codes
    b_fp4_for_desc = MXFP4Tensor(size=(N, K), device=device); b_fp4_for_desc.data = b_codes_NK

    a_packed = a_fp4.to_packed_tensor(dim=1)         # (M, K/2)
    b_packed = b_fp4_for_desc.to_packed_tensor(dim=1)  # (N, K/2)  <-- matches kernel indexing

    a_desc = TensorDescriptor.from_tensor(a_packed, [BLOCK_M, BLOCK_K // ELEM_PER_BYTE_A])
    b_desc = TensorDescriptor.from_tensor(b_packed, [BLOCK_N, BLOCK_K // ELEM_PER_BYTE_B])

    rep_m = BLOCK_M // 128
    rep_n = BLOCK_N // 128
    rep_k = BLOCK_K // VEC_SIZE // 4

    # reshape FP8 scales to 5D TMA blocks [1, rep_*, rep_k, 2, 256]
    a_scale_5d = a_scale_fp8.reshape(1, a_scale_fp8.shape[0], a_scale_fp8.shape[1], 2, 256)
    b_scale_5d = b_scale_fp8.reshape(1, b_scale_fp8.shape[0], b_scale_fp8.shape[1], 2, 256)
    a_scale_desc = TensorDescriptor.from_tensor(a_scale_5d, [1, rep_m, rep_k, 2, 256])
    b_scale_desc = TensorDescriptor.from_tensor(b_scale_5d, [1, rep_n, rep_k, 2, 256])

    # --- 6) dequantize back to fp32 for the reference (post-quant tensors) ---
    a_ref_fp4 = MXFP4Tensor(size=(M, K), device=device); a_ref_fp4.data = a_codes
    b_ref_fp4 = MXFP4Tensor(size=(K, N), device=device); b_ref_fp4.data = b_codes_KN

    a_ref_f32, b_ref_f32 = generate_fake_quant(a_f32_pre, b_f32_pre)

    reference = a_ref_f32 @ b_ref_f32 if compute_reference else None

    configs = {
        "BLOCK_SIZE_M": BLOCK_M,
        "BLOCK_SIZE_N": BLOCK_N,
        "BLOCK_SIZE_K": BLOCK_K,
        "num_stages": 4,
        "ELEM_PER_BYTE_A": ELEM_PER_BYTE_A,
        "ELEM_PER_BYTE_B": ELEM_PER_BYTE_B,
        "VEC_SIZE": VEC_SIZE,
    }
    # RETURN DESCRIPTORS (not raw tensors!)
    return a_desc, a_scale_desc, b_desc, b_scale_desc, rep_m, rep_n, rep_k, configs, reference


def validate_block_scaled(M, N, K, block_scale_type="nvfp4", input_output_dtype=torch.bfloat16):
    a_desc, a_scale, b_desc, b_scale, rep_m, rep_n, rep_k, configs, reference = initialize_block_scaled(
        M, N, K, block_scale_type, compute_reference=True)
    output = block_scaled_matmul(a_desc, a_scale, b_desc, b_scale, input_output_dtype, M, N, K, rep_m, rep_n, rep_k, configs)
    torch.testing.assert_close(reference, output.to(input_output_dtype), atol=1e-3, rtol=1e-3)
    print(f"✅ (pass {block_scale_type})")


def bench_block_scaled(K, block_scale_type="nvfp4", reps=10):
    assert K % 128 == 0
    M = 8192
    N = 8192
    print(f"Problem Shape = {M}x{N}x{K}")

    a_desc, a_scale, b_desc, b_scale, rep_m, rep_n, rep_k, configs, _ = initialize_block_scaled(
        M, N, K, block_scale_type, compute_reference=False)
    _ = block_scaled_matmul(a_desc, a_scale, b_desc, b_scale, torch.float16, M, N, K, rep_m, rep_n, rep_k, configs)

    proton.activate(0)
    for _ in range(reps):
        _ = block_scaled_matmul(a_desc, a_scale, b_desc, b_scale, torch.float16, M, N, K, rep_m, rep_n, rep_k, configs)
    proton.deactivate(0)
    print("Done benchmarking")


def show_profile(profile_name):
    import triton.profiler.viewer as proton_viewer

    metric_names = ["time/ms"]
    metric_names = ["tflop/s"] + metric_names
    file_name = f"{profile_name}.hatchet"
    tree, metrics = proton_viewer.parse(metric_names, file_name)
    proton_viewer.print_tree(tree, metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-K", type=int, required=False, default=512)
    parser.add_argument("--K_range", type=int, nargs=2)
    parser.add_argument("--K_step", type=int, default=512)
    parser.add_argument("--bench", action="store_true", default=True)
    parser.add_argument("--format", type=str, choices=["nvfp4"], default="nvfp4")
    parser.add_argument("--input_output_dtype", type=str, choices=["float32", "float16", "bfloat16"], default="bfloat16")
    args = parser.parse_args()

    if not supports_block_scaling():
        print("⛔ This example requires GPU support for block scaled matmul")
    else:
        if args.K and args.K_range is None:
            args.K_range = [args.K, args.K]
            args.K_step = 1  # doesn't matter as long as it's not 0

        torch.manual_seed(42)
        dtype =torch.float32 if args.input_output_dtype == "float32" else torch.float16 if args.input_output_dtype == "float16" else torch.bfloat16
        validate_block_scaled(8192, 8192, 8192, block_scale_type=args.format, input_output_dtype=dtype)


        if args.bench:
            proton.start("block_scaled_matmul", hook="triton")
            proton.deactivate(0)  # Skip argument creation
            for K in range(args.K_range[0], args.K_range[1] + 1, args.K_step):
                bench_block_scaled(K, reps=10000, block_scale_type=args.format)
            proton.finalize()
            show_profile("block_scaled_matmul")