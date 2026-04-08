import torch
from triton.tools.tensor_descriptor import TensorDescriptor
from triton.tools.mxfp import MXFP4Tensor
from triton_fp4 import block_scaled_matmul
from flashinfer import SfLayout, mm_fp4, nvfp4_quantize



def triton_nvfp4(A: torch.Tensor,
                 B: torch.Tensor,
                 out_dtype: torch.dtype = torch.float16):
    """
    Block-scaled nvfp4 GEMM: (A @ B) with A:(M,K), B:(K,N) given in fp32/bf16.
    Returns the GEMM output in `out_dtype`.

    Requirements (match your kernel):
      - M % 128 == 0, N % 128 == 0, K % 64 == 0 (since vec_size=16 and 4 groups → 64)
      - Device: CUDA, Blackwell (SM=10)
    """
    assert A.is_cuda and B.is_cuda, "Inputs must be on CUDA"
    assert A.dtype in (torch.float32, torch.bfloat16), "A must be fp32/bf16"
    assert B.dtype in (torch.float32, torch.bfloat16), "B must be fp32/bf16"
    M, K = A.shape
    Kb, N = B.shape
    assert Kb == K, "Inner dims must match"
    assert M % 128 == 0 and N % 128 == 0 and K % 64 == 0, "Shape must respect tiling"

    # ---- kernel constants / config ----
    BLOCK_M = 128
    BLOCK_N = 256
    BLOCK_K = 256
    VEC_SIZE = 16
    ELEM_PER_BYTE_A = 2
    ELEM_PER_BYTE_B = 2
    MAX_MXF4 = 6.0    # nvfp4 (E2M1) max magnitude
    EPS = 1e-8

    device = A.device

    # ---- tile-scale helpers (FP8 E4M3, finite-only) ----
    # A tiles are 128 x 64 along (M,K)
    def _tile_scales_A(a_fp32):
        m_chunks = M // 128
        k_chunks = K // (VEC_SIZE * 4)  # 64 wide
        x = a_fp32.view(m_chunks, 128, k_chunks, 4, VEC_SIZE)
        tile_max = x.abs().amax(dim=(1, 3, 4))                       # [m_chunks, k_chunks]
        scale = torch.clamp(tile_max / MAX_MXF4, min=EPS)            # fp32
        s = scale[:, :, None, None].expand(m_chunks, k_chunks, 32, 16)
        return s.to(torch.float8_e4m3fn)                              # [m_chunks, k_chunks, 32, 16]

    # B tiles are 64 x 128 over (K,N)
    def _tile_scales_B(b_fp32):
        k_chunks = K // (VEC_SIZE * 4)
        n_chunks = N // 128
        x = b_fp32.view(k_chunks, 4, VEC_SIZE, n_chunks, 128)
        tile_max = x.abs().amax(dim=(1, 2, 4))                        # [k_chunks, n_chunks]
        scale = torch.clamp(tile_max / MAX_MXF4, min=EPS).t()         # [n_chunks, k_chunks]
        s = scale[:, :, None, None].expand(n_chunks, k_chunks, 32, 16)
        return s.to(torch.float8_e4m3fn)                              # [n_chunks, k_chunks, 32, 16]

    def _expand_A_scale_full(a_scale_fp8):
        m_chunks, k_chunks, _, _ = a_scale_fp8.shape
        base = a_scale_fp8[..., 0, 0].to(A.dtype)               # [m_chunks, k_chunks]
        tile = base[:, :, None, None].expand(m_chunks, k_chunks, 128, 64)
        return tile.permute(0, 2, 1, 3).reshape(M, k_chunks * 64).contiguous()  # [M,K]

    def _expand_B_scale_full(b_scale_fp8):
        n_chunks, k_chunks, _, _ = b_scale_fp8.shape
        base = b_scale_fp8[..., 0, 0].to(B.dtype)               # [n_chunks, k_chunks]
        tile = base[:, :, None, None].expand(n_chunks, k_chunks, 64, 128)
        return tile.permute(1, 2, 0, 3).reshape(k_chunks * 64, n_chunks * 128).contiguous()  # [K,N]

    # ---- 1) scales from fp32 ----
    a_scale_fp8 = _tile_scales_A(A)             # [M//128, K//64, 32, 16]
    b_scale_fp8 = _tile_scales_B(B)             # [N//128, K//64, 32, 16]
    a_scale_full = _expand_A_scale_full(a_scale_fp8)   # [M,K] fp32
    b_scale_full = _expand_B_scale_full(b_scale_fp8)   # [K,N] fp32

    # ---- 2) quantize to nvfp4 codes (E2M1) ----
    a_codes = MXFP4Tensor(data=(A / a_scale_full)).data            # [M,K] uint8 (low nibble used)
    b_codes_KN = MXFP4Tensor(data=(B / b_scale_full)).data         # [K,N] uint8

    # Kernel expects B laid out as (N, K) packed along K (like the original example)
    b_codes_NK = b_codes_KN.transpose(0, 1).contiguous()             # [N,K]

    # ---- 3) pack to 2-per-byte along K and build descriptors ----
    a_fp4 = MXFP4Tensor(size=(M, K), device=device); a_fp4.data = a_codes
    b_fp4_desc = MXFP4Tensor(size=(N, K), device=device); b_fp4_desc.data = b_codes_NK

    a_packed = a_fp4.to_packed_tensor(dim=1)      # (M, K/2)
    b_packed = b_fp4_desc.to_packed_tensor(dim=1) # (N, K/2)    
    
    a_desc = TensorDescriptor.from_tensor(a_packed, [BLOCK_M, BLOCK_K // ELEM_PER_BYTE_A])
    b_desc = TensorDescriptor.from_tensor(b_packed, [BLOCK_N, BLOCK_K // ELEM_PER_BYTE_B])

    # reps for scale swizzle
    rep_m = BLOCK_M // 128
    rep_n = BLOCK_N // 128
    rep_k = BLOCK_K // VEC_SIZE // 4

    # reshape FP8 scales into 5D TMA blocks [1, rep_*, rep_k, 2, 256]
    a_scale_5d = a_scale_fp8.reshape(1, a_scale_fp8.shape[0], a_scale_fp8.shape[1], 2, 256)
    b_scale_5d = b_scale_fp8.reshape(1, b_scale_fp8.shape[0], b_scale_fp8.shape[1], 2, 256)
    a_scale_desc = TensorDescriptor.from_tensor(a_scale_5d, [1, rep_m, rep_k, 2, 256])
    b_scale_desc = TensorDescriptor.from_tensor(b_scale_5d, [1, rep_n, rep_k, 2, 256])

    a_packed_cutlass, a_scale_fp8_cutlass = nvfp4_quantize(
        A, torch.tensor(1.0, device=device, dtype=torch.float32), sfLayout=SfLayout.layout_128x4, do_shuffle=False
    )
    import pdb; pdb.set_trace()
    # ---- 4) launch GEMM ----
    configs = {
        "BLOCK_SIZE_M": BLOCK_M,
        "BLOCK_SIZE_N": BLOCK_N,
        "BLOCK_SIZE_K": BLOCK_K,
        "num_stages": 4,
        "ELEM_PER_BYTE_A": ELEM_PER_BYTE_A,
        "ELEM_PER_BYTE_B": ELEM_PER_BYTE_B,
        "VEC_SIZE": VEC_SIZE,
    }
    # Uses your already-defined block_scaled_matmul(...)
    C = block_scaled_matmul(
        a_desc, a_scale_desc,
        b_desc, b_scale_desc,
        out_dtype, M, N, K,
        rep_m, rep_n, rep_k, configs
    )
    return C



def cutlass_nvfp4(A: torch.Tensor,
                  B: torch.Tensor,
                  out_dtype: torch.dtype = torch.bfloat16,
                  sf_layout: SfLayout = SfLayout.layout_128x4,
                  block_size: int = 16,
                  do_shuffle: bool = False,
                  scale_factor_A: torch.Tensor | float | None = None,
                  scale_factor_B: torch.Tensor | float | None = None) -> torch.Tensor:
    """
    FlashInfer/CUTLASS nvfp4 GEMM: returns (A @ B) using 4-bit E2M1 operands with per-tile scaling.

    Args:
      A: (M, K) tensor, fp32 or bf16, CUDA.
      B: (K, N) tensor, fp32 or bf16, CUDA.
      out_dtype: torch.float32, torch.float16, or torch.bfloat16 (recommended).
      sf_layout: FlashInfer scale-factor layout (default SfLayout.layout_128x4).
      block_size: internal tiling, default 16.
      do_shuffle: pass-through to nvfp4_quantize.
      scale_factor_A, scale_factor_B: optional global scale factors (float or 0-dim tensor).
        If None, defaults to 1.0 (FlashInfer still applies its internal per-tile scales).

    Returns:
      C: (M, N) tensor in out_dtype
    """
    assert A.is_cuda and B.is_cuda, "Inputs must be CUDA tensors"
    assert A.dim() == 2 and B.dim() == 2, "A and B must be 2D"
    M, K = A.shape
    Kb, N = B.shape
    assert K == Kb, f"Incompatible shapes: A is (M={M},K={K}), B is (K={Kb},N={N})"

    # Cast inputs to supported accumulation input types for FlashInfer quantizer.
    # (fp32/bf16 are fine; we keep user dtype if already bf16 to save bandwidth.)
    A_in = A if A.dtype in (torch.bfloat16, torch.float32) else A.float()
    B_in = B if B.dtype in (torch.bfloat16, torch.float32) else B.float()
    B_in = B_in.T.contiguous()

    # Choose global scaling (FlashInfer also uses per-tile scales internally).
    if scale_factor_A is None:
        scale_factor_A = torch.tensor(1.0, device=A.device, dtype=torch.float32)
    elif not torch.is_tensor(scale_factor_A):
        scale_factor_A = torch.tensor(float(scale_factor_A), device=A.device, dtype=torch.float32)

    if scale_factor_B is None:
        scale_factor_B = torch.tensor(1.0, device=B.device, dtype=torch.float32)
    elif not torch.is_tensor(scale_factor_B):
        scale_factor_B = torch.tensor(float(scale_factor_B), device=B.device, dtype=torch.float32)

    # Quantize A and B to nvfp4; inv scales are FlashInfer’s per-tile inverse scales.
    # Returned quantized matrices are FP4-packed formats understood by mm_fp4.
    A_q, A_inv_sf = nvfp4_quantize(
        A_in, scale_factor_A, sfLayout=sf_layout, do_shuffle=do_shuffle
    )
    B_q, B_inv_sf = nvfp4_quantize(
        B_in, scale_factor_B, sfLayout=sf_layout, do_shuffle=do_shuffle
    )

    # Alpha rescales the product of the two global scale factors.
    alpha = 1.0 / (scale_factor_A * scale_factor_B)

    # Pick FlashInfer flag for the scale-factor layout
    use_8x4_sf_layout = (sf_layout == SfLayout.layout_8x4)

    # Allocate output and call CUTLASS-backed FP4 GEMM.
    C = torch.empty((M, N), device=A.device, dtype=out_dtype)
    mm_fp4(
        A_q, B_q.T,             # note the transpose on B_q (expecting KxN → NxK in kernel)
        A_inv_sf, B_inv_sf.T,   # matching transposition of per-tile inverse scales
        alpha,                  # global alpha
        out_dtype, C,           # output dtype & buffer
        block_size=block_size,
        use_8x4_sf_layout=use_8x4_sf_layout,
        backend="cutlass"
    )
    return C


def main():
    import argparse, math
    torch.manual_seed(0)

    parser = argparse.ArgumentParser()
    parser.add_argument("--M", type=int, default=256, help="Rows of A / C (multiple of 128)")
    parser.add_argument("--N", type=int, default=256, help="Cols of B / C (multiple of 128)")
    parser.add_argument("--K", type=int, default=512, help="Inner dim (multiple of 64)")
    parser.add_argument("--dtype", type=str, choices=["fp32", "bf16"], default="bf16")
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")

    M, N, K = args.M, args.N, args.K
    assert M % 128 == 0 and N % 128 == 0 and K % 64 == 0, "Shape must satisfy: M%128==0, N%128==0, K%64==0."

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    device = "cuda"

    def stats(name, X):
        X = X.float()
        return f"{name}: max={X.abs().max().item():.6e} mean={X.abs().mean().item():.6e} std={X.std().item():.6e}"

    for t in range(args.trials):
        # Inputs
        A = torch.randn((M, K), device=device, dtype=dtype)
        B = torch.randn((K, N), device=device, dtype=dtype)

        # GEMMs
        C_triton  = triton_nvfp4(A, B, out_dtype=torch.bfloat16 if dtype is torch.bfloat16 else torch.float32)
        C_cutlass = cutlass_nvfp4(A, B, out_dtype=torch.bfloat16 if dtype is torch.bfloat16 else torch.float32)
        # Bring to fp32 for error analysis
        Ct = C_triton.float()
        Cc = C_cutlass.float()
        Cref = (A.float() @ B.float())
        # Triton vs Cutlass
        diff_tc = (Ct - Cc)
        denom_tc = Cc.abs().clamp_min(1e-8)
        rel_tc = (diff_tc.abs() / denom_tc)

        # Each vs reference (optional sanity)
        diff_t_ref = (Ct - Cref)
        diff_c_ref = (Cc - Cref)
        rel_t_ref = diff_t_ref.abs() / Cref.abs().clamp_min(1e-8)
        rel_c_ref = diff_c_ref.abs() / Cref.abs().clamp_min(1e-8)


        print("\nTriton vs Cutlass:")
        print(f"  abs diff   -> max={diff_tc.abs().max().item():.6e}  mean={diff_tc.abs().mean().item():.6e}")
        print(f"  rel diff   -> max={rel_tc.max().item():.6e}       mean={rel_tc.mean().item():.6e}")

        print("\nTriton vs Reference:")
        print(f"  abs diff   -> max={diff_t_ref.abs().max().item():.6e}  mean={diff_t_ref.abs().mean().item():.6e}")
        print(f"  rel diff   -> max={rel_t_ref.max().item():.6e}         mean={rel_t_ref.mean().item():.6e}")

        print("\nCutlass vs Reference:")
        print(f"  abs diff   -> max={diff_c_ref.abs().max().item():.6e}  mean={diff_c_ref.abs().mean().item():.6e}")
        print(f"  rel diff   -> max={rel_c_ref.max().item():.6e}         mean={rel_c_ref.mean().item():.6e}")

        if args.verbose:
            # a tiny sample print to eyeball values
            i, j = 0, 0
            print(f"\nSample C[0,0]: triton={Ct[i,j].item():.6e}, cutlass={Cc[i,j].item():.6e}, ref={Cref[i,j].item():.6e}")

if __name__ == "__main__":
    main()