# SPDX-License-Identifier: Apache-2.0
"""Opt-in dense NVFP4 attention for the H3 Omni prefix on sm_12x (approximate)."""
import torch

DENSE_FP4_PREFIX_API = ("scale_and_quant_fp4", "scale_and_quant_fp4_permute", "scale_and_quant_fp4_transpose",
                        "blockscaled_fp4_attn", "_zero_delta_s")


def has_dense_fp4_prefix_api(api) -> bool:
    """Whether ``attn_qat_infer.api`` exposes everything the dense NVFP4 prefix path calls."""
    return all(hasattr(api, name) for name in DENSE_FP4_PREFIX_API)


def dense_fp4_prefix_attention(q: torch.Tensor, k: torch.Tensor | None, v: torch.Tensor | None,
                               k_prequant: tuple[torch.Tensor, torch.Tensor] | None = None,
                               v_prequant: tuple[torch.Tensor, torch.Tensor] | None = None,
                               q_prequant: tuple[torch.Tensor, torch.Tensor] | None = None,
                               kv_len: int | None = None, single_level_p_quant: bool = True,
                               ) -> torch.Tensor | None:
    """BHSD operands -> ``[B, H, Lq, D]`` from the in-tree sm_12x dense NVFP4 kernel, or None.

    ``k``/``v`` are the already-smoothed prefix K8/V8, so per-block Q/K smoothing stays off;
    ``single_level_p_quant`` follows the sparse path's P scheme. Approximate vs bf16 SDPA.
    ``kv_len`` is the number of valid keys when K/V carry zero rows padded to the kernel's
    128-row block; the kernel masks keys from it, so the padded call is bit-identical to the
    unpadded one.
    ``k_prequant``/``v_prequant`` optionally supply the FP4 K/V ``(codes, scales)`` the producer
    already emitted (bit-identical to ``scale_and_quant_fp4_permute(k)`` /
    ``scale_and_quant_fp4_transpose(v)``); ``k``/``v`` may then be ``None``. ``q_prequant``
    likewise supplies the FP4 query emitted by the query producer (bit-identical to
    ``scale_and_quant_fp4(q)``), removing the last dense-prefix support kernel.
    """
    if (torch.is_grad_enabled() or torch.compiler.is_compiling() or not q.is_cuda
            or q.dtype != torch.bfloat16
            or q.shape[-1] != 128
            or (v is not None and v.dtype != q.dtype)
            or (v is not None and v.shape[-1] != 128)
            or torch.cuda.get_device_capability(q.device)[0] != 12):
        return None
    from fastvideo.attention.backends.attn_qat_infer import _get_attn_qat_infer
    kernel = _get_attn_qat_infer()
    if kernel is None:
        return None
    # The FP4 quantizers read strided input, so 128-row-aligned operands skip the generic
    # wrapper, which pads and copies Q. Other lengths fall back to the wrapper.
    k_len = k_prequant[0].shape[2] if k_prequant is not None else (k.shape[2] if k is not None else -1)
    v_len = v_prequant[0].shape[3] * 2 if v_prequant is not None else (v.shape[2] if v is not None else -1)
    if q.shape[2] % 128 == 0 and k_len % 128 == 0 and v_len == k_len:
        try:
            import attn_qat_infer.api as _api
        except ImportError:
            _api = None
        if _api is not None and has_dense_fp4_prefix_api(_api):
            kv_len = k_len if kv_len is None else kv_len
            if (q_prequant is not None and q_prequant[0].shape[2] == q.shape[2]
                    and q_prequant[0].shape[-1] == q.shape[-1] // 2):
                qlist = q_prequant
            else:
                qlist = _api.scale_and_quant_fp4(q)
            klist = k_prequant if k_prequant is not None else _api.scale_and_quant_fp4_permute(k)
            vlist = v_prequant if v_prequant is not None else _api.scale_and_quant_fp4_transpose(v)
            delta_s = _api._zero_delta_s(q.shape[0], q.shape[1], k_len, q.device)
            out = _api.blockscaled_fp4_attn(qlist, klist, vlist, delta_s, kv_len, False, False,
                                            q.dtype == torch.bfloat16, single_level_p_quant)[0]
            return out[:, :, :q.shape[2], :].contiguous()
    if k is None or v is None:
        return None
    if kv_len is not None:
        k, v = k[:, :, :kv_len], v[:, :, :kv_len]
    return kernel(q, k, v, is_causal=False, per_block_mean=False, single_level_p_quant=single_level_p_quant)
