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
                               q_prequant: tuple[torch.Tensor, torch.Tensor] | None = None
                               ) -> torch.Tensor | None:
    """BHSD operands -> ``[B, H, Lq, D]`` from the in-tree sm_12x dense NVFP4 kernel, or None.

    ``k``/``v`` are the already-smoothed prefix K8/V8, so per-block Q/K smoothing stays off and
    ``single_level_p_quant=False`` matches the recipe's two-level P. Approximate vs bf16 SDPA.
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
    # Specialized dense path: the generic sageattn_blackwell wrapper runs
    # ``preprocess_qkv``, which pads and then ``.contiguous()``-copies Q even though
    # the FP4 quantizers read strided input. Q arrives here as a BLHD view, and the
    # copy measured 1.5-2.0 ms/call at the real Omni prefix shape (14848 x 29184);
    # quantizing the strided view directly is bit-identical. Only the 128-multiple
    # case (guaranteed for this kernel by the producer's pre-pad)
    # takes the shortcut; everything else falls back to the generic wrapper, which
    # also handles the pad-to-block-size case.
    k_len = k_prequant[0].shape[2] if k_prequant is not None else (k.shape[2] if k is not None else -1)
    v_len = v_prequant[0].shape[3] * 2 if v_prequant is not None else (v.shape[2] if v is not None else -1)
    if q.shape[2] % 128 == 0 and k_len % 128 == 0 and v_len == k_len:
        try:
            import attn_qat_infer.api as _api
        except ImportError:
            _api = None
        if _api is not None and has_dense_fp4_prefix_api(_api):
            kv_len = k_len
            if (q_prequant is not None and q_prequant[0].shape[2] == q.shape[2]
                    and q_prequant[0].shape[-1] == q.shape[-1] // 2):
                qlist = q_prequant
            else:
                qlist = _api.scale_and_quant_fp4(q)
            klist = k_prequant if k_prequant is not None else _api.scale_and_quant_fp4_permute(k)
            vlist = v_prequant if v_prequant is not None else _api.scale_and_quant_fp4_transpose(v)
            delta_s = _api._zero_delta_s(q.shape[0], q.shape[1], kv_len, q.device)
            out = _api.blockscaled_fp4_attn(qlist, klist, vlist, delta_s, kv_len, False, False,
                                            q.dtype == torch.bfloat16, False)[0]
            return out[:, :, :q.shape[2], :].contiguous()
    if k is None or v is None:
        return None
    return kernel(q, k, v, is_causal=False, per_block_mean=False, single_level_p_quant=False)
