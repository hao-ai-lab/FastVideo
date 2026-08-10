# SPDX-License-Identifier: Apache-2.0

import importlib
import sys
from collections.abc import Callable
from pathlib import Path

import torch

from fastvideo.attention.backends.abstract import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
    AttentionMetadataBuilder,
)
from fastvideo.logger import init_logger

logger = init_logger(__name__)

_project_root = Path(__file__).resolve().parent.parent.parent.parent
_kernel_root = _project_root / "fastvideo-kernel"
_kernel_python_root = _kernel_root / "python"
_attn_qat_infer: Callable[..., torch.Tensor] | None = None
_attn_qat_infer_import_attempted = False


def _ensure_kernel_paths() -> None:
    for path in (_project_root, _kernel_root, _kernel_python_root):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


def _get_attn_qat_infer() -> Callable[..., torch.Tensor] | None:
    global _attn_qat_infer
    global _attn_qat_infer_import_attempted

    if _attn_qat_infer_import_attempted:
        return _attn_qat_infer

    _attn_qat_infer_import_attempted = True
    _ensure_kernel_paths()

    try:
        # Prefer the in-repo kernel implementation during local development.
        _attn_qat_infer = importlib.import_module("attn_qat_infer").sageattn_blackwell
    except ImportError:
        _attn_qat_infer = None

    return _attn_qat_infer


# Consumer-Blackwell compute capabilities the modified SageAttention3 FP4
# kernel is compiled for (sm_120a / sm_121a -- see fastvideo-kernel/README.md).
_SUPPORTED_DEVICE_CAPABILITIES = frozenset({(12, 0), (12, 1)})

# Datacenter-Blackwell capabilities served by the FP4 FA4 kernel
# (flash-attention-fp4 @ fp4, sm_100a/sm_103a) through #1221's plumbing:
# per-16 block-scaled NVFP4 Q/K (E4M3 scale factors), BF16 P/V. This is a
# DIFFERENT quantization scheme from the sm_12x CUTLASS extension above --
# ATTN_QAT_TRAIN simulates the CUTLASS scheme, so sm_100/sm_103 deployment
# carries a train-sim mismatch that is measured (MS-SSIM gate), not assumed.
_FA4_FP4_CAPABILITIES = frozenset({(10, 0), (10, 3)})

# The fork is written against the cutlass-dsl 4.4 API surface; the validated
# install set (GB200-proven) is nvidia-cutlass-dsl==4.4.2 +
# nvidia-cutlass-dsl-libs-base==4.4.2 + quack-kernels==0.4.1 +
# flashinfer-python==0.6.8, with the fork on PYTHONPATH,
# CUTE_DSL_ENABLE_TVM_FFI=1, and FASTVIDEO_FA4=1 (the fork ships no compiled
# FA2, so dense attention paths need the FA4 opt-in). dsl 4.6-era installs
# fail at CuTe JIT trace (cute.make_fragment was removed at module level).
_FA4_INSTALL_HINT = ("install flash-attention-fp4 (branch fp4) from "
                     "https://github.com/hao-ai-lab/flash-attention-fp4 with "
                     "nvidia-cutlass-dsl==4.4.2, quack-kernels==0.4.1, "
                     "flashinfer-python==0.6.8 and FASTVIDEO_FA4=1; "
                     "see docs/inference/optimizations.md")

_fa4_fp4_import_ok: bool | None = None


def _fa4_fp4_available() -> bool:
    """flash_attn.cute (FA4) import probe, cached. Reuses #1221's guarded
    import chain in fastvideo.attention.utils.flash_attn_cute (which maps
    cutlass-dsl version skew to ImportError with a loud warning)."""
    global _fa4_fp4_import_ok
    if _fa4_fp4_import_ok is None:
        try:
            from fastvideo.attention.utils.flash_attn_cute import (  # noqa: F401
                flash_attn_fp4_func, )
            _fa4_fp4_import_ok = True
        except ImportError:
            _fa4_fp4_import_ok = False
    return _fa4_fp4_import_ok


def _active_capability() -> tuple[int, int] | None:
    if not torch.cuda.is_available():
        return None
    try:
        return tuple(torch.cuda.get_device_capability())
    except Exception:  # pragma: no cover - defensive: never break backend selection
        return None


def _resolved_kernel() -> str | None:
    """Which ATTN_QAT_INFER kernel serves the active device, or None.

    Per-arch resolution (single source of truth -- extend the capability
    sets above, do not add equality checks elsewhere):
      * sm_12x consumer Blackwell -> fastvideo-kernel CUTLASS extension
        (modified SageAttention3 FP4).
      * sm_100a/sm_103a datacenter Blackwell -> FP4 FA4 (flash-attention-fp4)
        via the merged #1221 plumbing.
    """
    cap = _active_capability()
    if cap in _SUPPORTED_DEVICE_CAPABILITIES and _get_attn_qat_infer() is not None:
        return "cutlass_sm12x"
    if cap in _FA4_FP4_CAPABILITIES and _fa4_fp4_available():
        return "fa4_fp4"
    return None


def attn_qat_infer_receipt() -> str:
    """One-line receipt of the resolution decision (arch + kernel + quant
    knobs), for the selection log and for tooling. The FA4 knobs are the
    repo's tuned defaults passed through verbatim: qk_mode=nvfp4
    (per-16 E4M3 SFs), pv_mode=bf16 -- see flash_attn/cute/README.md in
    the kernel repo."""
    cap = _active_capability()
    arch = f"sm_{cap[0]}{cap[1]}" if cap is not None else "no-cuda"
    kernel = _resolved_kernel()
    if kernel == "cutlass_sm12x":
        return f"arch={arch} kernel=fastvideo-kernel-cutlass scheme=sage3-fp4-sm120"
    if kernel == "fa4_fp4":
        return (f"arch={arch} kernel=flash-attention-fp4 qk_mode=nvfp4(per-16-e4m3-sf) "
                f"pv_mode=bf16 train_sim_mismatch=measured")
    supported = "sm_120a/sm_121a via fastvideo-kernel build.sh; sm_100a/sm_103a via flash-attention-fp4"
    if cap is not None and cap in _FA4_FP4_CAPABILITIES:
        return f"arch={arch} kernel=none (flash_attn.cute not importable -- {_FA4_INSTALL_HINT})"
    return f"arch={arch} kernel=none (supported: {supported})"


_FA4_ROUTE_OPS: tuple | None = None


def _import_fa4_route_ops() -> tuple:
    """Slow path (own function so tests pin it runs once per process):
    resolves the FA4 quantize helper and kernel entry point."""
    from fastvideo.attention.backends.flash_attn import (
        _nvfp4_quantize_for_fa4, )
    from fastvideo.attention.utils.flash_attn_cute import (
        flash_attn_fp4_func, )
    return (_nvfp4_quantize_for_fa4, flash_attn_fp4_func)


def _resolve_fa4_route_ops() -> tuple:
    # Lazy but memoized: per-forward resolution graph-breaks dynamo every
    # step and blocks fullgraph compilation of the NVFP4 path.
    global _FA4_ROUTE_OPS
    if _FA4_ROUTE_OPS is None:
        _FA4_ROUTE_OPS = _import_fa4_route_ops()
    return _FA4_ROUTE_OPS


_receipt_logged = False


def _log_receipt_once() -> None:
    # One line per process, not per layer (the validation swap constructs
    # one impl per attention layer).
    global _receipt_logged
    if not _receipt_logged:
        _receipt_logged = True
        logger.info("ATTN_QAT_INFER resolved: %s", attn_qat_infer_receipt())


def is_attn_qat_infer_available() -> bool:
    """True only when the active device has a built ATTN_QAT_INFER kernel.

    The import check alone is not sufficient: CUDA 13 wheel builds can
    carry the sm_12x extension on any host (e.g. H100, GB200), where the
    import succeeds, backend selection picks this backend, and the first
    kernel call then fails with an unsupported-capability error instead of
    ever reaching the documented FlashAttention fallback in
    fastvideo.platforms.cuda. Gating on the active device's capability
    keeps that fallback working on every unsupported GPU, while
    sm_100a/sm_103a now resolve to the FP4 FA4 kernel (#1221).
    """
    return _resolved_kernel() is not None


class AttnQatInferBackend(AttentionBackend):

    accept_output_buffer: bool = True

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        return [64, 128]

    @staticmethod
    def get_name() -> str:
        return "ATTN_QAT_INFER"

    @staticmethod
    def get_impl_cls() -> type["AttnQatInferImpl"]:
        return AttnQatInferImpl

    @staticmethod
    def get_metadata_cls() -> type["AttentionMetadata"]:
        raise NotImplementedError

    @staticmethod
    def get_builder_cls() -> type["AttentionMetadataBuilder[AttentionMetadata]"]:
        raise NotImplementedError


class AttnQatInferImpl(AttentionImpl[AttentionMetadata]):

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        causal: bool,
        softmax_scale: float,
        num_kv_heads: int | None = None,
        prefix: str = "",
        **extra_impl_args,
    ) -> None:
        self.causal = causal
        self.softmax_scale = softmax_scale
        dropout_p = extra_impl_args.get("dropout_p", 0.0)
        if dropout_p > 0:
            raise NotImplementedError(f"attn_qat_infer does not support dropout (got dropout_p={dropout_p}). "
                                      "The QAT inference kernel applies no stochastic dropout.")
        # Kernel resolution is per-forward, not per-construction: callers
        # (the validation swap, backend selection) gate on
        # is_attn_qat_infer_available() first, and constructing an impl on a
        # host without the kernel must stay legal (pre-existing contract the
        # validation-swap test pins).
        _log_receipt_once()

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        # Dispatch on the single per-arch resolution: importability of the
        # bundled sm_12x extension is NOT sufficient (CUDA 13 wheels carry it
        # on unsupported hosts, where calling it is the wrong binary).
        kernel = _resolved_kernel()
        if kernel == "fa4_fp4":
            return self._forward_fa4_fp4(query, key, value)
        if kernel is None:
            raise ImportError(f"attn_qat_infer is not available ({attn_qat_infer_receipt()}). "
                              "Please ensure an ATTN_QAT_INFER kernel is installed for this device.")

        attn_qat_infer = _get_attn_qat_infer()
        assert attn_qat_infer is not None  # kernel == "cutlass_sm12x" implies the import succeeded

        query = query.transpose(1, 2).contiguous()
        key = key.transpose(1, 2).contiguous()
        value = value.transpose(1, 2).contiguous()

        output = attn_qat_infer(
            query,
            key,
            value,
            attn_mask=None,
            is_causal=self.causal,
            sm_scale=self.softmax_scale,
        )
        return output.transpose(1, 2).contiguous()

    def _forward_fa4_fp4(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        """sm_100a/sm_103a path: FP4 FA4 with the repo's tuned defaults
        (NVFP4 per-16 block-scaled Q/K, BF16 V) -- mirrors
        FlashAttentionImpl._forward_nvfp4 (#1221). Inputs/outputs are
        (batch, seqlen, nheads, headdim); no transpose."""
        _nvfp4_quantize_for_fa4, flash_attn_fp4_func = _resolve_fa4_route_ops()

        orig_seqlen_q = query.shape[1]
        orig_seqlen_k = key.shape[1]

        q_fp4, q_sf = _nvfp4_quantize_for_fa4(query)
        k_fp4, k_sf = _nvfp4_quantize_for_fa4(key)

        # FP4/SF buffers are padded to a 128 multiple; FA4 masks to the
        # original lengths so padding never biases the softmax.
        q_fp4 = q_fp4[:, :orig_seqlen_q]
        k_fp4 = k_fp4[:, :orig_seqlen_k]

        output = flash_attn_fp4_func(
            q_fp4,
            k_fp4,
            value,
            q_sf,
            k_sf,
            softmax_scale=self.softmax_scale,
            causal=self.causal,
        )
        if isinstance(output, tuple):
            output = output[0]
        return output
