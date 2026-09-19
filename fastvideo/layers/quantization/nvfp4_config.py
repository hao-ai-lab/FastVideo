# SPDX-License-Identifier: Apache-2.0
"""NVFP4 linear quantization for supported transformer layer sets.

NVFP4 is NVIDIA's block-scaled FP4 format (e2m1 mantissa, fp32 alpha,
``layout_128x4`` scale layout, group size 16) — distinct from
generic FP4 / OCP-FP4 / MX-FP4. We name the public surface ``NVFP4``
explicitly so downstream callers don't conflate it with other FP4
variants that may land later (e.g. AMD's MX-FP4 or vendor-neutral
e3m0).

The registered config targets the curated LTX-2 deployment set, the
main MiniMax-H3 transformer-block FFN linears, and the packed MiniMax-H3
DiT export (``layer_profile="h3_dit"``) covering attention plus FFN.

`flashinfer` is imported lazily inside the call paths that need it.
This keeps ``import fastvideo`` cheap on hosts where flashinfer is
not installed; only the actual NVFP4 quantize / matmul ops fail at
use time, with a clear error.
"""
from __future__ import annotations

import logging
import os
import re
from typing import Any

import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from fastvideo.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from fastvideo.models.utils import set_weight_attrs

logger = logging.getLogger(__name__)


def _require_flashinfer() -> tuple[Any, Any, Any]:
    """Lazy flashinfer import — raised at use time, not import time.

    Returns the bound ``(SfLayout, mm_fp4, nvfp4_quantize)`` triple from
    flashinfer. Raises ``ImportError`` with an actionable hint if the
    package is not available.
    """
    try:
        from flashinfer import (  # type: ignore[import-not-found]
            SfLayout, mm_fp4, nvfp4_quantize,
        )
    except ImportError as exc:  # pragma: no cover - depends on host env
        raise ImportError("NVFP4 quantization requires flashinfer. "
                          "Install with `pip install flashinfer-python`.") from exc
    return SfLayout, mm_fp4, nvfp4_quantize


_LTX2_REFINE_ONLY_SUFFIXES = (
    ".audio_to_video_attn.to_q",
    ".video_to_audio_attn.to_k",
    ".video_to_audio_attn.to_v",
)

_LTX2_NVFP4_BLOCK_LINEAR_SUFFIXES = (
    "attn1.to_q",
    "attn1.to_k",
    "attn1.to_v",
    "attn1.to_out",
    "attn2.to_q",
    "attn2.to_out",
    "audio_to_video_attn.to_q",
    "audio_to_video_attn.to_out",
    "video_to_audio_attn.to_k",
    "video_to_audio_attn.to_v",
    "ffn.fc_in",
    "ffn.fc_out",
)
_LTX2_NVFP4_LINEAR_PREFIXES = frozenset(f"ltx2.blocks.{block_idx}.{suffix}" for block_idx in range(48)
                                        for suffix in _LTX2_NVFP4_BLOCK_LINEAR_SUFFIXES) | frozenset(
                                            ("ltx2.adaln_single.linear", ))
_MINIMAX_H3_NVFP4_FF_PREFIX = re.compile(r"(?:^|\.)transformer_blocks\.\d+\.ff\.(?:fc_in|fc_out)$")
_MINIMAX_H3_NVFP4_DIT_PREFIX = re.compile(
    r"(?:^|\.)transformer_blocks\.\d+\.(?:attn\.to_(?:q|k|v|out)|ff\.(?:fc_in|fc_out))$")
H3_NVFP4_DIT_EXPORT_FILENAME = "nvfp4_weights.safetensors"
H3_NVFP4_DIT_KEY_SEP = "::"
H3_NVFP4_DIT_BUFFER_NAMES = (
    "_nvfp4_weight",
    "_nvfp4_weight_scale",
    "_nvfp4_alpha",
    "_weight_global_sf",
)


def is_ltx2_nvfp4_linear_prefix(prefix: str) -> bool:
    """Return whether *prefix* belongs to the LTX-2 NVFP4 deployment set."""
    return prefix in _LTX2_NVFP4_LINEAR_PREFIXES


def is_minimax_h3_nvfp4_linear_prefix(prefix: str) -> bool:
    """Return whether *prefix* is a main MiniMax-H3 transformer-block FFN linear."""
    return _MINIMAX_H3_NVFP4_FF_PREFIX.search(prefix) is not None


def is_minimax_h3_nvfp4_dit_linear_prefix(prefix: str) -> bool:
    """Return whether *prefix* is a MiniMax-H3 DiT attention or FFN linear.

    This is the packed NVFP4H3 export set: ``attn.to_{q,k,v,out}`` and
    ``ff.{fc_in,fc_out}`` in each main transformer block. Token-refiner,
    AdaLN, and embedding linears stay dense.
    """
    return _MINIMAX_H3_NVFP4_DIT_PREFIX.search(prefix) is not None


def is_minimax_h3_nvfp4_dit_export_path(path: str) -> bool:
    return os.path.basename(path) == H3_NVFP4_DIT_EXPORT_FILENAME


def find_minimax_h3_nvfp4_dit_export(weight_paths: list[str]) -> str | None:
    seen: list[str] = []
    for path in weight_paths:
        if is_minimax_h3_nvfp4_dit_export_path(path) and os.path.isfile(path):
            return path
        directory = path if os.path.isdir(path) else os.path.dirname(path)
        if directory and directory not in seen:
            seen.append(directory)
    for directory in seen:
        candidate = os.path.join(directory, H3_NVFP4_DIT_EXPORT_FILENAME)
        if os.path.isfile(candidate):
            return candidate
    return None


def dense_transformer_safetensors(weight_paths: list[str]) -> list[str]:
    """Drop the packed NVFP4 DiT export so it is not loaded as bf16 weights."""
    return [path for path in weight_paths if not is_minimax_h3_nvfp4_dit_export_path(path)]


def _is_ltx2_refine_only_prefix(prefix: str) -> bool:
    return any(prefix.endswith(suffix) for suffix in _LTX2_REFINE_ONLY_SUFFIXES)


def _get_ltx2_fp4_stage_profile(default: str = "refine") -> str:
    """Read the active stage profile from the forward context.

    Streaming inference flips between ``base`` and ``refine`` between
    segments; the FP4 layer set differs across the two. Falls back to
    ``default`` whenever the context is not available — this keeps the
    op safe to run outside the streaming server (e.g. during eager
    tests).
    """
    try:
        from fastvideo.forward_context import get_forward_context

        forward_ctx = get_forward_context()
        forward_batch = getattr(forward_ctx, "forward_batch", None)
        if forward_batch is None:
            return default
        extra = getattr(forward_batch, "extra", None)
        if not isinstance(extra, dict):
            return default
        profile = extra.get("ltx2_fp4_stage_profile", default)
        if profile in ("base", "refine"):
            return profile
        return default
    except Exception:
        return default


_OPS_REGISTERED = False


def _register_ops_once() -> None:
    """Register the fastvideo_fp4 torch ops on first import that needs
    them. Each op binds to flashinfer at call time; this just sets up
    the dispatcher entries."""
    global _OPS_REGISTERED
    if _OPS_REGISTERED:
        return

    @torch.library.custom_op(
        "fastvideo_fp4::nvfp4_quantize",
        mutates_args=(),
        device_types="cuda",
    )
    def _nvfp4_quantize_op(
        x: torch.Tensor,
        global_sf: torch.Tensor,
        sf_layout: int,
        do_shuffle: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        SfLayout, _, nvfp4_quantize = _require_flashinfer()
        return nvfp4_quantize(x, global_sf, sfLayout=SfLayout(sf_layout), do_shuffle=do_shuffle)

    @_nvfp4_quantize_op.register_fake
    def _nvfp4_quantize_op_fake(
        x: torch.Tensor,
        global_sf: torch.Tensor,
        sf_layout: int,
        do_shuffle: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del global_sf, sf_layout, do_shuffle
        m, k = x.shape
        quantized = torch.empty((m, (k + 1) // 2), device=x.device, dtype=torch.uint8)
        scales = torch.empty((m, (k + 15) // 16), device=x.device, dtype=torch.uint8)
        return quantized, scales

    @torch.library.custom_op(
        "fastvideo_fp4::mm_fp4",
        mutates_args=(),
        device_types="cuda",
    )
    def _mm_fp4_op(
        a: torch.Tensor,
        b: torch.Tensor,
        a_scale: torch.Tensor,
        b_scale: torch.Tensor,
        alpha: torch.Tensor | None,
        out_dtype: torch.dtype = torch.bfloat16,
        out: torch.Tensor | None = None,
        block_size: int = 16,
        use_8x4_sf_layout: bool = False,
        backend: str = "auto",
        use_nvfp4: bool = True,
    ) -> torch.Tensor:
        _, mm_fp4, _ = _require_flashinfer()
        if a.dtype == torch.float4_e2m1fn_x2:
            a = a.view(torch.uint8) if a.is_contiguous() else a.contiguous().view(torch.uint8)
        if b.dtype == torch.float4_e2m1fn_x2:
            b = b.view(torch.uint8) if b.is_contiguous() else b.contiguous().view(torch.uint8)

        return mm_fp4(
            a,
            b,
            a_scale,
            b_scale,
            alpha,
            out_dtype,
            out,
            block_size=block_size,
            use_8x4_sf_layout=use_8x4_sf_layout,
            backend=backend,
            use_nvfp4=use_nvfp4,
        )

    @_mm_fp4_op.register_fake
    def _mm_fp4_op_fake(
        a: torch.Tensor,
        b: torch.Tensor,
        a_scale: torch.Tensor,
        b_scale: torch.Tensor,
        alpha: torch.Tensor | None,
        out_dtype: torch.dtype = torch.bfloat16,
        out: torch.Tensor | None = None,
        block_size: int = 16,
        use_8x4_sf_layout: bool = False,
        backend: str = "auto",
        use_nvfp4: bool = True,
    ) -> torch.Tensor:
        del a_scale, b_scale, alpha, block_size, use_8x4_sf_layout, backend
        del use_nvfp4
        if out is not None:
            return out
        out_shape = (*a.shape[:-1], b.shape[1])
        return torch.empty(out_shape, device=a.device, dtype=out_dtype)

    _OPS_REGISTERED = True


def _nvfp4_quantize(
    x: torch.Tensor,
    global_sf: Any,
    *,
    sfLayout: Any,
    do_shuffle: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    _register_ops_once()
    SfLayout, _, _ = _require_flashinfer()
    if isinstance(sfLayout, SfLayout):
        sf_layout = sfLayout.value
    elif hasattr(sfLayout, "value"):
        sf_layout = int(sfLayout.value)
    else:
        sf_layout = int(sfLayout)
    if not torch.is_tensor(global_sf):
        global_sf = torch.tensor(global_sf, device=x.device, dtype=torch.float32)
    elif global_sf.device != x.device:
        global_sf = global_sf.to(device=x.device)
    if sf_layout == SfLayout.layout_linear.value:
        x_for_quant = x
        logical_rows = x.shape[0]
    else:
        # Sequence-parallel can feed either logical rows or row-padded
        # rows. Normalize to the kernel tile shape for swizzled layouts
        # so both paths share a stable quantization contract.
        row_tile = 8 if sf_layout == SfLayout.layout_8x4.value else 128
        logical_rows = x.shape[0]
        pad_rows = (-logical_rows) % row_tile
        x_for_quant = F.pad(x, (0, 0, 0, pad_rows))

    quantized, scales = torch.ops.fastvideo_fp4.nvfp4_quantize(x_for_quant, global_sf, sf_layout, do_shuffle)
    if sf_layout != SfLayout.layout_linear.value:
        quantized = quantized.narrow(0, 0, logical_rows)
    return quantized, scales


def _mm_fp4(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    alpha: Any,
    out_dtype: torch.dtype,
    out: torch.Tensor | None,
    **kwargs: Any,
) -> torch.Tensor:
    _register_ops_once()
    block_size = kwargs.pop("block_size", 16)
    use_8x4_sf_layout = kwargs.pop("use_8x4_sf_layout", False)
    backend = kwargs.pop("backend", "auto")
    use_nvfp4 = kwargs.pop("use_nvfp4", True)
    if kwargs:
        raise TypeError(f"Unsupported kwargs for _mm_fp4: {sorted(kwargs)}")
    if alpha is not None and not torch.is_tensor(alpha):
        alpha = torch.tensor(alpha, device=a.device, dtype=torch.float32)
    return torch.ops.fastvideo_fp4.mm_fp4(
        a,
        b,
        a_scale,
        b_scale,
        alpha,
        out_dtype,
        out,
        block_size,
        use_8x4_sf_layout,
        backend,
        use_nvfp4,
    )


def _coerce_fp4_input_dtype(x: torch.Tensor) -> torch.Tensor:
    """Coerce an activation to a dtype the FP4 linear accepts.

    The pre-attention norm can emit fp32 (e.g. in eager mode, without the
    torch.compile fusion that keeps it bf16). The FP4 linear emits bf16
    regardless (see _mm_fp4 out dtype), so cast fp32 -> bf16 rather than
    failing, matching the sibling fastvideo/layers/fp4linear.py. Non-floating
    inputs (e.g. int/bool) are a genuine error and are rejected fast.
    """
    if not x.is_floating_point():
        raise TypeError(f"fp4 linear expects floating-point inputs, got {x.dtype}")
    if x.dtype not in (torch.bfloat16, torch.float16):
        x = x.to(torch.bfloat16)
    return x


class NVFP4QuantizeMethod(QuantizeMethodBase):

    def __init__(self, layer_prefix: str = ""):
        super().__init__()
        self.weight_fp4 = None
        self.weight_scale = None
        self.x_global_sf = torch.tensor(1.0, device="cuda", dtype=torch.float32)
        self.layer_prefix = layer_prefix
        self._is_refine_only_layer = _is_ltx2_refine_only_prefix(layer_prefix)
        # Set from NVFP4Config.retain_original_weights in get_quant_method:
        # True = retain every original bf16 weight; None/False (default) =
        # purge the purgeable set. Refine-only layers are always retained --
        # the base stage profile runs them dense by deployment contract.
        self._retain_original_weights: bool | None = None

    def create_weights(self, layer: torch.nn.Module, input_size_per_partition: int, output_partition_sizes: list[int],
                       input_size: int, output_size: int, params_dtype: torch.dtype, **extra_weight_attrs):
        weight = Parameter(torch.empty(
            sum(output_partition_sizes),
            input_size_per_partition,
            dtype=params_dtype,
        ),
                           requires_grad=False)
        set_weight_attrs(weight, {"input_dim": 1, "output_dim": 0})
        layer.register_parameter("weight", weight)
        set_weight_attrs(weight, extra_weight_attrs)

    def quantize_input(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        SfLayout, _, _ = _require_flashinfer()
        x = _coerce_fp4_input_dtype(x)
        x_2d = x.view(-1, x.shape[-1])
        x_fp4, x_scale = _nvfp4_quantize(
            x_2d,
            self.x_global_sf,
            sfLayout=SfLayout.layout_128x4,
            do_shuffle=False,
        )
        return x_fp4, x_scale, self.x_global_sf

    def wants_prequantized_input(self) -> bool:
        if not self._is_refine_only_layer:
            return True
        stage_profile = _get_ltx2_fp4_stage_profile(default="refine")
        return stage_profile != "base"

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
        pre_quantized: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        | None = None,
    ) -> torch.Tensor:
        SfLayout, _, _ = _require_flashinfer()
        # The original bf16 weight may have been purged after FP4 conversion
        # (see convert_model_to_nvfp4); the packed FP4 weight keeps the
        # output dim as its first dimension (only K is packed 2-per-byte).
        weight = getattr(layer, "weight", None)
        out_dim = weight.shape[0] if weight is not None else layer._nvfp4_weight.shape[0]
        original_shape = x.shape

        # Stage-aware profile: keep refine-only FP4 layers in dense mode
        # during stage-1 denoising so the base path doesn't pay the
        # quantize/dequantize tax for layers it never touches.
        stage_profile = _get_ltx2_fp4_stage_profile(default="refine")
        if self._is_refine_only_layer and stage_profile == "base":
            if weight is None:
                raise RuntimeError(f"NVFP4 layer {self.layer_prefix!r} hit the stage-profile dense path, "
                                   "but its original weights were purged "
                                   "(NVFP4Config(retain_original_weights=False)). Streaming/two-stage "
                                   "deploys must load with retain_original_weights left unset (auto) or True.")
            out = (F.linear(x, weight, bias) if torch.cuda.is_available() or bias is None else F.linear(
                x, weight, bias.to(x.dtype)))
            return out.view(*original_shape[:-1], out_dim)
        if pre_quantized is not None:
            x_fp4, x_scale, x_global_sf = pre_quantized
            # FlashInfer fused norm+quant APIs may return 3D tensors for
            # 3D inputs. mm_fp4 only accepts 2D tensors, so flatten
            # batch/sequence dims here.
            if x_fp4.dim() > 2:
                x_fp4 = x_fp4.view(-1, x_fp4.shape[-1])
            if x_scale.dim() > 2:
                x_scale = x_scale.view(-1, x_scale.shape[-1])
        else:
            x = _coerce_fp4_input_dtype(x)
            x = x.view(-1, x.shape[-1])
            x_global_sf = self.x_global_sf
            x_fp4, x_scale = _nvfp4_quantize(
                x,
                x_global_sf,
                sfLayout=SfLayout.layout_128x4,
                do_shuffle=False,
            )

        weight_fp4 = layer._nvfp4_weight
        weight_scale = layer._nvfp4_weight_scale
        weight_global_sf = layer._weight_global_sf

        if hasattr(layer, "_nvfp4_alpha"):
            alpha = layer._nvfp4_alpha / x_global_sf
        else:
            alpha = 1.0 / (x_global_sf * weight_global_sf)

        out = _mm_fp4(
            x_fp4,
            weight_fp4.T,
            x_scale,
            weight_scale.T,
            alpha,
            torch.bfloat16,
            None,
            backend='auto',
        )

        if bias is not None:
            out = out + bias
        out = out.view(*original_shape[:-1], out_dim)
        return out


class NVFP4Config(QuantizationConfig):
    """Select NVFP4 for the supported LTX-2 and MiniMax-H3 linear sets.

    NVFP4 is NVIDIA's block-scaled FP4 (e2m1 mantissa, fp32 alpha,
    ``layout_128x4`` scale layout, group size 16). LTX-2 uses its curated
    attention and FFN deployment set. MiniMax-H3's default profile uses only
    ``fc_in`` and ``fc_out`` in each main transformer-block FFN.
    ``layer_profile="h3_dit"`` expands that to the packed NVFP4H3 DiT set
    (attention ``to_{q,k,v,out}`` plus those FFN linears).
    """

    def __init__(self, layer_profile: str = "refine", retain_original_weights: bool | None = None):
        super().__init__()
        if layer_profile not in ("base", "refine", "h3_dit"):
            raise ValueError("NVFP4Config.layer_profile must be one of 'base', 'refine', or 'h3_dit', "
                             f"got {layer_profile!r}")
        self.layer_profile = layer_profile
        # Original bf16 ``layer.weight`` retention after FP4 conversion.
        # Default (None/False): purge the purgeable originals -- every
        # always-FP4 layer. Refine-only layers (the cross-modal AV
        # projections) are ALWAYS retained: the ``base`` stage profile runs
        # them dense by deployment contract, including the distilled
        # single-stage deploy. True: retain everything (debugging /
        # pre-purge behavior).
        self.retain_original_weights = retain_original_weights

    def get_name(self):
        return "nvfp4"

    def get_supported_act_dtypes(self):
        return [torch.bfloat16, torch.float16]

    @classmethod
    def get_min_capability(cls):
        return 100

    @staticmethod
    def get_config_filenames():
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> NVFP4Config:
        return cls(
            layer_profile=config.get("layer_profile", "refine"),
            retain_original_weights=config.get("retain_original_weights"),
        )

    def get_quant_method(self, layer: torch.nn.Module, prefix: str):
        from fastvideo.layers.linear import LinearBase

        if not isinstance(layer, LinearBase):
            return None
        if self.layer_profile == "h3_dit":
            tagged = is_minimax_h3_nvfp4_dit_linear_prefix(prefix)
        else:
            tagged = is_ltx2_nvfp4_linear_prefix(prefix) or is_minimax_h3_nvfp4_linear_prefix(prefix)
        if tagged:
            method = NVFP4QuantizeMethod(layer_prefix=prefix)
            method._retain_original_weights = self.retain_original_weights
            return method
        return None


def convert_model_to_nvfp4(model: torch.nn.Module) -> None:
    SfLayout, _, _ = _require_flashinfer()
    from torch.distributed.tensor import DTensor  # type: ignore

    purged = 0
    retained = 0
    purged_bytes = 0
    for mod in model.modules():
        qm = getattr(mod, "quant_method", None)
        if isinstance(qm, NVFP4QuantizeMethod):
            weight = getattr(mod, "weight", None)
            if weight is None:
                continue
            weight_local = weight.to_local() if isinstance(weight, DTensor) else weight  # type: ignore[arg-type]
            weight_global_sf = (448 * 6) / weight_local.float().abs().nan_to_num().max()
            fp4_w, fp4_s = _nvfp4_quantize(
                weight_local,
                weight_global_sf,
                sfLayout=SfLayout.layout_128x4,
                do_shuffle=False,
            )
            weight_global_sf_t = torch.as_tensor(
                weight_global_sf,
                device=weight_local.device,
                dtype=torch.float32,
            )
            mod.register_buffer("_nvfp4_weight", fp4_w, persistent=False)
            mod.register_buffer("_nvfp4_weight_scale", fp4_s, persistent=False)
            mod.register_buffer(
                "_weight_global_sf",
                weight_global_sf_t.to(dtype=torch.bfloat16),
                persistent=False,
            )
            mod.register_buffer(
                "_nvfp4_alpha",
                (1.0 / weight_global_sf_t).to(dtype=torch.float32),
                persistent=False,
            )

            retain_flag = getattr(qm, "_retain_original_weights", None)
            # Refine-only layers are NEVER purgeable: the "base" stage profile
            # runs them dense by deployment contract (the distilled
            # single-stage deploy included — its forward context is the base
            # profile, so e.g. audio_to_video_attn routes dense every step).
            # retain_original_weights therefore only widens retention
            # (True = keep everything); it cannot narrow it below the
            # dense-capable set.
            retain = qm._is_refine_only_layer or retain_flag is True
            if retain:
                retained += 1
            elif isinstance(weight, DTensor):
                raise RuntimeError("NVFP4 cannot purge FSDP-sharded bf16 weights. Use a packed NVFP4 "
                                   "export, or convert without FSDP sharding.")
            else:
                purged_bytes += weight.numel() * weight.element_size()
                purged += 1
                mod.register_parameter("weight", None)

    if purged or retained:
        logger.info(
            "NVFP4 weight purge receipt: purged %d original bf16 weight tensors "
            "(%.2f GiB freed); retained %d (refine-only dense fallback or "
            "retain_original_weights).",
            purged,
            purged_bytes / (1 << 30),
            retained,
        )


def nvfp4_linear_weight_param_names(model: torch.nn.Module) -> set[str]:
    """State-dict names of ``weight`` on layers tagged with ``NVFP4QuantizeMethod``."""
    names: set[str] = set()
    for module_name, module in model.named_modules():
        if isinstance(getattr(module, "quant_method", None), NVFP4QuantizeMethod):
            names.add(f"{module_name}.weight" if module_name else "weight")
    return names


def _module_by_nvfp4_export_prefix(modules: dict[str, torch.nn.Module], prefix: str) -> torch.nn.Module | None:
    module = modules.get(prefix)
    if module is not None:
        return module
    if prefix.startswith("minimax_h3."):
        return modules.get(prefix[len("minimax_h3."):])
    return modules.get(f"minimax_h3.{prefix}")


def load_minimax_h3_nvfp4_dit_export(
    model: torch.nn.Module,
    path: str,
    device: torch.device | str,
) -> int:
    """Load a packed NVFP4H3 DiT export onto already-tagged NVFP4 linears.

    Keys are ``<module>::<buffer>`` with the four buffers
    ``convert_model_to_nvfp4`` registers. The bf16 ``weight`` is dropped so a
    32 GB card never materializes the dense GEMMs. Every export prefix must
    match an NVFP4 linear, and every NVFP4 linear must appear in the export.
    """
    from safetensors import safe_open

    modules = dict(model.named_modules())
    tagged = {
        name
        for name, module in modules.items() if isinstance(getattr(module, "quant_method", None), NVFP4QuantizeMethod)
    }
    groups: dict[str, dict[str, str]] = {}
    with safe_open(path, framework="pt", device="cpu") as reader:
        for key in reader.keys():  # noqa: SIM118
            if H3_NVFP4_DIT_KEY_SEP not in key:
                raise ValueError(f"MiniMax-H3 NVFP4 DiT export key {key!r} is missing {H3_NVFP4_DIT_KEY_SEP!r}")
            prefix, buffer_name = key.split(H3_NVFP4_DIT_KEY_SEP, 1)
            groups.setdefault(prefix, {})[buffer_name] = key

        loaded_names: set[str] = set()
        for prefix, buffers in groups.items():
            missing_buffers = [name for name in H3_NVFP4_DIT_BUFFER_NAMES if name not in buffers]
            if missing_buffers:
                raise ValueError(f"MiniMax-H3 NVFP4 DiT export layer {prefix!r} is missing {missing_buffers}")
            extra_buffers = sorted(set(buffers) - set(H3_NVFP4_DIT_BUFFER_NAMES))
            if extra_buffers:
                raise ValueError(f"MiniMax-H3 NVFP4 DiT export layer {prefix!r} has unknown buffers {extra_buffers}")
            module = _module_by_nvfp4_export_prefix(modules, prefix)
            if module is None:
                raise ValueError(f"MiniMax-H3 NVFP4 DiT export layer {prefix!r} is not in the model")
            if not isinstance(getattr(module, "quant_method", None), NVFP4QuantizeMethod):
                raise RuntimeError("MiniMax-H3 NVFP4 DiT export layer "
                                   f"{prefix!r} is not an NVFP4 linear; set NVFP4Config(layer_profile='h3_dit')")
            for buffer_name in H3_NVFP4_DIT_BUFFER_NAMES:
                tensor = reader.get_tensor(buffers[buffer_name]).to(device=device)
                module.register_buffer(buffer_name, tensor, persistent=False)
            module.register_parameter("weight", None)
            loaded_names.add(next(name for name, candidate in modules.items() if candidate is module))

    missing_layers = tagged - loaded_names
    extra_layers = loaded_names - tagged
    if missing_layers or extra_layers:
        raise RuntimeError("MiniMax-H3 NVFP4 DiT export does not cover the tagged linear set; "
                           f"missing={sorted(missing_layers)[:8]} extra={sorted(extra_layers)[:8]}")
    logger.info("Loaded MiniMax-H3 NVFP4 DiT export: %d linears from %s", len(loaded_names), path)
    return len(loaded_names)


__all__ = [
    "H3_NVFP4_DIT_BUFFER_NAMES",
    "H3_NVFP4_DIT_EXPORT_FILENAME",
    "NVFP4Config",
    "NVFP4QuantizeMethod",
    "convert_model_to_nvfp4",
    "dense_transformer_safetensors",
    "find_minimax_h3_nvfp4_dit_export",
    "is_ltx2_nvfp4_linear_prefix",
    "is_minimax_h3_nvfp4_dit_export_path",
    "is_minimax_h3_nvfp4_dit_linear_prefix",
    "is_minimax_h3_nvfp4_linear_prefix",
    "load_minimax_h3_nvfp4_dit_export",
    "nvfp4_linear_weight_param_names",
]
