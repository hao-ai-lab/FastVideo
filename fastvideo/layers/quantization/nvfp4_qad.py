# SPDX-License-Identifier: Apache-2.0
"""NVFP4 quantization-aware distillation (QAD) for the MiniMax-H3 DiT block linears.

``NVFP4QADLinearMethod`` replaces a ``ReplicatedLinear``'s ``quant_method``
in place, so parameter names, FSDP sharding and checkpoints are unchanged.
Its forward *is* the deployed NVFP4 GEMM, computed from the layer's current
(BF16-cast) master weight exactly as the packed export and
``NVFP4QuantizeMethod.apply`` compute it:

* weight: RTN NVFP4 with global scale ``448 * 6 / amax(W)`` (the converter's
  ``quantize_dense_linear``; ModelOpt max calibration gives the same codes);
* activation: per-16 E4M3 block scales under a global scale that is
  ``static`` (``448 * 6 / calibrated amax``, the FFN linears) or ``unit``
  (1.0, attention projections and VSA gates);
* GEMM: FlashInfer ``mm_fp4`` with ``alpha = (1 / weight_sf) / input_sf``.

With gradients enabled the forward runs inside a straight-through estimator
whose backward is full precision. ``enabled =
False`` delegates to the original method, so a disabled student is the
teacher bit for bit.
"""
from __future__ import annotations

import contextlib
import json
import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

FP4_RANGE = 448.0 * 6.0
H3_BLOCK_LINEAR = re.compile(r"transformer_blocks\.(\d+)\.(attn\.to_q|attn\.to_k|attn\.to_v|attn\.to_out"
                             r"|attn\.to_gate_compress|ff\.fc_in|ff\.fc_out)$")
STATIC_SUFFIXES = ("ff.fc_in", "ff.fc_out")


def weight_global_sf(weight: torch.Tensor) -> torch.Tensor:
    """``448 * 6 / amax(|W|)`` in fp32, as the packed-export converter computes it.

    The max of a BF16 tensor is exact in BF16, so one BF16 reduction gives the converter's value
    (``weight.float().abs().nan_to_num().max()``) for finite weights without fp32 temporaries.
    """
    return FP4_RANGE / torch.nan_to_num(weight.abs().amax().float()).clamp(min=1e-12)


def nvfp4_gemm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None,
               input_global_sf: torch.Tensor) -> torch.Tensor:
    """The deployed NVFP4 linear on a BF16 weight: quantize both operands, FP4 GEMM."""
    from fastvideo.layers.quantization.nvfp4_config import _mm_fp4, _mm_fp4_backend, _nvfp4_quantize
    from fastvideo.layers.quantization.nvfp4_config import _require_flashinfer

    sf_layout, _, _ = _require_flashinfer()
    weight = weight.to(torch.bfloat16).contiguous()
    w_sf = weight_global_sf(weight)
    w_fp4, w_scale = _nvfp4_quantize(weight, w_sf, sfLayout=sf_layout.layout_128x4, do_shuffle=False)
    alpha = (1.0 / w_sf) / input_global_sf
    shape = x.shape
    x2d = x.to(torch.bfloat16).reshape(-1, shape[-1]).contiguous()
    x_fp4, x_scale = _nvfp4_quantize(x2d, input_global_sf, sfLayout=sf_layout.layout_128x4, do_shuffle=False)
    out = _mm_fp4(x_fp4, w_fp4.T, x_scale, w_scale.T, alpha, torch.bfloat16, None, backend=_mm_fp4_backend())
    if bias is not None:
        out = out + bias.to(out.dtype)
    return out.view(*shape[:-1], weight.shape[0])


class _NVFP4STE(torch.autograd.Function):
    """Forward = ``nvfp4_gemm``; backward = full-precision linear gradients (straight-through)."""

    @staticmethod
    def forward(ctx, x, weight, bias, input_global_sf):  # type: ignore[override]
        ctx.save_for_backward(x, weight)
        ctx.has_bias = bias is not None
        return nvfp4_gemm(x, weight, bias, input_global_sf)

    @staticmethod
    def backward(ctx, grad_out):  # type: ignore[override]
        x, weight = ctx.saved_tensors
        grad_2d = grad_out.reshape(-1, grad_out.shape[-1])
        grad_x = (grad_2d @ weight.to(grad_2d.dtype)).reshape(x.shape).to(x.dtype)
        grad_w = (grad_2d.t() @ x.reshape(-1, x.shape[-1]).to(grad_2d.dtype)).to(weight.dtype)
        grad_b = grad_2d.sum(dim=0) if ctx.has_bias else None
        return grad_x, grad_w, grad_b, None


class NVFP4QADLinearMethod:
    """Drop-in ``quant_method`` for one H3 block linear (see module docstring)."""

    def __init__(self,
                 original: Any,
                 prefix: str,
                 act_scale: str,
                 input_amax: float | None = None,
                 allow_uncalibrated: bool = False) -> None:
        if act_scale not in ("static", "unit"):
            raise ValueError(f"act_scale must be 'static' or 'unit', got {act_scale!r}")
        if act_scale == "static" and not allow_uncalibrated and (input_amax is None or not input_amax > 0):
            raise ValueError(f"{prefix}: a static activation scale needs a positive calibrated amax")
        self.original = original
        self.prefix = prefix
        self.act_scale = act_scale
        self.input_amax = float(input_amax) if input_amax is not None else None
        self.enabled = True
        # Calibration: run the original (dense) path and fold each input's abs-max into ``observed_amax``.
        self.calibrating = False
        self.observed_amax: torch.Tensor | None = None
        self._sf: dict[torch.device, torch.Tensor] = {}

    def set_input_amax(self, amax: float) -> None:
        if not amax > 0:
            raise ValueError(f"{self.prefix}: calibrated amax must be positive, got {amax}")
        self.input_amax = float(amax)
        self._sf.clear()

    def input_global_sf(self, device: torch.device) -> torch.Tensor:
        if self.act_scale == "static" and self.input_amax is None:
            raise RuntimeError(f"{self.prefix}: static activation scale used before calibration")
        if device not in self._sf:
            value = 1.0 if self.act_scale == "unit" else FP4_RANGE / max(self.input_amax or 0.0, 1e-12)
            self._sf[device] = torch.tensor(value, dtype=torch.float32, device=device)
        return self._sf[device]

    def apply(self, layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        if self.calibrating:
            amax = x.detach().abs().amax().float()
            self.observed_amax = amax if self.observed_amax is None else torch.maximum(self.observed_amax, amax)
            return self.original.apply(layer, x, bias)
        if not self.enabled:
            return self.original.apply(layer, x, bias)
        input_sf = self.input_global_sf(x.device)
        if torch.is_grad_enabled() and (layer.weight.requires_grad or x.requires_grad):
            return _NVFP4STE.apply(x, layer.weight, bias, input_sf)
        return nvfp4_gemm(x, layer.weight, bias, input_sf)

    def __getattr__(self, name: str) -> Any:
        # Callers probe optional quant-method hooks (e.g. wants_prequantized_input); fall back to the original.
        # Before __init__ (copy.deepcopy, unpickling) there is no original: AttributeError keeps hasattr working.
        try:
            original = self.__dict__["original"]
        except KeyError:
            raise AttributeError(name) from None
        return getattr(original, name)


@dataclass(frozen=True)
class NVFP4QADPlan:
    """Which block linears are quantized and how (``h3_dit_vsa``: attention, gates and FFN)."""
    quantize_attention: bool = True
    quantize_gate: bool = True
    quantize_ffn: bool = True
    skip_blocks: tuple[int, ...] = ()


def load_amax_table(path: str | Path) -> dict[str, float]:
    """``FASTVIDEO_NVFP4_ACT_AMAX``-format JSON (``b<block>.<sub>`` or full prefix -> amax or {"all": amax})."""
    raw = json.loads(Path(path).read_text())
    return {key: float(value["all"] if isinstance(value, dict) else value) for key, value in raw.items()}


def _amax_for(table: dict[str, float], name: str, block: str, sub: str) -> float | None:
    for key in (name, f"b{block}.{sub}", f"transformer_blocks.{block}.{sub}"):
        if key in table:
            return table[key]
    return None


def install_nvfp4_qad(transformer: torch.nn.Module,
                      amax_table: dict[str, float] | None,
                      plan: NVFP4QADPlan,
                      allow_uncalibrated: bool = False) -> dict[str, NVFP4QADLinearMethod]:
    """Install QAD methods on the selected block linears; returns them by module name.

    FFN linears take ``static`` scales from ``amax_table`` (required unless
    ``allow_uncalibrated``, which leaves them for ``calibrate``); attention
    projections and gates use the unit scale, as the deployed export does.
    """
    installed: dict[str, NVFP4QADLinearMethod] = {}
    for raw_name, module in transformer.named_modules():
        # Activation checkpointing wraps blocks; key everything by the checkpoint (unwrapped) name.
        name = raw_name.replace("_checkpoint_wrapped_module.", "")
        match = H3_BLOCK_LINEAR.search(name)
        if match is None or not hasattr(module, "quant_method") or int(match.group(1)) in plan.skip_blocks:
            continue
        block, sub = match.group(1), match.group(2)
        if sub.startswith("ff.") and not plan.quantize_ffn:
            continue
        if sub == "attn.to_gate_compress" and not plan.quantize_gate:
            continue
        if sub.startswith("attn.to_") and sub != "attn.to_gate_compress" and not plan.quantize_attention:
            continue
        original = module.quant_method
        if isinstance(original, NVFP4QADLinearMethod):
            original = original.original
        if sub in STATIC_SUFFIXES:
            amax = _amax_for(amax_table or {}, name, block, sub)
            if amax is None and not allow_uncalibrated:
                raise KeyError(f"no calibrated input amax for {name} (key b{block}.{sub}); run Stage A calibration")
            method = NVFP4QADLinearMethod(original, name, "static", amax, allow_uncalibrated=allow_uncalibrated)
        else:
            method = NVFP4QADLinearMethod(original, name, "unit")
        module.quant_method = method
        installed[name] = method
    if not installed:
        raise ValueError("no MiniMax-H3 block linears matched; is this a MiniMax-H3 transformer?")
    return installed


@contextlib.contextmanager
def calibrating(methods: dict[str, NVFP4QADLinearMethod]) -> Iterator[None]:
    """Max-calibrate the static linears that have no amax yet (dense forwards while active).

    On exit the observed maxima are all-reduced (MAX) over the default process
    group, so every rank, including each sequence-parallel shard, agrees.
    """
    pending = {name: m for name, m in methods.items() if m.act_scale == "static" and m.input_amax is None}
    for method in pending.values():
        method.calibrating, method.observed_amax = True, None
    try:
        yield
    finally:
        for method in pending.values():
            method.calibrating = False
    if not pending:
        return
    names = sorted(pending)
    device = next((m.observed_amax.device for m in pending.values() if m.observed_amax is not None),
                  torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else None)
    values = torch.stack([(pending[n].observed_amax if pending[n].observed_amax is not None else torch.zeros(
        (), device=device)).float() for n in names])
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(values, op=torch.distributed.ReduceOp.MAX)
    for name, value in zip(names, values.tolist(), strict=True):
        if not value > 0:
            raise RuntimeError(f"{name} saw no activations during calibration")
        pending[name].set_input_amax(value)


def set_nvfp4_qad_enabled(methods: dict[str, NVFP4QADLinearMethod], enabled: bool) -> None:
    for method in methods.values():
        method.enabled = enabled


def nvfp4_qad_export_scales(methods: dict[str, NVFP4QADLinearMethod]) -> dict[str, float]:
    """The static FFN amax values the export must carry (``--act-amax`` input of the packed converter)."""
    table = {}
    for name, method in methods.items():
        match = H3_BLOCK_LINEAR.search(name)
        if method.act_scale == "static" and match is not None:
            table[f"b{match.group(1)}.{match.group(2)}"] = float(method.input_amax or 0.0)
    return table


__all__ = [
    "NVFP4QADLinearMethod", "NVFP4QADPlan", "calibrating", "install_nvfp4_qad", "load_amax_table", "nvfp4_gemm",
    "nvfp4_qad_export_scales", "set_nvfp4_qad_enabled", "weight_global_sf"
]
