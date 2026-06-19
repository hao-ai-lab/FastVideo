# SPDX-License-Identifier: Apache-2.0
"""FP8 quantization-aware training for linear layers.

Mirror of ``fp4linear.py`` but for FP8 (e4m3). The forward pass quantizes both
activations and weights to FP8 and runs ``torch._scaled_mm``; the backward pass
is a bf16 straight-through estimator so the high-precision master weights stay
trainable. Falls back to a bf16 fake-quant forward on GPUs older than sm89.
"""
import torch

FP8_DTYPE = torch.float8_e4m3fn
FP8_MAX = float(torch.finfo(FP8_DTYPE).max)  # 448.0
FP8_MIN_SCALE = 1.0 / (FP8_MAX * 512.0)


def _supports_fp8_compute() -> bool:
    """Whether the active device supports FP8 ``_scaled_mm`` (sm89+)."""
    if not torch.cuda.is_available():
        return False
    cap = torch.cuda.get_device_capability()
    return cap[0] > 8 or (cap[0] == 8 and cap[1] >= 9)


def _quantize_tensorwise(x_2d: torch.Tensor, ) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns ``(x_fp8 [M, K], x_scale [1] float32)``."""
    x_absmax = x_2d.abs().amax().float()
    x_scale = (x_absmax / FP8_MAX).clamp(min=FP8_MIN_SCALE)
    x_fp8 = (x_2d / x_scale.to(x_2d.dtype)).clamp(-FP8_MAX, FP8_MAX).to(FP8_DTYPE)
    return x_fp8, x_scale.view(1)


def _quantize_rowwise(x_2d: torch.Tensor, ) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns ``(x_fp8 [M, K], x_scale [M, 1] float32)``."""
    x_absmax = x_2d.abs().amax(dim=-1, keepdim=True).float()
    x_scale = (x_absmax / FP8_MAX).clamp(min=FP8_MIN_SCALE)
    x_fp8 = (x_2d / x_scale.to(x_2d.dtype)).clamp(-FP8_MAX, FP8_MAX).to(FP8_DTYPE)
    return x_fp8, x_scale


def _fake_quant(x_2d: torch.Tensor, granularity: str) -> torch.Tensor:
    """bf16 fake-quant (quantize then dequantize) for pre-sm89 fallback."""
    if granularity == "channel":
        x_fp8, x_scale = _quantize_rowwise(x_2d)
    else:
        x_fp8, x_scale = _quantize_tensorwise(x_2d)
    return x_fp8.to(x_2d.dtype) * x_scale.to(x_2d.dtype)


class _LinearFWD8BWD16Fn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, granularity="tensor"):
        # assert/normalize activation dtype
        if x.dtype not in (torch.float16, torch.bfloat16):
            x = x.to(dtype=torch.bfloat16)

        # cast params (can be fp32) to activation dtype for quantization
        weight_cast = weight.to(dtype=x.dtype)
        bias_cast = bias.to(dtype=x.dtype) if bias is not None else None

        orig_shape = x.shape
        k = weight_cast.shape[1]
        n = weight_cast.shape[0]
        x2d = x.reshape(-1, k).contiguous()

        if not _supports_fp8_compute():
            # bf16 fake-quant fallback: simulate the FP8 rounding error but
            # compute the matmul in bf16.
            x_fq = _fake_quant(x2d, granularity)
            w_fq = _fake_quant(weight_cast, granularity)
            out2d = x_fq.matmul(w_fq.t())
            if bias_cast is not None:
                out2d = out2d + bias_cast
            ctx.save_for_backward(x2d, weight, bias)
            ctx.n = n
            ctx.orig_shape = orig_shape
            return out2d.reshape(*orig_shape[:-1], n)

        if granularity == "channel":
            x_fp8, x_scale = _quantize_rowwise(x2d)
            w_fp8, w_scale = _quantize_rowwise(weight_cast)
            scale_b = w_scale.view(1, -1)
        else:
            x_fp8, x_scale = _quantize_tensorwise(x2d)
            w_fp8, w_scale = _quantize_tensorwise(weight_cast)
            scale_b = w_scale

        out2d = torch._scaled_mm(
            x_fp8,
            w_fp8.t(),
            scale_a=x_scale,
            scale_b=scale_b,
            out_dtype=x.dtype,
        )
        if isinstance(out2d, tuple):
            out2d = out2d[0]

        if bias_cast is not None:
            out2d = out2d + bias_cast

        # save tensors for backward (keep original dtypes)
        ctx.save_for_backward(x2d, weight, bias)
        ctx.n = n
        ctx.orig_shape = orig_shape
        return out2d.reshape(*orig_shape[:-1], n)

    @staticmethod
    def backward(ctx, grad_out):
        x2d, weight, bias = ctx.saved_tensors
        M = x2d.shape[0]
        n = ctx.n

        grad_out_2d = grad_out.reshape(M, n).contiguous()

        # bf16 straight-through estimator: gradients flow through the
        # full-precision master weights, not the FP8 quantized values.
        weight_cast = weight.to(dtype=grad_out.dtype)
        x_cast = x2d.to(dtype=grad_out.dtype)

        grad_x = grad_out_2d.matmul(weight_cast).reshape(*ctx.orig_shape)
        grad_w = grad_out_2d.t().matmul(x_cast)
        grad_b = grad_out_2d.sum(dim=0) if bias is not None else None

        # None for the extra forward arg (granularity)
        return grad_x, grad_w, grad_b, None


def fp8_linear_forward(self, x: torch.Tensor) -> torch.Tensor:
    # pass config **positionally**; autograd.Function.apply ignores kwargs
    return _LinearFWD8BWD16Fn.apply(x, self.weight, self.bias, "tensor"), None
