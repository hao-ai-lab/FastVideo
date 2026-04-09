import torch
from flashinfer import SfLayout, mm_fp4, nvfp4_quantize


class _LinearFWD4BWD16Fn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, backend="cutlass", block_size=16, use_128x4_sf_layout=True):
        # assert activation dtype
        if x.dtype not in (torch.float16, torch.bfloat16):
            x = x.to(dtype=torch.bfloat16)

        # cast params (can be fp32) to activation dtype for quantization
        weight_cast = weight.to(dtype=x.dtype)
        bias_cast = bias.to(dtype=x.dtype) if bias is not None else None

        # shapes
        orig_shape = x.shape
        k = weight_cast.shape[1]
        n = weight_cast.shape[0]
        x2d = x.reshape(-1, k).contiguous()
        M = x2d.shape[0]

        out2d = torch.empty((M, n), device=x.device, dtype=x.dtype)

        @torch.compile
        def _global_sf(t: torch.Tensor) -> torch.Tensor:
            maxabs = t.float().abs().nan_to_num().max()
            maxabs = torch.maximum(maxabs, torch.tensor(1e-12, device=t.device, dtype=maxabs.dtype))
            return (448.0 * 6.0) / maxabs

        a_sf_layout = SfLayout.layout_128x4 if use_128x4_sf_layout else SfLayout.layout_8x4
        global_sf_a = _global_sf(x2d)
        global_sf_b = _global_sf(weight_cast)

        a_fp4, a_inv_s = nvfp4_quantize(x2d, global_sf_a, sfLayout=a_sf_layout, do_shuffle=False)
        b_fp4, b_inv_s = nvfp4_quantize(weight_cast, global_sf_b, sfLayout=SfLayout.layout_128x4, do_shuffle=False)

        alpha = 1.0 / (global_sf_a * global_sf_b)

        mm_fp4(
            a_fp4,
            b_fp4.T,
            a_inv_s,
            b_inv_s.T,
            alpha,
            x.dtype,
            out2d,
            block_size=block_size,
            use_8x4_sf_layout=(not use_128x4_sf_layout),
            backend=backend,
        )

        if bias_cast is not None:
            out2d.add_(bias_cast)

        # save tensors for backward (keep original dtypes)
        ctx.save_for_backward(x2d, weight, bias)
        ctx.k = k
        ctx.n = n
        ctx.orig_shape = orig_shape
        return out2d.reshape(*orig_shape[:-1], n)

    @staticmethod
    def backward(ctx, grad_out):
        x2d, weight, bias = ctx.saved_tensors
        M = x2d.shape[0]
        n = ctx.n

        grad_out_2d = grad_out.reshape(M, n).contiguous()

        # cast to grad dtype for matmuls
        weight_cast = weight.to(dtype=grad_out.dtype)
        x_cast = x2d.to(dtype=grad_out.dtype)

        grad_x = grad_out_2d.matmul(weight_cast).reshape(*ctx.orig_shape)
        grad_w = grad_out_2d.t().matmul(x_cast)
        grad_b = grad_out_2d.sum(dim=0) if bias is not None else None

        # None for the three extra forward args
        return grad_x, grad_w, grad_b, None, None, None


def fp4_linear_forward(self, x: torch.Tensor) -> torch.Tensor:
    # pass config **positionally**; autograd.Function.apply ignores kwargs
    return _LinearFWD4BWD16Fn.apply(x, self.weight, self.bias, "cutlass", 16, True), None
