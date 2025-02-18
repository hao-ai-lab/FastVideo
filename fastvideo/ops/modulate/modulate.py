import torch
import triton

from fastvideo.ops.modulate.k_modulate import _modulate_bwd, _modulate_fwd


class _FusedModulate(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, scale, shift):
        y = torch.empty_like(x)
        batch, seq_len, dim = x.shape
        M = batch * seq_len
        N = dim
        x = x.view(-1, dim).contiguous()
        scale = scale.view(-1, dim).contiguous()
        shift = shift.view(-1, dim).contiguous()

        def grid(meta):
            return (
                triton.cdiv(batch * seq_len, meta["BLOCK_M"]),
                triton.cdiv(dim, meta["BLOCK_N"]),
            )

        _modulate_fwd[grid](x, y, scale, shift, x.stride(0), scale.stride(0),
                            M, N, seq_len)

        ctx.save_for_backward(x, scale)
        ctx.batch = batch
        ctx.seq_len = seq_len
        ctx.dim = dim
        return y

    @staticmethod
    def backward(
        ctx, dy
    ):  # pragma: no cover  # this is covered, but called directly from C++
        x, scale = ctx.saved_tensors

        batch, seq_len, dim = ctx.batch, ctx.seq_len, ctx.dim
        M = batch * seq_len
        N = dim

        # allocate output
        dy = dy.contiguous()
        dx = torch.empty_like(dy)
        dscale = torch.empty_like(dy)
        dshift = torch.sum(dy, dim=1)

        def grid(meta):
            return (
                triton.cdiv(batch * seq_len, meta["BLOCK_M"]),
                triton.cdiv(dim, meta["BLOCK_N"]),
            )

        _modulate_bwd[grid](dx, x, dy, scale, dscale, x.stride(0),
                            scale.stride(0), M, N, seq_len)

        dscale = torch.sum(dscale, dim=1)
        return dx, dscale, dshift


def fused_modulate(
    x: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
) -> torch.Tensor:
    return _FusedModulate.apply(x, scale, shift)
