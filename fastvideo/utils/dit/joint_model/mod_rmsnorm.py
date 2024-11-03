import torch


class ModulatedRMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale, eps=1e-6):
        # Convert to fp32 for precision
        x_fp32 = x.float()
        scale_fp32 = scale.float()

        # Compute RMS
        mean_square = x_fp32.pow(2).mean(-1, keepdim=True)
        inv_rms = torch.rsqrt(mean_square + eps)

        # Normalize and modulate
        x_normed = x_fp32 * inv_rms
        x_modulated = x_normed * (1 + scale_fp32.unsqueeze(1))

        return x_modulated.type_as(x)


def modulated_rmsnorm(x, scale, eps=1e-6):
    return ModulatedRMSNorm.apply(x, scale, eps)
