import torch


class ResidualTanhGatedRMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, x_res, gate, eps=1e-6):
        # Convert to fp32 for precision
        x_res_fp32 = x_res.float()

        # Compute RMS
        mean_square = x_res_fp32.pow(2).mean(-1, keepdim=True)
        scale = torch.rsqrt(mean_square + eps)

        # Apply tanh to gate
        tanh_gate = torch.tanh(gate).unsqueeze(1)

        # Normalize and apply gated scaling
        x_normed = x_res_fp32 * scale * tanh_gate

        # Apply residual connection
        output = x + x_normed.type_as(x)

        return output


def residual_tanh_gated_rmsnorm(x, x_res, gate, eps=1e-6):
    return ResidualTanhGatedRMSNorm.apply(x, x_res, gate, eps)
