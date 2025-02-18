import torch
from torch import nn

from benchmark import (
    benchmark_combined,
    benchmark_forward,
    benchmark_backward,
    assert_close,
)
from fastvideo.ops.modulate.modulate import fused_modulate


def torch_modulate(x, scale, shift):
    return x * (scale.unsqueeze(1) + 1) + shift.unsqueeze(1)


# from flash_attn import flash_attn_func


def time_fwd(func, *args, **kwargs):
    time_fb = benchmark_forward(func, *args, **kwargs)
    return time_fb[1].mean


def time_fwd_bwd(func, *args, **kwargs):
    time_fb = benchmark_combined(func, *args, **kwargs)
    return time_fb[1].mean


def time_bwd(func, *args, **kwargs):
    time_fb = benchmark_backward(func, *args, **kwargs)
    return time_fb[1].mean


device = "cuda"
dtype = torch.bfloat16


batch_sizes = [8, 32]
seq_lengths = [128, 512, 1024]
hidden_dims = [768, 1024, 2048]

# methods = (["torch", "triton", "thunderkitten"])
methods = ["torch", "triton"]
time_f = {}
time_b = {}
time_f_b = {}
speed_f = {}
speed_b = {}
speed_f_b = {}
for B in batch_sizes:
    for T in seq_lengths:
        for D in hidden_dims:
            config = (B, T, D)
            torch.cuda.manual_seed_all(1)
            norm_func = nn.LayerNorm(D, elementwise_affine=False, eps=1e-6)
            x = torch.randn(B, T, D, device="cuda", requires_grad=True, dtype=dtype)
            x = norm_func(x.to(torch.float32)).to(dtype)
            shift = torch.randn(B, D, device="cuda", requires_grad=True, dtype=dtype)
            scale = torch.randn(B, D, device="cuda", requires_grad=True, dtype=dtype)
            # test torch
            o_ref = torch_modulate(x, scale, shift)
            o_ref.sum().backward(retain_graph=True)
            f_b = time_fwd_bwd(torch_modulate, x, scale, shift, verbose=False)
            time_f_b[config, "torch"] = f_b
            # test triton
            o2 = fused_modulate(x, scale, shift)
            o2.sum().backward(retain_graph=True)
            f_b = time_fwd_bwd(fused_modulate, x, scale, shift, verbose=False)
            time_f_b[config, "triton"] = f_b
            # test if the results are close
            assert_close("  o", o_ref, o2, 0.005)
            # time_f_b[config, "thunderkitten"] = f_b

            print(f"### batch size={B}, seq length={T}, B={B}, hidden dim={D} ###")
            for method in methods:
                # time_f_b[config, method] = time_f[config, method] + time_b[config, method]
                print(
                    f"{method:>50} fwd + bwd:\t {time_f_b[config, method]*1000:>6.4f} ms "
                )

# with open('flash2_attn_time.plk', 'wb') as fp:
#     pickle.dump((speed_f, speed_b, speed_f_b), fp, protocol=pickle.HIGHEST_PROTOCOL)
