"""Self-forcing distillation — the math of fastvideo-main's
``self_forcing_distillation_pipeline.py`` (SFWan training).

The rollout drives the SAME causal model the serving path uses
(``WanModelFVCausal`` + its KV/cross caches — proven bitwise by the SFWan
inference anchor); this module adds only main's TRAINING orchestration:
per-block random exit steps, no-grad non-exit denoising, gradients on the
exit forward within the last-21-frame window, and a noised context re-encode
pass after each block. DMD/critic losses come from ``DMD2Step`` with the
SelfForcingFlowMatchScheduler table (``self_forcing_table``) and the SF
recipe's guidance (3.0).

Randomness is replayed via ``DrawCursor`` — the capture records every
randn/randn_like/randint made inside main's rollout IN CALL ORDER; replay
pops them sequentially and validates shapes, so the gate cannot silently
survive an order mismatch.
"""
from __future__ import annotations

from typing import Any


class DrawCursor:
    """Sequential replayer for recorded RNG draws: list of (kind, value)."""

    def __init__(self, draws: list, device: Any):
        self.draws = draws
        self.i = 0
        self.device = device

    def randn(self, shape: tuple, dtype: Any) -> Any:
        import torch
        kind, val = self.draws[self.i]
        assert kind == "randn", (self.i, kind)
        t = torch.from_numpy(val).to(self.device, dtype)
        assert tuple(t.shape) == tuple(shape), (self.i, tuple(t.shape), tuple(shape))
        self.i += 1
        return t

    def randint(self) -> int:
        kind, val = self.draws[self.i]
        assert kind == "randint", (self.i, kind)
        self.i += 1
        return int(val)

    def randint_list(self, n: int) -> list[int]:
        return [self.randint() for _ in range(n)]


def sf_rollout(model: Any, cursor: "DrawCursor", embeds: Any, *,
               num_frames: int = 21, num_frame_per_block: int = 3,
               denoising_steps: tuple[float, ...], table: tuple[Any, Any],
               latent_hw: tuple[int, int], context_noise: int = 0,
               same_step_across_blocks: bool = False) -> Any:
    """main's `_generator_multi_step_simulation_forward` (T2V, no MoE, no
    image latent, generated frames pinned to the full window so the >21-frame
    VAE branch never fires). Returns [B, F, C, H, W] with grads flowing only
    through exit forwards inside the last-21-frame window."""
    import torch
    device = next(model.parameters()).device
    dtype = torch.bfloat16
    tt, ss = table
    tt64, ss64 = tt.double(), ss.double()

    def sigma(t_val: Any, fp64: bool = True) -> Any:
        idx = (tt.to(t_val.device) - t_val.float().unsqueeze(-1)).abs().argmin(dim=-1)
        s = ss.to(t_val.device)[idx]
        return s.double() if fp64 else s

    h_lat, w_lat = latent_hw
    frame_seqlen = (h_lat // 2) * (w_lat // 2)
    num_blocks_drawn = cursor.randint()  # block-count draw (deterministic at 21f)
    num_blocks = num_frames // num_frame_per_block
    assert num_blocks_drawn == num_blocks, (num_blocks_drawn, num_blocks)
    noise = cursor.randn((1, num_frames, 16, h_lat, w_lat), dtype)
    exit_flags = cursor.randint_list(num_blocks)

    kv_cache = type(model).make_kv_cache(len(model.blocks), 1, 21 * frame_seqlen,
                                         model.num_attention_heads, model.head_dim,
                                         dtype, device)
    xattn_cache = type(model).make_crossattn_cache(len(model.blocks))
    output = torch.zeros_like(noise)
    start_gradient_frame_index = max(0, num_frames - 21)
    nds = len(denoising_steps)
    start = 0

    def fwd(x_btchw, t_2d, grad: bool):
        ctx = torch.enable_grad() if grad else torch.no_grad()
        with ctx:
            out = model(x_btchw.permute(0, 2, 1, 3, 4), embeds, t_2d,
                        kv_cache=kv_cache, crossattn_cache=xattn_cache,
                        current_start=start * frame_seqlen, start_frame=start)
        return out.permute(0, 2, 1, 3, 4)

    def x0_of(pred, noisy, t_2d):
        s = sigma(t_2d.flatten()).view(-1, 1, 1, 1)
        p, n = pred.flatten(0, 1), noisy.flatten(0, 1)
        return (n.double() - s * p.double()).to(pred.dtype).unflatten(0, pred.shape[:2])

    def add_noise(x0, eps, t_1d):
        s = sigma(t_1d, fp64=False).to(eps.device).view(-1, 1, 1, 1)
        return ((1 - s) * x0.flatten(0, 1) + s * eps.flatten(0, 1)
                ).type_as(eps).unflatten(0, (x0.shape[0], x0.shape[1]))

    for block_index in range(num_blocks):
        nf = num_frame_per_block
        noisy_input = noise[:, start:start + nf]
        for index, t_cur in enumerate(denoising_steps):
            exit_flag = (index == exit_flags[0] if same_step_across_blocks
                         else index == exit_flags[block_index])
            # int64 ones * fp32 scalar tensor -> FP32 warped timesteps (937.5...)
            t_2d = (torch.ones((1, nf), device=device, dtype=torch.int64)
                    * torch.tensor(float(t_cur), device=device, dtype=torch.float32))
            if not exit_flag:
                pred = fwd(noisy_input.to(dtype), t_2d, grad=False)
                x0 = x0_of(pred, noisy_input, t_2d)
                t_next = (torch.ones((nf,), device=device, dtype=torch.long)
                          * torch.tensor(float(denoising_steps[index + 1]),
                                         device=device, dtype=torch.float32))
                eps = cursor.randn((nf, 16, h_lat, w_lat), x0.dtype)[None].flatten(0, 1)
                noisy_input = add_noise(x0, eps.unflatten(0, (1, nf)), t_next)
            else:
                grad = start >= start_gradient_frame_index
                pred = fwd(noisy_input.to(dtype), t_2d, grad=grad)
                x0 = x0_of(pred, noisy_input, t_2d)
                break
        output[:, start:start + nf] = x0

        # context re-encode at context_noise (noised with a recorded draw)
        t_ctx = torch.ones_like(t_2d) * context_noise
        eps = cursor.randn((nf, 16, h_lat, w_lat), x0.dtype)[None].flatten(0, 1)
        ctx_in = add_noise(x0, eps.unflatten(0, (1, nf)),
                           torch.ones((nf,), device=device, dtype=torch.long) * context_noise)
        fwd(ctx_in.to(dtype), t_ctx, grad=False)
        start += nf
    return output
