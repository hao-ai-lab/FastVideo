"""Wan2.1 flow-match finetuning — the math of fastvideo-main's legacy
``TrainingPipeline`` (the shipped-model authority), on the SAME card-loaded
DiT the inference path serves (``wan21/model_fv.py`` — training reuses
inference definitions, never forks them).

Mixed-precision chain mirrors main's FSDP2 policy at world size 1 (fp32
master params, bf16 param cast for compute, fp32 reduce):

    bf16 compute model (forward+backward) -> grads upcast fp32 onto master
    -> clip_grad_norm_ -> AdamW(master, lr 5e-5, wd 1e-4, betas 0.9/0.999)
    -> master cast back into the bf16 model

Loss (main training_pipeline.py:391): pred and target float()ed,
``mean((pred - (noise - latents))**2) / grad_accum``; target computed in
bf16 first. Inputs (:278): noisy = (1-sigma)*latents + sigma*noise, all
bf16; timesteps fed to the model as BF16.

Known nondeterminism: flash-attention backward uses atomics, so per-step
losses are bitwise only at step 1; later steps carry grad noise. The gate
tolerance is main-vs-main self-noise, measured by running the capture twice
(recorded in the goldens manifest) — never hand-picked.
"""
from __future__ import annotations

from typing import Any


class FinetuneStep:
    """One training step of main's finetune math. Deterministic given
    (latents, embeds, noise, timesteps, sigmas) — RNG lives with the caller
    (trainer replays recorded draws or reproduces the seeded generators)."""

    def __init__(self, model_fp32: Any, *, lr: float = 5e-5, weight_decay: float = 1e-4,
                 betas: tuple[float, float] = (0.9, 0.999), max_grad_norm: float = 1.0,
                 grad_accum: int = 1):
        """``model_fp32``: the card DiT loaded with torch_dtype=fp32 (main's
        dit_precision) — masters snapshot the fp32 checkpoint values, then the
        module is downcast IN PLACE to bf16 as the compute copy."""
        import torch
        self.max_grad_norm = max_grad_norm
        self.grad_accum = grad_accum
        # fp32 masters snapshot BEFORE the bf16 downcast (main: fp32 storage)
        self.master = {n: p.detach().to(torch.float32).clone().requires_grad_(False)
                       for n, p in model_fp32.named_parameters()}
        self.model = model_fp32.to(torch.bfloat16)
        self.opt = torch.optim.AdamW(list(self.master.values()), lr=lr,
                                     betas=betas, weight_decay=weight_decay)
        for p in self.model.parameters():
            p.requires_grad_(True)

    def step(self, latents: Any, embeds: Any, noise: Any, timesteps: Any,
             sigmas: Any) -> tuple[float, float]:
        """All tensor args bf16 on device (sigmas broadcastable). Returns
        (loss, grad_norm) exactly as main logs them."""
        import torch
        noisy = (1.0 - sigmas) * latents + sigmas * noise
        pred = self.model(noisy, embeds, timesteps.to(torch.bfloat16))
        target = noise - latents
        loss = torch.mean((pred.float() - target.float()) ** 2) / self.grad_accum
        loss.backward()

        # grads -> fp32 masters (FSDP2 reduce_dtype fp32 at world 1)
        named = dict(self.model.named_parameters())
        for n, m in self.master.items():
            g = named[n].grad
            m.grad = g.detach().to(torch.float32) if g is not None else None
        grad_norm = torch.nn.utils.clip_grad_norm_(
            list(self.master.values()), self.max_grad_norm).item()
        self.opt.step()
        self.opt.zero_grad(set_to_none=True)
        self.model.zero_grad(set_to_none=True)
        with torch.no_grad():
            for n, m in self.master.items():
                named[n].data.copy_(m.to(named[n].dtype))
        return float(loss.detach().item()), float(grad_norm)


def sample_inputs_like_main(latents_shape: Any, *, seed: int, rank: int = 0,
                            step_gens: Any = None, table: Any = None,
                            device: Any = None) -> tuple[Any, Any, Any, Any]:
    """Reproduce main's per-step RNG draws: noise from a CUDA generator
    (seed+rank), u from a CPU generator (seed+rank) with uniform weighting,
    timesteps = FlowUniPC table[(u*1000).long()], sigmas by exact table match.
    ``step_gens`` carries (cuda_gen, cpu_gen) across steps — one stream per
    run, like main. ``table`` is (timesteps_tensor, sigmas_tensor)."""
    import torch
    cuda_gen, cpu_gen = step_gens
    noise = torch.randn(latents_shape, generator=cuda_gen, device=device,
                        dtype=torch.bfloat16)
    u = torch.rand((latents_shape[0],), generator=cpu_gen)  # uniform scheme
    tt, ss = table
    indices = (u * 1000).long()
    timesteps = tt[indices].to(device)
    step_idx = [(tt == t).nonzero().item() for t in timesteps.cpu()]
    sigmas = ss[step_idx].to(device=device, dtype=torch.bfloat16).view(-1, 1, 1, 1)
    return noise, timesteps, sigmas, u
