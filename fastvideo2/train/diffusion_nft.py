"""DiffusionNFT RL — the math of fastvideo-main's merged
``train/methods/rl/diffusion_nft.py`` (student / old-policy / reference on
the SAME card DiT the serving path uses).

Authority facts (all verified against main at the gate commit):

  * models: three ``WanModel`` roles from the same base checkpoint, each
    loaded through ``maybe_load_fsdp_model`` with
    ``MixedPrecisionPolicy(param_dtype=bf16, reduce_dtype=fp32)`` over fp32
    storage — FORWARDS run on bf16-cast params (norm weights and
    scale_shift_table included; measured by the block0 bisection probe),
    while AdamW (``eps=1e-8``) and the old-policy EMA operate on the fp32
    storage. At world 1 this is exactly the fp32-master/bf16-compute chain
    (``_MasterOpt``) proven by the finetune/DMD2 gates;
  * forwards (``predict_noise``): BTCHW→BCTHW permute in/out, model input
    cast bf16, run under ``torch.autocast(bf16)`` (a no-op on the bf16
    compute params);
  * loss (``_training_timestep_loss``, ported verbatim): xt = (1-t)x0 + t·ε
    in x0's dtype; old/reference predictions no-grad; positive =
    β·fwd + (1-β)·old, implicit negative = (1+β)·old − β·fwd; per-sample
    fp64-weighted x0 MSEs (weight = |x̂0−x0|.mean per sample, clip 1e-5);
    r = clipped-normalized advantages; policy = mean((r·pos + (1−r)·neg)/β ·
    adv_clip_max); + kl_β·MSE(fwd, ref);
  * inner loop: effective grad accum = grad_accum × num_train_timesteps —
    ``backward(loss / accum)`` per call, clip → AdamW → zero once per round;
  * advantages (``_compute_advantages``): group rewards["avg"] by PROMPT,
    (r − mean)/(std_unbiased=False + 1e-4), repeat per timestep;
  * old-policy update (``_update_old_model``): old ← old·decay +
    student·(1−decay) over the LIVE parameters; decay =
    ``_return_decay(step, decay_type)`` (pure).

Randomness (the xt noise) is replayed from capture recordings.
"""
from __future__ import annotations

from typing import Any


def compute_advantages(rewards: Any, prompts: list[str], *,
                       num_train_timesteps: int) -> Any:
    """Pure: main's group-relative advantages at world size 1."""
    import torch
    from collections import defaultdict
    avg = rewards.detach().float()
    out = torch.empty_like(avg)
    groups: dict[str, list[int]] = defaultdict(list)
    for i, p in enumerate(prompts):
        groups[p].append(i)
    for idx in groups.values():
        it = torch.tensor(idx, device=avg.device, dtype=torch.long)
        g = avg[it]
        out[it] = (g - g.mean()) / (g.std(unbiased=False) + 1e-4)
    return out.unsqueeze(1).repeat(1, num_train_timesteps)


def return_decay(step: int, decay_type: int) -> float:
    """Verbatim ``DiffusionNFTMethod._return_decay``."""
    if decay_type == 0:
        flat, uprate, uphold = 0, 0.0, 0.0
    elif decay_type == 1:
        flat, uprate, uphold = 0, 0.001, 0.5
    elif decay_type == 2:
        flat, uprate, uphold = 75, 0.0075, 0.999
    else:
        raise ValueError(f"Unsupported decay_type: {decay_type}")
    if step < flat:
        return 0.0
    return min((step - flat) * uprate, uphold)


class NFTStep:
    def __init__(self, student_fp32: Any, old_fp32: Any, ref_fp32: Any, *,
                 lr: float = 3e-5, beta: float = 0.1, kl_beta: float = 1e-4,
                 adv_clip_max: float = 5.0, adv_mode: str = "all",
                 num_train_timesteps: int = 1000, max_grad_norm: float = 1.0,
                 weight_decay: float = 1e-4, betas=(0.9, 0.999)):
        import torch

        from fastvideo2.train.dmd2 import _MasterOpt
        self.student = _MasterOpt(student_fp32, lr, betas=betas,
                                  weight_decay=weight_decay,
                                  max_grad_norm=max_grad_norm)
        # old also keeps fp32 masters (main EMAs the fp32 storage) with a
        # bf16 compute copy; ref is frozen bf16 compute only
        self.old_master = {n: p.detach().to(torch.float32).clone()
                           for n, p in old_fp32.named_parameters()}
        self.old = old_fp32.to(torch.bfloat16).eval().requires_grad_(False)
        self.ref = ref_fp32.to(torch.bfloat16).eval().requires_grad_(False)
        self.beta = float(beta)
        self.kl_beta = float(kl_beta)
        self.adv_clip_max = float(adv_clip_max)
        self.adv_mode = adv_mode
        self.ntt = int(num_train_timesteps)

    def _fwd(self, model: Any, x_btchw: Any, t: Any, embeds: Any) -> Any:
        import torch
        x = x_btchw.permute(0, 2, 1, 3, 4)
        if x.is_floating_point():
            x = x.to(torch.bfloat16)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(x, embeds, t)
        return out.permute(0, 2, 1, 3, 4)

    def timestep_loss(self, x0: Any, embeds: Any, timestep: Any, noise: Any,
                      advantages: Any) -> dict:
        """One inner NFT step given recorded (x0, timestep, noise, adv).
        x0/noise BTCHW in main's loss dtype; timestep/advantages [B]."""
        import torch
        t = timestep.float() / float(self.ntt)
        t_exp = t.view(-1, *([1] * (x0.ndim - 1)))
        xt = ((1 - t_exp) * x0 + t_exp * noise).to(dtype=x0.dtype)

        with torch.no_grad():
            old_pred = self._fwd(self.old, xt, timestep, embeds).detach()
            ref_pred = self._fwd(self.ref, xt, timestep, embeds).detach()
        fwd_pred = self._fwd(self.student.model, xt, timestep, embeds)

        adv = torch.clamp(advantages, -self.adv_clip_max, self.adv_clip_max)
        if self.adv_mode == "positive_only":
            adv = torch.clamp(adv, 0, self.adv_clip_max)
        elif self.adv_mode == "negative_only":
            adv = torch.clamp(adv, -self.adv_clip_max, 0)
        elif self.adv_mode == "one_only":
            adv = torch.where(adv > 0, torch.ones_like(adv), torch.zeros_like(adv))
        elif self.adv_mode == "binary":
            adv = torch.sign(adv)
        r = torch.clamp((adv / self.adv_clip_max) / 2.0 + 0.5, 0, 1)

        pos = self.beta * fwd_pred + (1 - self.beta) * old_pred.detach()
        neg = (1.0 + self.beta) * old_pred.detach() - self.beta * fwd_pred

        dims = tuple(range(1, x0.ndim))
        x0_hat = xt - t_exp * pos
        with torch.no_grad():
            w = torch.abs(x0_hat.double() - x0.double()).mean(
                dim=dims, keepdim=True).clip(min=0.00001)
        pos_loss = ((x0_hat - x0) ** 2 / w).mean(dim=dims)

        nx0_hat = xt - t_exp * neg
        with torch.no_grad():
            nw = torch.abs(nx0_hat.double() - x0.double()).mean(
                dim=dims, keepdim=True).clip(min=0.00001)
        neg_loss = ((nx0_hat - x0) ** 2 / nw).mean(dim=dims)

        ori = r * pos_loss / self.beta + (1.0 - r) * neg_loss / self.beta
        policy = (ori * self.adv_clip_max).mean()
        kl = ((fwd_pred - ref_pred) ** 2).mean(dim=dims).mean()
        total = policy + self.kl_beta * kl
        return {"total_loss": total, "policy_loss": policy, "kl_div_loss": kl}

    def optimizer_step(self) -> float:
        """grads→fp32 masters, clip, AdamW, copy back to the bf16 compute
        params (main fires this once per accumulation round; the LR
        scheduler is constant/0-warmup at the gate config — no-op)."""
        return self.student.apply_grads_and_step()

    def update_old(self, decay: float) -> None:
        """old ← old·decay + student·(1−decay) over the fp32 STORAGE params
        (verbatim ``_update_old_model`` under FSDP2 mixed precision), then
        refresh old's bf16 compute copy."""
        import torch
        named = dict(self.old.named_parameters())
        with torch.no_grad():
            for n, m in self.old_master.items():
                m.mul_(decay).add_(self.student.master[n], alpha=1.0 - decay)
                named[n].data.copy_(m.to(named[n].dtype))
