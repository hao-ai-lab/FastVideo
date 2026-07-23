"""DiffusionNFT RL — the math of fastvideo-main's merged
``train/methods/rl/diffusion_nft.py`` (student / old-policy / reference on
the SAME card DiT the serving path uses).

Loss (`_training_timestep_loss`, ported verbatim): xt = (1-t)x0 + t·ε;
old/reference predictions no-grad; positive = β·fwd + (1-β)·old, implicit
negative = (1+β)·old − β·fwd; per-sample fp64-weighted x0 MSEs (weight =
|x̂0−x0|.mean per sample, clip 1e-5); r = clipped-normalized advantages;
policy = (r·pos + (1−r)·neg)/β · adv_clip_max, mean; + kl_β·MSE(fwd, ref).

Advantages (`_compute_advantages`): group rewards by PROMPT,
(r − mean)/(std_unbiased=False + 1e-4).

Old-policy update: EMA old ← decay·old + (1−decay)·student.

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
        self.old = old_fp32.to(torch.bfloat16).eval().requires_grad_(False)
        self.ref = ref_fp32.to(torch.bfloat16).eval().requires_grad_(False)
        self.beta = float(beta)
        self.kl_beta = float(kl_beta)
        self.adv_clip_max = float(adv_clip_max)
        self.adv_mode = adv_mode
        self.ntt = int(num_train_timesteps)

    def _fwd(self, model: Any, x_btchw: Any, t: Any, embeds: Any) -> Any:
        out = model(x_btchw.permute(0, 2, 1, 3, 4), embeds, t)
        return out.permute(0, 2, 1, 3, 4)

    def timestep_loss(self, x0: Any, embeds: Any, timestep: Any, noise: Any,
                      advantages: Any) -> dict:
        """One inner NFT step given recorded (x0, timestep, noise, adv).
        All tensors BTCHW bf16 except timestep [B] and advantages [B]."""
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

    def update_old(self, decay: float) -> None:
        """old <- decay*old + (1-decay)*student (main's `_update_old_model`);
        student values come from the fp32 masters (the source of truth)."""
        import torch
        with torch.no_grad():
            named = dict(self.old.named_parameters())
            for n, m in self.student.master.items():
                p = named[n]
                p.data.mul_(decay).add_(m.to(p.dtype), alpha=1.0 - decay)
