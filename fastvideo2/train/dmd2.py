"""DMD2 distillation — the math of fastvideo-main's LEGACY
``distillation_pipeline.py`` (authoritative for shipped FastWan artifacts).

Three roles on the SAME card DiT class the serving path uses (reuse
invariant): student (trainable, fp32 masters + bf16 compute), teacher
("real score", frozen bf16), critic ("fake score", trainable, fp32 masters
+ bf16 compute). Sigma machinery reuses the inference loop's
``dmd_inference_table`` (the shift-8 FlowMatch training table — the same
table main's scheduler exposes to ``pred_noise_to_pred_video``/``add_noise``).

Legacy conventions preserved exactly (differ from main's NEW modular stack):
  * teacher CFG in the DMD2 parameterization ``cond + w*(cond - uncond)``
    (offset by 1 from Ho&Salimans; w=3.5 in the shipped 1.3B recipe)
  * dmd_loss = 0.5*MSE(x0.float(), (x0 - grad).float().detach()) with
    grad = (x0_fake - x0_real) / |x0_student - x0_real|.mean(), nan_to_num
  * critic loss = mean((pred_noise - (noise - x0_student))**2) in BF16
    (no .float() — unlike the finetune loss)
  * timesteps enter the DiT as the raw (warped, clamped) values — LONG/fp32,
    never cast to bf16 (unlike finetune)
  * student updated every ``generator_update_interval`` steps; critic every
    step; separate AdamWs

All randomness is replayed from capture recordings (main draws from the
GLOBAL torch RNG — record-and-replay keeps the gate independent of stream
ordering).
"""
from __future__ import annotations

from typing import Any


class _MasterOpt:
    """fp32 masters + bf16 compute for one role (main's FSDP2 world-1 mixed
    precision, same chain proven bitwise by the finetune gate)."""

    def __init__(self, model_fp32: Any, lr: float, betas=(0.9, 0.999),
                 weight_decay: float = 1e-4, max_grad_norm: float = 1.0):
        import torch
        self.master = {n: p.detach().to(torch.float32).clone().requires_grad_(False)
                       for n, p in model_fp32.named_parameters()}
        self.model = model_fp32.to(torch.bfloat16)
        self.opt = torch.optim.AdamW(list(self.master.values()), lr=lr,
                                     betas=betas, weight_decay=weight_decay)
        self.max_grad_norm = max_grad_norm
        for p in self.model.parameters():
            p.requires_grad_(True)

    def apply_grads_and_step(self) -> float:
        import torch
        named = dict(self.model.named_parameters())
        for n, m in self.master.items():
            g = named[n].grad
            m.grad = g.detach().to(torch.float32) if g is not None else None
        gnorm = torch.nn.utils.clip_grad_norm_(
            list(self.master.values()), self.max_grad_norm).item()
        self.opt.step()
        self.opt.zero_grad(set_to_none=True)
        self.model.zero_grad(set_to_none=True)
        with torch.no_grad():
            for n, m in self.master.items():
                named[n].data.copy_(m.to(named[n].dtype))
        return float(gnorm)


class DMD2Step:
    def __init__(self, student_fp32: Any, teacher_fp32: Any, critic_fp32: Any, *,
                 denoising_steps: tuple[int, ...] = (1000, 757, 522),
                 shift: float = 8.0, guidance: float = 3.5,
                 lr: float = 2e-6, fake_lr: float = 2e-6,
                 min_t_ratio: float = 0.02, max_t_ratio: float = 0.98,
                 table: Any = None):
        """``table``: (timesteps, sigmas) override — self-forcing uses the
        SelfForcingFlowMatchScheduler table instead of the FlowMatch fork's."""
        import torch

        from fastvideo2.wan21.loop import dmd_inference_table
        self.student = _MasterOpt(student_fp32, lr)
        self.critic = _MasterOpt(critic_fp32, fake_lr)
        self.teacher = teacher_fp32.to(torch.bfloat16).eval().requires_grad_(False)
        if table is None:
            tt, ss = dmd_inference_table(shift)
            self.table_t = torch.tensor(tt)
            self.table_s = torch.tensor(ss)
        else:
            self.table_t, self.table_s = table[0].clone(), table[1].clone()
        self.denoising_steps = tuple(int(t) for t in denoising_steps)
        self.guidance = float(guidance)
        self.min_t, self.max_t = min_t_ratio * 1000, max_t_ratio * 1000

    # --- scheduler math (main's FlowMatchEulerDiscreteScheduler fork) ------ #
    def _sigma(self, t: Any, fp64: bool = False) -> Any:
        import torch
        tt = self.table_t.to(t.device)
        idx = (tt.unsqueeze(0) - t.float().unsqueeze(1)).abs().argmin(dim=1)
        s = self.table_s.to(t.device)[idx]
        return s.double() if fp64 else s

    def add_noise(self, x0: Any, noise: Any, t: Any) -> Any:
        sigma = self._sigma(t).to(noise.device).view(-1, 1, 1, 1)
        # fp32 sigma tensor promotes; single cast back (the 0-dim lesson)
        return ((1 - sigma) * x0.flatten(0, 1) + sigma * noise.flatten(0, 1)
                ).type_as(noise).unflatten(0, (x0.shape[0], x0.shape[1]))

    def pred_video(self, pred_noise: Any, noisy: Any, t: Any) -> Any:
        sigma = self._sigma(t, fp64=True).view(-1, 1, 1, 1)
        p, n = pred_noise.flatten(0, 1), noisy.flatten(0, 1)
        return (n.double() - sigma * p.double()).to(pred_noise.dtype
                                                    ).unflatten(0, pred_noise.shape[:2])

    # --- forwards (BTCHW state, BCTHW model calls — main's convention) ----- #
    def _fwd(self, model: Any, noisy_btchw: Any, t: Any, embeds: Any,
             vsa: Any = None) -> Any:
        out = model(noisy_btchw.permute(0, 2, 1, 3, 4), embeds, t, vsa=vsa)
        return out.permute(0, 2, 1, 3, 4)

    def student_rollout(self, draws: dict, embeds: Any, vsa: Any = None) -> Any:
        """main's `_generator_multi_step_simulation_forward`, replaying the
        recorded draws: target_idx (int), init_noise, step_noises (list)."""
        import torch
        target_idx = int(draws["target_idx"])
        x = draws["init_noise"]
        x_copy = x.clone()
        noise_latents = []
        with torch.no_grad():
            for k in range(len(self.denoising_steps) - 1):
                t_k = torch.tensor([self.denoising_steps[k]], device=x.device,
                                   dtype=torch.long)
                pred = self._fwd(self.student.model, x, t_k, embeds, vsa=vsa)
                clean = self.pred_video(pred, x, t_k)
                t_next = torch.tensor([self.denoising_steps[k + 1]], device=x.device,
                                      dtype=torch.long)
                x = self.add_noise(clean, draws["step_noises"][k], t_next)
                noise_latents.append(x.clone())
        noisy_input = noise_latents[target_idx - 1] if target_idx > 0 else x_copy
        t_tgt = torch.tensor([self.denoising_steps[target_idx]], device=x.device,
                             dtype=torch.long)
        pred = self._fwd(self.student.model, noisy_input, t_tgt, embeds, vsa=vsa)
        return self.pred_video(pred, noisy_input, t_tgt)

    def dmd_loss(self, x0_student: Any, draws: dict, cond: Any, uncond: Any,
                 vsa_dense: Any = None) -> Any:
        """``vsa_dense``: for VSA+DMD2, main scores with sparsity-0.0 VSA
        metadata (same kernel, all blocks, gate mixing active) — NOT flash."""
        import torch
        import torch.nn.functional as F
        with torch.no_grad():
            t = draws["dmd_timestep"].to(x0_student.device)  # warped+clamped, long
            noisy = self.add_noise(x0_student, draws["dmd_noise"], t)
            fake = self.pred_video(self._fwd(self.critic.model, noisy, t, cond,
                                             vsa=vsa_dense), noisy, t)
            real_c = self.pred_video(self._fwd(self.teacher, noisy, t, cond,
                                               vsa=vsa_dense), noisy, t)
            real_u = self.pred_video(self._fwd(self.teacher, noisy, t, uncond,
                                               vsa=vsa_dense), noisy, t)
            real = real_c + (real_c - real_u) * self.guidance  # DMD2 CFG
            grad = (fake - real) / torch.abs(x0_student - real).mean()
            grad = torch.nan_to_num(grad)
        return 0.5 * F.mse_loss(x0_student.float(),
                                (x0_student.float() - grad.float()).detach())

    def critic_loss(self, x0_student: Any, draws: dict, cond: Any,
                    vsa_dense: Any = None) -> Any:
        import torch
        t = draws["critic_timestep"].to(x0_student.device)
        noise = draws["critic_noise"]
        noisy = self.add_noise(x0_student, noise, t)
        pred = self._fwd(self.critic.model, noisy, t, cond, vsa=vsa_dense)
        target = noise - x0_student
        return torch.mean((pred - target) ** 2)  # bf16, no .float() (legacy)
