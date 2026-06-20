"""DMD2 distribution-matching distillation (repo: train/methods/distribution_matching).

Roles: student (trainable) · teacher (frozen real score) · critic (trainable fake score).
The student rollout is driven through the shared denoise loop (not a vendored sampler). Loss math
(repo dmd2.py):

  real_cfg_x0 = teacher_uncond_x0 + s·(teacher_cond_x0 − teacher_uncond_x0)
  faker_x0    = critic_x0
  grad        = (faker_x0 − real_cfg_x0) / mean(|gen_x0 − real_cfg_x0|)
  gen_loss    = 0.5·MSE(gen_x0, (gen_x0 − grad).detach())          # generator (student) update
  critic_loss = MSE(critic(noised_gen), ε − gen_x0)                # fake-score flow-matching
"""
from __future__ import annotations

import numpy as np

from v2.core.enums import ConsistencyLevel
from v2.recipes.common import cached_text_encode
from v2.core.request import DiffusionParams, TaskType, make_request
from v2.training.methods.base import TrainingMethod, new_instance, predict_x0


class DMD2Method(TrainingMethod):
    name = "dmd2"
    consistency = ConsistencyLevel.C1  # kernel-pinned: rollout under trainer kernels

    def __init__(self,
                 student_instance,
                 teacher_instance,
                 critic_instance,
                 *,
                 lr: float = 0.05,
                 critic_lr: float = 0.05,
                 guidance_scale: float = 4.5,
                 generator_update_interval: int = 5,
                 rollout_steps: int = 3,
                 **kw):
        super().__init__(student_instance, lr=lr, **kw)
        self.teacher = teacher_instance
        self.critic = critic_instance
        self.critic_lr = critic_lr
        self.guidance_scale = guidance_scale
        self.generator_update_interval = generator_update_interval
        self.rollout_steps = rollout_steps

    def get_grad_clip_targets(self, iteration: int = 0) -> dict:
        t = {"critic": self.critic.component("transformer")}
        if iteration % self.generator_update_interval == 0:
            t["student"] = self.student_dit
        return t

    def _rollout_sample(self, prompt: str, seed: int) -> np.ndarray:
        req = make_request(TaskType.T2V,
                           self.student.card.model_id,
                           prompt,
                           diffusion=DiffusionParams(num_steps=self.rollout_steps, seed=seed))
        return np.asarray(self._rollout(req).outputs["latents"], dtype="float32")

    def train_step(self, batch: dict, iteration: int) -> tuple[dict, dict]:
        self.iteration = iteration
        teacher = self.teacher.component("transformer")
        critic = self.critic.component("transformer")
        gen_losses, critic_losses, gnorm_s, gnorm_c = [], [], [], []
        for i, p in enumerate(batch["prompts"]):
            seed = batch.get("seeds", list(range(len(batch["prompts"]))))[i]
            emb = cached_text_encode(self.student, p)
            neg = cached_text_encode(self.student, "")
            x0g = self._rollout_sample(p, seed)  # student rollout via SHARED loop

            rng = np.random.default_rng(seed ^ 0xD2)
            sigma = float(rng.uniform(0.1, 0.9))
            noise = rng.standard_normal(x0g.shape).astype("float32")
            noised = ((1.0 - sigma) * x0g + sigma * noise).astype("float32")

            # real score (teacher, CFG) + fake score (critic), both in x0-space
            real_cond = predict_x0(teacher(noised, emb, sigma), noised, sigma)
            real_uncond = predict_x0(teacher(noised, neg, sigma), noised, sigma)
            real_cfg = real_uncond + self.guidance_scale * (real_cond - real_uncond)
            faker = predict_x0(critic(noised, emb, sigma), noised, sigma)
            denom = float(np.mean(np.abs(x0g - real_cfg))) + 1e-6
            grad = np.nan_to_num((faker - real_cfg) / denom).astype("float32")
            gen_losses.append(0.5 * float(np.mean(grad**2)))  # DMD generator loss value

            # generator (student) update toward (gen_x0 − grad), converted to velocity space
            if iteration % self.generator_update_interval == 0:
                gen_target_x0 = (x0g - grad).astype("float32")
                v_target = ((noised - gen_target_x0) / max(sigma, 1e-3)).astype("float32")
                _, gn = self.student_dit.mse_grad_step(noised, emb, sigma, v_target, self.lr)
                gnorm_s.append(gn)

            # critic (fake-score) flow-matching update on the generated sample
            sigma2 = float(rng.uniform(0.1, 0.9))
            noise2 = rng.standard_normal(x0g.shape).astype("float32")
            noised2 = ((1.0 - sigma2) * x0g + sigma2 * noise2).astype("float32")
            c_loss, gc = critic.mse_grad_step(noised2, emb, sigma2, (noise2 - x0g).astype("float32"), self.critic_lr)
            critic_losses.append(c_loss)
            gnorm_c.append(gc)

        metrics = {
            "dmd_loss": float(np.mean(gen_losses)),
            "critic_loss": float(np.mean(critic_losses)),
            "grad_norm/critic": float(np.mean(gnorm_c)),
        }
        if gnorm_s:
            metrics["grad_norm/student"] = float(np.mean(gnorm_s))
        return ({"dmd_loss": metrics["dmd_loss"], "critic_loss": metrics["critic_loss"]}, metrics)


def build_dmd2(card, **kw) -> DMD2Method:
    return DMD2Method(new_instance(card), new_instance(card), new_instance(card), **kw)
