"""WorkflowRLMethod — end-to-end RL over a cross-model workflow (design_v3 §10 + §13).

The stress test of the training-plane / Workflow boundary: train BOTH stages of the T2I→I2V workflow
(``flux-t2i`` and ``wan-i2v`` — two *separate* models) from **one final-video reward**. The rollout
spans two instances: roll out the T2I diffusion loop with SDE capture → decode an image → condition
and roll out the I2V diffusion loop with SDE capture → decode a video → score it. One group-relative
advantage from the final video then drives a **FlowGRPO PPO update on each stage's transformer**, with
**two independent WeightSyncPlans on two different instances**.

The interesting part is the credit assignment: the reward is at the *end* (the video), yet it credits an
*earlier* model (the T2I image generator) — end-to-end RL across a model boundary. The T2I stage learns
to produce images that lead to better videos. This probes whether "rollout == serve + capture" holds for
a *workflow* (multi-instance), not just a single card's loops — and it does, with no new primitive: the
method drives the same shared loops the engine serves, on each instance, and threads the artifact between
them exactly as the serve-time Workflow does.
"""
from __future__ import annotations

import numpy as np

from ..._enums import ConsistencyLevel, ExecutionProfile
from ...loop.sampler import flow_sde_ml_velocity, flow_sde_step_with_logprob
from ...models.common import cached_text_encode
from ...request import DiffusionParams, TaskType, make_request
from ..rollout import rollout_loop
from ..weight_sync import WeightRole, WeightSyncPlan
from .base import TrainingMethod, new_instance


def _flowgrpo_ppo_step(dit, ref_dit, behavior, emb, advantage, *, lr, ppo_clip, beta, noise_scale):
    """One FlowGRPO PPO pass over an SDE trajectory: per-step ratio + KL, then move the policy toward
    the max-likelihood velocity of each realized sample (advantage-weighted). Returns
    (ppo_loss, kl, ratio, grad_norm). Shared by both stages."""
    ppo_losses, kls, ratios, gnorms = [], [], [], []
    for rec in behavior or []:
        if "sde_logprob" not in rec:
            continue
        prev, sample = rec["prev"], rec["sample"]
        st, sn = float(rec["sigma_t"]), float(rec["sigma_next"])
        v_cur = dit(prev, emb, st)
        _, logp_cur, mean_cur, eff_std = flow_sde_step_with_logprob(
            prev, v_cur, st, sn, prev_sample=sample, noise_scale=noise_scale)
        v_ref = ref_dit(prev, emb, st)
        _, _, mean_ref, _ = flow_sde_step_with_logprob(
            prev, v_ref, st, sn, prev_sample=sample, noise_scale=noise_scale)
        ratio = float(np.exp(np.clip(logp_cur - float(rec["sde_logprob"]), -10.0, 10.0)))
        clipped = float(np.clip(ratio, 1.0 - ppo_clip, 1.0 + ppo_clip))
        ppo_losses.append(max(-advantage * ratio, -advantage * clipped))
        kls.append(float(np.mean((mean_cur - mean_ref) ** 2) / (2.0 * max(eff_std ** 2, 1e-8))))
        ratios.append(ratio)
        v_target = flow_sde_ml_velocity(prev, sample, st, sn, noise_scale=noise_scale)
        _, gn = dit.mse_grad_step(prev, emb, st, v_target, lr * float(np.clip(advantage, -1.0, 1.0)))
        gnorms.append(gn)
    m = lambda xs: float(np.mean(xs)) if xs else 0.0          # noqa: E731
    return m(ppo_losses), m(kls), m(ratios), m(gnorms)


class WorkflowRLMethod(TrainingMethod):
    name = "workflow_rl"
    consistency = ConsistencyLevel.C2

    def __init__(self, t2i_student, i2v_student, t2i_reference, i2v_reference, *,
                 num_samples_per_prompt: int = 4, rollout_steps: int = 4, t2i_lr: float = 0.02,
                 i2v_lr: float = 0.02, beta: float = 0.01, adv_clip_max: float = 5.0,
                 ppo_clip: float = 1e-2, sde_noise_scale: float = 0.7, **kw):
        super().__init__(t2i_student, lr=t2i_lr, **kw)        # base "student" = the T2I stage
        self.t2i, self.i2v = t2i_student, i2v_student
        self.t2i_ref, self.i2v_ref = t2i_reference, i2v_reference
        self.K = num_samples_per_prompt
        self.rollout_steps = rollout_steps
        self.t2i_lr, self.i2v_lr = t2i_lr, i2v_lr
        self.beta, self.adv_clip_max, self.ppo_clip = beta, adv_clip_max, ppo_clip
        self.sde_noise_scale = sde_noise_scale
        # one WeightSyncPlan per stage's generator — on two DIFFERENT instances (independent versions)
        self.t2i_sync = WeightSyncPlan(role=WeightRole.STUDENT, components=("transformer",))
        self.i2v_sync = WeightSyncPlan(role=WeightRole.STUDENT, components=("transformer",))
        self.t2i_ref.component("transformer").copy_from(self.t2i.component("transformer"))
        self.i2v_ref.component("transformer").copy_from(self.i2v.component("transformer"))

    def manages_optimization(self) -> bool:
        return True

    def get_grad_clip_targets(self, iteration: int = 0) -> dict:
        return {"t2i.transformer": self.t2i.component("transformer"),
                "i2v.transformer": self.i2v.component("transformer")}

    def _sde_params(self, **extra):
        return DiffusionParams(num_steps=self.rollout_steps, guidance_scale=1.0, sde_rollout=True,
                               sde_noise_scale=self.sde_noise_scale, **extra)

    def _rollout_t2i(self, prompt, seed):
        emb = cached_text_encode(self.t2i, prompt)
        neg = cached_text_encode(self.t2i, "")
        req = make_request(TaskType.T2I, self.t2i.card.model_id, prompt,
                           diffusion=self._sde_params(num_frames=1, seed=seed))
        res = rollout_loop(self.t2i, "t2i_denoise", req,
                           slots={"text_embeds": emb, "neg_text_embeds": neg},
                           profile=ExecutionProfile.ROLLOUT)
        image = self.t2i.component("vae").decode(res.outputs["latents"])
        return res, emb, image

    def _rollout_i2v(self, prompt, image, seed):
        emb = cached_text_encode(self.i2v, prompt)
        neg = cached_text_encode(self.i2v, "")
        cond = self.i2v.component("vae").encode(np.asarray(image, dtype="float32"))   # image conditioning
        emb = (np.asarray(emb, dtype="float32") + float(np.tanh(np.mean(cond)))).astype("float32")
        req = make_request(TaskType.I2V, self.i2v.card.model_id, prompt,
                           diffusion=self._sde_params(num_frames=81, seed=seed))
        res = rollout_loop(self.i2v, "i2v_denoise", req,
                           slots={"text_embeds": emb, "neg_text_embeds": neg},
                           profile=ExecutionProfile.ROLLOUT)
        video = self.i2v.component("vae").decode(res.outputs["latents"])
        return res, emb, video

    @staticmethod
    def _reward(video):
        return float(np.tanh(-np.std(np.asarray(video, dtype="float64"))))   # lower-spread video = better

    def managed_train_step(self, batch: dict, iteration: int) -> tuple[dict, dict]:
        self.iteration = iteration
        prompts = batch["prompts"]
        seeds = batch.get("seeds", list(range(len(prompts))))
        t2i_loss, i2v_loss, t2i_gn, i2v_gn, ratios, all_reward = [], [], [], [], [], []

        for pi, prompt in enumerate(prompts):
            samples = []
            for k in range(self.K):
                seed = seeds[pi] * 1000 + k
                res_t, emb_t, image = self._rollout_t2i(prompt, seed)             # stage 1 rollout (SDE)
                res_i, emb_i, video = self._rollout_i2v(prompt, image, seed)      # stage 2 rollout (SDE)
                samples.append((res_t, emb_t, res_i, emb_i, self._reward(video)))

            rewards = np.array([s[4] for s in samples], dtype="float64")
            adv = np.clip((rewards - rewards.mean()) / (rewards.std() + 1e-4),
                          -self.adv_clip_max, self.adv_clip_max)                  # one final-video advantage
            all_reward.extend(rewards.tolist())

            for k, (res_t, emb_t, res_i, emb_i, _r) in enumerate(samples):
                a = float(adv[k])
                # the SAME final advantage credits BOTH stages (end-to-end across the model boundary)
                pt, _kt, rt, gt = _flowgrpo_ppo_step(
                    self.t2i.component("transformer"), self.t2i_ref.component("transformer"),
                    res_t.behavior, emb_t, a, lr=self.t2i_lr, ppo_clip=self.ppo_clip,
                    beta=self.beta, noise_scale=self.sde_noise_scale)
                pi_, _ki, ri, gi = _flowgrpo_ppo_step(
                    self.i2v.component("transformer"), self.i2v_ref.component("transformer"),
                    res_i.behavior, emb_i, a, lr=self.i2v_lr, ppo_clip=self.ppo_clip,
                    beta=self.beta, noise_scale=self.sde_noise_scale)
                t2i_loss.append(pt); i2v_loss.append(pi_)
                t2i_gn.append(gt); i2v_gn.append(gi)
                ratios.extend([rt, ri])

        t2i_ver = self.t2i_sync.apply(self.t2i.component("transformer"),
                                      self.t2i.component("transformer"), self.t2i)
        i2v_ver = self.i2v_sync.apply(self.i2v.component("transformer"),
                                      self.i2v.component("transformer"), self.i2v)
        m = lambda xs: float(np.mean(xs)) if xs else 0.0      # noqa: E731
        metrics = {
            "reward_mean": m(all_reward),
            "grad_norm/t2i": m(t2i_gn),
            "grad_norm/i2v": m(i2v_gn),
            "ppo_ratio_mean": m(ratios) if ratios else 1.0,
            "t2i_weights_version": t2i_ver,
            "i2v_weights_version": i2v_ver,
        }
        return {"t2i_pg_loss": m(t2i_loss), "i2v_pg_loss": m(i2v_loss)}, metrics

    def train_step(self, batch: dict, iteration: int) -> tuple[dict, dict]:
        return self.managed_train_step(batch, iteration)


def build_workflow_rl(t2i_card, i2v_card, **kw) -> WorkflowRLMethod:
    """Four roles: trainable T2I + I2V students and their frozen references (two instances each side)."""
    return WorkflowRLMethod(new_instance(t2i_card), new_instance(i2v_card),
                            new_instance(t2i_card), new_instance(i2v_card), **kw)
