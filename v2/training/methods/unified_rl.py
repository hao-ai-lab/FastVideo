"""UnifiedRLMethod — joint LM + generator RL (UniRL / PromptRL).

One reward, computed once per sample, drives TWO trainable experts under one group-relative advantage:

  * the ``llm`` prompt-refiner — a token policy gradient (REINFORCE on the refinement action,
    ``−A·logπ(a) + β·KL(π‖π_ref)``); and
  * the ``transformer`` flow generator — FlowGRPO PPO over the SDE rollout's per-step log-probs
    (``max(−A·ρ, −A·clip(ρ, 1±ε)) + β·KL`` with ``ρ = exp(logp_cur − logp_rollout)``).

Notable: the training rollout spans two loop types (``ar_decode`` refine + ``diffusion_denoise``);
dual log-prob capture (categorical LM action + Gaussian per-step SDE) feeds one advantage; the two
experts use independent WeightSyncPlans versioned/cache-invalidated separately (the LM sync must not
flush the text-encoder feature cache); the rollout SDE sampler differs from the serve ODE sampler, a
controlled divergence gated by ``DiffusionParams.sde_rollout`` so the serve path is untouched.

``joint=False`` is PromptRL's prompt-only mode: the generator is frozen (ODE rollout, no DiT update),
only the LM learns. ``num_skip_refinement`` samples per group use the original prompt (partial
refinement) so the advantage contrasts refined vs unrefined — the LM's learning signal.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from v2.core.enums import ConsistencyLevel, ExecutionProfile
from v2.core.loop.sampler import flow_sde_ml_velocity as _ml_velocity
from v2.recipes.common import cached_text_encode
from v2.platform import FLOW_SDE_STEP
from v2.core.request import DiffusionParams, TaskType, make_request
from v2.training.rollout import rollout_loop
from v2.training.weight_sync import WeightRole, WeightSyncPlan
from v2.training.methods.base import TrainingMethod, new_instance


class UnifiedRLMethod(TrainingMethod):
    name = "unified_rl"
    consistency = ConsistencyLevel.C2  # likelihood-BASED (per-step log-prob identity)
    student_loop_id = "diffusion_denoise"

    def __init__(self,
                 student_instance,
                 reference_instance,
                 *,
                 joint: bool = True,
                 num_samples_per_prompt: int = 4,
                 num_skip_refinement: int = 1,
                 rollout_steps: int = 4,
                 llm_lr: float = 0.4,
                 dit_lr: float = 0.02,
                 beta: float = 0.01,
                 adv_clip_max: float = 5.0,
                 ppo_clip: float = 1e-2,
                 sde_noise_scale: float = 0.7,
                 target_action: int = 3,
                 action_bonus: float = 2.0,
                 **kw):
        super().__init__(student_instance, lr=dit_lr, **kw)
        self.reference = reference_instance
        self.joint = joint
        self.K = num_samples_per_prompt
        self.num_skip = max(0, min(num_skip_refinement, self.K - 1))
        self.rollout_steps = rollout_steps
        self.llm_lr, self.dit_lr = llm_lr, dit_lr
        self.beta, self.adv_clip_max, self.ppo_clip = beta, adv_clip_max, ppo_clip
        self.sde_noise_scale = sde_noise_scale
        self.target_action, self.action_bonus = target_action, action_bonus
        # one plan PER trainable expert — versioned + cache-scoped independently
        self.llm_sync = WeightSyncPlan(role=WeightRole.STUDENT, components=("llm", ))
        self.dit_sync = WeightSyncPlan(role=WeightRole.STUDENT, components=("transformer", ))
        # reference = frozen copy of BOTH experts at init (the two KL anchors)
        self.reference.component("llm").copy_from(self.llm)
        self.reference.component("transformer").copy_from(self.dit)

    # --- expert handles --------------------------------------------------------- #
    @property
    def llm(self):
        return self.student.component("llm")

    @property
    def dit(self):
        return self.student.component("transformer")

    def manages_optimization(self) -> bool:
        return True

    def get_grad_clip_targets(self, iteration: int = 0) -> dict:
        t = {"llm": self.llm}
        if self.joint:
            t["transformer"] = self.dit
        return t

    # --- toy reward: a diffusion signal + an LM action bonus -------------------- #
    def _reward(self, latent: np.ndarray, action: int | None) -> float:
        """Reward depends on BOTH experts: lower-spread latents (generator) + the LM picking the
        reward-favored ``target_action`` (refiner). So one scalar trains both."""
        diffusion_signal = float(np.tanh(-np.std(np.asarray(latent, dtype="float64"))))
        lm_signal = self.action_bonus if (action is not None and int(action) == self.target_action) else 0.0
        return diffusion_signal + lm_signal

    def _rollout_sample(self, prompt: str, emb, neg_emb, seed: int):
        """Drive the SHARED diffusion loop, conditioned on the (possibly refined) embedding.
        SDE rollout (with per-step log-prob capture) when joint; deterministic ODE when prompt-only."""
        # guidance_scale=1 ⇒ the CFG combine collapses to the cond branch, so the PPO recompute
        # (a single cond forward) replays the rollout velocity exactly — the likelihood-based C2
        # identity (FLUX is guidance-distilled: one effective forward, no CFG at RL time).
        req = make_request(TaskType.T2V,
                           self.student.card.model_id,
                           prompt,
                           diffusion=DiffusionParams(num_steps=self.rollout_steps,
                                                     seed=seed,
                                                     guidance_scale=1.0,
                                                     sde_rollout=self.joint,
                                                     sde_noise_scale=self.sde_noise_scale))
        slots = {
            "text_embeds": np.asarray(emb, dtype="float32"),
            "neg_text_embeds": np.asarray(neg_emb, dtype="float32")
        }
        return rollout_loop(self.student, self.student_loop_id, req, slots=slots, profile=ExecutionProfile.ROLLOUT)

    def _dit_ppo(self, behavior: Any, emb: Any, advantage: float) -> tuple[float, float, float, float]:
        """FlowGRPO PPO over one rollout's SDE trajectory: per-step ratio + KL, then a toy DiT step.
        Returns (mean_ppo_loss, mean_kl, mean_ratio, grad_norm)."""
        ref_dit = self.reference.component("transformer")
        # Pin the log-prob recompute to the SAME SDE kernel the rollout used (C2 kernel-pinning):
        # on a GPU box the rollout's logp and this ratio must come from one kernel or the PPO ratio biases.
        sde_step = self.student.platform.kernels.get(FLOW_SDE_STEP)
        ppo_losses, kls, ratios, gnorms = [], [], [], []
        for rec in behavior or []:
            if "sde_logprob" not in rec:
                continue
            prev, sample = rec["prev"], rec["sample"]
            st, sn = float(rec["sigma_t"]), float(rec["sigma_next"])
            v_cur = self.dit(prev, emb, st)
            _, logp_cur, mean_cur, eff_std = sde_step(prev,
                                                      v_cur,
                                                      st,
                                                      sn,
                                                      prev_sample=sample,
                                                      noise_scale=self.sde_noise_scale)
            v_ref = ref_dit(prev, emb, st)
            _, _, mean_ref, _ = sde_step(prev, v_ref, st, sn, prev_sample=sample, noise_scale=self.sde_noise_scale)
            ratio = float(np.exp(np.clip(logp_cur - float(rec["sde_logprob"]), -10.0, 10.0)))
            clipped = float(np.clip(ratio, 1.0 - self.ppo_clip, 1.0 + self.ppo_clip))
            ppo = max(-advantage * ratio, -advantage * clipped)  # FlowGRPO PPO (per element)
            kl = float(np.mean((mean_cur - mean_ref)**2) / (2.0 * max(eff_std**2, 1e-8)))
            ppo_losses.append(ppo)
            kls.append(kl)
            ratios.append(ratio)
            v_target = _ml_velocity(prev, sample, st, sn, noise_scale=self.sde_noise_scale)  # PG direction
            _, gn = self.dit.mse_grad_step(prev, emb, st, v_target, self.dit_lr * float(np.clip(advantage, -1.0, 1.0)))
            gnorms.append(gn)

        def m(xs: list[Any]) -> float:
            return float(np.mean(xs)) if xs else 0.0

        return m(ppo_losses), m(kls), m(ratios), m(gnorms)

    def managed_train_step(self, batch: dict, iteration: int) -> tuple[dict, dict]:
        self.iteration = iteration
        prompts = batch["prompts"]
        seeds = batch.get("seeds", list(range(len(prompts))))
        neg_base = cached_text_encode(self.student, "")  # shared negative embed
        lm_losses, dit_losses, kls, ratios = [], [], [], []
        lm_gnorms, dit_gnorms, all_reward, all_adv = [], [], [], []

        for pi, prompt in enumerate(prompts):
            base_emb = cached_text_encode(self.student, prompt)  # encoded ONCE for the group
            rng = np.random.default_rng((seeds[pi] + 1) * 7919 + iteration)
            action, logp_action = self.llm.sample_refinement(rng)  # the LM's RL action (sampled)
            refined_emb = self.llm.refined_embed(base_emb, action)

            samples = []  # (latent, behavior, emb, reward, refined?)
            for k in range(self.K):
                refined = k >= self.num_skip  # partial refinement
                emb = refined_emb if refined else base_emb
                act = action if refined else None
                res = self._rollout_sample(prompt, emb, neg_base, seed=seeds[pi] * 1000 + k)
                latent = np.asarray(res.outputs["latents"], dtype="float32")
                reward = self._reward(latent, act)
                samples.append((latent, res.behavior, emb, reward, refined))

            rewards = np.array([s[3] for s in samples], dtype="float64")
            adv = np.clip((rewards - rewards.mean()) / (rewards.std() + 1e-4), -self.adv_clip_max,
                          self.adv_clip_max)  # group-relative advantage
            all_reward.extend(rewards.tolist())
            all_adv.extend(adv.tolist())

            # --- LM token policy gradient: advantage of the REFINED samples (vs original) ------- #
            refined_adv = [adv[k] for k in range(self.K) if samples[k][4]]
            lm_adv = float(np.mean(refined_adv)) if refined_adv else 0.0
            lm_kl = self.llm.kl_to(self.reference.component("llm"))
            gn_lm = self.llm.pg_step(action, lm_adv, self.llm_lr)  # REAL REINFORCE step
            lm_losses.append(-lm_adv * logp_action + self.beta * lm_kl)
            lm_gnorms.append(gn_lm)

            # --- generator FlowGRPO PPO (joint only): one update per refined sample ------------- #
            if self.joint:
                for k in range(self.K):
                    if not samples[k][4]:
                        continue
                    ppo, kl, ratio, gn = self._dit_ppo(samples[k][1], samples[k][2], float(adv[k]))
                    dit_losses.append(ppo + self.beta * kl)
                    kls.append(kl)
                    ratios.append(ratio)
                    dit_gnorms.append(gn)

        # publish both experts' weights — independent versions + cache scopes
        lm_ver = self.llm_sync.apply(self.llm, self.llm, self.student)
        dit_ver = self.dit_sync.apply(self.dit, self.dit, self.student) if self.joint else "frozen"

        p_target = float(self.llm._probs()[self.target_action])  # LM learning signal
        metrics = {
            "lm_pg_loss": float(np.mean(lm_losses)),
            "grad_norm/llm": float(np.mean(lm_gnorms)),
            "reward_mean": float(np.mean(all_reward)),
            "advantage_std": float(np.std(all_adv)),
            "refine_target_prob": p_target,
            "joint": float(self.joint),
            "llm_weights_version": lm_ver,
            "transformer_weights_version": dit_ver,
        }
        loss_map = {"lm_pg_loss": metrics["lm_pg_loss"]}
        if self.joint:
            metrics.update({
                "dit_pg_loss": float(np.mean(dit_losses)) if dit_losses else 0.0,
                "kl_div_loss": float(np.mean(kls)) if kls else 0.0,
                "ppo_ratio_mean": float(np.mean(ratios)) if ratios else 1.0,
                "grad_norm/transformer": float(np.mean(dit_gnorms)) if dit_gnorms else 0.0,
            })
            loss_map["dit_pg_loss"] = metrics["dit_pg_loss"]
        return loss_map, metrics

    def train_step(self, batch: dict, iteration: int) -> tuple[dict, dict]:
        return self.managed_train_step(batch, iteration)


def build_unified_rl(card, *, joint: bool = True, **kw) -> UnifiedRLMethod:
    """Two roles: a trainable student (llm + transformer) and a frozen reference (both experts)."""
    return UnifiedRLMethod(new_instance(card), new_instance(card), joint=joint, **kw)
