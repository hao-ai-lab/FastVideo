"""JointMultiExpertRL — N-way joint RL over an arbitrary set of experts (design_v3 §10).

The generalization of ``UnifiedRLMethod`` (which is the N=2 special case: one refiner LM + one
generator) to **N refiner LMs + one generator**. It answers the question "can we do joint RL over more
than two experts?" — *yes, with no substrate rewrite*. The proof is structural:

  * the **card** already holds N components and N loops (Qwen-Omni shipped three) — no change;
  * **WeightSyncPlan** is already per-component, so N experts version + cache-scope independently —
    this method just builds a ``dict`` of N plans instead of two;
  * ``get_grad_clip_targets`` already returns a ``dict`` — N entries instead of two;
  * rollout / behavior capture / the consistency ladder are per-loop, agnostic to N.

So the ONLY thing that was "two" was this method's body. Here it is a loop over an expert list. One
reward → one group-relative advantage → N policy-gradient updates (N token-PG refiners + one FlowGRPO
PPO generator). The N refiners' refinements **compose** into the diffusion conditioning, so one reward
genuinely depends on all of them.

Credit assignment (the real research subtlety, not an architectural one):
  * ``credit="shared"`` — one reward, one advantage applied to every expert (faithful to UniRL).
    Works, but each refiner sees the others as noise (shared-reward multi-agent variance).
  * ``credit="per_expert"`` — each refiner gets an advantage from its OWN reward term (a clean
    decomposition). Also supported with no substrate change — it is a reward-shaping choice. The
    generator always uses the total reward's advantage.
"""
from __future__ import annotations

import numpy as np

from ..._enums import ConsistencyLevel, ExecutionProfile
from ...loop.sampler import flow_sde_ml_velocity as _ml_velocity
from ...loop.sampler import flow_sde_step_with_logprob
from ...models.common import cached_text_encode
from ...request import DiffusionParams, TaskType, make_request
from ..rollout import rollout_loop
from ..weight_sync import WeightRole, WeightSyncPlan
from .base import TrainingMethod, new_instance


class JointMultiExpertRL(TrainingMethod):
    name = "joint_multi_rl"
    consistency = ConsistencyLevel.C2
    student_loop_id = "diffusion_denoise"

    def __init__(self, student_instance, reference_instance, *, refiner_ids, generator_id="transformer",
                 target_actions, joint_generator: bool = True, credit: str = "shared",
                 num_samples_per_prompt: int = 4, num_skip_refinement: int = 1, rollout_steps: int = 4,
                 refiner_lr: float = 0.4, dit_lr: float = 0.02, beta: float = 0.01,
                 adv_clip_max: float = 5.0, ppo_clip: float = 1e-2, sde_noise_scale: float = 0.7,
                 action_bonus: float = 2.0, **kw):
        super().__init__(student_instance, lr=dit_lr, **kw)
        self.reference = reference_instance
        self.refiner_ids = list(refiner_ids)
        self.generator_id = generator_id
        self.targets = dict(target_actions)
        self.joint_generator = joint_generator
        self.credit = credit
        self.K = num_samples_per_prompt
        self.num_skip = max(0, min(num_skip_refinement, self.K - 1))
        self.rollout_steps = rollout_steps
        self.refiner_lr, self.dit_lr = refiner_lr, dit_lr
        self.beta, self.adv_clip_max, self.ppo_clip = beta, adv_clip_max, ppo_clip
        self.sde_noise_scale, self.action_bonus = sde_noise_scale, action_bonus
        # one WeightSyncPlan PER trainable expert — independent versions + cache scopes (§7.1, §10)
        self.sync = {rid: WeightSyncPlan(role=WeightRole.STUDENT, components=(rid,))
                     for rid in self.refiner_ids}
        if joint_generator:
            self.sync[generator_id] = WeightSyncPlan(role=WeightRole.STUDENT, components=(generator_id,))
        for rid in self.refiner_ids:                            # frozen reference per refiner
            self.reference.component(rid).copy_from(self.student.component(rid))
        self.reference.component(generator_id).copy_from(self.student.component(generator_id))

    def refiner(self, rid):
        return self.student.component(rid)

    @property
    def dit(self):
        return self.student.component(self.generator_id)

    def manages_optimization(self) -> bool:
        return True

    def get_grad_clip_targets(self, iteration: int = 0) -> dict:
        t = {rid: self.refiner(rid) for rid in self.refiner_ids}
        if self.joint_generator:
            t[self.generator_id] = self.dit
        return t

    def _compose(self, base_emb, actions):
        """Stack every refiner's chosen offset into the diffusion conditioning."""
        emb = base_emb
        for rid in self.refiner_ids:
            emb = self.refiner(rid).refined_embed(emb, actions[rid])
        return emb

    def _reward(self, latent, actions):
        """One reward; returns (total, per_expert). The diffusion signal is shared; each refiner adds
        its own bonus when it picks its target (so per-expert reward isolates that refiner's credit)."""
        diff = float(np.tanh(-np.std(np.asarray(latent, dtype="float64"))))
        per, total = {}, diff
        for rid in self.refiner_ids:
            hit = self.action_bonus if (actions is not None
                                        and actions.get(rid) == self.targets.get(rid)) else 0.0
            total += hit
            per[rid] = diff + hit
        return total, per

    def _rollout_sample(self, prompt, emb, neg_emb, seed):
        req = make_request(TaskType.T2V, self.student.card.model_id, prompt,
                           diffusion=DiffusionParams(num_steps=self.rollout_steps, seed=seed,
                                                     guidance_scale=1.0, sde_rollout=self.joint_generator,
                                                     sde_noise_scale=self.sde_noise_scale))
        slots = {"text_embeds": np.asarray(emb, dtype="float32"),
                 "neg_text_embeds": np.asarray(neg_emb, dtype="float32")}
        return rollout_loop(self.student, self.student_loop_id, req, slots=slots,
                            profile=ExecutionProfile.ROLLOUT)

    def _dit_ppo(self, behavior, emb, advantage):
        ref_dit = self.reference.component(self.generator_id)
        ppo_losses, kls, ratios, gnorms = [], [], [], []
        for rec in behavior or []:
            if "sde_logprob" not in rec:
                continue
            prev, sample = rec["prev"], rec["sample"]
            st, sn = float(rec["sigma_t"]), float(rec["sigma_next"])
            v_cur = self.dit(prev, emb, st)
            _, logp_cur, mean_cur, eff_std = flow_sde_step_with_logprob(
                prev, v_cur, st, sn, prev_sample=sample, noise_scale=self.sde_noise_scale)
            v_ref = ref_dit(prev, emb, st)
            _, _, mean_ref, _ = flow_sde_step_with_logprob(
                prev, v_ref, st, sn, prev_sample=sample, noise_scale=self.sde_noise_scale)
            ratio = float(np.exp(np.clip(logp_cur - float(rec["sde_logprob"]), -10.0, 10.0)))
            clipped = float(np.clip(ratio, 1.0 - self.ppo_clip, 1.0 + self.ppo_clip))
            ppo_losses.append(max(-advantage * ratio, -advantage * clipped))
            kls.append(float(np.mean((mean_cur - mean_ref) ** 2) / (2.0 * max(eff_std ** 2, 1e-8))))
            ratios.append(ratio)
            v_target = _ml_velocity(prev, sample, st, sn, noise_scale=self.sde_noise_scale)  # PG direction
            _, gn = self.dit.mse_grad_step(prev, emb, st, v_target,
                                           self.dit_lr * float(np.clip(advantage, -1.0, 1.0)))
            gnorms.append(gn)
        m = lambda xs: float(np.mean(xs)) if xs else 0.0          # noqa: E731
        return m(ppo_losses), m(kls), m(ratios), m(gnorms)

    @staticmethod
    def _advantage(rewards, clip_max):
        r = np.asarray(rewards, dtype="float64")
        return np.clip((r - r.mean()) / (r.std() + 1e-4), -clip_max, clip_max)

    def managed_train_step(self, batch: dict, iteration: int) -> tuple[dict, dict]:
        self.iteration = iteration
        prompts = batch["prompts"]
        seeds = batch.get("seeds", list(range(len(prompts))))
        neg_base = cached_text_encode(self.student, "")
        refiner_pg = {rid: [] for rid in self.refiner_ids}
        refiner_gn = {rid: [] for rid in self.refiner_ids}
        dit_losses, kls, ratios, dit_gn, all_reward = [], [], [], [], []

        for pi, prompt in enumerate(prompts):
            base = cached_text_encode(self.student, prompt)
            rng = np.random.default_rng((seeds[pi] + 1) * 7919 + iteration)
            actions, logps = {}, {}
            for rid in self.refiner_ids:                          # each refiner samples its action
                actions[rid], logps[rid] = self.refiner(rid).sample_refinement(rng)
            composed = self._compose(base, actions)

            samples = []
            for k in range(self.K):
                refined = k >= self.num_skip
                emb = composed if refined else base
                acts = actions if refined else None
                res = self._rollout_sample(prompt, emb, neg_base, seed=seeds[pi] * 1000 + k)
                total, per = self._reward(np.asarray(res.outputs["latents"], dtype="float32"), acts)
                samples.append((res, emb, refined, total, per))

            totals = [s[3] for s in samples]
            adv_total = self._advantage(totals, self.adv_clip_max)
            all_reward.extend(totals)

            # --- N refiner token-PG updates (one per expert) ------------------------ #
            for rid in self.refiner_ids:
                if self.credit == "per_expert":
                    adv = self._advantage([s[4][rid] for s in samples], self.adv_clip_max)
                else:
                    adv = adv_total
                refined_adv = [adv[k] for k in range(self.K) if samples[k][2]]
                lm_adv = float(np.mean(refined_adv)) if refined_adv else 0.0
                kl = self.refiner(rid).kl_to(self.reference.component(rid))
                refiner_gn[rid].append(self.refiner(rid).pg_step(actions[rid], lm_adv, self.refiner_lr))
                refiner_pg[rid].append(-lm_adv * logps[rid] + self.beta * kl)

            # --- generator FlowGRPO PPO (joint), credited by the total reward ------- #
            if self.joint_generator:
                for k in range(self.K):
                    if not samples[k][2]:
                        continue
                    ppo, kl, ratio, gn = self._dit_ppo(samples[k][0].behavior, samples[k][1],
                                                       float(adv_total[k]))
                    dit_losses.append(ppo + self.beta * kl)
                    kls.append(kl)
                    ratios.append(ratio)
                    dit_gn.append(gn)

        # publish every expert independently (N+? distinct versions + cache scopes)
        versions = {rid: self.sync[rid].apply(self.refiner(rid), self.refiner(rid), self.student)
                    for rid in self.refiner_ids}
        if self.joint_generator:
            versions[self.generator_id] = self.sync[self.generator_id].apply(self.dit, self.dit, self.student)

        m = lambda xs: float(np.mean(xs)) if xs else 0.0          # noqa: E731
        metrics = {
            "reward_mean": m(all_reward),
            "n_experts": float(len(self.refiner_ids) + (1 if self.joint_generator else 0)),
            "credit_mode": self.credit,
        }
        loss_map = {}
        for rid in self.refiner_ids:
            metrics[f"grad_norm/{rid}"] = m(refiner_gn[rid])
            metrics[f"target_prob/{rid}"] = float(self.refiner(rid)._probs()[self.targets[rid]])
            metrics[f"weights_version/{rid}"] = versions[rid]
            loss_map[f"pg_loss/{rid}"] = m(refiner_pg[rid])
        if self.joint_generator:
            metrics[f"grad_norm/{self.generator_id}"] = m(dit_gn)
            metrics["dit_pg_loss"] = m(dit_losses)
            metrics["ppo_ratio_mean"] = m(ratios) if ratios else 1.0
            metrics[f"weights_version/{self.generator_id}"] = versions[self.generator_id]
            loss_map["dit_pg_loss"] = m(dit_losses)
        return loss_map, metrics

    def train_step(self, batch: dict, iteration: int) -> tuple[dict, dict]:
        return self.managed_train_step(batch, iteration)


def build_joint_multi_rl(card, *, refiner_ids, target_actions, **kw) -> JointMultiExpertRL:
    """Two roles: a trainable student (N refiners + generator) and a frozen reference (all experts)."""
    return JointMultiExpertRL(new_instance(card), new_instance(card),
                              refiner_ids=refiner_ids, target_actions=target_actions, **kw)
