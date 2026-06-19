"""DiffusionNFT — likelihood-free RL (repo: train/methods/rl/diffusion_nft).

Likelihood-free (no log-probs to match), so consistency is the C2 behavioral rung: seeded final
sample plus prediction-space identity (``old_deviate`` / ref-MSE), declared on the recipe. Sampling
is from the old (decay-blended behavior) policy, NOT the student — a caveat the WeightSyncPlan must
carry. K samples of one prompt form a homogeneous group; group-relative advantages drive the update;
the shared prompt encodes once via the feature cache.

NFT objective (repo diffusion_nft.py):
  positive   = β·forward + (1−β)·old ;  implicit_neg = (1+β)·old − β·forward
  x0⁺ = xt − t·positive ;  x0⁻ = xt − t·implicit_neg
  policy_loss = r·‖x0⁺−x0‖²/wf⁺/β + (1−r)·‖x0⁻−x0‖²/wf⁻/β   (r = clamp(adv/clip/2+0.5, 0, 1))
  loss = policy_loss + kl_β·‖forward − ref‖² ;  old_deviate = ‖forward − old‖²
"""
from __future__ import annotations

import numpy as np

from v2._enums import ConsistencyLevel
from v2.recipes.common import cached_text_encode
from v2.request import DiffusionParams, TaskType, make_request
from v2.training.rewards import build_multi_reward_scorer
from v2.training.rollout import Trajectory, TrajectoryBuffer
from v2.training.weight_sync import WeightRole, WeightSyncPlan
from v2.training.methods.base import TrainingMethod, new_instance


class DiffusionNFTMethod(TrainingMethod):
    name = "diffusion_nft"
    consistency = ConsistencyLevel.C2  # likelihood-free behavioral identity

    def __init__(self,
                 student_instance,
                 old_instance,
                 reference_instance,
                 *,
                 reward_fn: dict | None = None,
                 num_video_per_prompt: int = 4,
                 rollout_steps: int = 3,
                 beta: float = 0.1,
                 kl_beta: float = 1e-4,
                 adv_clip_max: float = 5.0,
                 num_inner_timesteps: int = 2,
                 lr: float = 0.05,
                 decay_type: int = 1,
                 **kw):
        super().__init__(student_instance, lr=lr, **kw)
        self.old = old_instance  # behavior policy (sampled from!)
        self.reference = reference_instance  # frozen KL anchor
        self.K = num_video_per_prompt
        self.rollout_steps = rollout_steps
        self.beta = beta
        self.kl_beta = kl_beta
        self.adv_clip_max = adv_clip_max
        self.num_inner_timesteps = num_inner_timesteps
        self.decay_type = decay_type
        self.scorer = build_multi_reward_scorer(reward_fn or {"pickscore": 1.0, "clipscore": 1.0})
        self.old_sync = WeightSyncPlan(role=WeightRole.OLD_POLICY)
        # init old + reference from student (NFT inits old from student)
        self.old.component("transformer").copy_from(self.student_dit)
        self.reference.component("transformer").copy_from(self.student_dit)

    def manages_optimization(self) -> bool:  # RL owns its sample→score→train cadence
        return True

    def get_grad_clip_targets(self, iteration: int = 0) -> dict:
        return {"student": self.student_dit}  # NFT clips internally (matches repo)

    @staticmethod
    def _decay(step: int, decay_type: int) -> float:
        if decay_type == 1:
            return min(step * 0.001, 0.5)
        if decay_type == 2:
            return min(max(0, step - 75) * 0.0075, 0.999)
        return 0.0

    def _sample_group(self, prompt: str, base_seed: int) -> TrajectoryBuffer:
        """K rollouts of one prompt FROM THE OLD POLICY via the shared loop (homogeneous group)."""
        buf = TrajectoryBuffer()
        for k in range(self.K):
            seed = base_seed * 1000 + k
            req = make_request(TaskType.T2V,
                               self.old.card.model_id,
                               prompt,
                               diffusion=DiffusionParams(num_steps=self.rollout_steps, seed=seed))
            res = self._rollout(req, instance=self.old)  # behavior policy = OLD
            buf.add(
                Trajectory(request_id=req.request_id,
                           prompt=prompt,
                           seed=seed,
                           latents=np.asarray(res.outputs["latents"], dtype="float32"),
                           behavior=res.behavior,
                           weights_version=self.old.weights_version))
        return buf

    def managed_train_step(self, batch: dict, iteration: int) -> tuple[dict, dict]:
        self.iteration = iteration
        prompts = batch["prompts"]
        seeds = batch.get("seeds", list(range(len(prompts))))
        policy_losses, kl_losses, old_devs, gnorms, all_adv, all_reward = [], [], [], [], [], []

        for pi, prompt in enumerate(prompts):
            group = self._sample_group(prompt, seeds[pi])
            media = [t.latents for t in group]
            scored = self.scorer.score(media, [prompt] * len(group))
            rewards = scored["avg"]
            adv = (rewards - rewards.mean()) / (rewards.std() + 1e-4)  # group-relative
            adv = np.clip(adv, -self.adv_clip_max, self.adv_clip_max)
            all_reward.extend(rewards.tolist())
            all_adv.extend(adv.tolist())

            emb = cached_text_encode(self.student, prompt)  # encoded ONCE for the group
            old_dit, ref_dit = self.old.component("transformer"), self.reference.component("transformer")
            for k, traj in enumerate(group):
                x0 = traj.latents
                r = float(np.clip(adv[k] / self.adv_clip_max / 2.0 + 0.5, 0.0, 1.0))
                rng = np.random.default_rng(traj.seed ^ 0x4F7)
                for _ in range(self.num_inner_timesteps):
                    t = float(rng.uniform(0.05, 0.95))
                    noise = rng.standard_normal(x0.shape).astype("float32")
                    xt = ((1.0 - t) * x0 + t * noise).astype("float32")
                    fwd = self.student_dit(xt, emb, t)
                    old_p = old_dit(xt, emb, t)
                    ref_p = ref_dit(xt, emb, t)
                    positive = self.beta * fwd + (1 - self.beta) * old_p
                    neg = (1.0 + self.beta) * old_p - self.beta * fwd
                    x0p = xt - t * positive
                    x0n = xt - t * neg
                    wfp = max(float(np.mean(np.abs(x0p - x0))), 1e-5)
                    wfn = max(float(np.mean(np.abs(x0n - x0))), 1e-5)
                    pos_loss = float(np.mean((x0p - x0)**2) / wfp)
                    neg_loss = float(np.mean((x0n - x0)**2) / wfn)
                    policy_loss = (r * pos_loss / self.beta + (1 - r) * neg_loss / self.beta) * self.adv_clip_max
                    kl = float(np.mean((fwd - ref_p)**2))
                    policy_losses.append(policy_loss)
                    kl_losses.append(kl)
                    old_devs.append(float(np.mean((fwd - old_p)**2)))
                    # update student toward x0 along the (reward-weighted) positive direction
                    v_target = ((xt - x0) / max(t, 1e-3)).astype("float32")
                    _, gn = self.student_dit.mse_grad_step(xt, emb, t, v_target, self.lr * max(r, 0.05))
                    gnorms.append(gn)

        # update OLD ← decay-blend student (the behavior-policy role); bump version + invalidate caches
        self.old_sync.decay = self._decay(iteration, self.decay_type)
        new_ver = self.old_sync.apply(self.student_dit, self.old.component("transformer"), self.old)

        metrics = {
            "policy_loss": float(np.mean(policy_losses)),
            "kl_div_loss": float(np.mean(kl_losses)),
            "old_deviate": float(np.mean(old_devs)),
            "grad_norm/student": float(np.mean(gnorms)),
            "reward_mean": float(np.mean(all_reward)),
            "advantage_std": float(np.std(all_adv)),
            "old_decay": self.old_sync.decay,
            "old_weights_version": new_ver,
        }
        loss_map = {"policy_loss": metrics["policy_loss"], "kl_div_loss": metrics["kl_div_loss"]}
        return loss_map, metrics

    def train_step(self, batch: dict, iteration: int) -> tuple[dict, dict]:
        return self.managed_train_step(batch, iteration)


def build_diffusion_nft(card, **kw) -> DiffusionNFTMethod:
    return DiffusionNFTMethod(new_instance(card), new_instance(card), new_instance(card), **kw)
