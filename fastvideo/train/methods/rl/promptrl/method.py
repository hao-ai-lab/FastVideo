# SPDX-License-Identifier: Apache-2.0
"""PromptRL training method for Wan video generation.

Two milestones share one implementation:

* ``prompt_only`` — train the Qwen prompt-refiner LoRA while Wan stays
  frozen.  Rollout/transition storage is skipped and only the refiner
  optimizer steps.
* ``joint`` — train independent LoRA adapters for the refiner and Wan
  with shared group-relative advantages (detached, so no gradients
  cross between the models).

One original prompt is replicated across the group's ranks; each rank
produces independently seeded candidates for its assigned group slots.
Ranks holding retained-original slots generate from the original
prompt; refined slots use sampled refiner outputs (falling back to the
original when the completion misses ``<answer>...</answer>``).  Videos
are scored against the original prompt by the pluggable reward
provider, rewards are gathered and validated consistently on every
rank, normalized per ``(group, reward_tag)``, then consumed detached by
the refiner GRPO loss (refined slots only) and Wan's clipped
flow-policy loss (all slots).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any
from collections.abc import Iterator

import torch

from fastvideo.logger import init_logger
from fastvideo.train.methods.base import LogScalar, TrainingMethod
from fastvideo.train.methods.rl.common import RLValidationConfig
from fastvideo.train.methods.rl.promptrl.advantages import (
    group_relative_advantages,
    route_generator_advantages,
    route_refiner_advantages,
)
from fastvideo.train.methods.rl.promptrl.config import (
    PromptRLMethodConfig,
    RoleOptimizerConfig,
)
from fastvideo.train.methods.rl.promptrl.distributed import (
    RewardFailure,
    all_gather_objects,
    validate_group_reward_results,
    world_rank,
    world_size,
)
from fastvideo.train.methods.rl.promptrl.prompts import (
    GroupAssignment,
    PromptDataset,
    PromptRecord,
    group_assignments,
    parse_answer_tag,
    render_refinement_prompt,
)
from fastvideo.train.methods.rl.promptrl.rewards import (
    HttpRewardProvider,
    RewardProvider,
    RewardResult,
    RewardSample,
)
from fastvideo.train.methods.rl.promptrl.sde import transition_kl_to_reference
from fastvideo.train.methods.rl.promptrl.video_io import encode_video_bytes
from fastvideo.train.roles.base import TrainRoleBase
from fastvideo.train.utils.optimizer import clip_grad_norm_if_needed

logger = init_logger(__name__)

#: Distributed validation of gathered rewards lives in distributed.py;
#: imported here for re-export to tests.


@dataclass(slots=True)
class _SlotState:
    """Per-slot rollout state kept on the owning rank."""

    slot: int
    kind: str
    refiner_participation: bool
    instruction: str
    completion: str
    refined_prompt: str
    generation_prompt: str
    format_valid: bool
    format_reward: float
    reward_sample: RewardSample
    rollout: Any  # WanRolloutResult
    batch: Any  # TrainingBatch
    reward_result: RewardResult | None = None

    def to_payload(self) -> dict[str, Any]:
        """Picklable per-slot summary exchanged across ranks."""
        return {
            "slot": self.slot,
            "kind": self.kind,
            "reward_result": self.reward_result,
            "format_reward": self.format_reward,
            "format_valid": self.format_valid,
            "refined_prompt": self.refined_prompt,
            "generation_prompt": self.generation_prompt,
        }


class _TickDataloader:
    """Infinite placeholder dataloader.

    Prompt sampling is a deterministic function of the iteration (see
    ``_sample_prompt_record``), so resume continues the exact prompt
    stream without dataloader state; the trainer only needs an iterable
    to pace steps.
    """

    def __iter__(self) -> Iterator[dict[str, Any]]:
        while True:
            yield {}


def _has_lora_layers(module: torch.nn.Module) -> bool:
    from fastvideo.layers.lora.linear import BaseLayerWithLoRA

    return any(isinstance(m, BaseLayerWithLoRA) for m in module.modules())


class PromptRLMethod(TrainingMethod):
    """PromptRL: refiner GRPO + Wan clipped flow-policy optimization."""

    def __init__(
        self,
        *,
        cfg: Any,
        role_models: dict[str, TrainRoleBase],
    ) -> None:
        super().__init__(cfg=cfg, role_models=role_models)
        if "refiner" not in role_models:
            raise ValueError("PromptRLMethod requires role 'refiner'")

        self.refiner = role_models["refiner"]
        self.config = PromptRLMethodConfig.from_mapping(self.method_config)
        if not self.student._trainable and self.config.mode == "joint":
            raise ValueError("PromptRL joint mode requires a trainable student")
        self._validation_config = RLValidationConfig.from_mapping(
            self.method_config.get("validation"))
        self._assignments: list[GroupAssignment] = group_assignments(
            group_size=self.config.group_size,
            retained_originals=self.config.retained_originals,
        )
        if world_size() > 1 and self.config.group_size % world_size() != 0:
            raise ValueError(f"method.group_size ({self.config.group_size}) must be "
                             f"divisible by world size ({world_size()})")

        self.student.init_preprocessors(self.training_config)
        self.refiner.init_preprocessors(self.training_config)

        self._prompt_dataset: PromptDataset | None = None
        if self.config.data.data_path:
            self._prompt_dataset = PromptDataset.load(
                self.config.data.data_path,
                prompt_key=self.config.data.prompt_key,
                id_key=self.config.data.id_key,
                reward_tag_key=self.config.data.reward_tag_key,
            )

        self._reward_provider: RewardProvider = HttpRewardProvider(
            endpoint_url=self.config.reward.endpoint_url,
            timeout_sec=self.config.reward.timeout_sec,
            retries=self.config.reward.retries,
            score_path=self.config.reward.score_path,
            health_path=self.config.reward.health_path,
        )

        self._refiner_optimizer: torch.optim.Optimizer | None = None
        self._refiner_scheduler: Any = None
        self._generator_optimizer: torch.optim.Optimizer | None = None
        self._generator_scheduler: Any = None
        self._init_optimizers()

        # The trainer paces steps through this iterable; prompt sampling
        # itself is deterministic by iteration.
        if getattr(self.student, "dataloader", None) is None:
            self.student.dataloader = _TickDataloader()

    # ------------------------------------------------------------------
    # Wiring helpers
    # ------------------------------------------------------------------

    def set_reward_provider(self, provider: RewardProvider) -> None:
        """Inject a reward provider (tests / alternative scorers)."""
        self._reward_provider = provider

    def set_prompt_dataset(self, dataset: PromptDataset) -> None:
        """Inject a prompt dataset (tests / programmatic use)."""
        self._prompt_dataset = dataset

    def _init_optimizers(self) -> None:
        self._refiner_optimizer, self._refiner_scheduler = self._build_role_optimizer(
            self.refiner.trainable_parameters(),
            self.config.refiner_optimizer,
            role="refiner",
        )
        self._generator_optimizer, self._generator_scheduler = self._build_role_optimizer(
            self.student.trainable_parameters(),
            self.config.generator_optimizer,
            role="student",
        )

    def _build_role_optimizer(
        self,
        params: list[torch.nn.Parameter],
        cfg: RoleOptimizerConfig,
        *,
        role: str,
    ) -> tuple[torch.optim.Optimizer, Any]:
        if not params:
            raise ValueError(f"PromptRL role {role!r} has no trainable parameters")
        from fastvideo.train.utils.training_config import OptimizerConfig
        from fastvideo.train.utils.optimizer import build_optimizer_and_scheduler

        shim = OptimizerConfig(
            learning_rate=cfg.learning_rate,
            betas=cfg.betas,
            weight_decay=cfg.weight_decay,
            lr_scheduler=cfg.lr_scheduler,
            lr_warmup_steps=cfg.lr_warmup_steps,
        )
        return build_optimizer_and_scheduler(
            params=params,
            optimizer_config=shim,
            loop_config=self.training_config.loop,
            learning_rate=cfg.learning_rate,
            betas=cfg.betas,
            scheduler_name=cfg.lr_scheduler,
        )


    # ------------------------------------------------------------------
    # TrainingMethod plumbing
    # ------------------------------------------------------------------

    @property
    def _optimizer_dict(self) -> dict[str, torch.optim.Optimizer]:
        return {
            "student": self._generator_optimizer,
            "refiner": self._refiner_optimizer,
        }

    @property
    def _lr_scheduler_dict(self) -> dict[str, Any]:
        return {
            "student": self._generator_scheduler,
            "refiner": self._refiner_scheduler,
        }

    def get_optimizers(self, iteration: int) -> list[torch.optim.Optimizer]:
        optimizers = [self._refiner_optimizer]
        if self.config.mode == "joint":
            optimizers.append(self._generator_optimizer)
        return [opt for opt in optimizers if opt is not None]

    def get_lr_schedulers(self, iteration: int) -> list[Any]:
        schedulers = [self._refiner_scheduler]
        if self.config.mode == "joint":
            schedulers.append(self._generator_scheduler)
        return [sched for sched in schedulers if sched is not None]

    def on_train_start(self) -> None:
        """Initialize both the diffusion student and prompt refiner roles."""
        super().on_train_start()
        self.refiner.on_train_start()

    def manages_optimization(self) -> bool:
        return True

    def single_train_step(self, batch: dict[str, Any], iteration: int):
        raise NotImplementedError("PromptRLMethod uses managed_train_step")

    def get_grad_clip_targets(self, iteration: int) -> dict[str, torch.nn.Module]:
        targets: dict[str, torch.nn.Module] = {}
        refiner_modules = self.refiner.checkpoint_modules()
        if "model" in refiner_modules:
            targets["refiner"] = refiner_modules["model"]
        transformer = getattr(self.student, "transformer", None)
        if isinstance(transformer, torch.nn.Module):
            targets["student"] = transformer
        return targets

    # ------------------------------------------------------------------
    # Prompt + seed scheduling
    # ------------------------------------------------------------------

    def _sample_prompt_record(self, iteration: int) -> PromptRecord:
        """Deterministic per-iteration prompt shared by the whole group."""
        if self._prompt_dataset is None:
            raise RuntimeError("PromptRLMethod has no prompt dataset; set "
                               "method.data.data_path or call set_prompt_dataset()")
        from fastvideo.train.methods.rl.common import distributed_k_repeat_indices

        sample = distributed_k_repeat_indices(
            dataset_length=len(self._prompt_dataset),
            batch_size=1,
            repeats_per_prompt=self.config.group_size,
            world_size=self.config.group_size,
            rank=0,
            seed=int(self.training_config.data.seed) + int(iteration),
        )
        return self._prompt_dataset[sample.local_indices[0]]

    def _slot_seed(self, iteration: int, slot: int) -> int:
        """Independently seeded candidate stream for one group slot."""
        base = int(self.training_config.data.seed)
        return base + int(iteration) * self.config.group_size + int(slot)

    def _local_slots(self) -> list[int]:
        """Group slots owned by this rank (round-robin across ranks)."""
        rank = world_rank()
        world = world_size()
        return [slot for slot in range(self.config.group_size) if slot % world == rank]



    # ------------------------------------------------------------------
    # Rollout
    # ------------------------------------------------------------------

    def _rollout_slot(
        self,
        record: PromptRecord,
        slot: int,
        iteration: int,
        group_id: str,
    ) -> _SlotState:
        """Refine (or retain) the prompt, generate + upload one video."""
        cfg = self.config
        assignment = self._assignments[slot]
        instruction = render_refinement_prompt(
            record.prompt, template_version=cfg.refiner.template_version)

        # Every slot samples a refiner completion so all ranks execute
        # compatible refiner paths; retained originals discard it for
        # generation and receive zero refiner advantage.
        completion = self.refiner.generate_refinements(
            [instruction],
            max_new_tokens=cfg.refiner.max_new_tokens,
            temperature=cfg.refiner.temperature,
            top_p=cfg.refiner.top_p,
            seed=self._slot_seed(iteration, slot),
        )[0]
        parsed = parse_answer_tag(completion)

        refined_prompt = ""
        if assignment.kind == "original":
            generation_prompt = record.prompt
            format_reward = 1.0
        elif parsed.format_valid:
            generation_prompt = parsed.refined_prompt
            refined_prompt = parsed.refined_prompt
            format_reward = 1.0
        else:
            # Malformed: fall back to the original prompt for video
            # generation; the completion is preserved for the LM loss.
            generation_prompt = record.prompt
            format_reward = 0.0

        embeds, mask = self.student.encode_prompts([generation_prompt])
        batch = self.student.build_rollout_batch(
            embeds, mask, latent_shape=self.student.latent_shape(1))
        joint = cfg.mode == "joint"
        rollout = self.student.rollout(
            batch,
            batch_size=1,
            generator=self.cuda_generator,
            num_steps=cfg.rollout.num_steps,
            sde_steps=cfg.rollout.sde_steps if joint else 0,
            noise_scale=cfg.rollout.noise_scale,
            flow_shift=cfg.rollout.flow_shift,
            store_transitions=joint,
        )
        media = self.student.decode_latents(rollout.latents)
        video_bytes = encode_video_bytes(media, fps=cfg.reward.fps)
        reward_sample = RewardSample(
            group_id=group_id,
            request_id=group_id,
            sample_id=f"slot-{slot}",
            expected_group_size=cfg.group_size,
            original_prompt=record.prompt,  # always score the original
            reward_tag=record.reward_tag,
            fps=cfg.reward.fps,
            video_bytes=video_bytes,
        )
        return _SlotState(
            slot=slot,
            kind=assignment.kind,
            refiner_participation=assignment.refiner_participation,
            instruction=instruction,
            completion=completion,
            refined_prompt=refined_prompt,
            generation_prompt=generation_prompt,
            format_valid=parsed.format_valid,
            format_reward=format_reward,
            reward_sample=reward_sample,
            rollout=rollout,
            batch=batch,
        )

    # ------------------------------------------------------------------
    # Managed training step
    # ------------------------------------------------------------------

    def managed_train_step(
        self,
        data_stream: Any,
        iteration: int,
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, LogScalar]]:
        del data_stream
        cfg = self.config
        t0 = time.perf_counter()
        record = self._sample_prompt_record(iteration)
        group_id = f"step-{int(iteration)}:{record.sample_id}"

        # --- 1. Rollout: refiner + Wan per local slot ---
        rollout_t0 = time.perf_counter()
        slot_states: dict[int, _SlotState] = {}
        for slot in self._local_slots():
            slot_states[slot] = self._rollout_slot(record, slot, iteration, group_id)
        rollout_sec = time.perf_counter() - rollout_t0

        # --- 2. Score + gather/validate consistently on every rank ---
        reward_t0 = time.perf_counter()
        samples = [slot_states[slot].reward_sample for slot in sorted(slot_states)]
        local_reward_error: RewardFailure | None = None
        try:
            local_results = list(self._reward_provider.score(samples))
        except Exception as exc:  # noqa: BLE001 - propagated consistently after gather
            local_reward_error = RewardFailure(
                rank=world_rank(),
                error_type=type(exc).__name__,
                message=str(exc),
            )
            local_payloads = [local_reward_error]
        else:
            for slot, result in zip(sorted(slot_states), local_results, strict=False):
                slot_states[slot].reward_result = result
            local_payloads = [slot_states[slot].to_payload() for slot in sorted(slot_states)]
        gathered = all_gather_objects(local_payloads)
        payloads = [payload for rank_payloads in gathered for payload in rank_payloads]
        reward_failures = [p for p in payloads if isinstance(p, RewardFailure)]
        if reward_failures:
            validate_group_reward_results(
                reward_failures,
                group_id=group_id,
                expected_group_size=cfg.group_size,
            )
        payloads.sort(key=lambda p: p["slot"])
        results = [p["reward_result"] for p in payloads]
        validate_group_reward_results(
            results,
            group_id=group_id,
            expected_group_size=cfg.group_size,
        )
        reward_sec = time.perf_counter() - reward_t0

        # --- 3. Composite rewards + detached group-relative advantages ---
        format_rewards = torch.tensor([p["format_reward"] for p in payloads],
                                      dtype=torch.float32)
        video_scores = torch.tensor([r.score for r in results], dtype=torch.float32)
        composite = (cfg.format_reward_weight * format_rewards +
                     cfg.video_reward_weight * video_scores)
        group_keys = [f"{group_id}|{record.reward_tag}"] * len(payloads)
        advantages = group_relative_advantages(composite, group_keys)
        refiner_advantages = route_refiner_advantages(
            advantages, [self._assignments[p["slot"]].refiner_participation for p in payloads])
        wan_advantages = route_generator_advantages(advantages)

        # --- 4. Losses (advantages detached: no cross-model gradients) ---
        backward_t0 = time.perf_counter()
        refiner_loss, refiner_metrics = self._refiner_loss(slot_states, refiner_advantages)
        refiner_loss.backward()

        wan_loss = torch.zeros((), device=format_rewards.device)
        wan_metrics: dict[str, LogScalar] = {}
        if cfg.mode == "joint":
            wan_loss, wan_metrics = self._wan_loss(slot_states, wan_advantages)
        backward_sec = time.perf_counter() - backward_t0

        # --- 5. Clip + per-role optimizer steps ---
        grad_norms = self._clip_grads(iteration)
        self.optimizers_schedulers_step(iteration)
        self.optimizers_zero_grad(iteration)

        # --- 6. Metrics ---
        metrics = self._step_metrics(
            payloads=payloads,
            video_scores=video_scores,
            format_rewards=format_rewards,
            advantages=advantages,
            refiner_metrics=refiner_metrics,
            wan_metrics=wan_metrics,
            grad_norms=grad_norms,
            rollout_sec=rollout_sec,
            reward_sec=reward_sec,
            backward_sec=backward_sec,
            step_sec=time.perf_counter() - t0,
        )
        total_loss = refiner_loss.detach() + wan_loss.detach()
        return {"total_loss": total_loss}, {}, metrics


    # ------------------------------------------------------------------
    # Refiner GRPO loss
    # ------------------------------------------------------------------

    def _refiner_loss(
        self,
        slot_states: dict[int, _SlotState],
        refiner_advantages: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float | None]]:
        """Per-token GRPO surrogate + k3 KL to the frozen base refiner.

        Only refined slots carry nonzero advantage; retained-original
        slots run the same forward/backward with participation weight 0
        so replicated/DDP gradient collectives stay aligned.
        """
        cfg = self.config
        device = self.refiner.device
        weighted_losses: list[torch.Tensor] = []
        weighted_tokens: list[torch.Tensor] = []
        kl_means: list[torch.Tensor] = []
        token_counts: list[float] = []

        for slot in sorted(slot_states):
            state = slot_states[slot]
            participation = 1.0 if state.refiner_participation else 0.0
            advantage = refiner_advantages[slot].to(device)

            new_tok, mask = self.refiner.token_logprobs(
                [state.instruction], [state.completion], requires_grad=True)
            old_tok = new_tok.detach()
            ref_tok, _ = self.refiner.token_logprobs(
                [state.instruction], [state.completion], use_adapter=False)

            ratio = torch.exp(new_tok - old_tok)
            clipped = torch.clamp(ratio, 1.0 - cfg.ppo_clip, 1.0 + cfg.ppo_clip)
            surrogate = torch.minimum(ratio * advantage, clipped * advantage)
            # k3 KL estimator to the adapter-disabled reference policy.
            ref_gap = ref_tok - new_tok
            kl_tok = torch.exp(ref_gap) - ref_gap - 1.0
            token_loss = (-surrogate + cfg.refiner_kl_beta * kl_tok) * mask

            weight = torch.tensor(participation, device=device, dtype=token_loss.dtype)
            weighted_losses.append((token_loss * weight).sum())
            weighted_tokens.append((mask * weight).sum())
            kl_means.append(((kl_tok * mask).sum() / mask.sum().clamp_min(1.0)) * weight)
            token_counts.append(float(mask.sum().item()) * participation)

        total_tokens = torch.stack(weighted_tokens).sum().clamp_min(1.0)
        loss = torch.stack(weighted_losses).sum() / total_tokens
        positive_counts = [c for c in token_counts if c > 0]
        metrics: dict[str, float | None] = {
            "loss": float(loss.detach().float().item()),
            "kl": float(torch.stack(kl_means).sum().detach().item()),
            "completion_tokens": (sum(positive_counts) / len(positive_counts)
                                  if positive_counts else None),
        }
        return loss, metrics


    # ------------------------------------------------------------------
    # Wan clipped flow-policy loss (joint mode)
    # ------------------------------------------------------------------

    def _wan_loss(
        self,
        slot_states: dict[int, _SlotState],
        wan_advantages: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, LogScalar]]:
        """PPO-clipped transition loss + Gaussian KL to frozen-base Wan.

        Consumes *all* slots' advantages.  Transition probabilities are
        recomputed in microbatches; each microbatch backward
        accumulates into the Wan LoRA only (reference passes run under
        the adapter-disabled context with no grad).
        """
        cfg = self.config
        device = self.student.device
        microbatch = max(1, int(cfg.rollout.loss_microbatch_size))
        total_slots = max(1, len(slot_states))
        total_loss = torch.zeros((), device=device)
        policy_losses: list[float] = []
        kl_values: list[float] = []
        ratios: list[float] = []
        clip_fractions: list[float] = []

        for slot in sorted(slot_states):
            state = slot_states[slot]
            advantage = wan_advantages[slot].to(device)
            transitions = state.rollout.transitions
            if not transitions:
                continue
            sigma_max = (state.rollout.sigmas[1] if len(state.rollout.sigmas) > 1 else 1.0)
            slot_loss_scale = 1.0 / (total_slots * len(transitions))

            pending: list[tuple[torch.Tensor, Any]] = []

            def backward_pending() -> None:
                nonlocal pending, total_loss
                backward = getattr(self.student, "backward", None)
                for pending_loss, backward_ctx in pending:
                    if callable(backward):
                        # Wan activation checkpointing recomputes attention
                        # during backward. Route through the model's backward
                        # hook so its timestep/attention ForwardContext is live
                        # for every independently recomputed transition graph.
                        backward(
                            pending_loss,
                            backward_ctx,
                            grad_accum_rounds=1,
                        )
                    else:
                        # Model-agnostic fallback for lightweight role plugins.
                        pending_loss.backward()
                    total_loss = total_loss + pending_loss.detach()
                pending = []

            for transition in transitions:
                new_logp, new_mean = self.student.transition_logprobs(
                    [transition],
                    state.batch,
                    noise_scale=cfg.rollout.noise_scale,
                    sigma_max=sigma_max,
                    use_adapter=True,
                    requires_grad=True,
                )[0]
                _, ref_mean = self.student.transition_logprobs(
                    [transition],
                    state.batch,
                    noise_scale=cfg.rollout.noise_scale,
                    sigma_max=sigma_max,
                    use_adapter=False,
                    requires_grad=False,
                )[0]
                old_logp = transition.old_log_prob.to(new_logp.device)
                ratio = torch.exp(new_logp - old_logp)
                clipped = torch.clamp(ratio, 1.0 - cfg.ppo_clip, 1.0 + cfg.ppo_clip)
                surrogate = torch.minimum(ratio * advantage, clipped * advantage)
                kl = transition_kl_to_reference(
                    new_mean,
                    ref_mean,
                    sigma=transition.sigma,
                    sigma_next=transition.sigma_next,
                    noise_scale=cfg.rollout.noise_scale,
                    sigma_max=sigma_max,
                )
                transition_loss = (-surrogate.mean() + cfg.student_kl_beta * kl.mean())
                backward_timestep = torch.full(
                    (transition.sample.shape[0], ),
                    float(transition.timestep),
                    device=device,
                )
                backward_ctx = (
                    backward_timestep,
                    getattr(state.batch, "attn_metadata", None),
                )
                pending.append(
                    (transition_loss * slot_loss_scale, backward_ctx)
                )

                policy_losses.append(float((-surrogate.mean()).detach().item()))
                kl_values.append(float(kl.mean().detach().item()))
                ratios.append(float(ratio.mean().detach().item()))
                clip_fractions.append(
                    float(((ratio - 1.0).abs() > cfg.ppo_clip).float().mean().item()))

                if len(pending) >= microbatch:
                    backward_pending()
            if pending:
                backward_pending()

        metrics: dict[str, LogScalar] = {
            "loss": sum(policy_losses) / max(1, len(policy_losses)),
            "kl": sum(kl_values) / max(1, len(kl_values)),
            "ratio": sum(ratios) / max(1, len(ratios)),
            "clip_fraction": sum(clip_fractions) / max(1, len(clip_fractions)),
        }
        return total_loss, metrics


    # ------------------------------------------------------------------
    # Gradient clipping + metrics
    # ------------------------------------------------------------------

    def _clip_grads(self, iteration: int) -> dict[str, float]:
        norms: dict[str, float] = {}
        targets = self.get_grad_clip_targets(iteration)
        if "refiner" in targets:
            norms["refiner"] = clip_grad_norm_if_needed(
                targets["refiner"], self.config.refiner_optimizer.max_grad_norm)
        if self.config.mode == "joint" and "student" in targets:
            norms["student"] = clip_grad_norm_if_needed(
                targets["student"], self.config.generator_optimizer.max_grad_norm)
        return norms

    @staticmethod
    def _peak_memory_mb() -> float:
        if torch.cuda.is_available():
            return float(torch.cuda.max_memory_allocated()) / 1e6
        return 0.0

    def _distributed_mean(self, value: float | None) -> float | None:
        """Mean of an optional scalar across ranks (None-aware)."""
        if world_size() <= 1:
            return value
        gathered = all_gather_objects(value)
        present = [float(v) for v in gathered if v is not None]
        if not present:
            return None
        return sum(present) / len(present)

    def _step_metrics(
        self,
        *,
        payloads: list[dict[str, Any]],
        video_scores: torch.Tensor,
        format_rewards: torch.Tensor,
        advantages: torch.Tensor,
        refiner_metrics: dict[str, float | None],
        wan_metrics: dict[str, LogScalar],
        grad_norms: dict[str, float],
        rollout_sec: float,
        reward_sec: float,
        backward_sec: float,
        step_sec: float,
    ) -> dict[str, LogScalar]:
        metrics: dict[str, LogScalar] = {}

        # Group reward statistics.
        composite = (self.config.format_reward_weight * format_rewards +
                     self.config.video_reward_weight * video_scores)
        metrics["reward/group_mean"] = float(composite.mean())
        metrics["reward/group_std"] = float(composite.std(unbiased=False))
        metrics["reward/group_min"] = float(composite.min())
        metrics["reward/group_max"] = float(composite.max())
        metrics["reward/video_mean"] = float(video_scores.mean())
        metrics["reward/format_mean"] = float(format_rewards.mean())
        metrics["reward/advantage_abs_mean"] = float(advantages.abs().mean())

        # VideoScore2 component details.
        component_names: set[str] = set()
        for payload in payloads:
            result = payload.get("reward_result")
            if result is not None:
                component_names.update(result.details)
        for name in sorted(component_names):
            values = [
                payload["reward_result"].details[name] for payload in payloads
                if payload.get("reward_result") is not None
                and name in payload["reward_result"].details
            ]
            if values:
                metrics[f"reward/{name}"] = sum(values) / len(values)

        # Refinement statistics.
        refined = [p for p in payloads if p["kind"] == "refined"]
        originals = [p for p in payloads if p["kind"] == "original"]
        if refined:
            metrics["refine/valid_rate"] = (sum(1.0 for p in refined if p["format_valid"]) /
                                            len(refined))
        if refined and originals:
            refined_mean = sum(p["reward_result"].score for p in refined) / len(refined)
            original_mean = sum(p["reward_result"].score for p in originals) / len(originals)
            metrics["refine/refined_vs_original_gap"] = refined_mean - original_mean

        # Language-model metrics (produced only on refined-slot ranks;
        # average across the group so rank-0 logging stays meaningful).
        lm_loss = self._distributed_mean(refiner_metrics.get("loss"))
        if lm_loss is not None:
            metrics["lm/loss"] = lm_loss
        lm_kl = self._distributed_mean(refiner_metrics.get("kl"))
        if lm_kl is not None:
            metrics["lm/kl"] = lm_kl
        completion_tokens = self._distributed_mean(refiner_metrics.get("completion_tokens"))
        if completion_tokens is not None:
            metrics["lm/completion_tokens"] = completion_tokens

        # Wan flow-policy metrics.
        for key, value in wan_metrics.items():
            metrics[f"wan/{key}"] = value

        for name, norm in grad_norms.items():
            metrics[f"grad_norm/{name}"] = norm

        metrics["latency/rollout_sec"] = rollout_sec
        metrics["latency/reward_sec"] = reward_sec
        metrics["latency/backward_sec"] = backward_sec
        metrics["latency/step_sec"] = step_sec
        metrics["mem/peak_gpu_mb"] = self._peak_memory_mb()
        return metrics


    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def on_validation_begin(self, iteration: int = 0) -> dict[str, LogScalar]:
        """Held-out refined-vs-original evaluation.

        Shards validation prompts across ranks; each sharded prompt is
        generated once from the original prompt and once from its
        refinement, then scored against the original prompt.  Videos or
        prompt samples are only surfaced on group leaders by the
        tracker/callback layer.
        """
        config = self._validation_config
        if config.every_steps <= 0 or iteration % config.every_steps != 0:
            return {}
        if self._prompt_dataset is None:
            return {}

        from fastvideo.train.methods.rl.common import validation_shard_indices

        rank = world_rank()
        world = world_size()
        shard = validation_shard_indices(
            min(config.num_prompts, len(self._prompt_dataset)),
            rank=rank,
            world_size=world,
        )
        gaps: list[float] = []
        original_scores: list[float] = []
        refined_scores: list[float] = []
        valid_flags: list[float] = []
        for prompt_idx, is_real in shard:
            if not is_real:
                continue
            record = self._prompt_dataset[prompt_idx]
            instruction = render_refinement_prompt(
                record.prompt, template_version=self.config.refiner.template_version)
            completion = self.refiner.generate_refinements(
                [instruction],
                max_new_tokens=self.config.refiner.max_new_tokens,
                temperature=self.config.refiner.temperature,
                top_p=self.config.refiner.top_p,
                seed=self._slot_seed(iteration, prompt_idx),
            )[0]
            parsed = parse_answer_tag(completion)
            refined_prompt = parsed.refined_prompt if parsed.format_valid else record.prompt
            valid_flags.append(1.0 if parsed.format_valid else 0.0)

            scores: dict[str, float] = {}
            for tag, text in (("original", record.prompt), ("refined", refined_prompt)):
                embeds, mask = self.student.encode_prompts([text])
                batch = self.student.build_rollout_batch(
                    embeds, mask, latent_shape=self.student.latent_shape(1))
                rollout = self.student.rollout(
                    batch,
                    batch_size=1,
                    generator=self.cuda_generator,
                    num_steps=config.num_steps,
                    sde_steps=0,
                    noise_scale=self.config.rollout.noise_scale,
                    flow_shift=self.config.rollout.flow_shift,
                    store_transitions=False,
                )
                media = self.student.decode_latents(rollout.latents)
                sample = RewardSample(
                    group_id=f"val-{iteration}:{record.sample_id}:{tag}",
                    request_id=f"val-{iteration}:{record.sample_id}:{tag}:{rank}",
                    sample_id=f"rank-{rank}",
                    expected_group_size=1,
                    original_prompt=record.prompt,
                    reward_tag=record.reward_tag,
                    fps=self.config.reward.fps,
                    video_bytes=encode_video_bytes(media, fps=self.config.reward.fps),
                )
                scores[tag] = float(self._reward_provider.score([sample])[0].score)
            original_scores.append(scores["original"])
            refined_scores.append(scores["refined"])
            gaps.append(scores["refined"] - scores["original"])

        metrics: dict[str, LogScalar] = {}
        for key, value in (
            ("video_original", sum(original_scores) / max(1, len(original_scores))),
            ("video_refined", sum(refined_scores) / max(1, len(refined_scores))),
            ("refined_vs_original_gap", sum(gaps) / max(1, len(gaps))),
            ("refine_valid_rate", sum(valid_flags) / max(1, len(valid_flags))),
        ):
            if shard:
                mean = self._distributed_mean(float(value))
                if mean is not None:
                    metrics[f"validation/{key}"] = mean
        return metrics


    # ------------------------------------------------------------------
    # Bundle export
    # ------------------------------------------------------------------

    def export_bundle(self, output_dir: str) -> dict[str, str]:
        """Export a PromptRL inference bundle from the live roles.

        The bundle carries the refiner PEFT adapter, the Wan LoRA (when
        the student role has one), the prompt template/version, refiner
        sampling configuration, base model identifiers, and the current
        FastVideo version.
        """
        import fastvideo
        from fastvideo.train.methods.rl.promptrl.bundle import (
            BundleManifest,
            export_promptrl_bundle,
        )

        refiner_lora = getattr(self.refiner, "_lora_config", None)
        student_lora = getattr(self.student, "_lora_config", None)
        manifest = BundleManifest(
            base_refiner_model=str(getattr(self.refiner, "_init_from", "")),
            base_generator_model=str(self.training_config.model_path),
            fastvideo_version=str(getattr(fastvideo, "__version__", "unknown")),
            refiner_lora=({
                "rank": refiner_lora.rank,
                "alpha": refiner_lora.alpha,
                "target_modules": refiner_lora.target_modules,
            } if refiner_lora is not None else {}),
            generator_lora=({
                "rank": student_lora.rank,
                "alpha": student_lora.alpha,
                "target_modules": student_lora.target_modules,
            } if student_lora is not None else {}),
            prompt_template_version=self.config.refiner.template_version,
            refiner_sampling={
                "max_new_tokens": self.config.refiner.max_new_tokens,
                "temperature": self.config.refiner.temperature,
                "top_p": self.config.refiner.top_p,
            },
            mode=self.config.mode,
        )
        transformer = getattr(self.student, "transformer", None)
        export_generator = (self.student._trainable and isinstance(transformer, torch.nn.Module)
                            and _has_lora_layers(transformer))
        if self.student._trainable and not export_generator:
            logger.info("Student transformer has no LoRA layers; exporting refiner-only bundle")
        return export_promptrl_bundle(
            output_dir,
            manifest=manifest,
            refiner_role=self.refiner,
            refiner_tokenizer=getattr(self.refiner, "tokenizer", None),
            generator_transformer=transformer if export_generator else None,
        )
