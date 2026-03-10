# SPDX-License-Identifier: Apache-2.0
"""Video GRPO / PPO training method for diffusion models.

One trainer step == one GenRL outer epoch:
  1. Sample videos and compute rewards.
  2. Compute advantages (per-prompt normalization).
  3. PPO training across inner epochs.

The method handles backward / optimizer internally and
returns all stats.  The trainer's outer loop just calls
``single_train_step`` once per step.
"""

from __future__ import annotations

import copy
from collections import defaultdict
from concurrent import futures
from typing import Any

import numpy as np
import torch
import torch.distributed as dist

from fastvideo.distributed import get_world_group
from fastvideo.forward_context import set_forward_context
from fastvideo.logger import init_logger
from fastvideo.train.methods.base import (
    LogScalar,
    TrainingMethod,
)
from fastvideo.train.methods.rl.advantages import (
    compute_advantages,
)
from fastvideo.train.methods.rl.data import (
    build_prompt_dataloaders,
)
from fastvideo.train.methods.rl.diffusion import (
    compute_log_prob,
)
from fastvideo.train.methods.rl.embeddings import (
    compute_text_embeddings,
)
from fastvideo.train.methods.rl.rewards import (
    multi_score,
    reward_models_on_device,
)
from fastvideo.train.methods.rl.sampling import (
    sample_epoch,
)
from fastvideo.train.methods.rl.stat_tracking import (
    PerPromptStatTracker,
)
from fastvideo.train.models.base import ModelBase
from fastvideo.train.utils.optimizer import (
    build_optimizer_and_scheduler,
    clip_grad_norm_if_needed,
)

logger = init_logger(__name__)

ADVANTAGE_EPSILON = 1e-6
SEED_EPOCH_STRIDE = 10_000


def _gather_tensor(
    tensor: torch.Tensor,
    world_size: int,
) -> torch.Tensor:
    """Gather tensor from all ranks and concatenate."""
    if world_size <= 1:
        return tensor
    gathered = [torch.empty_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered, tensor)
    return torch.cat(gathered, dim=0)


class GenRLMethod(TrainingMethod):
    """Video GRPO / PPO method for diffusion models.

    Handles the full RL training loop internally
    within ``single_train_step``.
    """

    def __init__(
        self,
        *,
        cfg: Any,
        role_models: dict[str, ModelBase],
    ) -> None:
        super().__init__(cfg=cfg, role_models=role_models)

        mc = self.method_config
        tc = self.training_config

        # Optional reference model for KL.
        self._reference = role_models.get("reference")

        # Parse RL config.
        self._parse_config(mc)

        # Init student preprocessors (VAE, text encoder).
        self.student.init_preprocessors(tc)

        # Scheduler copy for RL pipeline.
        self._scheduler = copy.deepcopy(
            self.student.noise_scheduler
        )

        # Reward functions.
        self._reward_fn = multi_score(
            torch.device("cpu"),
            self._reward_cfg,
            mc.get("reward_module"),
            return_raw_scores=True,
        )

        # Prompt dataloaders.
        wg = get_world_group()
        self._world_size = wg.world_size
        self._rank = wg.rank
        self._is_main = wg.rank == 0

        train_dl, test_dl, train_sampler = (
            build_prompt_dataloaders(
                prompt_dataset_path=mc[
                    "prompt_dataset_path"
                ],
                prompt_fn=mc.get(
                    "prompt_fn", "general_ocr"
                ),
                sample_batch_size=self._sample_batch_size,
                eval_batch_size=self._eval_batch_size,
                num_video_per_prompt=(
                    self._num_video_per_prompt
                ),
                num_processes=self._world_size,
                process_index=self._rank,
                seed=self._seed,
            )
        )
        self._train_dataloader = train_dl
        self._test_dataloader = test_dl
        self._train_sampler = train_sampler
        self._train_iter = iter(train_dl)

        # Stat trackers.
        self._build_stat_trackers()

        # Negative prompt embeddings.
        self._compute_negative_embeds()

        # Optimizer and scheduler.
        self._init_optimizer()

        # Async reward executor.
        self._executor = futures.ThreadPoolExecutor(
            max_workers=8
        )

        # Training timestep indices.
        self._compute_train_timesteps()

    # ------------------------------------------------------------------
    # Config parsing
    # ------------------------------------------------------------------

    def _parse_config(self, mc: dict[str, Any]) -> None:
        # Sampling.
        self._sample_batch_size = int(
            mc.get("sample_batch_size", 8)
        )
        self._eval_batch_size = int(
            mc.get("eval_batch_size", 2)
        )
        self._num_batches_per_epoch = int(
            mc.get("num_batches_per_epoch", 2)
        )
        self._num_inference_steps = int(
            mc.get("num_inference_steps", 20)
        )
        self._guidance_scale = float(
            mc.get("guidance_scale", 4.5)
        )
        self._num_video_per_prompt = int(
            mc.get("num_video_per_prompt", 4)
        )
        self._noise_level = float(
            mc.get("noise_level", 0.7)
        )
        self._sde_type = str(
            mc.get("sde_type", "flow_sde")
        )
        self._sde_window_size = int(
            mc.get("sde_window_size", 0)
        )
        raw_range = mc.get("sde_window_range")
        self._sde_window_range = (
            tuple(raw_range) if raw_range else None
        )
        self._diffusion_clip = bool(
            mc.get("diffusion_clip", False)
        )
        self._diffusion_clip_value = float(
            mc.get("diffusion_clip_value", 0.45)
        )
        self._kl_reward = float(
            mc.get("kl_reward", 0.0)
        )
        self._same_latent = bool(
            mc.get("same_latent", False)
        )

        # Training.
        self._num_inner_epochs = int(
            mc.get("num_inner_epochs", 1)
        )
        self._clip_range = float(
            mc.get("clip_range", 1e-3)
        )
        self._adv_clip_max = float(
            mc.get("adv_clip_max", 5.0)
        )
        self._beta = float(mc.get("beta", 0.0))
        self._use_cfg = bool(mc.get("use_cfg", True))
        self._loss_reweighting = mc.get(
            "loss_reweighting"
        )
        self._weight_advantages = bool(
            mc.get("weight_advantages", False)
        )
        self._max_grad_norm = float(
            mc.get("max_grad_norm", 1.0)
        )
        self._train_batch_size = int(
            mc.get("train_batch_size", 8)
        )

        # Data / dimensions.
        self._height = int(mc.get("height", 480))
        self._width = int(mc.get("width", 832))
        self._num_frames = int(mc.get("num_frames", 81))
        self._seed = int(mc.get("seed", 42))

        # Per-prompt tracking.
        self._per_prompt_stat_tracking = bool(
            mc.get("per_prompt_stat_tracking", True)
        )
        if self._num_video_per_prompt == 1:
            self._per_prompt_stat_tracking = False

        # Reward config.
        self._reward_cfg = dict(mc.get("reward_fn", {}))

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    def _build_stat_trackers(self) -> None:
        self._stat_tracker = None
        self._reward_stat_trackers = None
        self._kl_stat_tracker = None

        if self._per_prompt_stat_tracking:
            self._stat_tracker = PerPromptStatTracker(
                use_global_std=bool(
                    self.method_config.get(
                        "global_std", False
                    )
                ),
                max_group_std=bool(
                    self.method_config.get(
                        "max_group_std", False
                    )
                ),
            )

        if (
            self._weight_advantages
            and self._per_prompt_stat_tracking
        ):
            self._reward_stat_trackers = {
                name: PerPromptStatTracker(
                    use_global_std=bool(
                        self.method_config.get(
                            "global_std", False
                        )
                    ),
                    max_group_std=bool(
                        self.method_config.get(
                            "max_group_std", False
                        )
                    ),
                )
                for name in self._reward_cfg
            }
            if self._kl_reward > 0:
                self._kl_stat_tracker = (
                    PerPromptStatTracker(
                        use_global_std=bool(
                            self.method_config.get(
                                "global_std", False
                            )
                        ),
                        max_group_std=bool(
                            self.method_config.get(
                                "max_group_std", False
                            )
                        ),
                    )
                )

    def _compute_negative_embeds(self) -> None:
        device = self.student.device
        neg = compute_text_embeddings(
            [""],
            self.student.text_encoder,
            self.student.tokenizer,
            max_sequence_length=512,
            device=device,
        )
        self._sample_neg_embeds = neg.repeat(
            self._sample_batch_size, 1, 1
        )
        self._train_neg_embeds = neg.repeat(
            self._train_batch_size, 1, 1
        )

    def _init_optimizer(self) -> None:
        tc = self.training_config
        params = [
            p
            for p in self.student.transformer.parameters()
            if p.requires_grad
        ]
        self._transformer_params = params
        (
            self._optimizer,
            self._lr_scheduler,
        ) = build_optimizer_and_scheduler(
            params=params,
            optimizer_config=tc.optimizer,
            loop_config=tc.loop,
            learning_rate=float(
                tc.optimizer.learning_rate
            ),
            betas=tc.optimizer.betas,
            scheduler_name=str(tc.optimizer.lr_scheduler),
        )

    def _compute_train_timesteps(self) -> None:
        if self._sde_window_size > 0:
            num_ts = self._sde_window_size
        else:
            num_ts = int(
                self._num_inference_steps * 0.99
            )
        self._num_train_timesteps = num_ts
        self._train_timesteps = list(range(num_ts))

    # ------------------------------------------------------------------
    # TrainingMethod interface
    # ------------------------------------------------------------------

    def single_train_step(
        self,
        batch: dict[str, Any],
        iteration: int,
    ) -> tuple[
        dict[str, torch.Tensor],
        dict[str, Any],
        dict[str, LogScalar],
    ]:
        """Run one full RL epoch (sample -> train).

        The *batch* argument (from the dummy dataloader)
        is ignored.
        """
        epoch = iteration - 1  # 0-indexed
        device = self.student.device
        all_metrics: dict[str, LogScalar] = {
            "epoch": float(epoch),
        }

        # 1. Sample epoch.
        self.student.transformer.eval()
        with reward_models_on_device(
            self._reward_cfg, device
        ):
            samples = sample_epoch(
                model=self.student,
                scheduler=self._scheduler,
                train_sampler=self._train_sampler,
                train_iter=self._train_iter,
                reward_fn=self._reward_fn,
                sample_neg_prompt_embeds=(
                    self._sample_neg_embeds
                ),
                text_encoder=self.student.text_encoder,
                tokenizer=self.student.tokenizer,
                executor=self._executor,
                epoch=epoch,
                global_step=iteration,
                sample_batch_size=self._sample_batch_size,
                num_batches_per_epoch=(
                    self._num_batches_per_epoch
                ),
                num_inference_steps=(
                    self._num_inference_steps
                ),
                guidance_scale=self._guidance_scale,
                height=self._height,
                width=self._width,
                num_frames=self._num_frames,
                noise_level=self._noise_level,
                sde_type=self._sde_type,
                diffusion_clip=self._diffusion_clip,
                diffusion_clip_value=(
                    self._diffusion_clip_value
                ),
                sde_window_size=self._sde_window_size,
                sde_window_range=self._sde_window_range,
                kl_reward=self._kl_reward,
                same_latent=self._same_latent,
                seed=self._seed,
                device=device,
                is_main_process=self._is_main,
                ref_transformer=(
                    self._reference.transformer
                    if self._reference
                    else None
                ),
                tracker=self.tracker,
            )

        # 2. Prepare samples (advantages).
        samples = self._prepare_samples(
            samples, epoch, iteration
        )

        # 3. PPO training.
        ppo_metrics = self._ppo_train(
            samples, epoch, iteration
        )
        all_metrics.update(ppo_metrics)

        # Return dummy loss (everything is internal).
        dummy_loss = torch.zeros(
            (), device=device, requires_grad=False
        )
        return (
            {"total_loss": dummy_loss},
            {},
            all_metrics,
        )

    def backward(
        self,
        loss_map: dict[str, torch.Tensor],
        outputs: dict[str, Any],
        *,
        grad_accum_rounds: int = 1,
    ) -> None:
        pass  # Handled internally.

    def get_optimizers(
        self,
        iteration: int,
    ) -> list[torch.optim.Optimizer]:
        return []  # Handled internally.

    def get_lr_schedulers(
        self,
        iteration: int,
    ) -> list[Any]:
        return []

    @property
    def _optimizer_dict(
        self,
    ) -> dict[str, torch.optim.Optimizer]:
        return {"student": self._optimizer}

    @property
    def _lr_scheduler_dict(self) -> dict[str, Any]:
        return {"student": self._lr_scheduler}

    def get_grad_clip_targets(
        self,
        iteration: int,
    ) -> dict[str, torch.nn.Module]:
        return {}  # We clip internally.

    def on_train_start(self) -> None:
        """Seed RNG and call student on_train_start."""
        from fastvideo.utils import set_random_seed

        set_random_seed(self._seed)
        self.cuda_generator = torch.Generator(
            device=self.student.device
        ).manual_seed(self._seed + self._rank)
        self.student.on_train_start()

    # ------------------------------------------------------------------
    # Prepare samples
    # ------------------------------------------------------------------

    def _prepare_samples(
        self,
        samples: list[dict[str, Any]],
        epoch: int,
        global_step: int,
    ) -> dict[str, torch.Tensor]:
        """Collate, compute advantages, and filter."""
        device = self.student.device

        # Collate list of per-batch dicts into one dict.
        collated: dict[str, Any] = {}
        for k in samples[0]:
            first = samples[0][k]
            if isinstance(first, dict):
                collated[k] = {
                    sk: torch.cat(
                        [s[k][sk] for s in samples],
                        dim=0,
                    )
                    for sk in first
                }
            else:
                collated[k] = torch.cat(
                    [s[k] for s in samples], dim=0
                )
        samples_t = collated

        # Apply KL penalty.
        samples_t["rewards"]["ori_avg"] = samples_t[
            "rewards"
        ]["avg"]
        kl_penalty = (
            self._kl_reward * samples_t["kl"]
        )
        samples_t["rewards"]["avg"] = (
            samples_t["rewards"]["avg"].unsqueeze(-1)
            - kl_penalty
        )

        # Broadcast raw rewards to timestep dimension.
        num_timesteps = samples_t["kl"].shape[1]
        for rn in self._reward_cfg:
            raw_key = f"{rn}_raw"
            samples_t["rewards"][f"ori_{raw_key}"] = (
                samples_t["rewards"][raw_key]
            )
            samples_t["rewards"][raw_key] = (
                samples_t["rewards"][raw_key]
                .unsqueeze(-1)
                .expand(-1, num_timesteps)
            )

        # Gather rewards / KL across processes.
        gathered_rewards = {
            k: _gather_tensor(v, self._world_size)
            .cpu()
            .numpy()
            for k, v in samples_t["rewards"].items()
        }
        gathered_kl = (
            _gather_tensor(samples_t["kl"], self._world_size)
            .cpu()
            .numpy()
        )

        # Log reward stats.
        raw_keys = [
            k
            for k in gathered_rewards
            if k.endswith("_raw")
            and not k.startswith("ori_")
        ]
        reward_logs = {
            f"reward_{k}": float(
                gathered_rewards[k].mean()
            )
            for k in raw_keys
        }
        kl_mean = float(gathered_kl.mean())
        reward_logs["kl"] = kl_mean
        reward_logs["kl_abs"] = float(
            np.abs(gathered_kl).mean()
        )
        if self._is_main and self.tracker is not None:
            self.tracker.log(reward_logs, global_step)

        # Decode prompts for advantage computation.
        prompts = None
        if self._per_prompt_stat_tracking:
            prompt_ids = (
                _gather_tensor(
                    samples_t["prompt_ids"],
                    self._world_size,
                )
                .cpu()
                .numpy()
            )
            prompts = (
                self.student.tokenizer.batch_decode(
                    prompt_ids,
                    skip_special_tokens=True,
                )
            )

        # Compute advantages.
        advantages, adv_log = compute_advantages(
            reward_fn_cfg=self._reward_cfg,
            weight_advantages=self._weight_advantages,
            per_prompt_stat_tracking=(
                self._per_prompt_stat_tracking
            ),
            kl_reward=self._kl_reward,
            samples=samples_t,
            gathered_rewards=gathered_rewards,
            gathered_kl=gathered_kl,
            prompts=prompts,
            stat_tracker=self._stat_tracker,
            reward_stat_trackers=(
                self._reward_stat_trackers
            ),
            kl_stat_tracker=self._kl_stat_tracker,
        )
        if adv_log and self._is_main and self.tracker:
            self.tracker.log(adv_log, global_step)

        # Shard advantages back to this rank.
        advantages = torch.as_tensor(advantages)
        advantages = advantages.reshape(
            self._world_size,
            -1,
            advantages.shape[-1],
        )[self._rank].to(device)
        samples_t["advantages"] = advantages

        # Cleanup unused fields.
        del samples_t["rewards"]
        del samples_t["prompt_ids"]

        # Filter zero-advantage samples.
        mask = (
            samples_t["advantages"].abs().sum(dim=1) != 0
        )
        num_batches = self._num_batches_per_epoch
        true_count = mask.sum()
        if true_count == 0:
            samples_t["advantages"] = (
                samples_t["advantages"] + ADVANTAGE_EPSILON
            )
            mask = (
                samples_t["advantages"].abs().sum(dim=1)
                != 0
            )
            true_count = mask.sum()
        if true_count % num_batches != 0:
            false_idx = torch.where(~mask)[0]
            need = num_batches - (
                true_count % num_batches
            )
            if len(false_idx) >= need:
                g = torch.Generator(device=device)
                g.manual_seed(self._seed + epoch)
                perm = torch.randperm(
                    len(false_idx),
                    device=device,
                    generator=g,
                )[:need]
                mask[false_idx[perm]] = True

        samples_t = {
            k: v[mask] for k, v in samples_t.items()
        }
        return samples_t

    # ------------------------------------------------------------------
    # PPO training
    # ------------------------------------------------------------------

    def _ppo_train(
        self,
        samples: dict[str, torch.Tensor],
        epoch: int,
        global_step: int,
    ) -> dict[str, LogScalar]:
        """Run inner PPO training loop."""
        device = self.student.device
        total_batch_size, num_timesteps = samples[
            "timesteps"
        ].shape
        num_ts = self._num_train_timesteps
        all_info: dict[str, list[float]] = defaultdict(
            list
        )

        for inner_epoch in range(self._num_inner_epochs):
            # Shuffle samples.
            g = torch.Generator(device=device)
            g.manual_seed(
                self._seed
                + epoch * SEED_EPOCH_STRIDE
                + inner_epoch
            )
            perm = torch.randperm(
                total_batch_size,
                device=device,
                generator=g,
            )
            samples = {
                k: v[perm] for k, v in samples.items()
            }
            # Shuffle timestep dimension per-sample.
            perms = torch.stack(
                [
                    torch.arange(
                        num_timesteps, device=device
                    )
                    for _ in range(total_batch_size)
                ]
            )
            row_idx = torch.arange(
                total_batch_size, device=device
            )[:, None]
            for key in [
                "timesteps",
                "latents",
                "next_latents",
                "log_probs",
            ]:
                samples[key] = samples[key][
                    row_idx, perms
                ]

            # Batch into micro-batches.
            micro = (
                total_batch_size
                // self._num_batches_per_epoch
            )
            batched = {
                k: v.reshape(
                    -1, micro, *v.shape[1:]
                )
                for k, v in samples.items()
            }
            batched_list = [
                dict(zip(batched, x, strict=False))
                for x in zip(
                    *batched.values(), strict=False
                )
            ]

            self.student.transformer.train()
            info: dict[str, list] = defaultdict(list)

            for sample in batched_list:
                # Get embeddings.
                embeds = sample["prompt_embeds"]
                neg_embeds = (
                    self._train_neg_embeds[
                        : len(embeds)
                    ]
                    if self._use_cfg
                    else None
                )

                self._optimizer.zero_grad()

                for j in self._train_timesteps:
                    # Reference model output (for KL).
                    prev_mean_ref = None
                    dt_sqrt_ref = None
                    if self._beta > 0:
                        ref_model = self._get_ref_model()
                        if ref_model is not None:
                            with torch.no_grad():
                                (
                                    _,
                                    _,
                                    prev_mean_ref,
                                    _,
                                    dt_sqrt_ref,
                                    _,
                                    _,
                                ) = compute_log_prob(
                                    ref_model,
                                    self._scheduler,
                                    sample,
                                    j,
                                    embeds,
                                    neg_embeds,
                                    self._guidance_scale,
                                    self._use_cfg,
                                    self._noise_level,
                                    self._sde_type,
                                    self._diffusion_clip,
                                    self._diffusion_clip_value,
                                )

                    # Policy forward.
                    (
                        _prev_sample,
                        log_prob,
                        prev_sample_mean,
                        std_dev_t,
                        dt_sqrt,
                        sigma,
                        sigma_max,
                    ) = compute_log_prob(
                        self.student,
                        self._scheduler,
                        sample,
                        j,
                        embeds,
                        neg_embeds,
                        self._guidance_scale,
                        self._use_cfg,
                        self._noise_level,
                        self._sde_type,
                        self._diffusion_clip,
                        self._diffusion_clip_value,
                    )

                    # PPO loss.
                    advantages = torch.clamp(
                        sample["advantages"][:, j],
                        -self._adv_clip_max,
                        self._adv_clip_max,
                    )
                    ratio = torch.exp(
                        log_prob
                        - sample["log_probs"][:, j]
                    )
                    unclipped = -advantages * ratio
                    clipped = -advantages * torch.clamp(
                        ratio,
                        1.0 - self._clip_range,
                        1.0 + self._clip_range,
                    )
                    policy_loss = torch.mean(
                        torch.maximum(unclipped, clipped)
                    )

                    # Loss reweighting.
                    rw_scale = 1.0
                    rw_scale_kl = 1.0
                    if (
                        self._loss_reweighting
                        == "longcat"
                        and self._sde_type == "flow_sde"
                    ):
                        rw_scale = (
                            torch.sqrt(
                                sigma
                                / (
                                    1
                                    - torch.where(
                                        sigma == 1,
                                        torch.tensor(
                                            sigma_max,
                                            device=(
                                                sigma.device
                                            ),
                                            dtype=(
                                                sigma.dtype
                                            ),
                                        ),
                                        sigma,
                                    )
                                )
                            )
                            / dt_sqrt
                        )
                        rw_scale = torch.mean(rw_scale)
                        rw_scale_kl = rw_scale**2

                    # KL loss.
                    if (
                        self._beta > 0
                        and prev_mean_ref is not None
                    ):
                        if (
                            self._sde_type == "flow_sde"
                        ):
                            kl_denom = (
                                std_dev_t * dt_sqrt_ref
                            ) ** 2
                        elif (
                            self._sde_type == "flow_cps"
                        ):
                            kl_denom = 0.5
                        else:
                            msg = (
                                "Unknown sde_type: "
                                f"{self._sde_type}"
                            )
                            raise ValueError(msg)
                        kl_loss = (
                            (
                                prev_sample_mean
                                - prev_mean_ref
                            )
                            ** 2
                        ).mean(
                            dim=(1, 2, 3), keepdim=True
                        ) / (
                            2 * kl_denom
                        )
                        kl_loss = torch.mean(kl_loss)
                        loss = (
                            rw_scale * policy_loss
                            + self._beta
                            * kl_loss
                            * rw_scale_kl
                        )
                    else:
                        loss = rw_scale * policy_loss

                    # Backward with gradient accumulation.
                    timestep_j = sample["timesteps"][
                        :, j
                    ]
                    with set_forward_context(
                        current_timestep=timestep_j,
                        attn_metadata=None,
                    ):
                        (loss / num_ts).backward()

                    # Track.
                    info["approx_kl"].append(
                        0.5
                        * torch.mean(
                            (
                                log_prob
                                - sample["log_probs"][
                                    :, j
                                ]
                            )
                            ** 2
                        )
                        .detach()
                        .item()
                    )
                    info["clip_frac"].append(
                        torch.mean(
                            (
                                torch.abs(ratio - 1.0)
                                > self._clip_range
                            ).float()
                        )
                        .detach()
                        .item()
                    )
                    info["policy_loss"].append(
                        policy_loss.detach().item()
                    )
                    if self._beta > 0 and prev_mean_ref is not None:
                        info["kl_loss"].append(
                            kl_loss.detach().item()
                        )
                    info["loss"].append(
                        loss.detach().item()
                    )

                # Clip + step after all timesteps.
                clip_grad_norm_if_needed(
                    self.student.transformer,
                    self._max_grad_norm,
                )
                self._optimizer.step()
                self._optimizer.zero_grad()

            # Aggregate info for this inner epoch.
            for k, v in info.items():
                all_info[k].extend(v)

        # Aggregate across inner epochs.
        metrics: dict[str, LogScalar] = {}
        for k, vals in all_info.items():
            metrics[k] = float(np.mean(vals))
        metrics["inner_epochs"] = float(
            self._num_inner_epochs
        )
        return metrics

    # ------------------------------------------------------------------
    # Reference model
    # ------------------------------------------------------------------

    def _get_ref_model(self):
        """Get reference model for KL computation."""
        if self._reference is not None:
            return self._reference
        # LoRA case: caller should use disable_adapter.
        return None
