# SPDX-License-Identifier: Apache-2.0
"""
RL training pipeline for FastVideo.

This module currently implements reinforcement learning training using GRPO (Group Relative Policy Optimization)
and related algorithms. It extends the base TrainingPipeline with RL-specific functionality:

Reference:
    Flow-GRPO: https://github.com/yifan123/flow_grpo
"""

import torch
import torch.nn as nn
from typing import Any

from fastvideo.fastvideo_args import TrainingArgs
from fastvideo.logger import init_logger
from fastvideo.pipelines import TrainingBatch
from fastvideo.training.training_pipeline import TrainingPipeline
from fastvideo.training.rl.rewards import (
    create_reward_models,
    MultiRewardAggregator,
    ValueModel
)
from .rl_utils import (
    compute_gae,
    normalize_advantages,
    compute_grpo_policy_loss,
    compute_value_loss,
    compute_policy_entropy,
    sample_random_timesteps,
    compute_reward_statistics,
    check_early_stopping
)
from fastvideo.training.training_utils import (
    get_scheduler,
    count_trainable
)

logger = init_logger(__name__)


class RLPipeline(TrainingPipeline):
    """
    RL training pipeline using GRPO for flow matching models.

    This pipeline implements online reinforcement learning for video generation models.
    It follows the Flow-GRPO approach with:
    - Fast trajectory collection (1-2 denoising steps)
    - Multi-reward aggregation
    - GRPO policy optimization
    - GRPO-Guard safety mechanisms
    """

    def __init__(
        self,
        model_path: str,
        fastvideo_args: TrainingArgs,
        required_config_modules: list[str] | None = None,
        loaded_modules: dict[str, nn.Module] | None = None
    ) -> None:
        """Initialize RL pipeline."""
        if not fastvideo_args.rl_args.rl_mode:
            logger.warning(
                "rl_mode is False, but RLPipeline is being initialized. "
                "Setting rl_mode=True."
            )
            fastvideo_args.rl_args.rl_mode = True

        super().__init__(model_path, fastvideo_args, required_config_modules, loaded_modules)

        # RL-specific components (will be initialized in initialize_training_pipeline)
        self.reward_models: MultiRewardAggregator | None = None
        self.value_model: ValueModel | None = None
        self.value_optimizer: torch.optim.Optimizer | None = None
        self.value_scheduler: Any | None = None

        logger.info(
            "Initialized RLPipeline with algorithm: %s",
            fastvideo_args.rl_args.rl_algorithm
        )

    def initialize_training_pipeline(self, training_args: TrainingArgs):
        """Initialize the RL training pipeline with reward and value models."""
        # First initialize base pipeline (transformer, optimizer, dataloader, etc.)
        super().initialize_training_pipeline(training_args)

        logger.info("Initializing RL-specific components...")

        # Initialize reward models
        self.reward_models = create_reward_models(
            reward_model_paths=training_args.rl_args.reward_model_paths,
            reward_weights=training_args.rl_args.reward_weights,
            reward_model_types=training_args.rl_args.reward_model_types,
            device=str(self.device)
        )
        logger.info("Loaded reward models: %s", self.reward_models)

        # Initialize value model
        if training_args.rl_args.value_model_share_backbone:
            # Share transformer backbone with policy
            logger.info("Value model will share backbone with policy transformer")
            self.value_model = ValueModel(
                self.transformer,
                share_backbone=True
            )
        else:
            # Separate value model (clone transformer architecture)
            logger.info("Creating separate value model")
            # TODO: Implement separate value model initialization
            # For now, use shared backbone
            self.value_model = ValueModel(
                self.transformer,
                share_backbone=True
            )

        # Create optimizer and scheduler for value model
        if not training_args.rl_args.value_model_share_backbone:
            value_params = list(self.value_model.parameters())
            self.value_optimizer = torch.optim.AdamW(
                value_params,
                lr=training_args.learning_rate,
                betas=(0.9, 0.999),
                weight_decay=training_args.weight_decay,
                eps=1e-8,
            )

            self.value_scheduler = get_scheduler(
                training_args.lr_scheduler,
                optimizer=self.value_optimizer,
                num_warmup_steps=training_args.lr_warmup_steps,
                num_training_steps=training_args.max_train_steps,
                num_cycles=training_args.lr_num_cycles,
                power=training_args.lr_power,
                min_lr_ratio=training_args.min_lr_ratio,
                last_epoch=self.init_steps - 1,
            )

            logger.info("Created separate optimizer and scheduler for value model")
            logger.info("Value model trainable parameters: %s B",
                       round(count_trainable(self.value_model) / 1e9, 3))

        logger.info("RL pipeline initialization complete")

    def initialize_validation_pipeline(self, training_args: TrainingArgs):
        """Initialize validation pipeline for RL (similar to base implementation)."""
        # For now, reuse the base implementation
        # In the future, we can add RL-specific validation (e.g., compare old vs new policy)
        logger.info("RL validation pipeline will be implemented based on task requirements")
        # Set validation_pipeline to None for now
        self.validation_pipeline = None

    def collect_trajectories(self, training_batch: TrainingBatch) -> TrainingBatch:
        """
        Collect on-policy trajectories using Flow-GRPO-Fast approach.

        This implements fast trajectory collection by:
        1. Sampling random intermediate timesteps for noise injection
        2. Running 1-2 denoising steps (not full trajectory)
        3. Computing log probabilities at sampled steps

        Args:
            training_batch: Current training batch

        Returns:
            Updated training_batch with trajectory information
        """
        logger.debug("Collecting trajectories with Flow-GRPO-Fast")

        # Parse rollout steps from config
        rollout_steps_str = self.training_args.rl_args.rl_rollout_steps
        rollout_steps = [int(s.strip()) for s in rollout_steps_str.split(",")]

        # Sample random timesteps for noise injection
        batch_size = training_batch.latents.shape[0]
        timesteps = sample_random_timesteps(
            batch_size=batch_size,
            min_timestep=self.training_args.rl_args.rl_noise_injection_min,
            max_timestep=self.training_args.rl_args.rl_noise_injection_max,
            device=training_batch.latents.device,
            generator=self.noise_random_generator
        )

        # TODO: Implement actual trajectory collection
        # use existing noisy inputs and compute log probs for now
        training_batch.timesteps = timesteps

        # Store old log probs for importance ratio
        training_batch.old_log_probs = training_batch.log_probs

        logger.debug("Trajectory collection complete")
        return training_batch

    def compute_rewards(self, training_batch: TrainingBatch) -> TrainingBatch:
        """
        Compute rewards using multi-reward aggregation.

        Args:
            training_batch: Training batch with generated videos

        Returns:
            Updated training_batch with reward scores
        """
        logger.debug("Computing rewards from reward models")

        # TODO: Implement actual asynchronous reward computation
        # This requires decoding latents to videos and running reward models
        # For now, use dummy rewards
        batch_size = training_batch.latents.shape[0]
        dummy_rewards = torch.randn(batch_size, device=training_batch.latents.device) * 0.1 + 0.5
        training_batch.reward_scores = dummy_rewards.clamp(0.0, 1.0)

        # Compute reward statistics
        reward_stats = compute_reward_statistics(training_batch.reward_scores)
        training_batch.reward_mean = reward_stats["reward_mean"]
        training_batch.reward_std = reward_stats["reward_std"]

        logger.debug("Rewards computed: mean=%.3f, std=%.3f",
                    reward_stats["reward_mean"],
                    reward_stats["reward_std"])

        return training_batch

    def compute_values(self, training_batch: TrainingBatch) -> TrainingBatch:
        """
        Compute value predictions using value model.

        Args:
            training_batch: Training batch

        Returns:
            Updated training_batch with value predictions
        """
        logger.debug("Computing value predictions")

        # TODO: Implement actual value computation
        # For now, use dummy values
        batch_size = training_batch.latents.shape[0]
        dummy_values = torch.randn(batch_size, device=training_batch.latents.device) * 0.1 + 0.5
        training_batch.values = dummy_values
        training_batch.old_values = training_batch.values.clone()

        training_batch.value_mean = training_batch.values.mean().item()

        logger.debug("Values computed: mean=%.3f", training_batch.value_mean)
        return training_batch

    def compute_advantages(self, training_batch: TrainingBatch) -> TrainingBatch:
        """
        Compute advantages using GAE-lambda.

        Args:
            training_batch: Training batch with rewards and values

        Returns:
            Updated training_batch with advantages and returns
        """
        logger.debug("Computing advantages using GAE")

        # For single-step (Flow-GRPO-Fast), we have simple advantage computation
        # advantages = rewards - values
        # In multi-step, we'd use GAE with gamma and lambda

        if training_batch.reward_scores is None or training_batch.values is None:
            raise ValueError("Rewards and values must be computed before advantages")

        # Simple advantage for single-step
        # For multi-step, use compute_gae()
        rewards = training_batch.reward_scores
        values = training_batch.values
        next_values = values  # For single-step, next_value = current_value

        advantages, returns = compute_gae(
            rewards=rewards,
            values=values,
            next_values=next_values,
            gamma=self.training_args.rl_args.rl_gamma,
            lambda_=self.training_args.rl_args.rl_lambda
        )

        # Normalize advantages
        if self.training_args.rl_args.rl_normalize_advantages:
            advantages = normalize_advantages(advantages)

        training_batch.advantages = advantages
        training_batch.returns = returns
        training_batch.advantage_mean = advantages.mean().item()
        training_batch.advantage_std = advantages.std().item()

        logger.debug("Advantages computed: mean=%.3f, std=%.3f",
                    training_batch.advantage_mean,
                    training_batch.advantage_std)

        return training_batch

    def train_one_step(self, training_batch: TrainingBatch) -> TrainingBatch:
        """
        Train one step using RL/GRPO algorithm.

        This overrides the base class train_one_step to implement RL-specific logic:
        1. Collect trajectories with current policy
        2. Compute rewards
        3. Compute values
        4. Compute advantages
        5. Update policy with GRPO loss
        6. Update value function

        Args:
            training_batch: Current training batch

        Returns:
            Updated training_batch with loss and metrics
        """
        # Check if we're in warmup phase (do SFT instead of RL)
        if training_batch.current_timestep < self.training_args.rl_args.rl_warmup_steps:
            logger.debug("In warmup phase, using standard SFT training")
            return super().train_one_step(training_batch)

        training_batch = self._prepare_training(training_batch)

        # Gradient accumulation loop
        for _ in range(self.training_args.gradient_accumulation_steps):
            training_batch = self._get_next_batch(training_batch)
            training_batch = self._normalize_dit_input(training_batch)
            training_batch = self._prepare_dit_inputs(training_batch)

            # === RL-specific steps ===

            # 1. Collect trajectories
            training_batch = self.collect_trajectories(training_batch)

            # 2. Compute rewards
            training_batch = self.compute_rewards(training_batch)

            # 3. Compute value predictions
            training_batch = self.compute_values(training_batch)

            # 4. Compute advantages using GAE
            training_batch = self.compute_advantages(training_batch)

            # 5. Compute policy loss (GRPO specific)
            if training_batch.log_probs is not None and training_batch.old_log_probs is not None:
                policy_loss, policy_info = compute_grpo_policy_loss(
                    log_probs=training_batch.log_probs,
                    old_log_probs=training_batch.old_log_probs,
                    advantages=training_batch.advantages,
                    clip_range=self.training_args.rl_args.rl_policy_clip_range,
                    use_ratio_norm=self.training_args.rl_args.rl_ratio_norm_correction,
                    max_importance_ratio=self.training_args.rl_args.rl_max_importance_ratio
                )

                training_batch.policy_loss = policy_info["policy_loss"]
                training_batch.kl_divergence = policy_info["kl_divergence"]
                training_batch.importance_ratio = policy_info["importance_ratio_mean"]
                training_batch.clip_fraction = policy_info["clip_fraction"]

                # Backward pass for policy
                (policy_loss / self.training_args.gradient_accumulation_steps).backward()

            # 6. Compute value loss
            if training_batch.values is not None and training_batch.returns is not None:
                value_loss, value_info = compute_value_loss(
                    values=training_batch.values,
                    returns=training_batch.returns,
                    old_values=training_batch.old_values,
                    clip_range=self.training_args.rl_args.rl_value_clip_range,
                    use_clipping=True
                )

                training_batch.value_loss = value_info["value_loss"]

                # Backward pass for value
                value_loss_scaled = value_loss * self.training_args.rl_args.rl_value_loss_coef
                (value_loss_scaled / self.training_args.gradient_accumulation_steps).backward()

            # 7. Entropy bonus (optional)
            if self.training_args.rl_args.rl_entropy_coef > 0.0 and training_batch.log_probs is not None:
                entropy = compute_policy_entropy(training_batch.log_probs)
                training_batch.entropy = entropy.item()
                entropy_loss = -self.training_args.rl_args.rl_entropy_coef * entropy
                (entropy_loss / self.training_args.gradient_accumulation_steps).backward()

            # Accumulate total loss
            total_loss = training_batch.policy_loss + training_batch.value_loss
            training_batch.total_loss += total_loss

        # Clip gradients
        training_batch = self._clip_grad_norm(training_batch)

        # Optimizer step
        with self.tracker.timed("timing/optimizer_step"):
            self.optimizer.step()
            self.lr_scheduler.step()

            if self.value_optimizer is not None:
                self.value_optimizer.step()
                self.value_scheduler.step()

        # Check for early stopping based on KL divergence
        if check_early_stopping(training_batch.kl_divergence, self.training_args.rl_args.rl_target_kl):
            logger.warning(
                "Early stopping at step %d due to high KL divergence",
                training_batch.current_timestep
            )

        return training_batch

    def set_trainable(self) -> None:
        """Set which parameters should be trainable."""
        # Policy (transformer) is trainable
        for param in self.transformer.parameters():
            param.requires_grad = True

        # Value model is trainable if not sharing backbone
        if self.value_model is not None and not self.training_args.rl_args.value_model_share_backbone:
            for param in self.value_model.parameters():
                param.requires_grad = True

        # Freeze reward models (they should not be trained)
        if self.reward_models is not None:
            for param in self.reward_models.parameters():
                param.requires_grad = False

        logger.info("Set trainable parameters for RL training")


def create_rl_pipeline(
    model_path: str,
    training_args: TrainingArgs
) -> RLPipeline:
    """
    Factory function to create RL pipeline.

    Args:
        model_path: Path to pretrained model
        training_args: Training arguments with RL configuration

    Returns:
        Initialized RLPipeline
    """
    pipeline = RLPipeline(model_path, training_args)
    pipeline.initialize_training_pipeline(training_args)
    return pipeline
