# SPDX-License-Identifier: Apache-2.0
"""
RL training pipeline for FastVideo.

This module implements reinforcement learning training using pluggable algorithms
(GRPO, PPO, DPO). It extends the base TrainingPipeline with RL-specific functionality.

The pipeline separates algorithm logic from pipeline infrastructure, allowing
easy switching between different RL algorithms.

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
from fastvideo.training.rl.algorithms import (
    BaseRLAlgorithm,
    create_algorithm,
)
from .rl_utils import (
    sample_random_timesteps,
    compute_reward_statistics,
)
from fastvideo.training.training_utils import (
    get_scheduler,
    count_trainable
)

logger = init_logger(__name__)


class RLPipeline(TrainingPipeline):
    """
    RL training pipeline for flow matching models.

    This pipeline implements online reinforcement learning for video generation models.
    It supports multiple RL algorithms through a pluggable architecture:
    - GRPO (Group Relative Policy Optimization) with GRPO-Guard safety
    - PPO (Proximal Policy Optimization) #TODO
    - DPO (Direct Preference Optimization) #TODO

    The algorithm is selected via the `rl_algorithm` configuration and handles:
    - Advantage computation
    - Policy loss computation
    - Value loss computation (if applicable)
    """

    # Algorithm instance that implements train_one_step logic
    algorithm: BaseRLAlgorithm

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
        """Initialize the RL training pipeline with algorithm, reward and value models."""
        super().initialize_training_pipeline(training_args)

        logger.info("Initializing RL-specific components...")

        # Initialize the RL algorithm
        self.algorithm = create_algorithm(
            algorithm_name=training_args.rl_args.rl_algorithm,
            config=training_args.rl_args
        )
        logger.info("Using RL algorithm: %s", self.algorithm.name)

        # Initialize reward models (excludes DPO)
        if not self.algorithm.name == "dpo":
            self.reward_models = create_reward_models(
                reward_models=training_args.rl_args.reward_models,
                device=str(self.device)
            )
            logger.info("Loaded reward models: %s", self.reward_models)

        # Initialize value model if algorithm requires it
        if self.algorithm.requires_value_model:
            self._initialize_value_model(training_args)

        logger.info("RL pipeline initialization complete")

    def _initialize_value_model(self, training_args: TrainingArgs) -> None:
        """Initialize the value model and its optimizer."""
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
        rewards = self.reward_models.compute_reward(training_batch.latents, training_batch.input_kwargs["prompts"])
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
        if not self.algorithm.requires_value_model:
            return training_batch

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
        Compute advantages using the algorithm's implementation.

        Args:
            training_batch: Training batch with rewards and values

        Returns:
            Updated training_batch with advantages and returns
        """
        if training_batch.reward_scores is None:
            raise ValueError("Rewards must be computed before advantages")

        # Get values (use zeros if algorithm doesn't use value model)
        if training_batch.values is not None:
            values = training_batch.values
        else:
            values = torch.zeros_like(training_batch.reward_scores)

        # For single-step, next_value = current_value
        next_values = values

        # Delegate to algorithm for advantage computation
        advantages, returns = self.algorithm.compute_advantages(
            rewards=training_batch.reward_scores,
            values=values,
            next_values=next_values,
            dones=None
        )

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
        Train one step using the configured RL algorithm.

        This method orchestrates the training step while delegating
        algorithm-specific loss computation to the algorithm instance.

        Steps:
        1. Collect trajectories with current policy
        2. Compute rewards
        3. Compute values (if algorithm requires)
        4. Compute advantages
        5. Compute losses using algorithm
        6. Update policy and value function

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

            # 3. Compute value predictions (if algorithm requires)
            training_batch = self.compute_values(training_batch)

            # 4. Compute advantages using algorithm
            training_batch = self.compute_advantages(training_batch)

            # 5. Compute losses using algorithm
            if training_batch.log_probs is not None and training_batch.old_log_probs is not None:
                # Use the algorithm to compute all losses
                algorithm_output = self.algorithm.compute_loss(training_batch)

                # Extract metrics from algorithm output
                if algorithm_output.metrics:
                    training_batch.policy_loss = algorithm_output.metrics.get("policy_loss", 0.0)
                    training_batch.kl_divergence = algorithm_output.metrics.get("kl_divergence", 0.0)
                    training_batch.importance_ratio = algorithm_output.metrics.get("importance_ratio_mean", 1.0)
                    training_batch.clip_fraction = algorithm_output.metrics.get("clip_fraction", 0.0)
                    training_batch.value_loss = algorithm_output.metrics.get("value_loss", 0.0)
                    training_batch.entropy = algorithm_output.metrics.get("entropy", 0.0)

                # Backward pass with scaled loss
                if algorithm_output.total_loss is not None:
                    scaled_loss = algorithm_output.total_loss / self.training_args.gradient_accumulation_steps
                    scaled_loss.backward()

                # Accumulate total loss
                if algorithm_output.total_loss is not None:
                    training_batch.total_loss += algorithm_output.total_loss.item()

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
        if self.algorithm.check_early_stopping(training_batch.kl_divergence):
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

        # Value model is trainable if not sharing backbone and algorithm requires it
        if (self.value_model is not None and
            self.algorithm.requires_value_model and
            not self.training_args.rl_args.value_model_share_backbone):
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
