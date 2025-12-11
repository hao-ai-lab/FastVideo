# SPDX-License-Identifier: Apache-2.0
"""
Abstract base class for RL algorithms.

This module defines the interface that all RL algorithms (GRPO, PPO, DPO) must implement.
Each algorithm is responsible for:
- Computing policy loss
- Computing value loss (if applicable)
- Handling algorithm-specific training logic
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import torch

from fastvideo.logger import init_logger
from fastvideo.pipelines import TrainingBatch

logger = init_logger(__name__)


@dataclass
class AlgorithmOutput:
    """Output from an RL algorithm's compute_loss method."""
    # Primary losses
    policy_loss: torch.Tensor
    value_loss: torch.Tensor | None = None

    # Total loss (policy + value + entropy bonus)
    total_loss: torch.Tensor | None = None

    # Metrics for logging
    metrics: dict[str, float] | None = None


class BaseRLAlgorithm(ABC):
    """
    Abstract base class for RL algorithms.

    All RL algorithms should inherit from this class and implement
    the required methods for computing losses and updating the policy.
    """

    def __init__(self, config: Any) -> None:
        """
        Initialize the algorithm with configuration.

        Args:
            config: Algorithm-specific configuration (typically RLArgs)
        """
        self.config = config
        self._validate_config()
        logger.info("Initialized %s algorithm", self.__class__.__name__)

    @abstractmethod
    def _validate_config(self) -> None:
        """Validate algorithm-specific configuration."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the algorithm name."""
        pass

    @property
    @abstractmethod
    def requires_value_model(self) -> bool:
        """Return whether this algorithm requires a value model."""
        pass

    @property
    @abstractmethod
    def requires_reference_model(self) -> bool:
        """Return whether this algorithm requires a reference model (for KL penalty)."""
        pass

    @abstractmethod
    def compute_advantages(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        next_values: torch.Tensor,
        dones: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute advantages and returns.

        Args:
            rewards: Rewards [B] or [B, T]
            values: Value predictions [B] or [B, T]
            next_values: Next value predictions [B] or [B, T]
            dones: Episode termination flags [B] or [B, T]

        Returns:
            advantages: Computed advantages
            returns: Computed returns (advantages + values)
        """
        pass

    @abstractmethod
    def compute_policy_loss(
        self,
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """
        Compute the policy loss.

        Args:
            log_probs: Log probabilities from current policy [B]
            old_log_probs: Log probabilities from old policy [B]
            advantages: Advantages [B]

        Returns:
            loss: Policy loss (scalar)
            info: Dictionary with diagnostic information
        """
        pass

    @abstractmethod
    def compute_value_loss(
        self,
        values: torch.Tensor,
        returns: torch.Tensor,
        old_values: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """
        Compute the value function loss.

        Args:
            values: Value predictions from current model [B]
            returns: Target returns [B]
            old_values: Value predictions from old model [B] (for clipping)

        Returns:
            loss: Value loss (scalar)
            info: Dictionary with diagnostic information
        """
        pass

    def compute_entropy_bonus(self, log_probs: torch.Tensor) -> torch.Tensor:
        """
        Compute entropy bonus for exploration.

        Args:
            log_probs: Log probabilities [B]

        Returns:
            entropy: Mean entropy (scalar)
        """
        # For continuous actions: H = -log_prob (assuming Gaussian)
        entropy = -log_probs.mean()
        return entropy

    def compute_loss(
        self,
        training_batch: TrainingBatch
    ) -> AlgorithmOutput:
        """
        Compute all losses for the training step.

        This method orchestrates the computation of policy loss, value loss,
        and entropy bonus. Subclasses can override this for algorithm-specific
        behavior.

        Args:
            training_batch: Training batch with all required tensors

        Returns:
            AlgorithmOutput with losses and metrics
        """
        metrics: dict[str, float] = {}

        # Compute policy loss
        if training_batch.log_probs is not None and training_batch.old_log_probs is not None:
            policy_loss, policy_info = self.compute_policy_loss(
                log_probs=training_batch.log_probs,
                old_log_probs=training_batch.old_log_probs,
                advantages=training_batch.advantages
            )
            metrics.update(policy_info)
        else:
            policy_loss = torch.tensor(0.0)

        # Compute value loss
        value_loss = None
        if self.requires_value_model and training_batch.values is not None:
            value_loss, value_info = self.compute_value_loss(
                values=training_batch.values,
                returns=training_batch.returns,
                old_values=training_batch.old_values
            )
            metrics.update(value_info)

        # Compute entropy bonus
        entropy = torch.tensor(0.0)
        if self.config.rl_entropy_coef > 0.0 and training_batch.log_probs is not None:
            entropy = self.compute_entropy_bonus(training_batch.log_probs)
            metrics["entropy"] = entropy.item()

        # Compute total loss
        total_loss = policy_loss
        if value_loss is not None:
            total_loss = total_loss + self.config.rl_value_loss_coef * value_loss
        if self.config.rl_entropy_coef > 0.0:
            total_loss = total_loss - self.config.rl_entropy_coef * entropy

        return AlgorithmOutput(
            policy_loss=policy_loss,
            value_loss=value_loss,
            total_loss=total_loss,
            metrics=metrics
        )

    def check_early_stopping(self, kl_divergence: float) -> bool:
        """
        Check if training should stop early based on KL divergence.

        Args:
            kl_divergence: Current KL divergence

        Returns:
            should_stop: True if KL exceeds target
        """
        if kl_divergence > self.config.rl_target_kl:
            logger.warning(
                "Early stopping triggered: KL divergence %.4f > target %.4f",
                kl_divergence,
                self.config.rl_target_kl
            )
            return True
        return False

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
