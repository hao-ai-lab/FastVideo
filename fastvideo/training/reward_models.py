# SPDX-License-Identifier: Apache-2.0
"""
Base infrastructure for VIDEO reward models in RL/GRPO training.

IMPORTANT: This module is designed exclusively for VIDEO generation models.
All reward models must operate on video sequences [B, T, C, H, W], not single frames.

This module provides:
1. Abstract base class for VIDEO reward models
2. Multi-reward aggregation for video
3. Value model wrapper
4. Integration with FastVideo video generation infrastructure

Scope: VIDEO ONLY - No image-only reward models
"""

from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn as nn

from fastvideo.logger import init_logger

logger = init_logger(__name__)


class BaseRewardModel(ABC, nn.Module):
    """
    Abstract base class for VIDEO reward models.

    All VIDEO reward models should inherit from this class and implement
    the compute_reward() method.

    IMPORTANT: Reward models must process FULL VIDEO SEQUENCES, not individual frames.
    Input shape is [B, T, C, H, W] where T is the temporal (frame) dimension.

    For video-specific rewards, consider:
    - Temporal coherence across frames
    - Motion quality and smoothness
    - Video-text alignment (not just frame-text)
    - Multi-frame aesthetic quality
    """

    def __init__(self, model_path: str | None = None, device: str = "cuda"):
        super().__init__()
        self.model_path = model_path
        self.device = device

    @abstractmethod
    def compute_reward(
        self,
        videos: torch.Tensor,  # [B, T, C, H, W] decoded video sequences
        prompts: list[str],  # Text prompts describing videos
        **kwargs: Any
    ) -> torch.Tensor:
        """
        Compute rewards for generated VIDEO sequences.

        IMPORTANT: This method must process the FULL temporal sequence [B, T, C, H, W].
        Do NOT evaluate individual frames independently and average.

        Args:
            videos: Decoded video tensors [B, T, C, H, W] in range [0, 1]
                   B = batch size
                   T = number of frames (temporal dimension)
                   C = channels (typically 3 for RGB)
                   H, W = height, width
            prompts: List of text prompts (length B) describing each video
            **kwargs: Additional model-specific arguments

        Returns:
            rewards: Tensor of shape [B] with reward scores for each video sequence

        Example:
            >>> videos = torch.rand(4, 17, 3, 256, 256)  # 4 videos, 17 frames each
            >>> prompts = ["A cat jumping", "A dog running", ...]
            >>> rewards = model.compute_reward(videos, prompts)
            >>> rewards.shape  # torch.Size([4])
        """
        raise NotImplementedError("Subclasses must implement compute_reward()")

    @abstractmethod
    def load_model(self) -> None:
        """Load the reward model from checkpoint."""
        raise NotImplementedError("Subclasses must implement load_model()")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model_path={self.model_path})"


class MultiRewardAggregator(nn.Module):
    """
    Aggregates multiple reward models with configurable weights.

    This implements the multi-reward aggregation strategy from flow_grpo,
    allowing combination of different reward signals (aesthetic quality,
    text-video alignment, compositional understanding, etc.)
    """

    def __init__(
        self,
        reward_models: list[BaseRewardModel],
        reward_weights: list[float] | None = None,
        normalize_rewards: bool = True
    ):
        """
        Initialize multi-reward aggregator.

        Args:
            reward_models: List of reward model instances
            reward_weights: Weights for each reward model (default: uniform)
            normalize_rewards: Whether to normalize rewards before aggregation
        """
        super().__init__()
        self.reward_models = nn.ModuleList(reward_models)

        if reward_weights is None:
            reward_weights = [1.0 / len(reward_models)] * len(reward_models)

        assert len(reward_weights) == len(reward_models), \
            f"Number of weights ({len(reward_weights)}) must match number of models ({len(reward_models)})"

        assert abs(sum(reward_weights) - 1.0) < 1e-6, \
            f"Reward weights must sum to 1.0, got {sum(reward_weights)}"

        self.reward_weights = reward_weights
        self.normalize_rewards = normalize_rewards

        logger.info(
            "Initialized MultiRewardAggregator with %d models: %s",
            len(reward_models),
            [(type(m).__name__, w) for m, w in zip(reward_models, reward_weights, strict=False)]
        )

    def compute_reward(
        self,
        videos: torch.Tensor,
        prompts: list[str],
        return_individual: bool = False,
        **kwargs: Any
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """
        Compute aggregated reward from multiple models.

        Args:
            videos: Decoded video tensors [B, C, T, H, W]
            prompts: List of text prompts
            return_individual: If True, return dict with individual rewards
            **kwargs: Additional arguments passed to reward models

        Returns:
            If return_individual=False: aggregated_rewards [B]
            If return_individual=True: dict with "aggregated" and individual model rewards
        """
        batch_size = videos.shape[0]
        individual_rewards: dict[str, torch.Tensor] = {}

        # Collect rewards from all models
        all_rewards = []
        for i, (model, weight) in enumerate(zip(self.reward_models, self.reward_weights, strict=False)):
            reward = model.compute_reward(videos, prompts, **kwargs)
            assert reward.shape == (batch_size,), \
                f"Reward model {i} returned shape {reward.shape}, expected ({batch_size},)"

            # Optionally normalize individual rewards
            if self.normalize_rewards:
                reward = (reward - reward.mean()) / (reward.std() + 1e-8)

            individual_rewards[f"reward_{type(model).__name__}"] = reward
            all_rewards.append(weight * reward)

        # Aggregate with weights
        aggregated = sum(all_rewards)

        if return_individual:
            individual_rewards["aggregated"] = aggregated
            return individual_rewards

        return aggregated

    def __repr__(self) -> str:
        models_str = ", ".join([
            f"{type(m).__name__}(w={w:.3f})"
            for m, w in zip(self.reward_models, self.reward_weights, strict=False)
        ])
        return f"MultiRewardAggregator({models_str})"


class ValueModel(nn.Module):
    """
    Value function model wrapper for RL training.

    The value model can either:
    1. Share the transformer backbone with the policy (memory efficient)
    2. Use a separate transformer (more flexible)

    For now, this is a placeholder that will be expanded based on
    the chosen architecture strategy.
    """

    def __init__(
        self,
        transformer: nn.Module,
        share_backbone: bool = False,
        hidden_size: int | None = None
    ):
        """
        Initialize value model.

        Args:
            transformer: Transformer model (policy or separate)
            share_backbone: Whether to share backbone with policy
            hidden_size: Hidden size for value head (inferred if None)
        """
        super().__init__()
        self.transformer = transformer
        self.share_backbone = share_backbone

        # Value head will be added later based on transformer architecture
        # For now, just store the transformer reference
        logger.info(
            "Initialized ValueModel (share_backbone=%s)",
            share_backbone
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        **kwargs: Any
    ) -> torch.Tensor:
        """
        Forward pass to compute value predictions.

        Args:
            hidden_states: Latent states [B, C, T, H, W]
            encoder_hidden_states: Text embeddings [B, L, D]
            timestep: Timesteps [B]
            **kwargs: Additional transformer arguments

        Returns:
            values: Value predictions [B]
        """
        # TODO: Implement value prediction
        # For now, return dummy values
        batch_size = hidden_states.shape[0]
        return torch.zeros(batch_size, device=hidden_states.device)


class DummyRewardModel(BaseRewardModel):
    """
    Dummy VIDEO reward model for testing and development.

    Returns random rewards in the range [0, 1] for VIDEO inputs.
    This is a placeholder for testing the RL pipeline before real video reward models
    are implemented.

    NOTE: This does NOT actually evaluate video quality - it's just for testing!
    """

    def __init__(self, mean: float = 0.5, std: float = 0.1):
        super().__init__(model_path=None)
        self.mean = mean
        self.std = std
        logger.info("Initialized DummyRewardModel (VIDEO) - mean=%.2f, std=%.2f", mean, std)
        logger.warning(
            "DummyRewardModel is for TESTING ONLY - does not evaluate actual video quality!"
        )

    def compute_reward(
        self,
        videos: torch.Tensor,  # [B, T, C, H, W]
        prompts: list[str],
        **kwargs: Any
    ) -> torch.Tensor:
        """
        Return random rewards for testing.

        Args:
            videos: Video sequences [B, T, C, H, W]
            prompts: Text prompts

        Returns:
            Random rewards [B] in range [0, 1]
        """
        batch_size = videos.shape[0]
        num_frames = videos.shape[1]

        logger.debug(
            "DummyRewardModel processing %d videos with %d frames each",
            batch_size,
            num_frames
        )

        # Generate random rewards (not based on actual video content!)
        rewards = torch.randn(batch_size, device=videos.device) * self.std + self.mean
        return rewards.clamp(0.0, 1.0)

    def load_model(self) -> None:
        """No model to load for dummy."""
        pass


def create_reward_models(
    reward_model_paths: str,
    reward_weights: str,
    reward_model_types: str,
    device: str = "cuda"
) -> MultiRewardAggregator:
    """
    Factory function to create VIDEO reward models from configuration strings.

    IMPORTANT: Only creates VIDEO reward models. Image-only reward models
    (PickScore, ImageReward, GenEval, etc.) are NOT supported.

    Args:
        reward_model_paths: Comma-separated paths to VIDEO reward models
        reward_weights: Comma-separated weights for aggregation
        reward_model_types: Comma-separated VIDEO reward types
        device: Device to load models on

    Returns:
        MultiRewardAggregator with loaded VIDEO reward models

    Supported VIDEO Reward Types (Phase 2+):
        - "video_score": Video aesthetic quality (multi-frame)
        - "video_text_alignment": CLIP-based video-text similarity
        - "temporal_coherence": Frame-to-frame consistency
        - "motion_quality": Motion smoothness and realism
        - "dummy": Random rewards for testing (VIDEO-aware)

    NOT Supported (Image-Only):
        - "pickscore": Image aesthetic (use "video_score" instead)
        - "imagereward": Image quality (use "video_score" instead)
        - "geneval": Image compositional (no video equivalent yet)
        - Any single-frame reward models

    Example:
        >>> models = create_reward_models(
        ...     reward_model_paths="/path/to/video_score,/path/to/video_clip",
        ...     reward_weights="0.5,0.5",
        ...     reward_model_types="video_score,video_text_alignment",
        ...     device="cuda"
        ... )
    """
    if not reward_model_paths:
        logger.warning("No reward models specified, using DummyRewardModel (VIDEO)")
        return MultiRewardAggregator([DummyRewardModel()], [1.0])

    paths = [p.strip() for p in reward_model_paths.split(",")]
    types = [t.strip() for t in reward_model_types.split(",")]

    if reward_weights:
        weights = [float(w.strip()) for w in reward_weights.split(",")]
    else:
        weights = [1.0 / len(paths)] * len(paths)

    assert len(paths) == len(types), \
        f"Number of paths ({len(paths)}) must match number of types ({len(types)})"

    assert len(paths) == len(weights), \
        f"Number of paths ({len(paths)}) must match number of weights ({len(weights)})"

    # Validate that no image-only reward types are specified
    image_only_types = {"pickscore", "imagereward", "geneval", "aesthetic_predictor"}
    for reward_type in types:
        if reward_type.lower() in image_only_types:
            raise ValueError(
                f"Image-only reward type '{reward_type}' is not supported. "
                f"This RL implementation is VIDEO-ONLY. "
                f"Please use video-specific reward models instead."
            )

    # Create reward models based on types
    reward_models: list[BaseRewardModel] = []
    for path, reward_type in zip(paths, types, strict=False):
        if reward_type == "dummy":
            model = DummyRewardModel()
        elif reward_type == "video_score":
            # TODO: Implement VideoScore reward model (Phase 2)
            logger.warning(
                "VideoScore reward not implemented yet, using DummyRewardModel"
            )
            model = DummyRewardModel()
        elif reward_type == "video_text_alignment":
            # TODO: Implement VideoTextAlignment reward model (Phase 2)
            logger.warning(
                "VideoTextAlignment reward not implemented yet, using DummyRewardModel"
            )
            model = DummyRewardModel()
        elif reward_type == "temporal_coherence":
            # TODO: Implement TemporalCoherence reward model (Phase 2)
            logger.warning(
                "TemporalCoherence reward not implemented yet, using DummyRewardModel"
            )
            model = DummyRewardModel()
        elif reward_type == "motion_quality":
            # TODO: Implement MotionQuality reward model (Phase 2)
            logger.warning(
                "MotionQuality reward not implemented yet, using DummyRewardModel"
            )
            model = DummyRewardModel()
        else:
            logger.warning(
                "Unknown VIDEO reward type '%s', using DummyRewardModel",
                reward_type
            )
            model = DummyRewardModel()

        reward_models.append(model)

    logger.info(
        "Created MultiRewardAggregator with %d VIDEO reward models",
        len(reward_models)
    )

    return MultiRewardAggregator(reward_models, weights, normalize_rewards=True)
