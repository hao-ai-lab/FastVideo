# SPDX-License-Identifier: Apache-2.0
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

from typing import Any

from abc import ABC, abstractmethod
import torch
import torch.nn as nn

class BaseRewardModel(ABC, nn.Module):
    def __init__(self, model_path: str | None = None, device: str = "cuda"):
        super().__init__()
        self.model_path = model_path
        self.device = device

    @abstractmethod
    def compute_reward(
        self,
        videos: torch.Tensor,  # [B, T, C, H, W] decoded video sequences
        prompts: list[str] | None,  # Text prompts
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
        """
        raise NotImplementedError("Subclasses must implement compute_reward()")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model_path={self.model_path})"