# SPDX-License-Identifier: Apache-2.0
"""Synchformer visual conditioner used by MMAudio.

The MotionFormer implementation is shared with FastVideo's existing
audio/video synchronization evaluator. This production adapter deliberately
owns only the visual feature extractor used by MMAudio, so its state-dict and
forward numerics match the official ``Synchformer`` module without carrying
the evaluator's unused audio and classification heads.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch

from fastvideo.configs.models.encoders import BaseEncoderOutput
from fastvideo.configs.models.encoders.mmaudio_synchformer import (
    MMAudioSynchformerConfig,
)
from fastvideo.models.encoders.base import ImageEncoder
from fastvideo.models.loader.weight_utils import default_weight_loader
from fastvideo.third_party.synchformer.motionformer import MotionFormer


class MMAudioSynchformerVisualEncoder(ImageEncoder):
    """Extract temporal synchronization tokens from 25 FPS video frames."""

    def __init__(self, config: MMAudioSynchformerConfig) -> None:
        super().__init__(config)
        self.vfeat_extractor = MotionFormer(
            extract_features=True,
            factorize_space_time=True,
            agg_space_module="TransformerEncoderLayer",
            agg_time_module="torch.nn.Identity",
            add_global_repr=False,
        )

    def forward_segmented(self, segments: torch.Tensor) -> torch.Tensor:
        """Encode ``[B, S, 16, 3, 224, 224]`` frame segments."""
        if segments.ndim != 6:
            raise ValueError(f"Synchformer segments must have shape [B, S, T, C, H, W], got {tuple(segments.shape)}")
        _, _, frames, channels, height, width = segments.shape
        expected = self.config.arch_config
        if (
            frames != expected.segment_size
            or channels != expected.num_channels
            or height != expected.image_size
            or width != expected.image_size
        ):
            raise ValueError(
                "Synchformer requires segments shaped "
                f"[B, S, {expected.segment_size}, {expected.num_channels}, "
                f"{expected.image_size}, {expected.image_size}], got "
                f"{tuple(segments.shape)}"
            )
        visual = segments.permute(0, 1, 3, 2, 4, 5)
        return self.vfeat_extractor(visual)

    def forward(
        self,
        pixel_values: torch.Tensor,
        **kwargs,
    ) -> BaseEncoderOutput:
        """Encode contiguous ``[B, T, 3, 224, 224]`` 25 FPS frames."""
        del kwargs
        if pixel_values.ndim != 5:
            raise ValueError(f"Synchformer video must have shape [B, T, C, H, W], got {tuple(pixel_values.shape)}")

        config = self.config.arch_config
        frame_count = pixel_values.shape[1]
        if frame_count < config.segment_size:
            raise ValueError(f"Synchformer needs at least {config.segment_size} frames, got {frame_count}")

        segments = (
            pixel_values.unfold(
                dimension=1,
                size=config.segment_size,
                step=config.segment_stride,
            )
            .permute(0, 1, 5, 2, 3, 4)
            .contiguous()
        )
        features = self.forward_segmented(segments)
        features = features.flatten(1, 2)
        return BaseEncoderOutput(last_hidden_state=features)

    def load_weights(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> set[str]:
        params = dict(self.named_parameters())
        loaded: set[str] = set()
        for name, tensor in weights:
            if not name.startswith("vfeat_extractor.") or name not in params:
                continue
            parameter = params[name]
            weight_loader = getattr(parameter, "weight_loader", default_weight_loader)
            weight_loader(parameter, tensor)
            loaded.add(name)
        return loaded


EntryClass = MMAudioSynchformerVisualEncoder
