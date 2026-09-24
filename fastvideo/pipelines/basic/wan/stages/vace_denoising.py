# SPDX-License-Identifier: Apache-2.0
"""VACE denoising hooks on top of the shared Wan dense loop."""

from dataclasses import dataclass

import torch

from fastvideo.pipelines.basic.wan.stages.denoising import WanDenoisingStage, WanDenoisingState
from fastvideo.pipelines.stages.denoising import DenoisingState


@dataclass
class WanVACEDenoisingState(WanDenoisingState):
    control_hidden_states: torch.Tensor | None = None
    control_hidden_states_scale: float | list[float] | torch.Tensor | None = None


class WanVACEDenoisingStage(WanDenoisingStage):

    def _format_conditioning_scale(
        self,
        scale: float | list[float] | torch.Tensor | None,
        target_dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """Normalize per-VACE-layer scales to a device tensor for the DiT forward."""
        num_layers = len(self.transformer.vace_layers)
        if scale is None:
            scale_tensor = torch.ones(num_layers, device=device, dtype=target_dtype)
        elif isinstance(scale, (int, float)):
            scale_tensor = torch.full((num_layers, ), float(scale), device=device, dtype=target_dtype)
        elif isinstance(scale, list):
            scale_tensor = torch.tensor([float(value) for value in scale], device=device, dtype=target_dtype)
        elif isinstance(scale, torch.Tensor):
            scale_tensor = scale.detach().to(device=device, dtype=target_dtype)
        else:
            raise TypeError(f"Unsupported VACE conditioning_scale type: {type(scale)!r}")
        if scale_tensor.numel() != num_layers:
            raise ValueError(f"VACE conditioning_scale length {scale_tensor.numel()} != {num_layers}")
        return scale_tensor.reshape(num_layers)

    def prepare_model_input(self, latents, batch, target_dtype, state):
        # VACE passes pixel control video via ``batch.video_latent`` and packs
        # DiT conditioning through ``control_hidden_states`` instead of channel concat.
        return latents.to(target_dtype)

    def prepare_denoising(self, batch, fastvideo_args, target_dtype) -> WanVACEDenoisingState:
        base_state = super().prepare_denoising(batch, fastvideo_args, target_dtype)
        return WanVACEDenoisingState(
            **vars(base_state),
            control_hidden_states=batch.vace_control_latents,
            control_hidden_states_scale=batch.conditioning_scale,
        )

    def prepare_family_transformer_kwargs(self, batch, state: DenoisingState, target_dtype) -> dict:
        if not isinstance(state, WanVACEDenoisingState) or state.control_hidden_states is None:
            return {}
        device = state.control_hidden_states.device
        control = state.control_hidden_states.to(target_dtype)
        return {
            "control_hidden_states": control,
            "control_hidden_states_scale": self._format_conditioning_scale(state.control_hidden_states_scale,
                                                                           target_dtype,
                                                                           device),
        }
