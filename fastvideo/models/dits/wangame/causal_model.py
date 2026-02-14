# SPDX-License-Identifier: Apache-2.0
from typing import Any

import torch

from fastvideo.configs.models.dits.wangamevideo import WanGameVideoConfig
from fastvideo.models.dits.matrixgame.causal_model import CausalMatrixGameWanModel

_DEFAULT_WANGAME_CAUSAL_CONFIG = WanGameVideoConfig()


class CausalWanTransformer3DModel(CausalMatrixGameWanModel):
    supports_action_input = False

    _fsdp_shard_conditions = _DEFAULT_WANGAME_CAUSAL_CONFIG._fsdp_shard_conditions
    _compile_conditions = _DEFAULT_WANGAME_CAUSAL_CONFIG._compile_conditions
    _supported_attention_backends = _DEFAULT_WANGAME_CAUSAL_CONFIG._supported_attention_backends
    param_names_mapping = _DEFAULT_WANGAME_CAUSAL_CONFIG.param_names_mapping
    reverse_param_names_mapping = _DEFAULT_WANGAME_CAUSAL_CONFIG.reverse_param_names_mapping
    lora_param_names_mapping = _DEFAULT_WANGAME_CAUSAL_CONFIG.lora_param_names_mapping

    def __init__(self,
                 config: WanGameVideoConfig,
                 hf_config: dict[str, Any],
                 **kwargs) -> None:
        super().__init__(config=config, hf_config=hf_config, **kwargs)

    def _normalize_action_inputs(
        self,
        mouse_cond: torch.Tensor | None,
        keyboard_cond: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        # drop
        if len(getattr(self, "action_config", {})) == 0:
            return None, None
        return mouse_cond, keyboard_cond

    def _forward_inference(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | list[torch.Tensor],
        timestep: torch.LongTensor,
        encoder_hidden_states_image: torch.Tensor | list[torch.Tensor] | None = None,
        mouse_cond: torch.Tensor | None = None,
        keyboard_cond: torch.Tensor | None = None,
        kv_cache: dict | None = None,
        kv_cache_mouse: dict | None = None,
        kv_cache_keyboard: dict | None = None,
        crossattn_cache: dict | None = None,
        current_start: int = 0,
        cache_start: int = 0,
        start_frame: int = 0,
        **kwargs,
    ) -> torch.Tensor:
        mouse_cond, keyboard_cond = self._normalize_action_inputs(
            mouse_cond, keyboard_cond)
        return super()._forward_inference(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep,
            encoder_hidden_states_image=encoder_hidden_states_image,
            mouse_cond=mouse_cond,
            keyboard_cond=keyboard_cond,
            kv_cache=kv_cache,
            kv_cache_mouse=kv_cache_mouse,
            kv_cache_keyboard=kv_cache_keyboard,
            crossattn_cache=crossattn_cache,
            current_start=current_start,
            cache_start=cache_start,
            start_frame=start_frame,
            **kwargs,
        )

    def _forward_train(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | list[torch.Tensor],
        timestep: torch.LongTensor,
        encoder_hidden_states_image: torch.Tensor | list[torch.Tensor] | None = None,
        mouse_cond: torch.Tensor | None = None,
        keyboard_cond: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        mouse_cond, keyboard_cond = self._normalize_action_inputs(
            mouse_cond, keyboard_cond)
        return super()._forward_train(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep,
            encoder_hidden_states_image=encoder_hidden_states_image,
            mouse_cond=mouse_cond,
            keyboard_cond=keyboard_cond,
            **kwargs,
        )


class CausalWanGameTransformer3DModel(CausalWanTransformer3DModel):
    pass
