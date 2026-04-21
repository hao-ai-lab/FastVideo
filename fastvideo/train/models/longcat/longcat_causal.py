# SPDX-License-Identifier: Apache-2.0
"""LongCat streaming model plugin for self-forcing rollouts."""

from __future__ import annotations

from typing import Any, Literal, TYPE_CHECKING

import torch

from fastvideo.forward_context import set_forward_context
from fastvideo.train.models.base import CausalModelBase
from fastvideo.train.models.longcat.longcat import LongCatModel

if TYPE_CHECKING:
    from fastvideo.pipelines import TrainingBatch
    from fastvideo.train.utils.training_config import TrainingConfig


class LongCatCausalModel(LongCatModel, CausalModelBase):
    """LongCat model with chunk-wise KV cache accumulation."""

    def __init__(
        self,
        *,
        init_from: str,
        training_config: TrainingConfig,
        trainable: bool = True,
        disable_custom_init_weights: bool = False,
        flow_shift: float = 0.0,
        enable_gradient_checkpointing_type: str | None = None,
        transformer_override_safetensor: str | None = None,
    ) -> None:
        super().__init__(
            init_from=init_from,
            training_config=training_config,
            trainable=trainable,
            disable_custom_init_weights=disable_custom_init_weights,
            flow_shift=flow_shift,
            enable_gradient_checkpointing_type=enable_gradient_checkpointing_type,
            transformer_override_safetensor=transformer_override_safetensor,
        )
        self._streaming_caches: dict[str, dict[int, tuple[torch.Tensor, torch.Tensor]]] = {}

    def clear_caches(self, *, cache_tag: str = "pos") -> None:
        self._streaming_caches.pop(str(cache_tag), None)

    def predict_noise_streaming(
        self,
        noisy_latents: torch.Tensor,
        timestep: torch.Tensor,
        batch: TrainingBatch,
        *,
        conditional: bool,
        cache_tag: str = "pos",
        store_kv: bool = False,
        cur_start_frame: int = 0,
        cfg_uncond: dict[str, Any] | None = None,
        attn_kind: Literal["dense", "vsa"] = "dense",
    ) -> torch.Tensor | None:
        if attn_kind not in {"dense", "vsa"}:
            raise ValueError(f"Unknown attn_kind: {attn_kind!r}")
        # SelfForcingMethod passes "vsa" for student rollout, but LongCat
        # selects sparse behavior through its native BSA config, not VSA metadata.
        cache_tag = str(cache_tag)
        cur_start_frame = int(cur_start_frame)
        if cur_start_frame < 0:
            raise ValueError("cur_start_frame must be >= 0")

        if conditional:
            text_dict = batch.conditional_dict
            if text_dict is None:
                raise RuntimeError("Missing conditional_dict in TrainingBatch")
        else:
            text_dict = self._get_uncond_text_dict(batch, cfg_uncond=cfg_uncond)

        transformer = self.transformer
        dtype = noisy_latents.dtype
        kv_cache_dict = self._streaming_caches.get(cache_tag)

        with torch.autocast(self.device.type, dtype=dtype), set_forward_context(
            current_timestep=batch.timesteps,
            attn_metadata=None,
        ):
            if store_kv:
                empty_embeds = self._empty_kv_cache_inputs(
                    transformer=transformer,
                    batch_size=int(noisy_latents.shape[0]),
                    device=noisy_latents.device,
                    dtype=dtype,
                )
                _, new_cache = transformer(
                    hidden_states=noisy_latents.permute(0, 2, 1, 3, 4),
                    encoder_hidden_states=empty_embeds,
                    timestep=timestep,
                    num_cond_latents=cur_start_frame,
                    return_kv=True,
                    skip_crs_attn=True,
                )
                if new_cache is None:
                    raise RuntimeError("LongCat transformer returned no KV cache")
                self._streaming_caches[cache_tag] = self._merge_kv_cache_dicts(
                    kv_cache_dict,
                    new_cache,
                )
                return None

            pred_noise = transformer(
                hidden_states=noisy_latents.permute(0, 2, 1, 3, 4),
                encoder_hidden_states=text_dict["encoder_hidden_states"],
                encoder_attention_mask=text_dict["encoder_attention_mask"],
                timestep=timestep,
                num_cond_latents=cur_start_frame,
                kv_cache_dict=kv_cache_dict,
            )
            if isinstance(pred_noise, tuple):
                raise RuntimeError("LongCat transformer returned a tuple "
                                   "when return_kv=False")
        return pred_noise.permute(0, 2, 1, 3, 4)

    def _empty_kv_cache_inputs(
        self,
        *,
        transformer: torch.nn.Module,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        base_transformer = getattr(transformer, "module", transformer)
        config = getattr(base_transformer, "config", None)
        caption_dim = int(getattr(config, "caption_channels", 4096))
        text_len = self._get_text_len(default=512)
        return torch.zeros(
            batch_size,
            text_len,
            caption_dim,
            device=device,
            dtype=dtype,
        )

    def _get_text_len(self, *, default: int) -> int:
        pipeline_config = getattr(self.training_config, "pipeline_config", None)
        text_configs = getattr(pipeline_config, "text_encoder_configs", None)
        if text_configs:
            arch_config = getattr(text_configs[0], "arch_config", None)
            value = getattr(arch_config, "text_len", None)
            if value is not None:
                return int(value)
        return int(default)

    def _merge_kv_cache_dicts(
        self,
        existing: dict[int, tuple[torch.Tensor, torch.Tensor]] | None,
        new: dict[int, tuple[torch.Tensor, torch.Tensor]],
    ) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
        if not existing:
            return dict(new)

        merged: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        for idx in sorted(set(existing) | set(new)):
            if idx not in existing:
                merged[idx] = new[idx]
                continue
            if idx not in new:
                merged[idx] = existing[idx]
                continue

            prev_k, prev_v = existing[idx]
            new_k, new_v = new[idx]
            merged[idx] = (
                torch.cat((prev_k, new_k), dim=2),
                torch.cat((prev_v, new_v), dim=2),
            )
        return merged
