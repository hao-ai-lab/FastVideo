# SPDX-License-Identifier: Apache-2.0
"""Wan causal model plugin (per-role instance, streaming/cache)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, TYPE_CHECKING

import torch

from fastvideo.forward_context import set_forward_context

from fastvideo.train.models.base import CausalModelBase
from fastvideo.train.models.wan.wan import WanModel

if TYPE_CHECKING:
    from fastvideo.train.utils.training_config import (
        TrainingConfig, )


@dataclass(slots=True)
class _StreamingCaches:
    kv_cache: list[dict[str, Any]]
    crossattn_cache: list[dict[str, Any]] | None
    frame_seq_length: int
    local_attn_size: int
    sliding_window_num_frames: int
    batch_size: int
    dtype: torch.dtype
    device: torch.device


class WanCausalModel(WanModel, CausalModelBase):
    """Wan per-role model with causal/streaming primitives."""

    _transformer_cls_name: str = ("CausalWanTransformer3DModel")

    def __init__(
        self,
        *,
        init_from: str,
        training_config: TrainingConfig,
        trainable: bool = True,
        disable_custom_init_weights: bool = False,
        flow_shift: float = 3.0,
        enable_gradient_checkpointing_type: str
        | None = None,
        transformer_override_safetensor: str
        | None = None,
        lora_rank: int | None = None,
        lora_alpha: int | None = None,
        lora_target_modules: list[str] | None = None,
    ) -> None:
        super().__init__(
            init_from=init_from,
            training_config=training_config,
            trainable=trainable,
            disable_custom_init_weights=(disable_custom_init_weights),
            flow_shift=flow_shift,
            enable_gradient_checkpointing_type=(enable_gradient_checkpointing_type),
            transformer_override_safetensor=(transformer_override_safetensor),
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_target_modules=lora_target_modules,
        )
        self._streaming_caches: (dict[tuple[int, str], _StreamingCaches]) = {}

    # --- CausalModelBase override: clear_caches ---
    def clear_caches(
        self,
        *,
        cache_tag: str = "pos",
    ) -> None:
        self._streaming_caches.pop((id(self), str(cache_tag)), None)

    # --- CausalModelBase override: predict_noise_streaming ---
    def predict_noise_streaming(
        self,
        noisy_latents: torch.Tensor,
        timestep: torch.Tensor,
        batch: Any,
        *,
        conditional: bool,
        cache_tag: str = "pos",
        store_kv: bool = False,
        cur_start_frame: int = 0,
        cfg_uncond: dict[str, Any] | None = None,
        attn_kind: Literal["dense", "vsa"] = "dense",
    ) -> torch.Tensor | None:
        if attn_kind == "dense":
            attn_metadata = batch.attn_metadata
        elif attn_kind == "vsa":
            attn_metadata = batch.attn_metadata_vsa
        else:
            raise ValueError(f"Unknown attn_kind: {attn_kind!r}")

        cache_tag = str(cache_tag)
        cur_start_frame = int(cur_start_frame)
        if cur_start_frame < 0:
            raise ValueError("cur_start_frame must be >= 0")

        batch_size = int(noisy_latents.shape[0])
        num_frames = int(noisy_latents.shape[1])
        timestep_full = self._ensure_per_frame_timestep(
            timestep=timestep,
            batch_size=batch_size,
            num_frames=num_frames,
            device=noisy_latents.device,
        )

        transformer = self._get_transformer(timestep_full)
        caches = self._get_or_init_streaming_caches(
            cache_tag=cache_tag,
            transformer=transformer,
            noisy_latents=noisy_latents,
        )

        frame_seq_length = int(caches.frame_seq_length)
        kv_cache = caches.kv_cache
        crossattn_cache = caches.crossattn_cache

        if (self._should_snapshot_streaming_cache() and torch.is_grad_enabled()):
            kv_cache = self._snapshot_kv_cache_indices(kv_cache)

        model_kwargs: dict[str, Any] = {
            "kv_cache": kv_cache,
            "crossattn_cache": crossattn_cache,
            "current_start": (cur_start_frame * frame_seq_length),
            "start_frame": cur_start_frame,
            "is_cache": bool(store_kv),
        }

        device_type = self.device.type
        dtype = noisy_latents.dtype

        if conditional:
            text_dict = batch.conditional_dict
            if text_dict is None:
                raise RuntimeError("Missing conditional_dict in "
                                   "TrainingBatch")
        else:
            text_dict = self._get_uncond_text_dict(batch, cfg_uncond=cfg_uncond)

        with (
                torch.autocast(device_type, dtype=dtype),
                set_forward_context(
                    current_timestep=batch.timesteps,
                    attn_metadata=attn_metadata,
                ),
        ):
            input_kwargs = (self._build_distill_input_kwargs(
                noisy_latents,
                timestep_full,
                text_dict,
            ))
            input_kwargs["timestep"] = (timestep_full.to(
                device=self.device,
                dtype=torch.long,
            ))
            input_kwargs.update(model_kwargs)

            if store_kv:
                with torch.no_grad():
                    _ = transformer(**input_kwargs)
                return None

            pred_noise = transformer(**input_kwargs, ).permute(0, 2, 1, 3, 4)
        return pred_noise

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_per_frame_timestep(
        self,
        *,
        timestep: torch.Tensor,
        batch_size: int,
        num_frames: int,
        device: torch.device,
    ) -> torch.Tensor:
        if timestep.ndim == 0:
            return (timestep.view(1, 1).expand(batch_size, num_frames).to(device=device))
        if timestep.ndim == 1:
            if int(timestep.shape[0]) == batch_size:
                return (timestep.view(batch_size, 1).expand(batch_size, num_frames).to(device=device))
            raise ValueError("streaming timestep must be scalar, "
                             "[B], or [B, T]; got shape="
                             f"{tuple(timestep.shape)}")
        if timestep.ndim == 2:
            return timestep.to(device=device)
        raise ValueError("streaming timestep must be scalar, "
                         "[B], or [B, T]; got ndim="
                         f"{int(timestep.ndim)}")

    def _get_or_init_streaming_caches(
        self,
        *,
        cache_tag: str,
        transformer: torch.nn.Module,
        noisy_latents: torch.Tensor,
    ) -> _StreamingCaches:
        key = (id(self), cache_tag)
        cached = self._streaming_caches.get(key)

        batch_size = int(noisy_latents.shape[0])
        dtype = noisy_latents.dtype
        device = noisy_latents.device

        frame_seq_length = (self._compute_frame_seq_length(transformer, noisy_latents))
        local_attn_size = self._get_local_attn_size(transformer)
        sliding_window_num_frames = (self._get_sliding_window_num_frames(transformer))

        meta = (
            frame_seq_length,
            local_attn_size,
            sliding_window_num_frames,
            batch_size,
            dtype,
            device,
        )

        if cached is not None:
            cached_meta = (
                cached.frame_seq_length,
                cached.local_attn_size,
                cached.sliding_window_num_frames,
                cached.batch_size,
                cached.dtype,
                cached.device,
            )
            if cached_meta == meta:
                return cached

        kv_cache = self._initialize_kv_cache(
            transformer=transformer,
            batch_size=batch_size,
            dtype=dtype,
            device=device,
            frame_seq_length=frame_seq_length,
            local_attn_size=local_attn_size,
            sliding_window_num_frames=(sliding_window_num_frames),
            checkpoint_safe=(self._should_use_checkpoint_safe_kv_cache()),
        )
        crossattn_cache = (self._initialize_crossattn_cache(
            transformer=transformer,
            device=device,
        ))

        caches = _StreamingCaches(
            kv_cache=kv_cache,
            crossattn_cache=crossattn_cache,
            frame_seq_length=frame_seq_length,
            local_attn_size=local_attn_size,
            sliding_window_num_frames=(sliding_window_num_frames),
            batch_size=batch_size,
            dtype=dtype,
            device=device,
        )
        self._streaming_caches[key] = caches
        return caches

    def _compute_frame_seq_length(
        self,
        transformer: torch.nn.Module,
        noisy_latents: torch.Tensor,
    ) -> int:
        latent_seq_length = (int(noisy_latents.shape[-1]) * int(noisy_latents.shape[-2]))
        patch_size = getattr(transformer, "patch_size", None)
        if patch_size is None:
            patch_size = getattr(
                getattr(
                    getattr(transformer, "config", None),
                    "arch_config",
                    None,
                ),
                "patch_size",
                None,
            )
        if patch_size is None:
            raise ValueError("Unable to determine "
                             "transformer.patch_size "
                             "for causal streaming")
        patch_ratio = (int(patch_size[-1]) * int(patch_size[-2]))
        if patch_ratio <= 0:
            raise ValueError("Invalid patch_size for causal "
                             "streaming")
        return latent_seq_length // patch_ratio

    def _get_sliding_window_num_frames(
        self,
        transformer: torch.nn.Module,
    ) -> int:
        cfg = getattr(transformer, "config", None)
        arch_cfg = getattr(cfg, "arch_config", None)
        value = (getattr(
            arch_cfg,
            "sliding_window_num_frames",
            None,
        ) if arch_cfg is not None else None)
        if value is None:
            return 15
        return int(value)

    def _get_local_attn_size(
        self,
        transformer: torch.nn.Module,
    ) -> int:
        try:
            value = getattr(transformer, "local_attn_size", -1)
        except Exception:
            value = -1
        if value is None:
            return -1
        return int(value)

    def _initialize_kv_cache(
        self,
        *,
        transformer: torch.nn.Module,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
        frame_seq_length: int,
        local_attn_size: int,
        sliding_window_num_frames: int,
        checkpoint_safe: bool,
    ) -> list[dict[str, Any]]:
        num_blocks = len(getattr(transformer, "blocks", []))
        if num_blocks <= 0:
            raise ValueError("Unexpected transformer.blocks "
                             "for causal streaming")

        try:
            num_attention_heads = int(transformer.num_attention_heads)  # type: ignore[attr-defined]
        except AttributeError as e:
            raise ValueError("Transformer is missing "
                             "num_attention_heads") from e

        try:
            attention_head_dim = int(transformer.attention_head_dim)  # type: ignore[attr-defined]
        except AttributeError:
            try:
                hidden_size = int(transformer.hidden_size)  # type: ignore[attr-defined]
            except AttributeError as e:
                raise ValueError("Transformer is missing "
                                 "attention_head_dim and "
                                 "hidden_size") from e
            attention_head_dim = (hidden_size // max(1, num_attention_heads))

        if local_attn_size != -1:
            kv_cache_size = (int(local_attn_size) * int(frame_seq_length))
        else:
            kv_cache_size = (int(frame_seq_length) * int(sliding_window_num_frames))

        if checkpoint_safe:
            tc = getattr(self, "training_config", None)
            total_frames = int(tc.data.num_frames if tc is not None else 0)
            if total_frames <= 0:
                raise ValueError("training.num_frames must be set "
                                 "to enable checkpoint-safe "
                                 "streaming KV cache; got "
                                 f"{total_frames}")
            kv_cache_size = max(
                kv_cache_size,
                int(frame_seq_length) * total_frames,
            )

        kv_cache: list[dict[str, Any]] = []
        for _ in range(num_blocks):
            kv_cache.append({
                "k":
                torch.zeros(
                    [
                        batch_size,
                        kv_cache_size,
                        num_attention_heads,
                        attention_head_dim,
                    ],
                    dtype=dtype,
                    device=device,
                ),
                "v":
                torch.zeros(
                    [
                        batch_size,
                        kv_cache_size,
                        num_attention_heads,
                        attention_head_dim,
                    ],
                    dtype=dtype,
                    device=device,
                ),
                "global_end_index":
                torch.zeros((), dtype=torch.long, device=device),
                "local_end_index":
                torch.zeros((), dtype=torch.long, device=device),
            })

        return kv_cache

    def _should_use_checkpoint_safe_kv_cache(self, ) -> bool:
        tc = getattr(self, "training_config", None)
        checkpointing_type = tc.model.enable_gradient_checkpointing_type if tc is not None else None
        return (bool(checkpointing_type) and bool(self._trainable))

    def _should_snapshot_streaming_cache(self, ) -> bool:
        return (self._should_use_checkpoint_safe_kv_cache())

    def _snapshot_kv_cache_indices(
        self,
        kv_cache: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        snapshot: list[dict[str, Any]] = []
        for block_cache in kv_cache:
            global_end_index = block_cache.get("global_end_index")
            local_end_index = block_cache.get("local_end_index")
            if (not isinstance(global_end_index, torch.Tensor) or not isinstance(local_end_index, torch.Tensor)):
                raise ValueError("Unexpected kv_cache index "
                                 "tensors; expected tensors at "
                                 "kv_cache[*].{global_end_index, "
                                 "local_end_index}")

            copied = dict(block_cache)
            copied["global_end_index"] = (global_end_index.detach().clone())
            copied["local_end_index"] = (local_end_index.detach().clone())
            snapshot.append(copied)
        return snapshot

    def _initialize_crossattn_cache(
        self,
        *,
        transformer: torch.nn.Module,
        device: torch.device,
    ) -> list[dict[str, Any]] | None:
        num_blocks = len(getattr(transformer, "blocks", []))
        if num_blocks <= 0:
            return None
        return [{
            "is_init": False,
            "k": torch.empty(0, device=device),
            "v": torch.empty(0, device=device),
        } for _ in range(num_blocks)]
