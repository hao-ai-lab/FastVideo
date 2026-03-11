# SPDX-License-Identifier: Apache-2.0
"""MatrixGame causal training model plugin."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import torch

from fastvideo.forward_context import set_forward_context
from fastvideo.models.dits.matrixgame.kv_cache import KVCache

from fastvideo.train.models.matrixgame.matrixgame import MatrixGameModel
from fastvideo.train.models.wan.wan_causal import WanCausalModel


@dataclass(slots=True)
class _MatrixGameStreamingCaches:
    kv_cache: list[dict[str, Any]]
    kv_cache_mouse: list[KVCache | None]
    kv_cache_keyboard: list[KVCache | None]
    crossattn_cache: list[dict[str, Any]] | None
    frame_seq_length: int
    local_attn_size: int
    sliding_window_num_frames: int
    batch_size: int
    dtype: torch.dtype
    device: torch.device


class MatrixGameCausalModel(MatrixGameModel, WanCausalModel):
    """MatrixGame per-role model with causal/streaming primitives."""

    _transformer_cls_name: str = "CausalMatrixGameWanModel"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._streaming_caches: (
            dict[tuple[int, str], _MatrixGameStreamingCaches]
        ) = {}

    def clear_caches(self, *, cache_tag: str = "pos") -> None:
        self._streaming_caches.pop((id(self), str(cache_tag)), None)

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
        del cfg_uncond
        attn_metadata = self._get_attn_metadata(batch, attn_kind=attn_kind)

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

        kv_cache = caches.kv_cache
        kv_cache_mouse = caches.kv_cache_mouse
        kv_cache_keyboard = caches.kv_cache_keyboard
        crossattn_cache = caches.crossattn_cache

        (
            kv_cache,
            kv_cache_mouse,
            kv_cache_keyboard,
        ) = self._maybe_snapshot_streaming_caches(
            kv_cache=kv_cache,
            kv_cache_mouse=kv_cache_mouse,
            kv_cache_keyboard=kv_cache_keyboard,
        )
        model_kwargs = self._build_streaming_model_kwargs(
            kv_cache=kv_cache,
            kv_cache_mouse=kv_cache_mouse,
            kv_cache_keyboard=kv_cache_keyboard,
            crossattn_cache=crossattn_cache,
            cur_start_frame=cur_start_frame,
            frame_seq_length=int(caches.frame_seq_length),
            store_kv=store_kv,
            num_frames=num_frames,
        )
        cond_dict = self._get_streaming_conditioning(
            batch,
            conditional=conditional,
        )

        device_type = self.device.type
        dtype = noisy_latents.dtype
        with (
            torch.autocast(device_type, dtype=dtype),
            set_forward_context(
                current_timestep=batch.timesteps,
                attn_metadata=attn_metadata,
            ),
        ):
            input_kwargs = self._build_streaming_input_kwargs(
                noisy_latents=noisy_latents,
                timestep=timestep_full,
                cond_dict=cond_dict,
                batch=batch,
                cur_start_frame=cur_start_frame,
            )
            input_kwargs["timestep"] = timestep_full.to(
                device=self.device,
                dtype=torch.long,
            )
            input_kwargs.update(model_kwargs)

            if store_kv:
                with torch.no_grad():
                    _ = transformer(**input_kwargs)
                return None

            pred_noise = transformer(**input_kwargs).permute(0, 2, 1, 3, 4)
        return pred_noise

    def _get_attn_metadata(
        self,
        batch: Any,
        *,
        attn_kind: Literal["dense", "vsa"],
    ) -> Any:
        if attn_kind == "dense":
            return batch.attn_metadata
        if attn_kind == "vsa":
            return batch.attn_metadata_vsa
        raise ValueError(f"Unknown attn_kind: {attn_kind!r}")

    def _maybe_snapshot_streaming_caches(
        self,
        *,
        kv_cache: list[dict[str, Any]],
        kv_cache_mouse: list[KVCache | None],
        kv_cache_keyboard: list[KVCache | None],
    ) -> tuple[list[dict[str, Any]], list[KVCache | None], list[KVCache | None]]:
        if self._should_snapshot_streaming_cache() and torch.is_grad_enabled():
            kv_cache = self._snapshot_kv_cache_indices(kv_cache)
            kv_cache_mouse = self._snapshot_action_kv_cache_lengths(kv_cache_mouse)
            kv_cache_keyboard = self._snapshot_action_kv_cache_lengths(
                kv_cache_keyboard
            )
        return kv_cache, kv_cache_mouse, kv_cache_keyboard

    def _build_streaming_model_kwargs(
        self,
        *,
        kv_cache: list[dict[str, Any]],
        kv_cache_mouse: list[KVCache | None],
        kv_cache_keyboard: list[KVCache | None],
        crossattn_cache: list[dict[str, Any]] | None,
        cur_start_frame: int,
        frame_seq_length: int,
        store_kv: bool,
        num_frames: int,
    ) -> dict[str, Any]:
        return {
            "kv_cache": kv_cache,
            "kv_cache_mouse": kv_cache_mouse,
            "kv_cache_keyboard": kv_cache_keyboard,
            "crossattn_cache": crossattn_cache,
            "current_start": cur_start_frame * frame_seq_length,
            "start_frame": cur_start_frame,
            "update_kv_cache": bool(store_kv),
            "num_frame_per_block": num_frames,
        }

    def _get_streaming_conditioning(
        self,
        batch: Any,
        *,
        conditional: bool,
    ) -> dict[str, Any]:
        cond_dict = batch.conditional_dict
        if cond_dict is None:
            raise RuntimeError("Missing conditional_dict in TrainingBatch")
        if conditional:
            return cond_dict
        return batch.unconditional_dict or cond_dict

    def _get_or_init_streaming_caches(
        self,
        *,
        cache_tag: str,
        transformer: torch.nn.Module,
        noisy_latents: torch.Tensor,
    ) -> _MatrixGameStreamingCaches:
        key = (id(self), cache_tag)
        cached = self._streaming_caches.get(key)

        batch_size = int(noisy_latents.shape[0])
        dtype = noisy_latents.dtype
        device = noisy_latents.device

        frame_seq_length = self._compute_frame_seq_length(transformer, noisy_latents)
        local_attn_size = self._get_local_attn_size(transformer)
        sliding_window_num_frames = self._get_sliding_window_num_frames(transformer)

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
            sliding_window_num_frames=sliding_window_num_frames,
            checkpoint_safe=self._should_use_checkpoint_safe_kv_cache(),
        )
        kv_cache_mouse = self._initialize_action_kv_cache(
            transformer=transformer,
            batch_size=batch_size,
            dtype=dtype,
            device=device,
            local_attn_size=local_attn_size,
            sliding_window_num_frames=sliding_window_num_frames,
            channel="mouse",
        )
        kv_cache_keyboard = self._initialize_action_kv_cache(
            transformer=transformer,
            batch_size=batch_size,
            dtype=dtype,
            device=device,
            local_attn_size=local_attn_size,
            sliding_window_num_frames=sliding_window_num_frames,
            channel="keyboard",
        )
        crossattn_cache = self._initialize_crossattn_cache(
            transformer=transformer,
            device=device,
        )

        caches = _MatrixGameStreamingCaches(
            kv_cache=kv_cache,
            kv_cache_mouse=kv_cache_mouse,
            kv_cache_keyboard=kv_cache_keyboard,
            crossattn_cache=crossattn_cache,
            frame_seq_length=frame_seq_length,
            local_attn_size=local_attn_size,
            sliding_window_num_frames=sliding_window_num_frames,
            batch_size=batch_size,
            dtype=dtype,
            device=device,
        )
        self._streaming_caches[key] = caches
        return caches

    def _build_streaming_input_kwargs(
        self,
        *,
        noisy_latents: torch.Tensor,
        timestep: torch.Tensor,
        cond_dict: dict[str, Any],
        batch: Any,
        cur_start_frame: int,
    ) -> dict[str, Any]:
        if batch.image_latents is None:
            raise RuntimeError("MatrixGame causal rollout requires image_latents")
        if batch.image_embeds is None:
            raise RuntimeError("MatrixGame causal rollout requires image_embeds")

        num_frames = int(noisy_latents.shape[1])
        frame_end = cur_start_frame + num_frames
        cond_latents = self._slice_image_latents_for_chunk(
            batch.image_latents,
            start=cur_start_frame,
            end=frame_end,
        )
        hidden_states = torch.cat(
            [noisy_latents.permute(0, 2, 1, 3, 4), cond_latents],
            dim=1,
        )

        return {
            "hidden_states": hidden_states,
            "encoder_hidden_states": None,
            "timestep": timestep,
            "encoder_hidden_states_image": cond_dict["encoder_hidden_states_image"],
            "keyboard_cond": self._slice_action_prefix(
                batch.keyboard_cond,
                frame_end=frame_end,
            ),
            "mouse_cond": self._slice_action_prefix(
                batch.mouse_cond,
                frame_end=frame_end,
            ),
            "return_dict": False,
        }

    def _slice_image_latents_for_chunk(
        self,
        image_latents: torch.Tensor,
        *,
        start: int,
        end: int,
    ) -> torch.Tensor:
        num_frames = end - start
        if image_latents.ndim != 5:
            raise ValueError(
                "image_latents must have shape [B, C, T, H, W], "
                f"got {tuple(image_latents.shape)}"
            )
        if image_latents.shape[2] >= end:
            return image_latents[:, :, start:end]
        if image_latents.shape[2] > start:
            cond = image_latents[:, :, start:]
            pad_frames = num_frames - int(cond.shape[2])
            if pad_frames <= 0:
                return cond
            pad = torch.zeros(
                cond.shape[0],
                cond.shape[1],
                pad_frames,
                cond.shape[3],
                cond.shape[4],
                device=cond.device,
                dtype=cond.dtype,
            )
            return torch.cat([cond, pad], dim=2)
        return torch.zeros(
            image_latents.shape[0],
            image_latents.shape[1],
            num_frames,
            image_latents.shape[3],
            image_latents.shape[4],
            device=image_latents.device,
            dtype=image_latents.dtype,
        )

    def _slice_action_prefix(
        self,
        action: torch.Tensor | None,
        *,
        frame_end: int,
    ) -> torch.Tensor | None:
        if action is None:
            return None
        action_frame_end = ((frame_end - 1) * self._temporal_compression_ratio()) + 1
        if action.shape[1] < action_frame_end:
            raise ValueError(
                "Action tensor is shorter than required for causal rollout: "
                f"got={action.shape[1]}, required>={action_frame_end}"
            )
        return action[:, :action_frame_end]

    def _initialize_action_kv_cache(
        self,
        *,
        transformer: torch.nn.Module,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
        local_attn_size: int,
        sliding_window_num_frames: int,
        channel: Literal["mouse", "keyboard"],
    ) -> list[KVCache | None]:
        blocks = list(getattr(transformer, "blocks", []))
        if not blocks:
            return []

        cache_size = (
            int(local_attn_size)
            if int(local_attn_size) != -1
            else int(sliding_window_num_frames)
        )
        if cache_size <= 0:
            raise ValueError(
                f"Invalid action cache size for channel {channel!r}: {cache_size}"
            )

        caches: list[KVCache | None] = []
        for block in blocks:
            action_model = getattr(block, "action_model", None)
            if action_model is None:
                caches.append(None)
                continue

            if channel == "mouse":
                enabled = bool(getattr(action_model, "enable_mouse", False))
                hidden_dim = getattr(
                    getattr(action_model, "proj_mouse", None),
                    "in_features",
                    None,
                )
            else:
                enabled = bool(getattr(action_model, "enable_keyboard", False))
                hidden_dim = getattr(
                    getattr(action_model, "proj_keyboard", None),
                    "in_features",
                    None,
                )

            if not enabled:
                caches.append(None)
                continue
            if hidden_dim is None:
                raise ValueError(
                    f"Unable to determine {channel} action hidden size for "
                    "MatrixGame causal cache initialization"
                )

            num_heads = int(getattr(action_model, "heads_num"))
            head_dim = int(hidden_dim) // max(1, num_heads)
            caches.append(
                KVCache.zeros(
                    batch_size=batch_size,
                    cache_size=cache_size,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    dtype=dtype,
                    device=device,
                )
            )
        return caches

    def _snapshot_action_kv_cache_lengths(
        self,
        caches: list[KVCache | None],
    ) -> list[KVCache | None]:
        snapshot: list[KVCache | None] = []
        for cache in caches:
            if cache is None:
                snapshot.append(None)
                continue
            snapshot.append(
                KVCache(
                    k=cache.k,
                    v=cache.v,
                    length=cache.length.detach().clone(),
                )
            )
        return snapshot
