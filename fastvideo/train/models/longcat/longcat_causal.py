# SPDX-License-Identifier: Apache-2.0
"""LongCat streaming model plugin for self-forcing rollouts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, TYPE_CHECKING

import torch

from fastvideo.forward_context import set_forward_context
from fastvideo.train.models.base import CausalModelBase
from fastvideo.train.models.longcat.longcat import LongCatModel

if TYPE_CHECKING:
    from fastvideo.pipelines import TrainingBatch
    from fastvideo.train.utils.training_config import TrainingConfig


@dataclass(slots=True)
class _LongCatStreamingCache:
    kv_cache: dict[int, tuple[torch.Tensor, torch.Tensor]]
    start_frame: int
    cached_frames: int
    tokens_per_frame: int


class LongCatCausalModel(LongCatModel, CausalModelBase):
    """LongCat model with chunk-wise KV cache accumulation."""

    def __init__(
        self,
        *,
        init_from: str,
        training_config: TrainingConfig,
        trainable: bool = True,
        disable_custom_init_weights: bool = False,
        flow_shift: float = 12.0,
        enable_gradient_checkpointing_type: str | None = None,
        transformer_override_safetensor: str | None = None,
        max_kv_cache_frames: int | None = None,
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
        if max_kv_cache_frames is None:
            self._max_kv_cache_frames = int(training_config.data.num_frames)
        else:
            if int(max_kv_cache_frames) <= 0:
                raise ValueError("max_kv_cache_frames must be > 0")
            self._max_kv_cache_frames = int(max_kv_cache_frames)
        self._streaming_caches: dict[str, _LongCatStreamingCache] = {}

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
        cache_state = self._streaming_caches.get(cache_tag)
        kv_cache_dict = cache_state.kv_cache if cache_state is not None else None
        kv_cache_start_frame = (
            cache_state.start_frame if cache_state is not None else 0
        )
        cached_frames = cache_state.cached_frames if cache_state is not None else 0

        if self._should_snapshot_streaming_cache() and torch.is_grad_enabled():
            kv_cache_dict = self._snapshot_kv_cache_dict(kv_cache_dict)

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
                    num_cond_latents=cached_frames,
                    return_kv=True,
                    skip_crs_attn=True,
                )
                if new_cache is None:
                    raise RuntimeError("LongCat transformer returned no KV cache")
                self._streaming_caches[cache_tag] = self._merge_kv_cache_state(
                    existing=cache_state,
                    new=new_cache,
                    new_frames=int(noisy_latents.shape[1]),
                )
                return None

            pred_noise = transformer(
                hidden_states=noisy_latents.permute(0, 2, 1, 3, 4),
                encoder_hidden_states=text_dict["encoder_hidden_states"],
                encoder_attention_mask=text_dict["encoder_attention_mask"],
                timestep=timestep,
                num_cond_latents=cached_frames,
                kv_cache_dict=kv_cache_dict,
                kv_cache_start_frame=kv_cache_start_frame,
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

    def _should_use_checkpoint_safe_kv_cache(self) -> bool:
        tc = getattr(self, "training_config", None)
        checkpointing_type = (
            tc.model.enable_gradient_checkpointing_type
            if tc is not None else None
        )
        return bool(checkpointing_type) and bool(self._trainable)

    def _should_snapshot_streaming_cache(self) -> bool:
        return self._should_use_checkpoint_safe_kv_cache()

    def _snapshot_kv_cache_dict(
        self,
        kv_cache_dict: dict[int, tuple[torch.Tensor, torch.Tensor]] | None,
    ) -> dict[int, tuple[torch.Tensor, torch.Tensor]] | None:
        if kv_cache_dict is None:
            return None

        return {
            idx: (k.detach().clone(), v.detach().clone())
            for idx, (k, v) in kv_cache_dict.items()
        }

    def _merge_kv_cache_state(
        self,
        existing: _LongCatStreamingCache | None,
        new: dict[int, tuple[torch.Tensor, torch.Tensor]],
        new_frames: int,
    ) -> _LongCatStreamingCache:
        if int(new_frames) <= 0:
            raise ValueError("new_frames must be > 0")

        tokens_per_frame = self._infer_tokens_per_frame(
            cache_dict=new,
            num_frames=int(new_frames),
        )

        if existing is None:
            merged = {
                idx: (k.contiguous(), v.contiguous())
                for idx, (k, v) in new.items()
            }
            start_frame = 0
            cached_frames = int(new_frames)
        else:
            if existing.tokens_per_frame != tokens_per_frame:
                raise ValueError(
                    "LongCat KV cache token density changed between chunks: "
                    f"{existing.tokens_per_frame} vs {tokens_per_frame}"
                )
            merged = {}
            for idx in sorted(set(existing.kv_cache) | set(new)):
                if idx not in existing.kv_cache:
                    merged[idx] = new[idx]
                    continue
                if idx not in new:
                    merged[idx] = existing.kv_cache[idx]
                    continue

                prev_k, prev_v = existing.kv_cache[idx]
                new_k, new_v = new[idx]
                merged[idx] = (
                    torch.cat((prev_k, new_k), dim=2),
                    torch.cat((prev_v, new_v), dim=2),
                )
            start_frame = int(existing.start_frame)
            cached_frames = int(existing.cached_frames + new_frames)

        max_cache_frames = min(
            self._max_kv_cache_frames,
            int(self.training_config.data.num_frames),
        )
        if cached_frames > max_cache_frames:
            frames_to_drop = int(cached_frames - max_cache_frames)
            tokens_to_drop = int(frames_to_drop * tokens_per_frame)
            for idx, (k, v) in merged.items():
                if k.shape[2] < tokens_to_drop or v.shape[2] < tokens_to_drop:
                    raise ValueError(
                        "LongCat KV cache eviction would drop more tokens "
                        "than are present in the cache"
                    )
                merged[idx] = (
                    k[:, :, tokens_to_drop:, :].contiguous(),
                    v[:, :, tokens_to_drop:, :].contiguous(),
                )
            start_frame += frames_to_drop
            cached_frames = max_cache_frames

        return _LongCatStreamingCache(
            kv_cache=merged,
            start_frame=start_frame,
            cached_frames=cached_frames,
            tokens_per_frame=tokens_per_frame,
        )

    def _infer_tokens_per_frame(
        self,
        *,
        cache_dict: dict[int, tuple[torch.Tensor, torch.Tensor]],
        num_frames: int,
    ) -> int:
        if not cache_dict:
            raise ValueError("LongCat returned an empty KV cache")
        if num_frames <= 0:
            raise ValueError("num_frames must be > 0")

        sample_k, _ = next(iter(cache_dict.values()))
        if sample_k.ndim != 4:
            raise ValueError(
                "Unexpected LongCat KV cache shape; expected [B, H, N, D], "
                f"got ndim={sample_k.ndim}"
            )
        num_tokens = int(sample_k.shape[2])
        if num_tokens % num_frames != 0:
            raise ValueError(
                "LongCat KV cache token count is not divisible by the number "
                f"of frames in the cached chunk: {num_tokens} vs {num_frames}"
            )
        return num_tokens // num_frames
