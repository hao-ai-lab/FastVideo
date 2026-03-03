# SPDX-License-Identifier: Apache-2.0

"""WanGame causal model plugin (streaming/cache primitives).

This module provides the *causal* extension for the WanGame model family.

Key differences vs. `models/wangame/wangame.py`:
- Supports `roles.<role>.variant: causal` by loading a causal transformer class.
- Implements `CausalModelBase` APIs (`clear_caches`, `predict_*_streaming`) so
  methods can drive streaming rollouts without passing KV-cache tensors around.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import torch

from fastvideo.forward_context import set_forward_context
from fastvideo.models.utils import pred_noise_to_pred_video

from fastvideo.distillation.models.base import CausalModelBase
from fastvideo.distillation.roles import RoleHandle
from fastvideo.distillation.utils.config import DistillRunConfig

from fastvideo.distillation.models.wangame.wangame import (
    WanGameModel,
    _build_wangame_role_handles,
)


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


class WanGameCausalModel(WanGameModel, CausalModelBase):
    """WanGame model plugin with optional causal/streaming primitives."""

    def __init__(self, *, cfg: DistillRunConfig) -> None:
        training_args = cfg.training_args
        roles_cfg = cfg.roles

        if getattr(training_args, "seed", None) is None:
            raise ValueError("training.seed must be set for distillation")
        if not getattr(training_args, "data_path", ""):
            raise ValueError("training.data_path must be set for distillation")

        # Load shared components (student base path).
        vae = self._load_shared_vae(training_args)
        noise_scheduler = self._build_noise_scheduler(training_args)

        def _transformer_cls_name_for_role(role: str, role_spec: Any) -> str:
            variant_raw = (role_spec.extra or {}).get("variant", None)
            if variant_raw is None or str(variant_raw).strip() == "":
                return "WanGameActionTransformer3DModel"

            variant = str(variant_raw).strip().lower()
            if variant in {"bidirectional", "bidi"}:
                return "WanGameActionTransformer3DModel"
            if variant == "causal":
                return "CausalWanGameActionTransformer3DModel"
            raise ValueError(
                f"Unknown roles.{role}.variant for wangame: {variant_raw!r}. "
                "Expected 'causal' or 'bidirectional'."
            )

        role_handles = _build_wangame_role_handles(
            roles_cfg=roles_cfg,
            training_args=training_args,
            transformer_cls_name_for_role=_transformer_cls_name_for_role,
        )

        # NOTE: re-run the rest of WanGameModel init without rebuilding roles.
        self._init_from_built_roles(
            cfg=cfg,
            role_handles=role_handles,
            vae=vae,
            noise_scheduler=noise_scheduler,
        )

        self._streaming_caches: dict[tuple[int, str], _StreamingCaches] = {}

    # --- CausalModelBase override: clear_caches ---
    def clear_caches(self, handle: RoleHandle, *, cache_tag: str = "pos") -> None:
        self._streaming_caches.pop((id(handle), str(cache_tag)), None)

    # --- CausalModelBase override: predict_noise_streaming ---
    def predict_noise_streaming(
        self,
        handle: RoleHandle,
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

        # Ensure per-frame timestep shape [B, T] (mirrors pipeline causal inference).
        batch_size = int(noisy_latents.shape[0])
        num_frames = int(noisy_latents.shape[1])
        timestep_full = self._ensure_per_frame_timestep(
            timestep=timestep,
            batch_size=batch_size,
            num_frames=num_frames,
            device=noisy_latents.device,
        )

        transformer = self._get_transformer(handle, timestep_full)
        caches = self._get_or_init_streaming_caches(
            handle=handle,
            cache_tag=cache_tag,
            transformer=transformer,
            noisy_latents=noisy_latents,
        )

        frame_seq_length = int(caches.frame_seq_length)
        model_kwargs: dict[str, Any] = {
            "kv_cache": caches.kv_cache,
            "crossattn_cache": caches.crossattn_cache,
            "current_start": cur_start_frame * frame_seq_length,
            "start_frame": cur_start_frame,
            "is_cache": bool(store_kv),
        }

        device_type = self.device.type
        dtype = noisy_latents.dtype
        with torch.autocast(device_type, dtype=dtype), set_forward_context(
            current_timestep=batch.timesteps,
            attn_metadata=attn_metadata,
        ):
            cond_inputs = self._select_cfg_condition_inputs(
                batch,
                conditional=conditional,
                cfg_uncond=cfg_uncond,
            )
            cond_inputs = self._slice_cond_inputs_for_streaming(
                cond_inputs=cond_inputs,
                cur_start_frame=cur_start_frame,
                num_frames=num_frames,
            )
            input_kwargs = self._build_distill_input_kwargs(
                noisy_latents,
                timestep_full,
                image_embeds=cond_inputs["image_embeds"],
                image_latents=cond_inputs["image_latents"],
                mask_lat_size=cond_inputs["mask_lat_size"],
                viewmats=cond_inputs["viewmats"],
                Ks=cond_inputs["Ks"],
                action=cond_inputs["action"],
                mouse_cond=cond_inputs["mouse_cond"],
                keyboard_cond=cond_inputs["keyboard_cond"],
            )

            # Override timestep dtype: causal inference expects integer timesteps.
            input_kwargs["timestep"] = timestep_full.to(
                device=self.device, dtype=torch.long
            )
            input_kwargs.update(model_kwargs)

            if store_kv:
                with torch.no_grad():
                    _ = transformer(**input_kwargs)
                return None

            pred_noise = transformer(**input_kwargs).permute(0, 2, 1, 3, 4)
        return pred_noise

    # --- CausalModelBase override: predict_x0_streaming ---
    def predict_x0_streaming(
        self,
        handle: RoleHandle,
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
        pred_noise = self.predict_noise_streaming(
            handle,
            noisy_latents,
            timestep,
            batch,
            conditional=conditional,
            cache_tag=cache_tag,
            store_kv=store_kv,
            cur_start_frame=cur_start_frame,
            cfg_uncond=cfg_uncond,
            attn_kind=attn_kind,
        )
        if pred_noise is None:
            return None

        pred_x0 = pred_noise_to_pred_video(
            pred_noise=pred_noise.flatten(0, 1),
            noise_input_latent=noisy_latents.flatten(0, 1),
            timestep=self.shift_and_clamp_timestep(
                self._ensure_per_frame_timestep(
                    timestep=timestep,
                    batch_size=int(noisy_latents.shape[0]),
                    num_frames=int(noisy_latents.shape[1]),
                    device=noisy_latents.device,
                ).flatten()
            ),
            scheduler=self.noise_scheduler,
        ).unflatten(0, pred_noise.shape[:2])
        return pred_x0

    # --- internal helpers ---
    def _load_shared_vae(self, training_args: Any) -> torch.nn.Module:
        from fastvideo.distillation.utils.moduleloader import load_module_from_path

        return load_module_from_path(
            model_path=str(training_args.model_path),
            module_type="vae",
            training_args=training_args,
        )

    def _build_noise_scheduler(self, training_args: Any):
        from fastvideo.models.schedulers.scheduling_flow_match_euler_discrete import (
            FlowMatchEulerDiscreteScheduler,
        )

        return FlowMatchEulerDiscreteScheduler(
            shift=float(training_args.pipeline_config.flow_shift or 0.0)
        )

    def _init_from_built_roles(
        self,
        *,
        cfg: DistillRunConfig,
        role_handles: dict[str, RoleHandle],
        vae: torch.nn.Module,
        noise_scheduler: Any,
    ) -> None:
        # This is a small, explicit extraction of `WanGameModel.__init__` so the
        # causal model can reuse all non-causal primitives without duplicating
        # the full class body.
        training_args = cfg.training_args

        self.bundle = self._build_bundle(role_handles)

        self.validator = None
        validation_cfg = getattr(cfg, "validation", {}) or {}
        validation_enabled = bool(validation_cfg.get("enabled", bool(validation_cfg)))
        if validation_enabled:
            from fastvideo.distillation.validators.wangame import WanGameValidator

            self.validator = WanGameValidator(training_args=training_args)

        self.training_args = training_args
        self.noise_scheduler = noise_scheduler
        self.vae = vae

        from fastvideo.distributed import (
            get_local_torch_device,
            get_sp_group,
            get_world_group,
        )

        self.world_group = get_world_group()
        self.sp_group = get_sp_group()
        self.device = get_local_torch_device()

        self.noise_random_generator = None
        self.noise_gen_cuda = None

        self._init_timestep_mechanics()

        from fastvideo.dataset.dataloader.schema import pyarrow_schema_wangame
        from fastvideo.distillation.utils.dataloader import (
            build_parquet_wangame_train_dataloader,
        )

        self.dataloader = build_parquet_wangame_train_dataloader(
            training_args,
            parquet_schema=pyarrow_schema_wangame,
        )
        self.start_step = 0

    def _build_bundle(self, role_handles: dict[str, RoleHandle]):
        from fastvideo.distillation.roles import RoleManager

        return RoleManager(roles=role_handles)

    def _ensure_per_frame_timestep(
        self,
        *,
        timestep: torch.Tensor,
        batch_size: int,
        num_frames: int,
        device: torch.device,
    ) -> torch.Tensor:
        if timestep.ndim == 0:
            return timestep.view(1, 1).expand(batch_size, num_frames).to(device=device)
        if timestep.ndim == 1:
            if int(timestep.shape[0]) == batch_size:
                return timestep.view(batch_size, 1).expand(batch_size, num_frames).to(device=device)
            raise ValueError(
                "streaming timestep must be scalar, [B], or [B, T]; got "
                f"shape={tuple(timestep.shape)}"
            )
        if timestep.ndim == 2:
            return timestep.to(device=device)
        raise ValueError(
            "streaming timestep must be scalar, [B], or [B, T]; got "
            f"ndim={int(timestep.ndim)}"
        )

    def _slice_cond_inputs_for_streaming(
        self,
        *,
        cond_inputs: dict[str, Any],
        cur_start_frame: int,
        num_frames: int,
    ) -> dict[str, Any]:
        start = int(cur_start_frame)
        num_frames = int(num_frames)
        if num_frames <= 0:
            raise ValueError("num_frames must be positive for streaming")
        if start < 0:
            raise ValueError("cur_start_frame must be >= 0 for streaming")
        end = start + num_frames

        sliced: dict[str, Any] = dict(cond_inputs)

        image_latents = cond_inputs.get("image_latents")
        if isinstance(image_latents, torch.Tensor):
            sliced["image_latents"] = image_latents[:, :, start:end]

        mask_lat_size = cond_inputs.get("mask_lat_size")
        if isinstance(mask_lat_size, torch.Tensor):
            sliced["mask_lat_size"] = mask_lat_size[:, :, start:end]

        viewmats = cond_inputs.get("viewmats")
        if isinstance(viewmats, torch.Tensor):
            sliced["viewmats"] = viewmats[:, start:end]

        Ks = cond_inputs.get("Ks")
        if isinstance(Ks, torch.Tensor):
            sliced["Ks"] = Ks[:, start:end]

        action = cond_inputs.get("action")
        if isinstance(action, torch.Tensor):
            sliced["action"] = action[:, start:end]

        temporal_compression_ratio = int(
            self.training_args.pipeline_config.vae_config.arch_config.temporal_compression_ratio
        )
        raw_end_frame_idx = 1 + temporal_compression_ratio * max(0, end - 1)

        mouse_cond = cond_inputs.get("mouse_cond")
        if isinstance(mouse_cond, torch.Tensor):
            sliced["mouse_cond"] = mouse_cond[:, :raw_end_frame_idx]

        keyboard_cond = cond_inputs.get("keyboard_cond")
        if isinstance(keyboard_cond, torch.Tensor):
            sliced["keyboard_cond"] = keyboard_cond[:, :raw_end_frame_idx]

        return sliced

    def _get_or_init_streaming_caches(
        self,
        *,
        handle: RoleHandle,
        cache_tag: str,
        transformer: torch.nn.Module,
        noisy_latents: torch.Tensor,
    ) -> _StreamingCaches:
        key = (id(handle), cache_tag)
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
        )
        crossattn_cache = self._initialize_crossattn_cache(transformer=transformer, device=device)

        caches = _StreamingCaches(
            kv_cache=kv_cache,
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

    def _compute_frame_seq_length(self, transformer: torch.nn.Module, noisy_latents: torch.Tensor) -> int:
        latent_seq_length = int(noisy_latents.shape[-1]) * int(noisy_latents.shape[-2])
        patch_size = getattr(transformer, "patch_size", None)
        if patch_size is None:
            patch_size = getattr(getattr(transformer, "config", None), "arch_config", None)
            patch_size = getattr(patch_size, "patch_size", None)
        if patch_size is None:
            raise ValueError("Unable to determine transformer.patch_size for causal streaming")
        patch_ratio = int(patch_size[-1]) * int(patch_size[-2])
        if patch_ratio <= 0:
            raise ValueError("Invalid patch_size for causal streaming")
        return latent_seq_length // patch_ratio

    def _get_sliding_window_num_frames(self, transformer: torch.nn.Module) -> int:
        cfg = getattr(transformer, "config", None)
        arch_cfg = getattr(cfg, "arch_config", None)
        value = getattr(arch_cfg, "sliding_window_num_frames", None) if arch_cfg is not None else None
        if value is None:
            return 15
        return int(value)

    def _get_local_attn_size(self, transformer: torch.nn.Module) -> int:
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
    ) -> list[dict[str, Any]]:
        num_blocks = len(getattr(transformer, "blocks", []))
        if num_blocks <= 0:
            raise ValueError("Unexpected transformer.blocks for causal streaming")

        try:
            num_attention_heads = int(transformer.num_attention_heads)  # type: ignore[attr-defined]
        except AttributeError as e:
            raise ValueError("Transformer is missing num_attention_heads") from e

        try:
            attention_head_dim = int(transformer.attention_head_dim)  # type: ignore[attr-defined]
        except AttributeError:
            try:
                hidden_size = int(transformer.hidden_size)  # type: ignore[attr-defined]
            except AttributeError as e:
                raise ValueError("Transformer is missing attention_head_dim and hidden_size") from e
            attention_head_dim = hidden_size // max(1, num_attention_heads)

        if local_attn_size != -1:
            kv_cache_size = int(local_attn_size) * int(frame_seq_length)
        else:
            kv_cache_size = int(frame_seq_length) * int(sliding_window_num_frames)

        kv_cache: list[dict[str, Any]] = []
        for _ in range(num_blocks):
            kv_cache.append(
                {
                    "k": torch.zeros(
                        [batch_size, kv_cache_size, num_attention_heads, attention_head_dim],
                        dtype=dtype,
                        device=device,
                    ),
                    "v": torch.zeros(
                        [batch_size, kv_cache_size, num_attention_heads, attention_head_dim],
                        dtype=dtype,
                        device=device,
                    ),
                    "global_end_index": torch.zeros((), dtype=torch.long, device=device),
                    "local_end_index": torch.zeros((), dtype=torch.long, device=device),
                }
            )

        return kv_cache

    def _initialize_crossattn_cache(
        self,
        *,
        transformer: torch.nn.Module,
        device: torch.device,
    ) -> list[dict[str, Any]] | None:
        # WanGame uses image conditioning; caching the image K/V is optional but
        # helps avoid repeated projections across timesteps in a rollout.
        num_blocks = len(getattr(transformer, "blocks", []))
        if num_blocks <= 0:
            return None
        return [
            {"is_init": False, "k": torch.empty(0, device=device), "v": torch.empty(0, device=device)}
            for _ in range(num_blocks)
        ]
