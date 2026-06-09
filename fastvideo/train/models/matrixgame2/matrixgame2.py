# SPDX-License-Identifier: Apache-2.0
"""Matrix-Game 2.0 training model plugin."""

from __future__ import annotations

import copy
from typing import Any, Literal

import torch

from fastvideo.dataset.dataloader.schema import pyarrow_schema_matrixgame2
from fastvideo.distributed import (
    get_sp_group,
    get_world_group,
)
from fastvideo.models.dits.matrixgame2.utils import scale_keyboard_condition
from fastvideo.pipelines import TrainingBatch
from fastvideo.training.training_utils import normalize_dit_input

from fastvideo.train.models.wan.wan import WanModel
from fastvideo.train.utils.dataloader import (
    build_parquet_matrixgame2_train_dataloader, )
from fastvideo.train.utils.moduleloader import (
    load_module_from_path, )


class MatrixGame2Model(WanModel):
    """Matrix-Game 2.0 per-role model for finetuning in the new trainer."""

    _transformer_cls_name: str = "MatrixGame2WanModel"

    def init_preprocessors(self, training_config: Any) -> None:
        self.vae = load_module_from_path(
            model_path=str(training_config.model_path),
            module_type="vae",
            training_config=training_config,
        )
        self.world_group = get_world_group()
        self.sp_group = get_sp_group()
        self._init_timestep_mechanics()
        self.dataloader = build_parquet_matrixgame2_train_dataloader(
            training_config.data,
            parquet_schema=pyarrow_schema_matrixgame2,
        )
        self.start_step = 0

    def on_train_start(self) -> None:
        # Matrix-Game 2.0 finetuning does not use negative text conditioning.
        return

    def prepare_batch(
        self,
        raw_batch: dict[str, Any],
        *,
        generator: torch.Generator,
        latents_source: Literal["data", "zeros"] = "data",
    ) -> TrainingBatch:
        assert self.training_config is not None
        tc = self.training_config
        dtype = self._get_training_dtype()
        device = self.device

        training_batch = TrainingBatch()
        infos = raw_batch.get("info_list")
        batch_size = self._infer_batch_size(raw_batch)

        if latents_source == "zeros":
            latents = self._make_zero_latents(batch_size=batch_size)
        elif latents_source == "data":
            latents = raw_batch["vae_latent"][:, :, :tc.data.num_latent_t]
            latents = latents.to(device=device, dtype=dtype)
        else:
            raise ValueError(f"Unknown latents_source: {latents_source!r}")

        clip_feature = raw_batch["clip_feature"].to(device=device, dtype=dtype)
        first_frame_latent = raw_batch["first_frame_latent"]
        first_frame_latent = first_frame_latent[:, :, :tc.data.num_latent_t]
        first_frame_latent = first_frame_latent.to(device=device, dtype=dtype)

        pil_image = raw_batch.get("pil_image")
        if pil_image is not None:
            pil_image = pil_image.to(device=device)

        keyboard_cond = self._get_optional_action(
            raw_batch,
            key="keyboard_cond",
            expected_frames=self._expected_action_frames(tc.data.num_latent_t),
            expected_dim=self._expected_action_dim("keyboard"),
            dtype=dtype,
        )
        mouse_cond = self._get_optional_action(
            raw_batch,
            key="mouse_cond",
            expected_frames=self._expected_action_frames(tc.data.num_latent_t),
            expected_dim=self._expected_action_dim("mouse"),
            dtype=dtype,
        )

        training_batch.latents = latents
        training_batch.encoder_hidden_states = None
        training_batch.encoder_attention_mask = None
        training_batch.preprocessed_image = pil_image
        training_batch.image_embeds = clip_feature
        training_batch.image_latents = first_frame_latent
        training_batch.keyboard_cond = keyboard_cond
        training_batch.mouse_cond = mouse_cond
        training_batch.infos = infos

        training_batch.latents = normalize_dit_input(
            "wan",
            training_batch.latents,
            self.vae,
        )
        training_batch = self._prepare_dit_inputs(training_batch, generator)
        training_batch = self._build_attention_metadata(training_batch)

        # Shallow copy keeps the lru_cache'd LongTensor index fields shared
        # with the original metadata; only the float ``VSA_sparsity`` differs
        # between the two views. deepcopy here would materialize a fresh copy
        # of all four cached index tensors on every training step.
        training_batch.attn_metadata_vsa = copy.copy(training_batch.attn_metadata)
        if training_batch.attn_metadata is not None:
            training_batch.attn_metadata.VSA_sparsity = 0.0  # type: ignore[attr-defined]

        return training_batch

    def _prepare_dit_inputs(
        self,
        training_batch: TrainingBatch,
        generator: torch.Generator,
    ) -> TrainingBatch:
        training_batch = super()._prepare_dit_inputs(training_batch, generator)

        image_latents = training_batch.image_latents
        image_embeds = training_batch.image_embeds
        if image_latents is None or image_embeds is None:
            raise RuntimeError("Matrix-Game 2.0 requires image_latents and image_embeds")

        cond_latents = self._build_matrixgame_cond_concat(image_latents)
        training_batch.image_latents = cond_latents
        training_batch.noisy_model_input = torch.cat(
            [training_batch.noisy_model_input, cond_latents],
            dim=1,
        )
        training_batch.conditional_dict = {
            "encoder_hidden_states": None,
            "encoder_attention_mask": None,
            "encoder_hidden_states_image": image_embeds,
            "keyboard_cond": training_batch.keyboard_cond,
            "mouse_cond": training_batch.mouse_cond,
            "image_latents": cond_latents,
        }
        training_batch.unconditional_dict = dict(training_batch.conditional_dict)
        return training_batch

    def _get_uncond_text_dict(
        self,
        batch: TrainingBatch,
        *,
        cfg_uncond: dict[str, Any] | None,
    ) -> dict[str, Any]:
        del cfg_uncond
        cond_dict = batch.conditional_dict
        if cond_dict is None:
            raise RuntimeError("Missing conditional_dict in TrainingBatch")
        return batch.unconditional_dict or cond_dict

    def _build_distill_input_kwargs(
        self,
        noise_input: torch.Tensor,
        timestep: torch.Tensor,
        text_dict: dict[str, Any] | None,
    ) -> dict[str, Any]:
        if text_dict is None:
            raise ValueError("text_dict cannot be None for Matrix-Game 2.0")
        hidden_states = noise_input.permute(0, 2, 1, 3, 4)
        if hidden_states.shape[1] == 16:
            # Streaming long tuning scores a window at a time; use the
            # window-specific first-image latent when the method provides it.
            cond_latents = self._get_score_window_image_latents(
                text_dict,
                batch_info=text_dict.get("_batch_info"),
                num_latents=hidden_states.shape[2],
            )
            if cond_latents is None:
                raise RuntimeError("Matrix-Game 2.0 requires image_latents in conditional_dict "
                                   "when noise_input has 16 channels")
            hidden_states = torch.cat([hidden_states, cond_latents], dim=1)
        # For streaming long tuning, align actions to the same score window as
        # the latent/image condition; otherwise this falls back to the prefix.
        keyboard_cond, mouse_cond = self._get_score_window_actions(
            text_dict,
            batch_info=text_dict.get("_batch_info"),
            num_latents=hidden_states.shape[2],
        )
        return {
            "hidden_states": hidden_states,
            "encoder_hidden_states": None,
            "timestep": timestep.to(device=self.device, dtype=torch.bfloat16),
            "encoder_hidden_states_image": text_dict["encoder_hidden_states_image"],
            "keyboard_cond": keyboard_cond,
            "mouse_cond": mouse_cond,
            "return_dict": False,
        }

    def predict_noise(
        self,
        noisy_latents: torch.Tensor,
        timestep: torch.Tensor,
        batch: TrainingBatch,
        *,
        conditional: bool,
        cfg_uncond: dict[str, Any] | None = None,
        attn_kind: Literal["dense", "vsa"] = "dense",
    ) -> torch.Tensor:
        # Streaming long tuning re-anchors each score window on its first
        # image/latent, so pass window metadata through existing dict plumbing.
        info = getattr(batch, "matrixgame_streaming_chunk_info", None)
        for text_dict in (batch.conditional_dict, batch.unconditional_dict):
            if isinstance(text_dict, dict):
                text_dict["_batch_info"] = info
        try:
            return super().predict_noise(
                noisy_latents,
                timestep,
                batch,
                conditional=conditional,
                cfg_uncond=cfg_uncond,
                attn_kind=attn_kind,
            )
        finally:
            for text_dict in (batch.conditional_dict, batch.unconditional_dict):
                if isinstance(text_dict, dict):
                    text_dict.pop("_batch_info", None)

    def _build_matrixgame_cond_concat(
        self,
        image_latents: torch.Tensor,
    ) -> torch.Tensor:
        if image_latents.ndim != 5:
            raise ValueError("first_frame_latent must have shape [B, C, T, H, W], "
                             f"got {tuple(image_latents.shape)}")
        if image_latents.shape[1] == 20:
            return image_latents
        if image_latents.shape[1] != 16:
            raise ValueError("Matrix-Game 2.0 expects first_frame_latent with 16 or 20 channels, "
                             f"got {image_latents.shape[1]}")

        temporal_compression_ratio = self._temporal_compression_ratio()
        batch_size, _, num_latent_t, latent_height, latent_width = image_latents.shape
        num_frames = (num_latent_t - 1) * temporal_compression_ratio + 1

        mask_lat_size = torch.ones(
            batch_size,
            1,
            num_frames,
            latent_height,
            latent_width,
            device=image_latents.device,
            dtype=image_latents.dtype,
        )
        mask_lat_size[:, :, 1:] = 0
        first_frame_mask = mask_lat_size[:, :, :1]
        first_frame_mask = torch.repeat_interleave(
            first_frame_mask,
            dim=2,
            repeats=temporal_compression_ratio,
        )
        mask_lat_size = torch.cat(
            [first_frame_mask, mask_lat_size[:, :, 1:]],
            dim=2,
        )
        mask_lat_size = mask_lat_size.view(
            batch_size,
            -1,
            temporal_compression_ratio,
            latent_height,
            latent_width,
        ).transpose(1, 2)
        return torch.cat([mask_lat_size, image_latents], dim=1)

    def _get_score_window_image_latents(
        self,
        text_dict: dict[str, Any],
        *,
        batch_info: dict[str, Any] | None,
        num_latents: int,
    ) -> torch.Tensor | None:
        # Prefer a window-specific first-image latent when long tuning has
        # recomputed one; otherwise keep the original MG first-frame condition.
        if batch_info is not None:
            condition_latents = batch_info.get("condition_image_latents")
            if condition_latents is not None:
                if condition_latents.shape[2] != num_latents:
                    raise ValueError("condition_image_latents temporal length must match "
                                     "noise_input: "
                                     f"{condition_latents.shape[2]} vs {num_latents}")
                return condition_latents

        image_latents = text_dict.get("image_latents")
        if image_latents is None:
            return None
        return image_latents[:, :, :num_latents]

    def _get_score_window_actions(
        self,
        text_dict: dict[str, Any],
        *,
        batch_info: dict[str, Any] | None,
        num_latents: int,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        frame_start = 0
        if batch_info is not None:
            frame_start = int(batch_info.get("frame_start", 0) or 0)
        frame_end = frame_start + int(num_latents)
        action_frame_start = frame_start * self._temporal_compression_ratio()
        action_frame_end = ((frame_end - 1) * self._temporal_compression_ratio()) + 1

        def _slice_action(action: torch.Tensor | None) -> torch.Tensor | None:
            if action is None:
                return None
            if action.shape[1] < action_frame_end:
                raise ValueError("Matrix-Game 2.0 score action tensor is shorter than "
                                 "the requested streaming window: "
                                 f"got={action.shape[1]}, required>={action_frame_end}")
            return action[:, action_frame_start:action_frame_end]

        return (
            _slice_action(text_dict.get("keyboard_cond")),
            _slice_action(text_dict.get("mouse_cond")),
        )

    def _get_optional_action(
        self,
        raw_batch: dict[str, Any],
        *,
        key: str,
        expected_frames: int,
        expected_dim: int,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        value = raw_batch.get(key)
        if value is None or value.numel() == 0:
            return None
        # Actions must follow the same first-image window as the latent
        # condition; allow only padded zero channels during format alignment.
        value = self._align_action_feature_dim(
            value,
            name=key,
            expected_dim=expected_dim,
        )
        if value.shape[1] < expected_frames:
            raise ValueError(f"{key} has {value.shape[1]} frames but requires at least {expected_frames}")
        return value[:, :expected_frames].to(device=self.device, dtype=dtype)

    def prepare_validation_action_condition(
        self,
        action: torch.Tensor,
        *,
        name: str,
    ) -> torch.Tensor:
        channel = name.removesuffix("_cond")
        action = self._align_action_feature_dim(
            action,
            name=name,
            expected_dim=self._expected_action_dim(channel),
        )
        if name == "keyboard_cond":
            pipeline_config = getattr(self.training_config, "pipeline_config", None)
            keyboard_value_scale = getattr(pipeline_config, "keyboard_value_scale", 1.0)
            action = scale_keyboard_condition(action, float(keyboard_value_scale))
        return action

    def post_process_validation_frames(
        self,
        frames: list[Any],
        *,
        action: dict[str, Any] | None = None,
    ) -> list[Any] | None:
        if action is None:
            return None
        from fastvideo.models.dits.matrixgame2.utils import (
            overlay_validation_actions_on_frames, )

        return overlay_validation_actions_on_frames(
            frames,
            keyboard_cond=action.get("keyboard"),
            mouse_cond=action.get("mouse"),
        )

    def _expected_action_dim(self, channel: str) -> int:
        action_config = getattr(self.transformer, "action_config", {}) or {}
        return int(action_config.get(f"{channel}_dim_in", 0) or 0)

    def _align_action_feature_dim(
        self,
        action: torch.Tensor,
        *,
        name: str,
        expected_dim: int,
    ) -> torch.Tensor:
        if expected_dim <= 0:
            return action
        actual_dim = int(action.shape[-1])
        if actual_dim == expected_dim:
            return action
        if actual_dim > expected_dim:
            extra = action[..., expected_dim:]
            if bool(torch.count_nonzero(extra).item() == 0):
                return action[..., :expected_dim]
            raise ValueError(f"{name} feature dim mismatch: got={actual_dim}, "
                             f"expected={expected_dim}, and trailing channels are not all zero.")
        raise ValueError(f"{name} feature dim mismatch: got={actual_dim}, expected={expected_dim}")

    def _expected_action_frames(self, num_latent_t: int) -> int:
        return (num_latent_t - 1) * self._temporal_compression_ratio() + 1

    def _temporal_compression_ratio(self) -> int:
        assert self.training_config is not None
        return int(self.training_config.pipeline_config.vae_config.arch_config.
                   temporal_compression_ratio  # type: ignore[union-attr]
                   )

    def _infer_batch_size(self, raw_batch: dict[str, Any]) -> int:
        if "vae_latent" in raw_batch:
            return int(raw_batch["vae_latent"].shape[0])
        if "clip_feature" in raw_batch:
            return int(raw_batch["clip_feature"].shape[0])
        raise ValueError("Unable to infer batch size from Matrix-Game 2.0 batch")

    def _make_zero_latents(self, *, batch_size: int) -> torch.Tensor:
        assert self.training_config is not None
        vae_config = self.training_config.pipeline_config.vae_config.arch_config  # type: ignore[union-attr]
        num_channels = vae_config.z_dim
        spatial_compression_ratio = vae_config.spatial_compression_ratio
        latent_height = self.training_config.data.num_height // spatial_compression_ratio
        latent_width = self.training_config.data.num_width // spatial_compression_ratio
        return torch.zeros(
            batch_size,
            num_channels,
            self.training_config.data.num_latent_t,
            latent_height,
            latent_width,
            device=self.device,
            dtype=self._get_training_dtype(),
        )
