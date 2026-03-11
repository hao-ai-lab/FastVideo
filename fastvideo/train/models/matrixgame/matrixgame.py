# SPDX-License-Identifier: Apache-2.0
"""MatrixGame training model plugin."""

from __future__ import annotations

import copy
from typing import Any, Literal

import torch

from fastvideo.dataset.dataloader.schema import pyarrow_schema_matrixgame
from fastvideo.forward_context import set_forward_context
from fastvideo.pipelines import TrainingBatch
from fastvideo.training.training_utils import normalize_dit_input

from fastvideo.train.models.wan.wan import WanModel
from fastvideo.train.utils.dataloader import (
    build_parquet_matrixgame_train_dataloader,
)


class MatrixGameModel(WanModel):
    """MatrixGame per-role model for finetuning in the new trainer."""

    _transformer_cls_name: str = "MatrixGameWanModel"

    def init_preprocessors(self, training_config: Any) -> None:
        self.vae = self._load_vae(training_config)
        self.world_group = self._get_world_group()
        self.sp_group = self._get_sp_group()
        self._init_timestep_mechanics()
        self.dataloader = build_parquet_matrixgame_train_dataloader(
            training_config.data,
            parquet_schema=pyarrow_schema_matrixgame,
        )
        self.start_step = 0

    def on_train_start(self) -> None:
        # MatrixGame finetuning does not use negative text conditioning.
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
            dtype=dtype,
        )
        mouse_cond = self._get_optional_action(
            raw_batch,
            key="mouse_cond",
            expected_frames=self._expected_action_frames(tc.data.num_latent_t),
            dtype=dtype,
        )

        training_batch.latents = normalize_dit_input("wan", latents, self.vae)
        training_batch.encoder_hidden_states = None
        training_batch.encoder_attention_mask = None
        training_batch.preprocessed_image = pil_image
        training_batch.image_embeds = clip_feature
        training_batch.image_latents = first_frame_latent
        training_batch.keyboard_cond = keyboard_cond
        training_batch.mouse_cond = mouse_cond
        training_batch.infos = infos

        training_batch = self._prepare_dit_inputs(training_batch, generator)
        training_batch = self._build_attention_metadata(training_batch)

        training_batch.attn_metadata_vsa = copy.deepcopy(training_batch.attn_metadata)
        if training_batch.attn_metadata is not None:
            training_batch.attn_metadata.VSA_sparsity = 0.0  # type: ignore[attr-defined]

        return training_batch

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
        del cfg_uncond
        if attn_kind == "dense":
            attn_metadata = batch.attn_metadata
        elif attn_kind == "vsa":
            attn_metadata = batch.attn_metadata_vsa
        else:
            raise ValueError(f"Unknown attn_kind: {attn_kind!r}")

        cond_dict = batch.conditional_dict
        if cond_dict is None:
            raise RuntimeError("Missing conditional_dict in TrainingBatch")

        if not conditional:
            cond_dict = batch.unconditional_dict or cond_dict

        device_type = self.device.type
        dtype = noisy_latents.dtype
        with torch.autocast(device_type, dtype=dtype), set_forward_context(
            current_timestep=batch.timesteps,
            attn_metadata=attn_metadata,
        ):
            input_kwargs = self._build_distill_input_kwargs(
                noisy_latents,
                timestep,
                cond_dict,
            )
            pred_noise = self.transformer(**input_kwargs).permute(0, 2, 1, 3, 4)
        return pred_noise

    def _prepare_dit_inputs(
        self,
        training_batch: TrainingBatch,
        generator: torch.Generator,
    ) -> TrainingBatch:
        training_batch = super()._prepare_dit_inputs(training_batch, generator)

        image_latents = training_batch.image_latents
        image_embeds = training_batch.image_embeds
        if image_latents is None or image_embeds is None:
            raise RuntimeError("MatrixGame requires image_latents and image_embeds")

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
        }
        training_batch.unconditional_dict = dict(training_batch.conditional_dict)
        return training_batch

    def _build_distill_input_kwargs(
        self,
        noise_input: torch.Tensor,
        timestep: torch.Tensor,
        text_dict: dict[str, Any] | None,
    ) -> dict[str, Any]:
        if text_dict is None:
            raise ValueError("text_dict cannot be None for MatrixGame")
        return {
            "hidden_states": noise_input.permute(0, 2, 1, 3, 4),
            "encoder_hidden_states": None,
            "timestep": timestep.to(device=self.device, dtype=torch.bfloat16),
            "encoder_hidden_states_image": text_dict["encoder_hidden_states_image"],
            "keyboard_cond": text_dict["keyboard_cond"],
            "mouse_cond": text_dict["mouse_cond"],
            "return_dict": False,
        }

    def _build_matrixgame_cond_concat(
        self,
        image_latents: torch.Tensor,
    ) -> torch.Tensor:
        if image_latents.ndim != 5:
            raise ValueError(
                "first_frame_latent must have shape [B, C, T, H, W], "
                f"got {tuple(image_latents.shape)}"
            )
        if image_latents.shape[1] == 20:
            return image_latents
        if image_latents.shape[1] != 16:
            raise ValueError(
                "MatrixGame expects first_frame_latent with 16 or 20 channels, "
                f"got {image_latents.shape[1]}"
            )

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

    def _get_optional_action(
        self,
        raw_batch: dict[str, Any],
        *,
        key: str,
        expected_frames: int,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        value = raw_batch.get(key)
        if value is None or value.numel() == 0:
            return None
        if value.shape[1] < expected_frames:
            raise ValueError(
                f"{key} has {value.shape[1]} frames but requires at least {expected_frames}"
            )
        return value[:, :expected_frames].to(device=self.device, dtype=dtype)

    def _expected_action_frames(self, num_latent_t: int) -> int:
        return (num_latent_t - 1) * self._temporal_compression_ratio() + 1

    def _temporal_compression_ratio(self) -> int:
        assert self.training_config is not None
        return int(
            self.training_config.pipeline_config.vae_config.arch_config.temporal_compression_ratio  # type: ignore[union-attr]
        )

    def _infer_batch_size(self, raw_batch: dict[str, Any]) -> int:
        if "vae_latent" in raw_batch:
            return int(raw_batch["vae_latent"].shape[0])
        if "clip_feature" in raw_batch:
            return int(raw_batch["clip_feature"].shape[0])
        raise ValueError("Unable to infer batch size from MatrixGame batch")

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

    def _load_vae(self, training_config: Any) -> Any:
        from fastvideo.train.utils.moduleloader import load_module_from_path

        return load_module_from_path(
            model_path=str(training_config.model_path),
            module_type="vae",
            training_config=training_config,
        )

    def _get_world_group(self) -> Any:
        from fastvideo.distributed import get_world_group

        return get_world_group()

    def _get_sp_group(self) -> Any:
        from fastvideo.distributed import get_sp_group

        return get_sp_group()
