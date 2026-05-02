# SPDX-License-Identifier: Apache-2.0
"""MatrixGame model plugin for YAML training."""

from __future__ import annotations

import copy
from typing import Any, Literal

import torch

from fastvideo.pipelines import TrainingBatch
from fastvideo.training.training_utils import (
    normalize_dit_input,
)
from fastvideo.train.models.wan.wan import (
    WanModel,
)


class MatrixGameModel(WanModel):
    """MatrixGame model plugin.

    This reuses the Wan training mechanics but swaps in the MatrixGame
    transformer class and action-conditioned batch handling.
    """

    _transformer_cls_name: str = "MatrixGameWanModel"

    def init_preprocessors(self, training_config) -> None:  # type: ignore[override]
        self.vae = self._load_vae(training_config)

        from fastvideo.distributed import (
            get_sp_group,
            get_world_group,
        )
        from fastvideo.dataset.dataloader.schema import (
            pyarrow_schema_matrixgame,
        )
        from fastvideo.train.utils.dataloader import (
            build_parquet_matrixgame_train_dataloader,
        )

        self.world_group = get_world_group()
        self.sp_group = get_sp_group()
        self.training_config = training_config
        self._init_timestep_mechanics()

        text_len = (
            training_config.pipeline_config.text_encoder_configs[0].arch_config.text_len  # type: ignore[union-attr]
        )
        self.dataloader = build_parquet_matrixgame_train_dataloader(
            training_config.data,
            text_len=int(text_len),
            parquet_schema=pyarrow_schema_matrixgame,
        )
        self.start_step = 0

    def _load_vae(self, training_config):
        from fastvideo.train.utils.moduleloader import (
            load_module_from_path,
        )

        return load_module_from_path(
            model_path=str(training_config.model_path),
            module_type="vae",
            training_config=training_config,
        )

    def _uses_image_conditioning(self) -> bool:
        return True

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
        image_embeds = raw_batch.get("clip_feature")
        image_latents = raw_batch.get("first_frame_latent")
        preprocessed_image = raw_batch.get("pil_image")
        mouse_cond = raw_batch.get("mouse_cond")
        keyboard_cond = raw_batch.get("keyboard_cond")

        if image_embeds is None or image_latents is None:
            raise ValueError(
                "MatrixGame training requires clip_feature and "
                "first_frame_latent in the batch. Please use a "
                "matrixgame-preprocessed parquet dataset."
            )

        if latents_source == "zeros":
            batch_size = image_embeds.shape[0]
            vae_config = (
                tc.pipeline_config.vae_config.arch_config  # type: ignore[union-attr]
            )
            num_channels = vae_config.z_dim
            spatial_compression_ratio = vae_config.spatial_compression_ratio
            latent_height = tc.data.num_height // spatial_compression_ratio
            latent_width = tc.data.num_width // spatial_compression_ratio
            latents = torch.zeros(
                batch_size,
                num_channels,
                tc.data.num_latent_t,
                latent_height,
                latent_width,
                device=device,
                dtype=dtype,
            )
        elif latents_source == "data":
            if "vae_latent" not in raw_batch:
                raise ValueError(
                    "vae_latent not found in batch and "
                    "latents_source='data'"
                )
            latents = raw_batch["vae_latent"][:, :, :tc.data.num_latent_t]
            latents = latents.to(device, dtype=dtype)
        else:
            raise ValueError(f"Unknown latents_source: {latents_source!r}")

        training_batch.latents = latents
        training_batch.encoder_hidden_states = None
        training_batch.encoder_attention_mask = None
        training_batch.infos = infos
        training_batch.image_embeds = image_embeds.to(device, dtype=dtype)
        training_batch.image_latents = image_latents[
            :,
            :,
            :tc.data.num_latent_t,
        ].to(device, dtype=dtype)

        if torch.is_tensor(preprocessed_image) and preprocessed_image.numel() > 0:
            training_batch.preprocessed_image = preprocessed_image.to(
                device,
                dtype=dtype,
            )

        if torch.is_tensor(mouse_cond) and mouse_cond.numel() > 0:
            training_batch.mouse_cond = mouse_cond[:, :tc.data.num_frames].to(
                device,
                dtype=dtype,
            )

        if torch.is_tensor(keyboard_cond) and keyboard_cond.numel() > 0:
            training_batch.keyboard_cond = keyboard_cond[
                :,
                :tc.data.num_frames,
            ].to(device, dtype=dtype)

        training_batch.latents = normalize_dit_input(
            "wan",
            training_batch.latents,
            self.vae,
        )
        training_batch = self._prepare_dit_inputs(training_batch, generator)
        training_batch = self._build_attention_metadata(training_batch)

        training_batch.attn_metadata_vsa = copy.deepcopy(
            training_batch.attn_metadata
        )
        if training_batch.attn_metadata is not None:
            training_batch.attn_metadata.VSA_sparsity = 0.0  # type: ignore[attr-defined]

        return training_batch

    def _prepare_dit_inputs(
        self,
        training_batch: TrainingBatch,
        generator: torch.Generator,
    ) -> TrainingBatch:
        training_batch = super()._prepare_dit_inputs(
            training_batch,
            generator,
        )

        cond = training_batch.conditional_dict or {}
        if training_batch.mouse_cond is not None:
            cond["mouse_cond"] = training_batch.mouse_cond
        if training_batch.keyboard_cond is not None:
            cond["keyboard_cond"] = training_batch.keyboard_cond
        training_batch.conditional_dict = cond

        uncond = training_batch.unconditional_dict
        if uncond is not None:
            if training_batch.mouse_cond is not None:
                uncond["mouse_cond"] = training_batch.mouse_cond
            if training_batch.keyboard_cond is not None:
                uncond["keyboard_cond"] = training_batch.keyboard_cond
            training_batch.unconditional_dict = uncond

        return training_batch

    def _build_distill_input_kwargs(
        self,
        noise_input: torch.Tensor,
        timestep: torch.Tensor,
        text_dict: dict[str, Any] | None,
    ) -> dict[str, Any]:
        kwargs = super()._build_distill_input_kwargs(
            noise_input,
            timestep,
            text_dict,
        )
        if text_dict is None:
            return kwargs
        mouse_cond = text_dict.get("mouse_cond")
        keyboard_cond = text_dict.get("keyboard_cond")
        if mouse_cond is not None:
            kwargs["mouse_cond"] = mouse_cond
        if keyboard_cond is not None:
            kwargs["keyboard_cond"] = keyboard_cond
        return kwargs

    def _get_uncond_text_dict(
        self,
        batch: TrainingBatch,
        *,
        cfg_uncond: dict[str, Any] | None,
    ) -> dict[str, torch.Tensor]:
        out = super()._get_uncond_text_dict(
            batch,
            cfg_uncond=cfg_uncond,
        )
        if batch.mouse_cond is not None:
            out["mouse_cond"] = batch.mouse_cond
        if batch.keyboard_cond is not None:
            out["keyboard_cond"] = batch.keyboard_cond
        return out
