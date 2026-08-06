# SPDX-License-Identifier: Apache-2.0
"""FastVideo pipeline for MAGI-2 Preview text-to-video and image-to-video."""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from fastvideo.fastvideo_args import FastVideoArgs, TrainingArgs
from fastvideo.models.dits.magi2_runtime import psm
from fastvideo.models.schedulers.scheduling_flow_unipc_multistep import (
    FlowUniPCMultistepScheduler,
)
from fastvideo.pipelines.basic.magi2.stages import (
    Magi2AudioDecodingStage,
    Magi2DataProxyConfig,
    Magi2InputValidationStage,
    Magi2LatentPreparationStage,
    Magi2LatentSavingStage,
    Magi2PreviewDenoisingStage,
    Magi2ReferenceImageStage,
    Magi2RefinerDataProxyConfig,
    Magi2RefinerStage,
    Magi2TextEncodingStage,
    Magi2VideoDecodingStage,
)
from fastvideo.pipelines.composed_pipeline_base import ComposedPipelineBase


def _configure_deterministic_kernels(seed: int) -> None:
    """Enable the official deterministic attention, MoE, and PyTorch paths."""
    os.environ["MAGI2_DETERMINISTIC"] = "1"
    os.environ["MAGI_ATTENTION_DETERMINISTIC_MODE"] = "1"
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


class Magi2Pipeline(ComposedPipelineBase):
    """Generate a 10-second video and stereo audio with MAGI-2 Preview."""

    is_video_pipeline = True
    _required_config_modules = [
        "transformer",
        "transformer_2",
        "text_encoder",
        "image_encoder",
        "vae",
        "audio_vae",
        "scheduler",
    ]

    def __init__(
        self,
        model_path: str,
        fastvideo_args: FastVideoArgs | TrainingArgs,
        required_config_modules: list[str] | None = None,
        loaded_modules: dict[str, torch.nn.Module] | None = None,
    ) -> None:
        """Configure deterministic kernels before distributed model imports."""
        deterministic_environment = os.environ.get(
            "MAGI2_DETERMINISTIC",
            "0",
        ) == "1"
        if fastvideo_args.deterministic or deterministic_environment:
            _configure_deterministic_kernels(seed=42)
        fastvideo_args.dit_layerwise_offload = False
        fastvideo_args.dit_cpu_offload = True
        super().__init__(
            model_path=model_path,
            fastvideo_args=fastvideo_args,
            required_config_modules=required_config_modules,
            loaded_modules=loaded_modules,
        )

    def load_modules(
        self,
        fastvideo_args: FastVideoArgs,
        loaded_modules: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Strictly load official components while keeping decoder copies on one rank."""
        self._load_config(self.model_path)
        from fastvideo.configs.models.encoders import Magi2Qwen35Config
        from fastvideo.configs.models.vaes import Magi2TurboVAEConfig
        from fastvideo.models.dits.magi2_loader import (
            load_magi2_preview_model,
            load_magi2_refiner_model,
        )
        from fastvideo.models.dits.magi2_runtime.fastvideo_parallel import (
            bind_fastvideo_parallel_state,
        )
        from fastvideo.models.encoders.qwen3_5 import Magi2Qwen35TextEncoder
        from fastvideo.models.vaes.magi2_audio_vae import load_magi2_audio_vae
        from fastvideo.models.vaes.magi2_turbo_vae import Magi2TurboVAEModel
        from fastvideo.models.vaes.magi2_wan_loader import (
            load_magi2_wan_image_encoder,
        )

        bind_fastvideo_parallel_state()
        checkpoint_root = Path(self.model_path)
        pipeline_config = fastvideo_args.pipeline_config
        provided_modules = loaded_modules or {}
        modules: dict[str, Any] = {}

        modules["transformer"] = provided_modules.get("transformer")
        if modules["transformer"] is None:
            modules["transformer"] = load_magi2_preview_model(
                str(checkpoint_root / "transformer"),
                pipeline_config.dit_config,
                "cpu",
            )
        modules["transformer_2"] = provided_modules.get("transformer_2")
        if modules["transformer_2"] is None:
            modules["transformer_2"] = load_magi2_refiner_model(
                str(checkpoint_root / "transformer_2"),
                pipeline_config.refiner_dit_config,
                "cpu",
            )

        is_decode_rank = psm.is_group_first_rank("cp")
        modules["text_encoder"] = provided_modules.get("text_encoder")
        modules["image_encoder"] = provided_modules.get("image_encoder")
        modules["vae"] = provided_modules.get("vae")
        modules["audio_vae"] = provided_modules.get("audio_vae")
        if is_decode_rank and modules["text_encoder"] is None:
            modules["text_encoder"] = Magi2Qwen35TextEncoder.from_pretrained_local(
                str(checkpoint_root / "text_encoder"),
                Magi2Qwen35Config(),
                torch.bfloat16,
                torch.device("cpu"),
            )
        if is_decode_rank and modules["image_encoder"] is None:
            modules["image_encoder"] = load_magi2_wan_image_encoder(
                checkpoint_root / "image_encoder" / "Wan2.2_VAE.pth",
                "cpu",
            )
        if is_decode_rank and modules["vae"] is None:
            turbo_config = Magi2TurboVAEConfig(
                config_path=str(
                    checkpoint_root
                    / "vae"
                    / "TurboV3-Wan22-TinyShallow_7_7.json"
                ),
                checkpoint_path=str(checkpoint_root / "vae" / "checkpoint.ckpt"),
                pretrained_dtype="bfloat16",
            )
            modules["vae"] = Magi2TurboVAEModel(turbo_config).to("cpu")
            torch.cuda.empty_cache()
        if is_decode_rank and modules["audio_vae"] is None:
            modules["audio_vae"] = load_magi2_audio_vae(
                checkpoint_root / "audio_vae",
                "cpu",
            )
        modules["scheduler"] = provided_modules.get(
            "scheduler",
            FlowUniPCMultistepScheduler(),
        )
        return modules

    def create_pipeline_stages(self, fastvideo_args: FastVideoArgs) -> None:
        """Compose conditioning, denoising, refinement, and decoding stages."""
        pipeline_config = fastvideo_args.pipeline_config
        self.add_stage(
            "input_validation_stage",
            Magi2InputValidationStage(
                output_frames=pipeline_config.output_frames,
                output_height=pipeline_config.output_height,
                output_width=pipeline_config.output_width,
                output_fps=pipeline_config.output_fps,
            ),
        )
        self.add_stage(
            "reference_image_stage",
            Magi2ReferenceImageStage(
                image_encoder=self.get_module("image_encoder"),
                preview_height=pipeline_config.preview_height,
                preview_width=pipeline_config.preview_width,
            ),
        )
        self.add_stage(
            "text_encoding_stage",
            Magi2TextEncodingStage(self.get_module("text_encoder")),
        )
        self.add_stage(
            "latent_preparation_stage",
            Magi2LatentPreparationStage(
                video_channels=pipeline_config.preview_video_channels,
                video_length=pipeline_config.preview_video_length,
                video_height=pipeline_config.preview_latent_height,
                video_width=pipeline_config.preview_latent_width,
                audio_length=pipeline_config.audio_latent_length,
                audio_channels=pipeline_config.audio_channels,
            ),
        )
        self.add_stage(
            "preview_denoising_stage",
            Magi2PreviewDenoisingStage(
                transformer=self.get_module("transformer"),
                data_proxy_config=Magi2DataProxyConfig(),
                flow_shift=pipeline_config.preview_flow_shift,
                video_guidance_scale=pipeline_config.preview_video_guidance_scale,
                audio_guidance_scale=pipeline_config.preview_audio_guidance_scale,
            ),
        )
        self.add_stage(
            "refiner_stage",
            Magi2RefinerStage(
                transformer=self.get_module("transformer_2"),
                data_proxy_config=Magi2RefinerDataProxyConfig(
                    t_patch_size=1,
                    patch_size=1,
                    frame_receptive_field=11,
                    spatial_rope_interpolation="extra",
                    text_offset=0,
                    coords_style="v1",
                    attn_config={
                        "mode": "window",
                        "block_t_size": 8,
                        "block_size": 4,
                        "window": {
                            "level": "block",
                            "block_mode": "grid",
                            "block_t_radius": 2,
                            "block_h_radius": 2,
                            "block_w_radius": 2,
                            "win_size": 384,
                            "frame_receptive_field": -1,
                            "auto_range_merge": True,
                            "sparse_load": False,
                            "full_attn_layers": [],
                        },
                    },
                    magi2_refiner_condition_input="none",
                ),
                latent_height=pipeline_config.refiner_latent_height,
                latent_width=pipeline_config.refiner_latent_width,
                noise_index=pipeline_config.refiner_noise_index,
                flow_shift=pipeline_config.refiner_flow_shift,
                video_guidance_scale=pipeline_config.refiner_video_guidance_scale,
                audio_guidance_scale=pipeline_config.refiner_audio_guidance_scale,
                audio_channels=pipeline_config.audio_channels,
            ),
        )
        self.add_stage("latent_saving_stage", Magi2LatentSavingStage())
        self.add_stage(
            "video_decoding_stage",
            Magi2VideoDecodingStage(self.get_module("vae")),
        )
        self.add_stage(
            "audio_decoding_stage",
            Magi2AudioDecodingStage(self.get_module("audio_vae")),
        )


EntryClass = Magi2Pipeline

__all__ = ["Magi2Pipeline"]
