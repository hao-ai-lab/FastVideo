# SPDX-License-Identifier: Apache-2.0
"""Shared component-loader extensions required by V2A pipelines."""

from __future__ import annotations

import json
from types import SimpleNamespace


def test_indexed_image_encoder_uses_matching_config_and_precision(tmp_path) -> None:
    from fastvideo.configs.models.encoders.mmaudio_clip import (
        MMAudioDFNCLIPVisionConfig,
    )
    from fastvideo.configs.models.encoders.mmaudio_synchformer import (
        MMAudioSynchformerConfig,
    )
    from fastvideo.models.loader.component_loader import ImageEncoderLoader

    component = tmp_path / "image_encoder_2"
    component.mkdir()
    with (component / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "architectures": ["MMAudioSynchformerVisualEncoder"],
                "segment_stride": 4,
            },
            handle,
        )

    class CaptureLoader(ImageEncoderLoader):
        def load_model(
            self,
            model_path,
            model_config,
            target_device,
            fastvideo_args,
            dtype="fp16",
            use_text_encoder_override=False,
            cpu_offload=None,
        ):
            del model_path, target_device, fastvideo_args, use_text_encoder_override
            return model_config, dtype, cpu_offload

    vision_config = MMAudioDFNCLIPVisionConfig()
    sync_config = MMAudioSynchformerConfig()
    pipeline_config = SimpleNamespace(
        image_encoder_config=vision_config,
        image_encoder_precision="fp32",
        image_encoder_configs=(vision_config, sync_config),
        image_encoder_precisions=("bf16", "fp16"),
    )
    args = SimpleNamespace(
        pipeline_config=pipeline_config,
        image_encoder_cpu_offload=True,
    )

    selected, precision, cpu_offload = CaptureLoader().load(str(component), args)
    assert selected is sync_config
    assert selected.arch_config.segment_stride == 4
    assert vision_config.arch_config.image_size == 378
    assert precision == "fp16"
    assert cpu_offload is True
