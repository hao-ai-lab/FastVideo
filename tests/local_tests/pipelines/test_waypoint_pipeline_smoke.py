# SPDX-License-Identifier: Apache-2.0
"""Waypoint pipeline structure and control-contract tests."""

from types import SimpleNamespace

import torch


def test_waypoint_pipeline_is_stage_composed():
    from fastvideo.configs.pipelines.waypoint import WaypointT2VConfig
    from fastvideo.pipelines.basic.waypoint import WaypointPipeline

    class Transformer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1, dtype=torch.bfloat16))
            self.denoise_step_emb = torch.nn.Linear(1, 1)

    class VAE(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))

    pipeline = object.__new__(WaypointPipeline)
    pipeline.modules = {
        "transformer": Transformer(),
        "vae": VAE(),
        "text_encoder": torch.nn.Linear(1, 1),
        "tokenizer": object(),
    }
    pipeline._stages = []
    pipeline._stage_name_mapping = {}
    args = SimpleNamespace(pipeline_config=WaypointT2VConfig())

    pipeline.create_pipeline_stages(args)

    assert list(pipeline._stage_name_mapping) == [
        "prompt_encoding_stage",
        "denoising_stage",
        "decoding_stage",
    ]


def test_waypoint_controls_include_scroll():
    from fastvideo.pipelines.stages.waypoint_stages import WaypointDenoisingStage

    keyboard = torch.zeros(2, 256)
    mouse = torch.zeros(2, 2)
    scroll = torch.tensor([-3.0, 2.0])
    keyboard, mouse, scroll = WaypointDenoisingStage._controls(
        keyboard,
        mouse,
        scroll,
        torch.device("cpu"),
        torch.float32,
    )

    assert keyboard.shape == (1, 2, 256)
    assert mouse.shape == (1, 2, 2)
    assert scroll.tolist() == [[[-1.0], [1.0]]]


def test_waypoint_decoder_uses_native_uint8_output():
    from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
    from fastvideo.pipelines.stages.waypoint_stages import WaypointDecodingStage

    class VAE(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))

        def decode(self, latent):
            return torch.full((4, 6, 3), 128, dtype=torch.uint8)

    batch = ForwardBatch(data_type="t2v")
    batch.latents = torch.zeros(1, 16, 2, 2, 2)
    output = WaypointDecodingStage(VAE()).forward(batch, SimpleNamespace())

    assert output.output.shape == (1, 3, 2, 4, 6)
    torch.testing.assert_close(
        output.output,
        torch.full_like(output.output, 128 / 255),
    )


def test_waypoint_config_matches_published_schedule():
    from fastvideo.configs.pipelines.waypoint import WaypointT2VConfig
    from fastvideo.registry import get_pipeline_config_cls_from_name

    config = WaypointT2VConfig()
    assert get_pipeline_config_cls_from_name(
        "FastVideo/Waypoint-1-Small-Diffusers"
    ) is WaypointT2VConfig
    assert config.scheduler_sigmas == [
        1.0,
        0.8609585762023926,
        0.729332447052002,
        0.3205108940601349,
        0.0,
    ]
