# SPDX-License-Identifier: Apache-2.0

from fastvideo.api.presets import get_preset
from fastvideo.api.sampling_param import SamplingParam
from fastvideo.configs.pipelines.wan import LucyEditDevConfig
from fastvideo.fastvideo_args import WorkloadType
from fastvideo.pipelines.basic.wan.lucy_edit_pipeline import LucyEditPipeline
from fastvideo.pipelines.pipeline_registry import PipelineType, get_pipeline_registry
from fastvideo.registry import get_default_preset, get_pipeline_config_cls_from_name


def test_lucy_edit_registry_and_preset() -> None:
    import fastvideo.registry  # noqa: F401

    preset = get_preset("lucy_edit_dev", "wan")
    assert preset.model_family == "wan"
    assert preset.defaults["height"] == 480
    assert preset.defaults["width"] == 832
    assert preset.defaults["num_frames"] == 81

    config = LucyEditDevConfig()
    assert config.lucy_edit_task is True
    assert config.ti2v_task is False
    assert config.dit_config.arch_config.out_channels == 48
    assert config.dit_config.arch_config.in_channels == 96
    assert config.vae_config.arch_config.z_dim == 48
    assert config.dit_config.arch_config.in_channels == config.vae_config.arch_config.z_dim * 2
    assert get_default_preset("decart-ai/Lucy-Edit-Dev") == "lucy_edit_dev"
    assert get_default_preset("decart-ai/Lucy-Edit-1.1-Dev") == "lucy_edit_dev"
    assert get_pipeline_config_cls_from_name("decart-ai/Lucy-Edit-Dev") is LucyEditDevConfig

    sampling_param = SamplingParam.from_pretrained("decart-ai/Lucy-Edit-Dev")
    assert sampling_param.height == 480
    assert sampling_param.width == 832
    assert sampling_param.num_frames == 81
    assert sampling_param.fps == 24
    assert sampling_param.guidance_scale == 5.0
    assert sampling_param.negative_prompt == ""

    # FastVideo has no V2V workload enum today; model_index dispatches Lucy by pipeline class name.
    registry = get_pipeline_registry(PipelineType.BASIC)
    assert registry.resolve_pipeline_cls("LucyEditPipeline", PipelineType.BASIC, WorkloadType.T2V) is LucyEditPipeline
