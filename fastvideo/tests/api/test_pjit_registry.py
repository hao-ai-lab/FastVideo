# SPDX-License-Identifier: Apache-2.0
import json

import pytest

from fastvideo.api.compat import generator_config_to_fastvideo_args, normalize_generation_request, request_to_sampling_param
from fastvideo.api.sampling_param import SamplingParam
from fastvideo.configs.pipelines.pjit import PJiTPipelineConfig
from fastvideo.fastvideo_args import WorkloadType
from fastvideo.registry import get_model_info, get_preset_selection


@pytest.mark.parametrize("local_copy", [True, False])
def test_pjit_registry_and_sampling_defaults(tmp_path, local_copy):
    path = "Linum-AI/pjit-21k"
    if local_copy:
        (tmp_path / "config.json").write_text(json.dumps({"model": "P-JiT 21K"}))
        (tmp_path / "model.safetensors").touch()
        path = str(tmp_path)

    info = get_model_info(path, workload_type=WorkloadType.T2I)
    assert info.pipeline_cls.__name__ == "PJiTPipeline"
    assert info.pipeline_config_cls is PJiTPipelineConfig
    assert get_preset_selection(path) == ("pjit_21k", "pjit")
    params = SamplingParam.from_pretrained(path)
    assert (params.num_frames, params.height, params.width) == (1, 1024, 1024)
    assert (params.num_inference_steps, params.guidance_scale, params.negative_prompt) == (50, 15.0, "")


def test_default_generator_exports_one_image():
    model_id = "Linum-AI/pjit-21k"
    args = generator_config_to_fastvideo_args({"model_path": model_id})
    assert args.workload_type == WorkloadType.T2I
    request = normalize_generation_request({"prompt": "A red teapot", "sampling": {"seed": 2}})
    params = request_to_sampling_param(request, model_path=model_id)
    assert (params.num_frames, params.guidance_scale, params.negative_prompt, params.seed) == (1, 15.0, "", 2)
