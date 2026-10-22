# SPDX-License-Identifier: Apache-2.0
import json
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from fastvideo.api.sampling_param import SamplingParam
from fastvideo.configs.pipelines.bernini import BerniniRPipelineConfig
from fastvideo.configs.pipelines.bernini import clean_bernini_prompt
from fastvideo.models.wan.pipeline_config import WanT2V480PConfig
from fastvideo.registry import get_model_info, get_pipeline_config_cls_from_name


def test_bernini_manifest_routing_and_sampling_defaults(tmp_path):
    (tmp_path / "model_index.json").write_text(json.dumps({
        "_class_name": "BerniniRPipeline", "_diffusers_version": "0.35.2"}))
    assert get_pipeline_config_cls_from_name(str(tmp_path)) is BerniniRPipelineConfig
    info = get_model_info(str(tmp_path))
    assert info.pipeline_cls.__name__ == "BerniniRPipeline"
    params = SamplingParam.from_pretrained(str(tmp_path))
    assert params.guidance_scale == 4.0
    assert params.num_inference_steps == 40
    assert params.num_frames == 81 and params.fps == 16


def test_bernini_directory_name_does_not_override_other_model_manifests(tmp_path):
    checkpoint = tmp_path / "bernini-r-1.3b-experiments" / "other-model"
    checkpoint.mkdir(parents=True)
    (checkpoint / "model_index.json").write_text(json.dumps({
        "_class_name": "WanPipeline", "_diffusers_version": "0.35.2"}))
    assert get_pipeline_config_cls_from_name(str(checkpoint)) is WanT2V480PConfig


def test_bernini_does_not_change_wan_defaults_or_share_configs():
    a, b, wan = BerniniRPipelineConfig(), BerniniRPipelineConfig(), WanT2V480PConfig()
    assert a.vae_config.load_encoder and a.vae_config.load_decoder
    assert not wan.vae_config.load_encoder
    assert a.vae_decode_precision == "fp32" and wan.vae_decode_precision == "bf16"
    encoder = a.text_encoder_configs[0]
    encoder.update_model_arch({"d_model": 4096, "num_layers": 24})
    assert encoder.tokenizer_kwargs["padding"] == "max_length"
    assert encoder.tokenizer_kwargs["max_length"] == 512
    assert "padding" not in wan.text_encoder_configs[0].tokenizer_kwargs
    assert a.text_encoder_precisions == ("bf16", )
    assert a.dit_config.num_layers == 30 and a.dit_config.num_attention_heads == 12
    a.dit_config.arch_config.num_layers = 1
    assert b.dit_config.num_layers == 30


def test_bernini_prompt_cleaning():
    assert clean_bernini_prompt("  a&amp;amp;b\n\t c  ") == "a&b c"


@pytest.mark.parametrize("changes,match", [
    ({"sp_size": 2}, "num_gpus=sp_size=tp_size=1"),
    ({"tp_size": 2}, "num_gpus=sp_size=tp_size=1"),
    ({"num_gpus": 2}, "num_gpus=sp_size=tp_size=1"),
    ({"pipeline_config": SimpleNamespace(vae_precision="bf16")}, "FP32 VAE"),
])
def test_bernini_rejects_unsupported_runtime_before_loading(changes, match):
    from fastvideo.pipelines.basic.bernini.bernini_pipeline import BerniniRPipeline
    fields = dict(sp_size=1, tp_size=1, num_gpus=1, pipeline_config=SimpleNamespace(vae_precision="fp32"))
    fields.update(changes)
    with pytest.raises(ValueError, match=match):
        BerniniRPipeline.load_modules(None, SimpleNamespace(**fields))


@pytest.mark.parametrize("fail", [False, True])
def test_bernini_example_uses_typed_api_and_shuts_down(monkeypatch, capsys, fail):
    path = Path(__file__).resolve().parents[3] / "examples/inference/basic/basic_bernini_r.py"
    spec = importlib.util.spec_from_file_location("bernini_example", path)
    example = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example)
    observed = {}

    class Generator:
        @classmethod
        def from_config(cls, config):
            observed["config"] = config
            return cls()

        def generate(self, *, request):
            observed["request"] = request
            if fail:
                raise RuntimeError("generation failure")
            return SimpleNamespace(video_path="edited.mp4")

        def shutdown(self):
            observed["shutdown"] = True

    monkeypatch.setattr(example, "VideoGenerator", Generator)
    monkeypatch.setattr("sys.argv", [str(path), "--model", "prepared-checkpoint", "--video", "source.mp4",
                                     "--prompt", "add snow"])
    if fail:
        with pytest.raises(RuntimeError, match="generation failure"):
            example.main()
    else:
        example.main()
        assert capsys.readouterr().out.strip() == "edited.mp4"
    config, request = observed["config"], observed["request"]
    assert config.engine.num_gpus == 1 and not config.engine.use_fsdp_inference
    assert not config.engine.offload.dit_layerwise
    assert request.inputs.video_path == "source.mp4"
    assert request.sampling.num_inference_steps == 40 and request.sampling.guidance_scale == 4.0
    assert request.negative_prompt == "" and observed["shutdown"]
