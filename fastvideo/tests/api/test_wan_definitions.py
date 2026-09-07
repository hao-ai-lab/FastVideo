# SPDX-License-Identifier: Apache-2.0
"""Weight-free compatibility contracts for the Wan family catalog.

Expected resolutions and numerical defaults below come from the registry and
configs at 275cf421, before the definition extraction. No Hub calls are needed.
"""

import dataclasses
import importlib
import json
import pickle
from types import SimpleNamespace

import pytest
import torch

from fastvideo import registry
from fastvideo.api.presets import get_preset
from fastvideo.api.sampling_param import SamplingParam
from fastvideo.models.wan import pipeline_config
from fastvideo.models.wan.definition import DMD_TRAINING_NOISE_SHIFT, WAN_MODEL_DEFINITIONS
from fastvideo.pipelines.basic.wan.presets import ALL_PRESETS


# Keep this independent of the candidate catalog: dropping or misassigning an
# alias must fail even if the remaining definitions are internally consistent.
LEGACY_VARIANTS = (
    ("Wan-AI/Wan2.1-T2V-1.3B-Diffusers", "WanT2V480PConfig", "wan_t2v_1_3b", ("t2v",)),
    ("Wan-AI/Wan2.1-T2V-14B-Diffusers", "WanT2V720PConfig", "wan_t2v_14b", ("t2v",)),
    ("FastVideo/Wan2.1-VSA-T2V-14B-720P-Diffusers", "WanT2V720PConfig", "wan_t2v_14b", ("t2v",)),
    ("Wan-AI/Wan2.1-I2V-14B-480P-Diffusers", "WanI2V480PConfig", "wan_i2v_14b_480p", ("i2v",)),
    ("Wan-AI/Wan2.1-I2V-14B-720P-Diffusers", "WanI2V720PConfig", "wan_i2v_14b_720p", ("i2v",)),
    ("weizhou03/Wan2.1-Fun-1.3B-InP-Diffusers", "WanI2V480PConfig", "wan_fun_1_3b_inp", ("i2v",)),
    ("IRMChen/Wan2.1-Fun-1.3B-Control-Diffusers", "WANV2VConfig", "wan_fun_1_3b_control", ()),
    ("FastVideo/FastWan2.1-T2V-1.3B-Diffusers", "FastWan2_1_T2V_480P_Config", "fast_wan_t2v_480p", ("t2v",)),
    ("FastVideo/FastWan2.1-T2V-14B-480P-Diffusers", "FastWan2_1_T2V_480P_Config", "fast_wan_t2v_480p", ("t2v",)),
    ("Wan-AI/Wan2.2-TI2V-5B-Diffusers", "Wan2_2_TI2V_5B_Config", "wan_2_2_ti2v_5b", ("t2v", "i2v")),
    ("FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers", "FastWan2_2_TI2V_5B_Config", "fast_wan_2_2_ti2v_5b", ("t2v", "i2v")),
    ("FastVideo/FastWan2.2-TI2V-5B-Diffusers", "FastWan2_2_TI2V_5B_Config", "fast_wan_2_2_ti2v_5b", ("t2v", "i2v")),
    ("decart-ai/Lucy-Edit-Dev", "LucyEditDevConfig", "lucy_edit_dev", ()),
    ("decart-ai/Lucy-Edit-1.1-Dev", "LucyEditDevConfig", "lucy_edit_dev", ()),
    ("Wan-AI/Wan2.2-T2V-A14B-Diffusers", "Wan2_2_T2V_A14B_Config", "wan_2_2_t2v_a14b", ("t2v",)),
    ("Wan-AI/Wan2.2-I2V-A14B-Diffusers", "Wan2_2_I2V_A14B_Config", "wan_2_2_i2v_a14b", ("i2v",)),
    ("wlsaidhi/SFWan2.1-T2V-1.3B-Diffusers", "SelfForcingWanT2V480PConfig", "sf_wan_t2v_1_3b", ("t2v",)),
    ("rand0nmr/SFWan2.2-T2V-A14B-Diffusers", "SelfForcingWan2_2_T2V480PConfig", "sf_wan_2_2_t2v_a14b", ("t2v",)),
    ("FastVideo/SFWan2.2-I2V-A14B-Preview-Diffusers", "SelfForcingWan2_2_T2V480PConfig", "sf_wan_2_2_i2v_a14b", ("i2v",)),
)


@pytest.fixture(autouse=True)
def no_hub_access(monkeypatch):
    def fail(*args, **kwargs):
        pytest.fail("Wan config contracts must not access the Hub")

    monkeypatch.setattr(registry, "maybe_download_model_index", fail)
    registry.get_model_info.cache_clear()
    yield
    registry.get_model_info.cache_clear()


@pytest.mark.parametrize("model_path,config_name,preset_name,workloads", LEGACY_VARIANTS)
def test_legacy_aliases_and_sampling_defaults(model_path, config_name, preset_name, workloads):
    expected_config = getattr(pipeline_config, config_name)
    preset = get_preset(preset_name, "wan")
    # Exact HF ID, mirror ID, and renamed parent directory of a local checkpoint
    # all use the existing exact/short-name resolution before any manifest IO.
    short_name = model_path.rsplit("/", 1)[-1]
    for path in (model_path, f"mirror/{short_name}", f"/checkpoints/{short_name}"):
        info = registry._get_config_info(path)
        assert info.pipeline_config_cls is expected_config
        assert info.model_family == "wan"
        assert info.default_preset == preset_name
        assert tuple(value.value for value in info.workload_types) == workloads
        assert info.sampling_param_cls is None
        assert info.pipeline_cls_name is None
        sampling = SamplingParam.from_pretrained(path)
        for name, value in preset.defaults.items():
            assert getattr(sampling, name) == value


@pytest.mark.parametrize("directory,manifest_class,expected_family,expected_preset", [
    ("custom", "WanPipeline", "wan", "wan_t2v_1_3b"),
    ("custom", "WanImageToVideoPipeline", "wan", "wan_i2v_14b_480p"),
    ("custom", "WanDMDPipeline", "wan", "fast_wan_t2v_480p"),
    ("custom", "WanCausalDMDPipeline", "wan", "sf_wan_t2v_1_3b"),
    ("Lucy-Edit-custom", "UnknownPipeline", "wan", "lucy_edit_dev"),
    ("SFWan2.2-custom", "UnknownPipeline", "wan", "sf_wan_2_2_t2v_a14b"),
    ("SFWan2_2-I2V-custom", "UnknownPipeline", "wan", "sf_wan_2_2_i2v_a14b"),
    # Counterintuitive but supported legacy precedence: do not silently fix it
    # during a relocation, including when path and class detectors disagree.
    ("lucy-edit-custom", "WanPipeline", "wan", "wan_t2v_1_3b"),
    ("sfwan2.2-i2v-custom", "WanCausalDMDPipeline", "wan", "sf_wan_t2v_1_3b"),
    ("dreamx-world-5b-lucy-edit-custom", "UnknownPipeline", "dreamx_world", "dreamx_world_5b_ar"),
    ("dreamx-world-5b-custom", "WanPipeline", "wan", "wan_t2v_1_3b"),
])
def test_local_manifest_detectors_keep_first_match(tmp_path, directory, manifest_class, expected_family, expected_preset):
    checkpoint = tmp_path / directory
    checkpoint.mkdir()
    (checkpoint / "model_index.json").write_text(json.dumps({
        "_class_name": manifest_class, "_diffusers_version": "0.33.0",
    }))
    info = registry._get_config_info(str(checkpoint))
    assert (info.model_family, info.default_preset) == (expected_family, expected_preset)


def test_manifest_selection_and_explicit_override_are_not_pinned(monkeypatch, tmp_path):
    from fastvideo.pipelines import pipeline_registry

    selected = []
    def resolve(name, *args):
        selected.append(name)
        return type(name, (), {})

    monkeypatch.setattr(pipeline_registry, "get_pipeline_registry",
                        lambda *args: SimpleNamespace(resolve_pipeline_cls=resolve))
    checkpoint = tmp_path / "Wan2.1-T2V-1.3B-Diffusers"
    checkpoint.mkdir()
    manifest = checkpoint / "model_index.json"
    manifest.write_text(json.dumps({"_class_name": "CustomWanPipeline", "_diffusers_version": "0.33.0"}))
    info = registry.get_model_info(str(checkpoint))
    assert info.pipeline_cls.__name__ == "CustomWanPipeline"
    assert info.pipeline_config_cls is pipeline_config.WanT2V480PConfig
    # An explicit override works even without a manifest, as before.
    manifest.unlink()
    info = registry.get_model_info(str(checkpoint), override_pipeline_cls_name="WanDMDPipeline")
    assert info.pipeline_cls.__name__ == "WanDMDPipeline"
    assert selected == ["CustomWanPipeline", "WanDMDPipeline"]


@pytest.mark.parametrize("config_name,flow_shift,dmd_steps,load_encoder", [
    ("WanT2V480PConfig", 3.0, None, False),
    ("WanT2V720PConfig", 5.0, None, False),
    ("WanI2V480PConfig", 3.0, None, True),
    ("WanI2V720PConfig", 5.0, None, True),
    ("WANV2VConfig", 3.0, None, True),
    ("FastWan2_1_T2V_480P_Config", 8.0, [1000, 757, 522], False),
    ("Wan2_2_TI2V_5B_Config", 5.0, None, True),
    ("FastWan2_2_TI2V_5B_Config", 5.0, [1000, 757, 522], True),
    ("LucyEditDevConfig", 5.0, None, True),
    ("Wan2_2_T2V_A14B_Config", 12.0, [1000, 750, 500, 250], True),
    ("Wan2_2_I2V_A14B_Config", 5.0, None, True),
    ("SelfForcingWanT2V480PConfig", 5.0, [1000, 750, 500, 250], False),
    ("SelfForcingWan2_2_T2V480PConfig", 12.0, [1000, 850, 700, 550, 350, 275, 200, 125], True),
])
def test_component_precision_and_sampling_defaults(config_name, flow_shift, dmd_steps, load_encoder):
    config_cls = getattr(pipeline_config, config_name)
    first, second = config_cls(), config_cls()
    assert first.flow_shift == flow_shift
    assert first.dmd_denoising_steps == dmd_steps
    assert first.dit_precision == "bf16"
    assert first.vae_precision == "fp32"
    assert first.vae_decode_precision == "bf16"
    assert first.text_encoder_precisions == ("fp32",)
    assert first.vae_config.load_encoder is load_encoder
    assert first.vae_config.load_decoder is True
    # Each request/instance still gets independent mutable component configs.
    first.vae_config.load_encoder = not load_encoder
    assert second.vae_config.load_encoder is load_encoder
    first.dit_config.boundary_ratio = 0.123
    assert second.dit_config.boundary_ratio != 0.123
    if first.dmd_denoising_steps is not None:
        first.dmd_denoising_steps.append(1)
        assert second.dmd_denoising_steps == dmd_steps


def test_definitions_are_complete_data_only_and_reference_existing_configs():
    paths = [path for definition in WAN_MODEL_DEFINITIONS for path in definition.hf_model_paths]
    assert len(paths) == len(set(paths))
    assert set(paths) == {case[0] for case in LEGACY_VARIANTS}
    assert {definition.preset for definition in WAN_MODEL_DEFINITIONS} == {preset.name for preset in ALL_PRESETS}
    assert len(WAN_MODEL_DEFINITIONS) == len(ALL_PRESETS)
    assert DMD_TRAINING_NOISE_SHIFT == 8.0
    for definition in WAN_MODEL_DEFINITIONS:
        assert json.loads(json.dumps(dataclasses.asdict(definition)))["preset"] == definition.preset
        config = getattr(pipeline_config, definition.pipeline_config)()
        if definition.sampling == "causal_dmd":
            assert config.is_causal
            assert config.warp_denoising_step
            assert config.dmd_denoising_steps
        elif definition.sampling == "dmd":
            assert not config.is_causal
            assert config.dmd_denoising_steps == [1000, 757, 522]
        else:
            assert definition.sampling == "unipc"
            assert not config.is_causal
        with pytest.raises(dataclasses.FrozenInstanceError):
            definition.preset = "changed"


def test_legacy_pipeline_config_aliases_and_pickle_transport():
    def assert_same_state(actual, expected):
        assert type(actual) is type(expected)
        if dataclasses.is_dataclass(expected):
            # vars also covers derived attributes omitted by dataclasses.asdict,
            # including the VAE normalization shift_factor.
            assert_same_state(vars(actual), vars(expected))
        elif isinstance(expected, dict):
            assert actual.keys() == expected.keys()
            for key in expected:
                assert_same_state(actual[key], expected[key])
        elif isinstance(expected, (tuple, list)):
            assert len(actual) == len(expected)
            for actual_item, expected_item in zip(actual, expected, strict=True):
                assert_same_state(actual_item, expected_item)
        elif isinstance(expected, torch.Tensor):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        else:
            assert actual == expected

    legacy = importlib.import_module("fastvideo.configs.pipelines.wan")
    for name in legacy.__all__:
        assert getattr(legacy, name) is getattr(pipeline_config, name)
    for name in {definition.pipeline_config for definition in WAN_MODEL_DEFINITIONS}:
        original = getattr(pipeline_config, name)()
        payload = pickle.dumps(original, protocol=0)
        legacy_payload = payload.replace(b"fastvideo.models.wan.pipeline_config\n", b"fastvideo.configs.pipelines.wan\n")
        assert legacy_payload != payload
        restored = pickle.loads(legacy_payload)
        assert type(restored) is type(original)
        assert_same_state(restored, original)
