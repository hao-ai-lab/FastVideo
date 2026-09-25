# SPDX-License-Identifier: Apache-2.0
"""Weight-free contracts adapted from #1494 / #1563 for component resolution."""
import json
import pickle
from unittest.mock import Mock

import pytest
import torch

from fastvideo import registry
from fastvideo.api.sampling_param import SamplingParam
from fastvideo.attention import selector
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.models.loader import component_loader
from fastvideo.models.wan import transformer as wan
from fastvideo.models.wan.config import WanVideoConfig
from fastvideo.platforms import AttentionBackendEnum as Backend, current_platform
from fastvideo.pipelines.stages.denoising import DenoisingStage

FULL = "FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers"
ALIAS = "FastVideo/FastWan2.2-TI2V-5B-Diffusers"
SDPA = Backend.TORCH_SDPA
VSA = Backend.VIDEO_SPARSE_ATTN


class FakeImpl:
    def __init__(self, **kwargs):
        pass

    def forward(self, *args, **kwargs):
        raise AssertionError("Construction tests must not run a fake kernel")


class FakeSDPA:
    @staticmethod
    def get_name():
        return "TORCH_SDPA"

    @staticmethod
    def get_impl_cls():
        return FakeImpl


class FakeVSA(FakeSDPA):
    @staticmethod
    def get_name():
        return "VIDEO_SPARSE_ATTN"


class FakeSage(FakeSDPA):
    @staticmethod
    def get_name():
        return "SAGE_ATTN"


@pytest.fixture(autouse=True)
def _offline_registry(monkeypatch):
    monkeypatch.delenv("FASTVIDEO_ATTENTION_BACKEND", raising=False)
    monkeypatch.setattr(registry, "maybe_download_model_index",
                        Mock(side_effect=AssertionError("No Hub access allowed")))


@pytest.fixture(params=["full", "alias", "short_full", "short_alias", "renamed", "snapshot_full", "snapshot_alias"])
def fullattn_identity(request, tmp_path):
    ids = {"full": FULL, "alias": ALIAS, "short_full": FULL.split("/")[-1],
           "short_alias": ALIAS.split("/")[-1]}
    if request.param in ids:
        return ids[request.param]
    name = "renamed"
    if request.param.startswith("snapshot"):
        model_id = FULL if request.param == "snapshot_full" else ALIAS
        name = "models--" + model_id.replace("/", "--") + "/snapshots/abcdef"
    path = tmp_path / name
    path.mkdir(parents=True)
    (path / "model_index.json").write_text(json.dumps({
        "_class_name": "WanDMDPipeline", "_diffusers_version": "0.35.0.dev0", "expand_timesteps": True,
    }))
    return str(path)


def test_fullattn_identity_and_preset(fullattn_identity, monkeypatch):
    monkeypatch.setenv("FASTVIDEO_ATTENTION_BACKEND", "VIDEO_SPARSE_ATTN")
    info = registry._get_config_info(fullattn_identity)
    assert info.pipeline_config_cls.__name__ == "FastWan2_2_TI2V_5B_FullAttn_Config"
    assert info.default_preset == "fast_wan_2_2_ti2v_5b"
    assert [w.value for w in info.workload_types] == ["t2v"]
    args = FastVideoArgs.from_kwargs(model_path=fullattn_identity,
                                     workload_type="t2v", attention_backend="TORCH_SDPA")
    config = args.pipeline_config
    assert not config.ti2v_task
    assert not config.vae_config.load_encoder
    assert config.vae_config.load_decoder
    assert config.dit_config.expand_timesteps
    assert VSA not in config.dit_config._supported_attention_backends
    assert SamplingParam.from_pretrained(fullattn_identity) == SamplingParam.from_pretrained(FULL)


@pytest.mark.parametrize("conflict", ["i2v", "VIDEO_SPARSE_ATTN"])
def test_fullattn_rejects_before_loader(fullattn_identity, conflict, monkeypatch):
    spy = Mock(side_effect=AssertionError("Weights must not be loaded"))
    monkeypatch.setattr(component_loader, "maybe_load_fsdp_model", spy)
    monkeypatch.setenv("FASTVIDEO_ATTENTION_BACKEND", "TORCH_SDPA")
    with pytest.raises(ValueError, match="FullAttn.*" + conflict):
        args = FastVideoArgs.from_kwargs(
            model_path=fullattn_identity, workload_type="i2v" if conflict == "i2v" else "t2v",
            attention_backend="VIDEO_SPARSE_ATTN" if conflict != "i2v" else "TORCH_SDPA")
        component_loader.PipelineComponentLoader.load_module("transformer", "unused", "diffusers", args)
    spy.assert_not_called()


@pytest.mark.parametrize("marker", [None, False, "true", 1])
def test_sparse_manifest_keeps_original_route(tmp_path, marker):
    manifest = {"_class_name": "WanDMDPipeline", "_diffusers_version": "0.35.0"}
    if marker is not None:
        manifest["expand_timesteps"] = marker
    (tmp_path / "model_index.json").write_text(json.dumps(manifest))
    assert registry._get_config_info(str(tmp_path)).pipeline_config_cls.__name__ == "FastWan2_1_T2V_480P_Config"


def test_fullattn_manifest_name_match_is_case_insensitive(tmp_path):
    (tmp_path / "model_index.json").write_text(json.dumps({
        "_class_name": "wAnDmDpIpElInE", "_diffusers_version": "0.35.0", "expand_timesteps": True,
    }))
    assert registry._get_config_info(str(tmp_path)).pipeline_config_cls.__name__ == "FastWan2_2_TI2V_5B_FullAttn_Config"


def test_fullattn_config_json_roundtrip(tmp_path):
    config_cls = registry._get_config_info(FULL).pipeline_config_cls
    original = config_cls()
    path = tmp_path / "config.json"
    original.dump_to_json(str(path))
    restored = config_cls()
    restored.load_from_json(str(path))
    assert not restored.ti2v_task
    assert not restored.vae_config.load_encoder
    assert VSA not in restored.dit_config._supported_attention_backends
    # FullAttn constraints cannot be undone by a stale serialized TI2V flag.
    restored.update_pipeline_config({"ti2v_task": True})
    assert not restored.ti2v_task


@pytest.fixture
def backend_construction(monkeypatch):
    def backend(selected_backend, head_size, dtype):
        name = {VSA: "FakeVSA", Backend.SAGE_ATTN: "FakeSage"}.get(selected_backend, "FakeSDPA")
        return __name__ + "." + name

    monkeypatch.setattr(current_platform, "get_attn_backend_cls", backend)
    monkeypatch.setattr(wan, "get_sp_world_size", lambda: 1)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    selector._cached_get_attn_backend.cache_clear()
    yield
    selector._cached_get_attn_backend.cache_clear()


def _tiny_wan_config(config):
    config.arch_config.num_attention_heads = 1
    config.arch_config.attention_head_dim = 4
    config.arch_config.in_channels = 4
    config.arch_config.out_channels = 4
    config.arch_config.num_layers = 1
    config.arch_config.ffn_dim = 8
    config.arch_config.text_dim = 4
    config.arch_config.freq_dim = 4
    config.arch_config.patch_size = (1, 1, 1)
    config.arch_config.__post_init__()
    return config


def _assert_backend_construction(model, expected):
    block = model.blocks[0]
    assert type(block) is (wan.WanTransformerBlock_VSA if expected is VSA else wan.WanTransformerBlock)
    assert block.attn1.backend is expected
    assert any("to_gate_compress" in key for key in model.state_dict()) is (expected is VSA)
    # Auxiliary cross-attention remains dense, including for a VSA transformer.
    assert block.attn2.attn.backend is SDPA
    assert DenoisingStage(model, scheduler=None).attn_backend.get_name() == expected.name


@pytest.mark.parametrize("requested,environment,expected", [
    (SDPA, "VIDEO_SPARSE_ATTN", SDPA),
    (VSA, None, VSA),
    (None, "VIDEO_SPARSE_ATTN", SDPA),
    (Backend.SAGE_ATTN, "VIDEO_SPARSE_ATTN", Backend.SAGE_ATTN),
])
def test_recorded_component_constructs_consistently(backend_construction, monkeypatch, requested, environment,
                                                    expected):
    config = _tiny_wan_config(WanVideoConfig())
    with selector._component_attention_backend_scope(requested, component="transformer"):
        selector.record_resolved_attention_backend(config)
    if environment:
        monkeypatch.setenv("FASTVIDEO_ATTENTION_BACKEND", environment)
    model = wan.WanTransformer3DModel(config, {})
    _assert_backend_construction(model, expected)


def test_interleaved_components_do_not_reread_environment(backend_construction, monkeypatch):
    models = []
    cases = ((FULL, SDPA), ("FastVideo/FastWan2.1-T2V-1.3B-Diffusers", VSA),
             ("Wan-AI/Wan2.1-T2V-1.3B-Diffusers", None), (ALIAS, SDPA))
    for model_id, requested in cases:
        config = _tiny_wan_config(registry._get_config_info(model_id).pipeline_config_cls().dit_config)
        with selector._component_attention_backend_scope(requested, component="transformer"):
            selector.record_resolved_attention_backend(config)
        models.append(wan.WanTransformer3DModel(config, {}))
    monkeypatch.setenv("FASTVIDEO_ATTENTION_BACKEND", "VIDEO_SPARSE_ATTN")
    for model, expected in zip(models, (SDPA, VSA, SDPA, SDPA), strict=True):
        _assert_backend_construction(model, expected)


def test_direct_construction_keeps_env_compatibility(backend_construction, monkeypatch):
    monkeypatch.setenv("FASTVIDEO_ATTENTION_BACKEND", "VIDEO_SPARSE_ATTN")
    _assert_backend_construction(wan.WanTransformer3DModel(_tiny_wan_config(WanVideoConfig()), {}), VSA)


def test_fullattn_direct_construction_rejects_before_parameters(backend_construction, monkeypatch):
    config = registry._get_config_info(FULL).pipeline_config_cls().dit_config
    monkeypatch.setenv("FASTVIDEO_ATTENTION_BACKEND", "VIDEO_SPARSE_ATTN")
    spy = Mock(side_effect=AssertionError("Must fail before allocating model parameters"))
    monkeypatch.setattr(wan, "PatchEmbed", spy)
    with pytest.raises(ValueError, match="FullAttn.*VIDEO_SPARSE_ATTN"):
        wan.WanTransformer3DModel(_tiny_wan_config(config), {})
    spy.assert_not_called()


def test_recorded_auto_survives_serialization(backend_construction, monkeypatch):
    config = _tiny_wan_config(WanVideoConfig())
    selector.record_resolved_attention_backend(config)
    restored = pickle.loads(pickle.dumps(config))
    assert restored._attention_backend_resolved is True
    monkeypatch.setenv("FASTVIDEO_ATTENTION_BACKEND", "VIDEO_SPARSE_ATTN")
    _assert_backend_construction(wan.WanTransformer3DModel(restored, {}), SDPA)


def test_legacy_subclass_does_not_inherit_wan_backend_opt_in(backend_construction, monkeypatch):
    from fastvideo.models.dits.dreamx_world import DreamXWorldTransformer3DModel

    # DreamX inherits Wan methods but owns its constructor and unported layers.
    model = DreamXWorldTransformer3DModel.__new__(DreamXWorldTransformer3DModel)
    torch.nn.Module.__init__(model)
    model.config = _tiny_wan_config(WanVideoConfig())
    model.hidden_size = 4
    model.num_attention_heads = 1
    selector.record_resolved_attention_backend(model.config)
    monkeypatch.setenv("FASTVIDEO_ATTENTION_BACKEND", "VIDEO_SPARSE_ATTN")
    assert selector.component_attention_backend(model) is selector.NO_REQUEST
    assert DenoisingStage(model, scheduler=None).attn_backend.get_name() == VSA.name


def test_constructed_wan_keeps_backend_after_class_wrapping(backend_construction, monkeypatch):
    config = _tiny_wan_config(WanVideoConfig())
    selector.record_resolved_attention_backend(config)
    model = wan.WanTransformer3DModel(config, {})

    class WrappedWan(wan.WanTransformer3DModel):
        pass

    # FSDP-style class replacement must not discard the instance's decision.
    model.__class__ = WrappedWan
    monkeypatch.setenv("FASTVIDEO_ATTENTION_BACKEND", "VIDEO_SPARSE_ATTN")
    _assert_backend_construction(model, SDPA)


def test_instance_support_is_used(backend_construction, monkeypatch):
    config = _tiny_wan_config(WanVideoConfig())
    config.arch_config._supported_attention_backends = (SDPA,)
    captured = []
    original = wan.WanTransformerBlock

    def capture(*args, **kwargs):
        captured.append(args[7])
        return original(*args, **kwargs)

    monkeypatch.setattr(wan, "WanTransformerBlock", capture)
    wan.WanTransformer3DModel(config, {})
    assert captured == [(SDPA,)]


def test_metadata_uses_actual_fallback_from_instance_support(backend_construction):
    config = _tiny_wan_config(WanVideoConfig())
    config.arch_config._supported_attention_backends = (SDPA,)
    with selector._component_attention_backend_scope(Backend.SAGE_ATTN, component="transformer"):
        selector.record_resolved_attention_backend(config)
    _assert_backend_construction(wan.WanTransformer3DModel(config, {}), SDPA)


@pytest.mark.parametrize("conflict", ["i2v", "VIDEO_SPARSE_ATTN"])
def test_loader_revalidates_before_any_weight_access(monkeypatch, conflict):
    args = FastVideoArgs.from_kwargs(model_path=FULL, attention_backend="TORCH_SDPA")
    if conflict == "i2v":
        from fastvideo.fastvideo_args import WorkloadType
        args.workload_type = WorkloadType.I2V
    config_spy = Mock(side_effect=AssertionError("Must validate before component IO"))
    weight_spy = Mock(side_effect=AssertionError("Must validate before weights"))
    monkeypatch.setattr(component_loader, "get_diffusers_config", config_spy)
    monkeypatch.setattr(component_loader, "maybe_load_fsdp_model", weight_spy)
    with selector._component_attention_backend_scope(VSA if conflict != "i2v" else SDPA):
        with pytest.raises(ValueError, match="FullAttn.*" + conflict):
            component_loader.PipelineComponentLoader.load_module("transformer", "unused", "diffusers", args)
    config_spy.assert_not_called()
    weight_spy.assert_not_called()


def test_dense_cross_attention_small_cpu_tensor(backend_construction, monkeypatch):
    from fastvideo.forward_context import set_forward_context

    monkeypatch.setattr(current_platform, "get_attn_backend_cls",
                        lambda *args: "fastvideo.attention.backends.sdpa.SDPABackend")
    selector._cached_get_attn_backend.cache_clear()
    config = _tiny_wan_config(WanVideoConfig())
    with selector._component_attention_backend_scope(SDPA):
        selector.record_resolved_attention_backend(config)
    monkeypatch.setenv("FASTVIDEO_ATTENTION_BACKEND", "VIDEO_SPARSE_ATTN")
    model = wan.WanTransformer3DModel(config, {})
    q = torch.randn(1, 2, 1, 4)
    k = torch.randn(1, 3, 1, 4)
    v = torch.randn(1, 3, 1, 4)
    with set_forward_context(current_timestep=0, attn_metadata=None):
        actual = model.blocks[0].attn2.attn(q, k, v)
    expected = torch.nn.functional.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)).transpose(1, 2)
    torch.testing.assert_close(actual, expected)
    assert DenoisingStage(model, scheduler=None).attn_backend.get_name() == SDPA.name


@pytest.mark.parametrize("model_id,requested,environment,expected", [
    (FULL, "TORCH_SDPA", "VIDEO_SPARSE_ATTN", SDPA),
    (FULL, None, "VIDEO_SPARSE_ATTN", SDPA),
    ("FastVideo/FastWan2.1-T2V-1.3B-Diffusers", "VIDEO_SPARSE_ATTN", "TORCH_SDPA", VSA),
    ("Wan-AI/Wan2.1-T2V-1.3B-Diffusers", "TORCH_SDPA", "VIDEO_SPARSE_ATTN", SDPA),
])
@pytest.mark.parametrize("direct_loader", [False, True])
def test_public_request_reaches_real_transformer_loader(
        backend_construction, monkeypatch, model_id, requested, environment, expected, direct_loader):
    args = FastVideoArgs.from_kwargs(model_path=model_id, attention_backend=requested)
    args.pipeline_config.dit_precision = "fp32"
    _tiny_wan_config(args.pipeline_config.dit_config)
    monkeypatch.setenv("FASTVIDEO_ATTENTION_BACKEND", environment)
    monkeypatch.setattr(component_loader, "get_diffusers_config",
                        lambda **kwargs: {"_class_name": "WanTransformer3DModel"})
    monkeypatch.setattr(component_loader.glob, "glob", lambda *args: ["fake.safetensors"])
    monkeypatch.setattr(component_loader.ModelRegistry, "resolve_model_cls",
                        lambda *args: (wan.WanTransformer3DModel, None))
    monkeypatch.setattr(component_loader, "get_local_torch_device", lambda: torch.device("cpu"))
    spy = Mock(side_effect=lambda model_cls, init_params, **kwargs: model_cls(**init_params))
    monkeypatch.setattr(component_loader, "maybe_load_fsdp_model", spy)
    if direct_loader:
        model = component_loader.TransformerLoader().load("unused", args)
    else:
        model = component_loader.PipelineComponentLoader.load_module("transformer", "unused", "diffusers", args)
    assert spy.call_count == 1
    _assert_backend_construction(model, expected)
