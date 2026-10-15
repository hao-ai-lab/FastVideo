# SPDX-License-Identifier: Apache-2.0
"""Normal-loader IPC dispatch fails closed and scopes leases to the pipeline."""
from contextlib import ExitStack, contextmanager
from types import SimpleNamespace

import pytest
import torch

from fastvideo.fastvideo_args import ExecutionMode, FastVideoArgs
from fastvideo.models.loader.component_loader import ComponentLoader, PipelineComponentLoader
from fastvideo.models.loader import inference_weight_cache as cache
from fastvideo.models.wan.pipeline_config import WanT2V480PConfig


def args(**overrides):
    config = dict(model_path="unused", pipeline_config=WanT2V480PConfig(), num_gpus=1,
                  tp_size=1, sp_size=1, hsdp_replicate_dim=1, hsdp_shard_dim=1,
                  dit_cpu_offload=False, dit_layerwise_offload=False, vae_cpu_offload=False,
                  text_encoder_cpu_offload=False, image_encoder_cpu_offload=False, lazy_module_load=False,
                  inference_torch_compile=False, weight_cache_checkpoint="immutable-revision")
    config.update(overrides)
    return FastVideoArgs(**config)


@pytest.mark.parametrize("option,value", [
    ("inference_mode", False), ("use_fsdp_inference", True), ("num_gpus", 2),
    ("dit_cpu_offload", True), ("dit_layerwise_offload", True), ("text_encoder_cpu_offload", True),
    ("vae_cpu_offload", True), ("lora_path", "adapter"), ("enable_torch_compile", True),
    ("init_weights_from_safetensors", "override"), ("weight_cache_checkpoint", ""),
])
def test_unsupported_modes_rejected_before_disk_loading(option, value):
    runtime = args(weight_cache_dir="unused")
    setattr(runtime, option, value)
    with pytest.raises(ValueError):
        with cache.component_weight_cache("transformer", runtime, ExitStack()):
            pytest.fail("Unsupported mode reached a loader")


def test_training_mode_rejected_even_if_inference_boolean_is_stale():
    runtime = args()
    runtime.mode = ExecutionMode.FINETUNING
    with pytest.raises(ValueError, match="inference only"):
        cache.validate_weight_cache_args(runtime)


def test_ipc_requires_explicit_lifetime_owner():
    with pytest.raises(ValueError, match="ExitStack"):
        with cache.component_weight_cache("vae", args(weight_cache_dir="unused"), None):
            pytest.fail("IPC cannot escape an unmanaged lifetime")


def test_unknown_adapter_cannot_silently_load_from_disk():
    with pytest.raises(ValueError, match="No native"):
        with cache.component_weight_cache("vae", args(), None):
            pass


def test_context_restores_after_failed_loader():
    with pytest.raises(RuntimeError):
        with cache.component_weight_cache("vae", args(), None):
            raise RuntimeError("Failed constructor")
    assert cache.attach_cached_model(None, None, None, lambda: None) is None


@pytest.mark.parametrize("component", cache.COMPONENTS)
def test_normal_dispatch_attaches_all_components_without_disk_hydration(monkeypatch, component):
    runtime = args(weight_cache_dir="/private/cache")
    cls = type(cache._CLASSES[component], (torch.nn.Linear,), {})
    calls = []

    class Loader:
        def load(self, path, runtime):
            model = cache.attach_cached_model(cls, {"width": 2}, torch.float32, lambda: cls(2, 2))
            assert model is not None, "Would have reached disk hydration"
            return model

    class Client:
        def __init__(self, path, key):
            assert str(path) == f"/private/cache/{component}.sock"
            self.key = key

        @contextmanager
        def load_model(self, factory):
            with torch.device("meta"):
                model = factory()
            calls.append("attach")
            try:
                yield SimpleNamespace(model=model, donor_pid=123, handles={"weight": "handle"})
            finally:
                calls.append("release")

    monkeypatch.setattr(ComponentLoader, "for_module_type", lambda *a: Loader())
    monkeypatch.setattr(cache, "WeightCacheClient", Client)
    leases = ExitStack()
    model = PipelineComponentLoader.load_module(component, "unused", "diffusers", runtime,
                                               weight_cache_leases=leases)
    assert calls == ["attach"]
    assert model._fastvideo_weight_cache_key.component == component
    assert model._fastvideo_weight_cache_witness["donor_pid"] == 123
    leases.close()
    assert calls == ["attach", "release"]


def test_key_changes_for_dtype_configuration_and_checkpoint():
    cls = type("AutoencoderKLWan", (torch.nn.Linear,), {})
    def key(config, dtype=torch.float32, checkpoint="revision"):
        with cache.component_weight_cache("vae", args(weight_cache_checkpoint=checkpoint), None) as context:
            assert cache.attach_cached_model(cls, config, dtype, lambda: None) is None
        return context.key
    baseline = key({"width": 2})
    assert baseline == key({"width": 2})
    assert baseline != key({"width": 3})
    assert baseline != key({"width": 2}, torch.bfloat16)
    assert baseline != key({"width": 2}, checkpoint="other")


@pytest.mark.parametrize("method", ["set_lora_adapter", "merge_lora_weights", "unmerge_lora_weights"])
def test_runtime_lora_apis_cannot_modify_shared_weights(method):
    from fastvideo.pipelines.lora_pipeline import LoRAPipeline
    pipeline = SimpleNamespace(fastvideo_args=args(weight_cache_dir="unused"))
    kwargs = {"lora_nickname": "test"} if method == "set_lora_adapter" else {}
    with pytest.raises(ValueError, match="cannot mutate"):
        getattr(LoRAPipeline, method)(pipeline, **kwargs)


def test_pipeline_close_releases_leases_and_is_idempotent():
    from fastvideo.pipelines.composed_pipeline_base import ComposedPipelineBase
    closed = []
    leases = ExitStack()
    leases.callback(lambda: closed.append(True))
    pipeline = SimpleNamespace(_weight_cache_leases=leases, _trace_mgr=None)
    ComposedPipelineBase.close(pipeline)
    ComposedPipelineBase.close(pipeline)
    assert closed == [True]


def test_partial_pipeline_load_failure_releases_existing_attachments(monkeypatch):
    from contextlib import nullcontext
    from fastvideo.pipelines import composed_pipeline_base as base
    released = []
    pipeline = SimpleNamespace()
    def failing_load(*args):
        pipeline._weight_cache_leases.callback(lambda: released.append("first-component"))
        raise ValueError("Later component incompatible")
    pipeline.load_modules = failing_load
    monkeypatch.setattr(base, "maybe_init_distributed_environment_and_model_parallel", lambda *a: None)
    monkeypatch.setattr(base, "get_world_group", lambda: SimpleNamespace(local_rank=0))
    monkeypatch.setattr(base, "get_or_create_profiler", lambda *a: SimpleNamespace(region=lambda *a: nullcontext()))
    runtime = SimpleNamespace(inference_mode=False, tp_size=1, sp_size=1)
    with pytest.raises(ValueError, match="Later component"):
        base.ComposedPipelineBase.__init__(pipeline, "unused", runtime, required_config_modules=[])
    assert released == ["first-component"]


def test_unsupported_vae_special_constructor_rejected_before_checkpoint_access(monkeypatch):
    from fastvideo.models.loader import component_loader
    runtime = args(weight_cache_dir="unused")
    monkeypatch.setattr(component_loader, "get_diffusers_config",
                        lambda **kwargs: {"_class_name": "LingBotWorld2WanVAE"})
    with ExitStack() as leases, pytest.raises(ValueError, match="Unsupported cached vae class"):
        PipelineComponentLoader.load_module("vae", "unused", "diffusers", runtime,
                                           weight_cache_leases=leases)
