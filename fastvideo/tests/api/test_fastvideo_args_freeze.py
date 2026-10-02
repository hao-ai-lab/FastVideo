# SPDX-License-Identifier: Apache-2.0
"""A FastVideoArgs built from a resolved config is read-only, and every later change is a recorded override."""
import pickle

import pytest

from fastvideo.api.compat import generator_config_to_fastvideo_args
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.tests.api.config_snapshot import isolated_environment

WAN_T2V = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"


def _resolved_args(raw=None, env_values=None):
    with isolated_environment(env_values):
        return generator_config_to_fastvideo_args({"model_path": WAN_T2V, **(raw or {})})


def test_config_fields_are_read_only_after_resolution():
    args = _resolved_args()

    with pytest.raises(AttributeError, match="use fastvideo_args.override"):
        args.dit_cpu_offload = False
    with pytest.raises(AttributeError, match="pipeline_config.flow_shift"):
        args.pipeline_config.flow_shift = 1.0

    args.model_loaded["transformer"] = False
    args.model_paths = {"transformer": "/tmp/transformer"}
    args.pipeline_config.vae_config.load_encoder = True


def test_directly_built_args_stay_writable():
    with isolated_environment():
        args = FastVideoArgs(model_path=WAN_T2V)

    args.dit_cpu_offload = False
    assert args.resolved_config is None


def test_override_is_logged_and_recorded_on_the_resolved_config():
    args = _resolved_args()

    args.override("test:source", {"dit_cpu_offload": False, "pipeline_config.flow_shift": 2.0})

    assert args.dit_cpu_offload is False
    assert args.pipeline_config.flow_shift == 2.0
    assert args.override_log == [("test:source", {"dit_cpu_offload": False, "pipeline_config.flow_shift": 2.0})]
    assert args.resolved_config.provenance("engine.offload.dit").source == "test:source"
    assert args.resolved_config.pipeline.flow_shift == 2.0


def test_unified_memory_policy_is_a_recorded_override(monkeypatch):
    from fastvideo.platforms import current_platform

    args = _resolved_args({"engine": {"offload": {"dit_layerwise": False}}})
    monkeypatch.setattr(current_platform, "has_unified_memory", lambda device_id: True)
    monkeypatch.setattr(current_platform, "get_device_name", lambda device_id: "test device")

    args.finalize_device_offload_policy(0)

    assert args.dit_cpu_offload is False and args.vae_cpu_offload is False
    assert args.lazy_module_load is True
    sources = [source for source, _ in args.override_log]
    assert sources == ["device_policy:unified_memory", "device_policy:lazy_module_load"]
    assert args.resolved_config.provenance("engine.offload.vae").source == "device_policy:unified_memory"


def test_explicit_false_wins_over_the_environment():
    env = {"FASTVIDEO_INFERENCE_TORCH_COMPILE": True, "FASTVIDEO_VAE_PARALLEL_DECODE": True}

    explicit = _resolved_args({
        "engine": {"compile": {"regional": False}},
        "pipeline": {"minimax_h3": {"vae_parallel_decode": False}},
    }, env)
    unset = _resolved_args(env_values=env)

    assert (explicit.inference_torch_compile, explicit.vae_parallel_decode) == (False, False)
    assert (unset.inference_torch_compile, unset.vae_parallel_decode) == (True, True)


def test_pickled_args_stay_read_only():
    args = pickle.loads(pickle.dumps(_resolved_args()))

    assert args.resolved_config.engine.num_gpus == 1
    with pytest.raises(AttributeError):
        args.num_gpus = 2
