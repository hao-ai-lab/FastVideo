# SPDX-License-Identifier: Apache-2.0
"""CPU tests for resolving a GeneratorConfig into a frozen ResolvedGeneratorConfig."""
import pickle

import pytest
import yaml

from fastvideo.api.resolution import (
    INPUT_SOURCE,
    ResolutionError,
    ResolutionView,
    ResolvedGeneratorConfig,
    resolve_generator_config,
)
from fastvideo.api.schema import GeneratorConfig, QuantizationConfig

MODEL_PATH = "/models/wan"


def fill_dit_offload_if_unset(view: ResolutionView) -> dict:
    if view.is_explicit("engine.offload.dit"):
        return {}
    return {"engine.offload.dit": False}


def set_sp_size_two(view: ResolutionView) -> dict:
    return {"engine.parallelism.sp_size": 2}


def set_sp_size_four(view: ResolutionView) -> dict:
    return {"engine.parallelism.sp_size": 4}


def test_explicit_input_beats_fill_if_unset_step() -> None:
    explicit = resolve_generator_config(
        {"model_path": MODEL_PATH, "engine": {"offload": {"dit": True}}},
        [fill_dit_offload_if_unset],
    )
    assert explicit.engine.offload.dit is True
    assert explicit.provenance("engine.offload.dit").source == INPUT_SOURCE
    assert explicit.provenance("engine.offload.dit").explicit is True
    assert explicit.decisions == ()

    unset = resolve_generator_config({"model_path": MODEL_PATH}, [fill_dit_offload_if_unset])
    assert unset.engine.offload.dit is False
    provenance = unset.provenance("engine.offload.dit")
    assert provenance.source == "fill_dit_offload_if_unset"
    assert provenance.raw_value is True
    assert provenance.explicit is False


def test_later_step_overrides_earlier_step_and_provenance_names_it() -> None:
    seen: dict = {}

    def check_previous_decision(view: ResolutionView) -> dict:
        seen["sp_size"] = view.get("engine.parallelism.sp_size")
        seen["decided_by"] = view.decided_by("engine.parallelism.sp_size")
        return {}

    resolved = resolve_generator_config(
        {"model_path": MODEL_PATH},
        [set_sp_size_two, check_previous_decision, set_sp_size_four],
    )

    assert seen == {"sp_size": 2, "decided_by": "set_sp_size_two"}
    assert resolved.engine.parallelism.sp_size == 4
    assert resolved.provenance("engine.parallelism.sp_size").source == "set_sp_size_four"
    assert resolved.provenance("engine.parallelism.sp_size").raw_value == -1
    assert resolved.provenance("engine.parallelism.tp_size").source == INPUT_SOURCE
    assert resolved.decisions == (
        ("set_sp_size_two", {"engine.parallelism.sp_size": 2}),
        ("set_sp_size_four", {"engine.parallelism.sp_size": 4}),
    )


@pytest.mark.parametrize(
    "path",
    [
        "engine.not_a_field",
        "engine.num_gpus.extra",
        "engine.quantization.transformer_quant",
        "pipeline.experimental.missing_section.key",
        "engine..num_gpus",
    ],
)
def test_unknown_paths_are_rejected_with_step_name(path: str) -> None:

    def write_bad_path(view: ResolutionView) -> dict:
        return {path: 1}

    with pytest.raises(ResolutionError, match="write_bad_path"):
        resolve_generator_config({"model_path": MODEL_PATH}, [write_bad_path])


def test_wildcard_dicts_and_optional_configs_accept_new_values() -> None:

    def add_experimental_key(view: ResolutionView) -> dict:
        return {"pipeline.experimental.attention_backend": "SAGE_ATTN"}

    def install_quantization(view: ResolutionView) -> dict:
        return {"engine.quantization": QuantizationConfig()}

    def set_quantization_field(view: ResolutionView) -> dict:
        return {"engine.quantization.transformer_quant": "fp8"}

    resolved = resolve_generator_config(
        {"model_path": MODEL_PATH},
        [add_experimental_key, install_quantization, set_quantization_field],
    )

    assert resolved.pipeline.experimental["attention_backend"] == "SAGE_ATTN"
    provenance = resolved.provenance("pipeline.experimental.attention_backend")
    assert provenance.raw_present is False
    assert provenance.source.endswith("add_experimental_key")
    assert resolved.engine.quantization.transformer_quant == "fp8"


def test_step_must_return_a_mapping() -> None:

    def return_list(view: ResolutionView) -> list:
        return ["engine.num_gpus"]

    with pytest.raises(ResolutionError, match="return_list"):
        resolve_generator_config({"model_path": MODEL_PATH}, [return_list])


def test_assignment_raises_at_every_nesting_level() -> None:
    resolved = resolve_generator_config(
        {"model_path": MODEL_PATH, "pipeline": {"experimental": {"flag": [1, 2]}}},
        [],
    )

    with pytest.raises(AttributeError):
        resolved.model_path = "/other"
    with pytest.raises(AttributeError):
        resolved.engine = None
    with pytest.raises(AttributeError):
        resolved.engine.num_gpus = 8
    with pytest.raises(AttributeError):
        resolved.engine.offload.dit = False
    with pytest.raises(AttributeError):
        del resolved.engine.parallelism
    with pytest.raises(TypeError):
        resolved.pipeline.experimental["flag"] = 3
    with pytest.raises(AttributeError):
        resolved.pipeline.experimental["flag"].append(3)
    assert resolved.pipeline.experimental["flag"] == (1, 2)

    def mutate_view_value(view: ResolutionView) -> dict:
        view.get("pipeline.experimental")["flag"] = 3
        return {}

    with pytest.raises(TypeError):
        resolve_generator_config({"model_path": MODEL_PATH}, [mutate_view_value])


def test_post_freeze_change_is_validated_and_logged() -> None:
    resolved = resolve_generator_config({"model_path": MODEL_PATH}, [set_sp_size_two])

    changed = resolved.with_override("worker.bind_device", {"engine.offload.dit": False})

    assert isinstance(changed, ResolvedGeneratorConfig)
    assert changed.engine.offload.dit is False
    assert resolved.engine.offload.dit is True
    assert resolved.override_log == ()
    assert changed.override_log == (("worker.bind_device", {"engine.offload.dit": False}), )
    assert changed.decisions == resolved.decisions
    assert changed.provenance("engine.offload.dit").source == "worker.bind_device"
    assert changed.provenance("engine.parallelism.sp_size").source == "set_sp_size_two"
    assert changed.to_config().engine.offload.dit is False

    with pytest.raises(ResolutionError, match="worker.bind_device"):
        changed.with_override("worker.bind_device", {"engine.offload.dit": False, "engine.not_a_field": 1})
    assert changed.override_log == (("worker.bind_device", {"engine.offload.dit": False}), )


def test_pickle_round_trip_keeps_values_and_provenance() -> None:
    resolved = resolve_generator_config(
        {"model_path": MODEL_PATH, "pipeline": {"experimental": {"flag": True}}},
        [fill_dit_offload_if_unset, set_sp_size_two],
    ).with_override("worker.bind_device", {"engine.parallelism.sp_size": 8})

    restored = pickle.loads(pickle.dumps(resolved))

    assert isinstance(restored, ResolvedGeneratorConfig)
    assert restored.to_dict() == resolved.to_dict()
    assert restored.to_config() == resolved.to_config()
    assert restored.provenance_table() == resolved.provenance_table()
    assert restored.decisions == resolved.decisions
    assert restored.override_log == resolved.override_log
    assert restored.is_explicit("pipeline.experimental.flag") is True
    with pytest.raises(AttributeError):
        restored.engine.num_gpus = 2


def test_view_reports_explicitness_for_yaml_mapping() -> None:
    raw = yaml.safe_load("""
model_path: /models/wan
engine:
  num_gpus: 1
  parallelism:
    sp_size: 2
  quantization:
    transformer_quant: fp8
pipeline:
  preset_overrides: {}
  experimental:
    attention_backend: SAGE_ATTN
""")
    queries = [
        "model_path",
        "revision",
        "engine.num_gpus",
        "engine.parallelism",
        "engine.parallelism.sp_size",
        "engine.parallelism.tp_size",
        "engine.offload",
        "engine.offload.dit",
        "engine.quantization.transformer_quant",
        "engine.quantization.text_encoder_quant",
        "pipeline.preset_overrides",
        "pipeline.experimental.attention_backend",
        "pipeline.experimental.other_key",
    ]
    seen: dict = {}

    def record_explicitness(view: ResolutionView) -> dict:
        seen.update({path: view.is_explicit(path) for path in queries})
        return {}

    resolved = resolve_generator_config(raw, [record_explicitness])

    assert seen == {
        "model_path": True,
        "revision": False,
        # Written with its default value; still explicit because the user wrote it.
        "engine.num_gpus": True,
        "engine.parallelism": True,
        "engine.parallelism.sp_size": True,
        "engine.parallelism.tp_size": False,
        "engine.offload": False,
        "engine.offload.dit": False,
        "engine.quantization.transformer_quant": True,
        "engine.quantization.text_encoder_quant": False,
        "pipeline.preset_overrides": True,
        "pipeline.experimental.attention_backend": True,
        "pipeline.experimental.other_key": False,
    }
    assert all(resolved.is_explicit(path) is explicit for path, explicit in seen.items())
    assert isinstance(resolved.to_config(), GeneratorConfig)
