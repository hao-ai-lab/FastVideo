"""Cookbook metadata checks without model imports, GPU execution, or quick-start YAMLs."""

import json
import shutil

import pytest
import yaml
from pydantic import ValidationError

from docs.cookbook_serving import SOURCE_DIR, build_serving_example, generate_serving_example


@pytest.fixture
def source_dir(tmp_path):
    # The isolated tree has no examples/ directory: it is not a runtime dependency.
    destination = tmp_path / "serving"
    shutil.copytree(SOURCE_DIR, destination)
    return destination


def change_yaml(path, mutate):
    data = yaml.safe_load(path.read_text())
    mutate(data)
    path.write_text(yaml.safe_dump(data))


def test_metadata_is_self_contained_and_keeps_common_and_model_defaults(source_dir):
    data = build_serving_example(source_dir)
    assert data["defaults"]["options"]["seed"]["default"] == 1024
    model = data["recipes"][0]
    assert model["options"]["seed"] == {"default": 1000}
    assert model["options"]["num_gpus"]["also_set"] == ["generator.engine.parallelism.sp_size"]
    assert model["options"]["num_inference_steps"]["choices"] == [9]
    assert model["options"]["vsa_sparsity"]["default"] == 0.8
    assert model["default_hardware"] == "gb200"
    assert data["hardware"]["gb200"] == {"label": "NVIDIA GB200", "runtime": "cuda"}
    assert model["options"]["num_gpus"]["default"] == 4
    assert "source" not in model


def test_generated_json_preserves_authored_metadata(tmp_path):
    output = tmp_path / "assets" / "metadata.json"
    data = generate_serving_example(output)
    assert json.loads(output.read_text()) == data == build_serving_example()


def test_models_preserve_exact_authored_data_without_injecting_defaults(source_dir):
    authored = {
        "defaults": yaml.safe_load((source_dir / "defaults.yaml").read_text()),
        "hardware": yaml.safe_load((source_dir / "hardware.yaml").read_text())["profiles"],
        "recipes": [yaml.safe_load(path.read_text()) for path in sorted((source_dir / "models").glob("*.yaml"))],
    }
    # Comparing JSON also catches numeric coercion such as guidance scale 0 -> 0.0.
    assert json.dumps(build_serving_example(source_dir), sort_keys=True) == json.dumps(authored, sort_keys=True)


@pytest.mark.parametrize(
    "mutate, error",
    [
        (lambda model: model["options"]["num_inference_steps"].update(default=50), "default must be one of choices"),
        (lambda model: model["options"]["seed"].update(default=-1), "below min"),
        (lambda model: model["options"]["seed"].update(default=True), "int_type"),
        (lambda model: model.update(task="unknown"), "unknown task"),
        (lambda model: model.update(runtime="mlx"), "unknown runtime"),
        (lambda model: model.update(default_hardware="other"), "Default hardware"),
        (lambda model: model.update(hardware=["other"], default_hardware="other"), "Unknown hardware"),
        (lambda model: model["options"].update(unknown={"default": 1}), "union_tag_not_found"),
        (lambda model: model["options"]["seed"].update(path="training.seed"), "string_pattern_mismatch"),
        (lambda model: model["settings"]["server"].update(port=9000), "two owners"),
        (lambda model: model["options"]["num_gpus"].update(default=2), "default must be one of choices"),
    ],
)
def test_invalid_model_metadata_fails_before_publish(source_dir, mutate, error):
    change_yaml(source_dir / "models" / "fasth3-v2.yaml", mutate)
    with pytest.raises(ValueError, match=error):
        build_serving_example(source_dir)


def test_unknown_common_task_option_fails_before_publish(source_dir):
    change_yaml(source_dir / "defaults.yaml", lambda data: data["tasks"]["t2v"]["options"].append("unknown"))
    with pytest.raises(ValueError, match="Task references an unknown option"):
        build_serving_example(source_dir)


def test_model_can_define_gpu_count_independently_of_gpu_type(source_dir):
    # Synthetic metadata checks independence; it does not claim two-GPU support.
    change_yaml(source_dir / "models" / "fasth3-v2.yaml",
                lambda model: model["options"]["num_gpus"].update(default=2, choices=[2, 4]))
    data = build_serving_example(source_dir)
    model = data["recipes"][0]
    assert model["default_hardware"] == "gb200"
    assert model["options"]["num_gpus"]["default"] == 2
    assert "gpu_count" not in data["hardware"]["gb200"]
    assert "defaults" not in data["hardware"]["gb200"]


def test_complete_model_only_option_extends_task_options(source_dir):
    data = build_serving_example(source_dir)
    assert "vsa_sparsity" not in data["defaults"]["options"]
    assert "vsa_sparsity" not in data["defaults"]["tasks"]["t2v"]["options"]
    option = data["recipes"][0]["options"]["vsa_sparsity"]
    assert option["type"] == "number"
    assert option["choices"] == [0.8]
    assert option["path"] == "generator.pipeline.experimental.VSA_sparsity"


@pytest.mark.parametrize("value,error", [(True, "int_type"), (float("inf"), "finite"), (float("nan"), "finite")])
def test_number_options_reject_booleans_and_nonfinite_values(source_dir, value, error):
    change_yaml(source_dir / "models" / "fasth3-v2.yaml",
                lambda model: model["options"]["vsa_sparsity"].update(default=value, choices=[value]))
    with pytest.raises(ValueError, match=error):
        build_serving_example(source_dir)


def test_number_options_accept_integer_values(source_dir):
    change_yaml(source_dir / "models" / "fasth3-v2.yaml",
                lambda model: model["options"]["vsa_sparsity"].update(default=1, choices=[1]))
    assert build_serving_example(source_dir)["recipes"][0]["options"]["vsa_sparsity"]["default"] == 1


def test_three_models_select_task_controls_and_keep_independent_defaults(source_dir):
    data = build_serving_example(source_dir)
    models = {model["id"]: model for model in data["recipes"]}
    tasks = data["defaults"]["tasks"]
    assert set(models) == {"fasth3-v2", "wan21-i2v", "zimage-turbo"}
    assert set(tasks) == {"t2v", "i2v", "t2i"}
    assert tasks["t2i"]["client"] == "image"
    assert "num_frames" not in tasks["t2i"]["options"]
    assert tasks["i2v"]["requires_image"] is True
    assert tasks["i2v"]["input_reference"] == "/path/to/first-frame.png"
    assert "num_frames" in tasks["i2v"]["options"]
    assert "num_frames" in tasks["t2v"]["options"]
    for model_id, task, count, steps in [("fasth3-v2", "t2v", 4, 9), ("wan21-i2v", "i2v", 2, 40),
                                         ("zimage-turbo", "t2i", 1, 8)]:
        model = models[model_id]
        assert model["task"] == task
        assert model["options"]["num_gpus"]["default"] == count
        assert model["options"]["num_inference_steps"]["default"] == steps
        assert model["evidence"] == "source-configured"
    assert models["fasth3-v2"]["options"]["num_frames"]["default"] == 124
    assert models["wan21-i2v"]["options"]["num_frames"]["default"] == 77
    assert "input_reference" not in models["wan21-i2v"]["settings"]["default_request"]
    zimage = models["zimage-turbo"]
    assert zimage["settings"]["generator"]["revision"] == "f332072aa78be7aecdf3ee76d5c247082da564a6"
    assert zimage["options"]["guidance_scale"]["choices"] == [0]
    assert zimage["options"]["seed"]["default"] == 42
    assert zimage["settings"]["default_request"]["sampling"]["num_frames"] == 1


@pytest.mark.parametrize(
    "task_name,patch,error",
    [
        ("t2i", {"client": "unknown"}, "literal_error"),
        ("t2v", {"prompt": ""}, "string_too_short"),
        ("i2v", {"requires_image": "true"}, "bool_type"),
        ("i2v", {"input_reference": ""}, "string_too_short"),
        ("t2i", {"requires_image": True}, "only by the video client"),
    ],
)
def test_invalid_task_client_metadata_fails_before_publish(source_dir, task_name, patch, error):
    change_yaml(source_dir / "defaults.yaml", lambda data: data["tasks"][task_name].update(patch))
    with pytest.raises(ValueError, match=error):
        build_serving_example(source_dir)


def test_wan_sampling_steps_obey_shared_limits(source_dir):
    change_yaml(source_dir / "models" / "wan21-i2v.yaml",
                lambda model: model["options"]["num_inference_steps"].update(default=201))
    with pytest.raises(ValueError, match="exceeds max"):
        build_serving_example(source_dir)


@pytest.mark.parametrize(
    "patch,error",
    [
        ({"default": "1000"}, "int_type"),
        ({"default": 1000.0}, "int_type"),
        ({"defualt": 42}, "extra_forbidden"),
        ({"type": "unknown"}, "union_tag_invalid"),
        ({"also_set": "server.extra_seed"}, "list_type"),
        ({"also_set": [42]}, "string_type"),
        ({"also_set": ["training.seed"]}, "string_pattern_mismatch"),
        ({"also_set": ["server.__proto__.seed"]}, "reserved browser property"),
        ({"also_set": ["server.port"]}, "two owners"),
        ({"path": "generator.model_path"}, "two owners"),
        ({"path": "generator.engine"}, "two owners"),
        ({"choices": []}, "default must be one of choices"),
        ({"min": "0"}, "int_type"),
        ({"min": None}, "bounds cannot be null"),
    ],
)
def test_option_schema_rejects_invalid_overrides_with_recipe_context(source_dir, patch, error):
    change_yaml(source_dir / "models" / "fasth3-v2.yaml", lambda model: model["options"]["seed"].update(patch))
    with pytest.raises(ValidationError, match=error) as failure:
        build_serving_example(source_dir)
    assert "fasth3-v2" in str(failure.value)


def test_baseline_scalar_cannot_own_parent_of_option_path(source_dir):
    change_yaml(source_dir / "models" / "fasth3-v2.yaml",
                lambda model: model["settings"]["generator"].update(engine=1))
    with pytest.raises(ValidationError, match="two owners"):
        build_serving_example(source_dir)


def test_disabled_option_does_not_claim_a_baseline_path(source_dir):
    def disable(model):
        model["options"]["vae_offload"] = {"supported": False}
        model["settings"]["generator"]["engine"]["offload"]["vae"] = False

    change_yaml(source_dir / "models" / "fasth3-v2.yaml", disable)
    model = build_serving_example(source_dir)["recipes"][0]
    assert model["options"]["vae_offload"] == {"supported": False}
    assert model["settings"]["generator"]["engine"]["offload"]["vae"] is False


def test_common_option_errors_report_the_field_location(source_dir):
    change_yaml(source_dir / "defaults.yaml", lambda data: data["options"]["port"].update(default="8000"))
    with pytest.raises(ValidationError) as failure:
        build_serving_example(source_dir)
    assert any(error["loc"] == ("defaults", "options", "port", "integer", "default") and error["type"] == "int_type"
               for error in failure.value.errors())


@pytest.mark.parametrize("name", ["defaults.yaml", "hardware.yaml", "models/fasth3-v2.yaml"])
def test_yaml_documents_must_be_mappings(source_dir, name):
    (source_dir / name).write_text("- not-a-mapping\n")
    with pytest.raises(ValidationError, match="model_type"):
        build_serving_example(source_dir)
