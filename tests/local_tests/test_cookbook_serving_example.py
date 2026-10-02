"""Cookbook metadata checks without model imports, GPU execution, or quick-start YAMLs."""

import json
import shutil

import pytest
import yaml
from pydantic import ValidationError

from docs.cookbook_serving import OPTION_ADAPTER, SOURCE_DIR, build_serving_example, generate_serving_example


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
    assert data["defaults"]["options"]["num_gpus"]["default"] == 1
    model = data["recipes"][0]
    assert model["options"]["num_gpus"]["also_set"] == ["generator.engine.parallelism.sp_size"]
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
        (lambda model: model["options"].update(port={"choices": [9000]}), "default must be one of choices"),
        (lambda model: model["options"].update(port={"default": 0}), "below min"),
        (lambda model: model["options"].update(port={"default": True}), "int_type"),
        (lambda model: model.update(task="unknown"), "unknown task"),
        (lambda model: model.update(runtime="mlx"), "unknown runtime"),
        (lambda model: model.update(default_hardware="other"), "Default hardware"),
        (lambda model: model.update(hardware=["other"], default_hardware="other"), "Unknown hardware"),
        (lambda model: model["options"].update(unknown={"default": 1}), "union_tag_not_found"),
        (lambda model: model["options"].update(port={"path": "training.port"}), "string_pattern_mismatch"),
        (lambda model: model["settings"]["server"].update(port=9000), "two owners"),
        (lambda model: model["options"]["num_gpus"].update(default=3), "default must be one of choices"),
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
    assert option["choices"] == [0.0, 0.5, 0.8, 0.9]
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
    assert tasks["i2v"]["requires_image"] is True
    assert set(data["defaults"]["options"]) == {"host", "port", "num_gpus", "vae_offload"}
    for task in tasks.values():
        assert task["options"] == ["host", "port", "num_gpus", "vae_offload"]
        assert "prompt" not in task
        assert "input_reference" not in task
    for model_id, task, count, steps in [("fasth3-v2", "t2v", 4, 9), ("wan21-i2v", "i2v", 2, 40),
                                         ("zimage-turbo", "t2i", 1, 8)]:
        model = models[model_id]
        assert model["task"] == task
        gpu_option = {**data["defaults"]["options"]["num_gpus"], **model["options"]["num_gpus"]}
        assert gpu_option["default"] == count
        assert model["example_request"]["num_inference_steps"] == steps
        assert set(model["settings"]) == {"generator", "server"}
        assert not {"seed", "num_frames", "num_inference_steps", "guidance_scale"}.intersection(model["options"])
        assert "evidence" not in model
        assert "notes" not in model
    assert models["fasth3-v2"]["example_request"]["num_frames"] == 124
    assert models["wan21-i2v"]["example_request"]["num_frames"] == 77
    assert models["wan21-i2v"]["example_request"]["input_reference"] == "/path/to/first-frame.png"
    zimage = models["zimage-turbo"]
    assert zimage["settings"]["generator"]["revision"] == "f332072aa78be7aecdf3ee76d5c247082da564a6"
    assert zimage["example_request"]["guidance_scale"] == 0
    assert zimage["example_request"]["seed"] == 42
    assert zimage["example_request"]["size"] == "1024x1024"


def test_partial_override_inherits_shared_properties_and_replaces_choices(source_dir):
    data = build_serving_example(source_dir)
    shared = data["defaults"]["options"]
    models = {model["id"]: model for model in data["recipes"]}
    # H3 changes the default while inheriting the shared GPU control definition.
    gpu = OPTION_ADAPTER.validate_python({**shared["num_gpus"], **models["fasth3-v2"]["options"]["num_gpus"]})
    assert (gpu.label, gpu.type, gpu.path) == ("GPU count", "integer", "generator.engine.num_gpus")
    assert gpu.default == 4
    assert gpu.choices == [1, 2, 4, 8]
    # Real Wan metadata replaces the shared GPU choices instead of adding to them.
    gpu = OPTION_ADAPTER.validate_python({**shared["num_gpus"], **models["wan21-i2v"]["options"]["num_gpus"]})
    assert shared["num_gpus"]["choices"] == [1, 2, 4, 8]
    assert gpu.choices == [2, 4, 8]
    assert gpu.default == 2


@pytest.mark.parametrize("model_id,option,default,error", [
    ("zimage-turbo", "port", 65536, "exceeds max"),
    ("zimage-turbo", "num_gpus", 2, "default must be one of choices"),
    ("wan21-i2v", "num_gpus", 1, "default must be one of choices"),
])
def test_partial_overrides_keep_inherited_constraints(source_dir, model_id, option, default, error):
    change_yaml(source_dir / "models" / f"{model_id}.yaml",
                lambda model: model["options"].setdefault(option, {}).update(default=default))
    with pytest.raises(ValidationError, match=error):
        build_serving_example(source_dir)


def test_model_can_override_only_max_without_repeating_common_option(source_dir):
    change_yaml(source_dir / "models" / "fasth3-v2.yaml", lambda model: model["options"].update(port={"max": 9000}))
    data = build_serving_example(source_dir)
    patch = data["recipes"][0]["options"]["port"]
    assert patch == {"max": 9000}
    port = OPTION_ADAPTER.validate_python({**data["defaults"]["options"]["port"], **patch})
    assert (port.type, port.label, port.path, port.min, port.max, port.default) == (
        "integer", "Server port", "server.port", 1, 9000, 8000)


@pytest.mark.parametrize(
    "task_name,patch,error",
    [
        ("t2i", {"client": "unknown"}, "literal_error"),
        ("t2v", {"prompt": "unexpected request setting"}, "extra_forbidden"),
        ("i2v", {"requires_image": "true"}, "bool_type"),
        ("i2v", {"input_reference": "/tmp/image.png"}, "extra_forbidden"),
        ("t2i", {"requires_image": True}, "only by the video client"),
    ],
)
def test_invalid_task_client_metadata_fails_before_publish(source_dir, task_name, patch, error):
    change_yaml(source_dir / "defaults.yaml", lambda data: data["tasks"][task_name].update(patch))
    with pytest.raises(ValueError, match=error):
        build_serving_example(source_dir)


def test_static_smoke_requests_keep_model_requirements_out_of_deployment(source_dir):
    models = {model["id"]: model for model in build_serving_example(source_dir)["recipes"]}
    h3 = models["fasth3-v2"]["example_request"]
    assert {key: h3[key] for key in ("width", "height", "num_frames", "fps", "num_inference_steps")} == {
        "width": 1344, "height": 768, "num_frames": 124, "fps": 24, "num_inference_steps": 9
    }
    for model in models.values():
        assert model["example_request"]["prompt"].strip()
        assert not {"model", "batch_cfg", "return_frames"}.intersection(model["example_request"])
        assert "default_request" not in model["settings"]


@pytest.mark.parametrize("patch,error", [
    ({"prompt": ""}, "nonempty prompt"),
    ({"prompt": 42}, "nonempty prompt"),
    ({"model": "other"}, "model is supplied by the resolver"),
])
def test_static_request_requires_prompt_and_resolver_owned_model(source_dir, patch, error):
    change_yaml(source_dir / "models" / "fasth3-v2.yaml", lambda model: model["example_request"].update(patch))
    with pytest.raises(ValidationError, match=error):
        build_serving_example(source_dir)


def test_static_request_is_required(source_dir):
    change_yaml(source_dir / "models" / "fasth3-v2.yaml", lambda model: model.pop("example_request"))
    with pytest.raises(ValidationError, match="example_request"):
        build_serving_example(source_dir)


@pytest.mark.parametrize("image", [None, "", "  ", 42])
def test_i2v_static_request_requires_image_reference(source_dir, image):
    change_yaml(source_dir / "models" / "wan21-i2v.yaml",
                lambda model: model["example_request"].update(input_reference=image))
    with pytest.raises(ValidationError, match="nonempty input_reference"):
        build_serving_example(source_dir)


@pytest.mark.parametrize("mutate", [
    lambda model: model["settings"].update(default_request={"sampling": {"seed": 42}}),
    lambda model: model["options"].update(port={"path": "default_request.sampling.seed"}),
    lambda model: model["options"].update(port={"also_set": ["default_request.sampling.seed"]}),
])
def test_request_settings_cannot_enter_deployment_paths(source_dir, mutate):
    change_yaml(source_dir / "models" / "fasth3-v2.yaml", mutate)
    with pytest.raises(ValidationError, match="string_pattern_mismatch"):
        build_serving_example(source_dir)


@pytest.mark.parametrize(
    "patch,error",
    [
        ({"default": "8000"}, "int_type"),
        ({"default": 8000.0}, "int_type"),
        ({"defualt": 42}, "extra_forbidden"),
        ({"type": "unknown"}, "union_tag_invalid"),
        ({"also_set": "server.extra_port"}, "list_type"),
        ({"also_set": [42]}, "string_type"),
        ({"also_set": ["training.port"]}, "string_pattern_mismatch"),
        ({"also_set": ["server.__proto__.port"]}, "reserved browser property"),
        ({"also_set": ["server.port"]}, "two owners"),
        ({"path": "generator.model_path"}, "two owners"),
        ({"path": "generator.engine"}, "two owners"),
        ({"choices": []}, "default must be one of choices"),
        ({"min": "0"}, "int_type"),
        ({"min": None}, "bounds cannot be null"),
    ],
)
def test_option_schema_rejects_invalid_overrides_with_recipe_context(source_dir, patch, error):
    change_yaml(source_dir / "models" / "fasth3-v2.yaml", lambda model: model["options"].update(port=patch))
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
