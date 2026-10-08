"""Recipe authoring and metadata delivery checks; no weights or GPU inference."""

import copy
import json
import subprocess
from pathlib import Path

import pytest
import yaml

from docs import cookbook_config

MODEL = "FastVideo/FastWan2.1-T2V-1.3B-Diffusers"


def fixture_recipe(tmp_path, **updates):
    baseline = yaml.safe_load((cookbook_config.ROOT / "examples/serving/openai_fastwan21_1_3b.yaml").read_text())
    config_path = tmp_path / "examples/serving/config.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(baseline))
    recipes_dir = tmp_path / "recipes"
    recipes_dir.mkdir(exist_ok=True)
    manifest = {
        "runtime": "fastvideo-cuda-rest", "workload": "t2v",
        "config": "examples/serving/config.yaml",
        "controls": ["server.port", "generator.engine.compile.enabled", "default_request.sampling.num_frames"],
        "env": {"FASTVIDEO_ATTENTION_BACKEND": "VIDEO_SPARSE_ATTN"},
    }
    manifest.update(updates)
    (recipes_dir / "example.yaml").write_text(yaml.safe_dump({"title": "Example", "deployments": {"cuda-rest": manifest}}, sort_keys=False))
    return recipes_dir, baseline, config_path


def test_manifest_controls_select_fields_without_filtering_baseline(tmp_path):
    recipes, baseline, _ = fixture_recipe(tmp_path)
    result = cookbook_config.build_catalogs(recipes, tmp_path)[0]
    assert result["base_config"] == baseline
    assert result["base_config"]["generator"]["pipeline"]["experimental"]["dmd_denoising_steps"] == [1000, 757, 522]
    assert result["model"] == {"id": MODEL, "key": "example", "title": "Example"}
    assert result["deployment"] == {"id": "cuda-rest", "label": "CUDA REST"}
    assert [control["path"] for control in result["controls"]] == [
        "server.port", "generator.engine.compile.enabled", "default_request.sampling.num_frames"]
    assert result["controls"][0]["schema"]["type"] == "integer"
    assert result["controls"][1]["schema"]["type"] == "boolean"
    assert result["runtime"]["server_defaults"] == {"host": "0.0.0.0", "port": 8000, "served_model_name": None}
    assert result["env"]["FASTVIDEO_ATTENTION_BACKEND"] == "VIDEO_SPARSE_ATTN"
    assert "schema" not in result
    assert "hardware" not in result


def test_inherited_defaults_only_describe_controls_and_never_expand_baseline(tmp_path):
    recipes, baseline, config = fixture_recipe(tmp_path, controls=["default_request.sampling.num_frames", "server.port"])
    del baseline["default_request"]
    del baseline["server"]
    config.write_text(yaml.safe_dump(baseline))
    result = cookbook_config.build_catalogs(recipes, tmp_path)[0]
    assert result["base_config"] == baseline
    assert "default_request" not in result["base_config"]
    assert result["controls"][0]["schema"]["default"] == 125
    assert result["controls"][1]["schema"]["default"] == 8000
    assert result["controls"][0]["schema"]["minimum"] == -cookbook_config.JS_SAFE_INTEGER
    assert "multipleOf" not in result["controls"][0]["schema"]


@pytest.mark.parametrize("updates, message", [
    ({"runtime": "mlx"}, "unsupported runtime"),
    ({"workload": "streaming"}, "unsupported workload"),
    ({"workload": "t2i"}, "unsupported workload"),
    ({"controls": {"add": ["server.port"]}}, "explicit list"),
    ({"controls": ["server.port", "server.port"]}, "duplicate control"),
    ({"controls": [1]}, "explicit list"),
    ({"env": {"BAD-NAME": "1"}}, "environment names"),
    ({"env": {"VALUE": True}}, "environment names"),
    ({"requirements": "install"}, "list of text"),
    ({"unknown": 1}, "expected"),
    ({"hardware": {}}, "expected"),
    ({"config": "../secret.yaml"}, "under examples/serving"),
])
def test_bad_manifests_fail_clearly(tmp_path, updates, message):
    recipes, _, _ = fixture_recipe(tmp_path, **updates)
    with pytest.raises(ValueError, match=message):
        cookbook_config.load_recipes(recipes, tmp_path)


@pytest.mark.parametrize("path", [
    "server.missing", "generator.model_path", "generator.pipeline.preset", "generator.pipeline.preset_version",
    "generator.pipeline.components.vae_weights", "generator.pipeline.workload_type", "streaming.warmup.enabled",
    "generator.pipeline.experimental.VSA_sparsity", "generator.engine",
    "default_request.output.save_video", "default_request.output.return_frames", "default_request.output.output_path", "server.__proto__.polluted",
])
def test_controls_need_typed_metadata_and_cannot_change_identity_or_runtime(tmp_path, path):
    recipes, _, _ = fixture_recipe(tmp_path, controls=[path])
    with pytest.raises(ValueError, match="control|metadata"):
        cookbook_config.build_catalogs(recipes, tmp_path)


@pytest.mark.parametrize("mutation, message", [
    (lambda raw: raw.update(streaming={}), "streaming"),
    (lambda raw: raw["generator"].update(model_path="Unknown/Model"), "unknown registered model"),
    (lambda raw: raw["generator"]["pipeline"].update(preset="unimplemented"), "compatibility adapter"),
    (lambda raw: raw["generator"]["engine"].update(num_gpus="two"), "num_gpus"),
    (lambda raw: raw["generator"]["engine"]["parallelism"].update(sp_size=3), "divisible"),
    (lambda raw: raw["generator"]["pipeline"].update(workload_type="i2v"), "disagrees"),
])
def test_baseline_passes_native_parser_and_serving_adapter(tmp_path, mutation, message):
    recipes, baseline, config = fixture_recipe(tmp_path)
    mutation(baseline)
    config.write_text(yaml.safe_dump(baseline))
    with pytest.raises((ValueError, NotImplementedError, AssertionError), match=message):
        cookbook_config.build_catalogs(recipes, tmp_path)


def test_another_registered_model_uses_the_same_builder(tmp_path):
    recipes, _, config = fixture_recipe(tmp_path, workload="i2v")
    config.write_text((cookbook_config.ROOT / "examples/serving/openai_wan22_ti2v_5b.yaml").read_text())
    result = cookbook_config.build_catalogs(recipes, tmp_path)[0]
    assert result["model"]["id"] == "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
    assert result["workload"] == "i2v"


def test_export_order_manifest_urls_and_pruning(tmp_path):
    recipes, _, _ = fixture_recipe(tmp_path)
    manifest = yaml.safe_load((recipes / "example.yaml").read_text())
    manifest["deployments"]["cuda-rest"]["config"] = "examples/serving/another.yaml"
    (tmp_path / "examples/serving/another.yaml").write_text(
        (cookbook_config.ROOT / "examples/serving/openai_fasth3_8step.yaml").read_text())
    (recipes / "another.yaml").write_text(yaml.safe_dump(manifest))
    output = tmp_path / "site/assets/cookbook-config"
    index = cookbook_config.export_catalogs(output, recipes, tmp_path)
    assert [row["id"] for row in index["models"]] == ["another", "example"]
    assert set(index) == {"models"}
    for row in [deployment for model in index["models"] for deployment in model["deployments"]]:
        catalog = json.loads((output / row["catalog_url"]).read_text())
        assert catalog["id"] == row["id"]
        assert "base_config" not in row
    assert json.loads((output / "recipes/example/cuda-rest.json").read_text()) == cookbook_config.build_catalogs(recipes, tmp_path)[1]
    marker = output / "keep.txt"
    marker.write_text("not an exporter file")
    (recipes / "another.yaml").unlink()
    cookbook_config.export_catalogs(output, recipes, tmp_path)
    assert not (output / "recipes/another/cuda-rest.json").exists()
    assert marker.exists()


def test_failed_export_preserves_previous_files(tmp_path):
    recipes, _, _ = fixture_recipe(tmp_path)
    output = tmp_path / "output"
    cookbook_config.export_catalogs(output, recipes, tmp_path)
    before = {path: path.read_bytes() for path in output.rglob("*.json")}
    (recipes / "broken.yaml").write_text("title: incomplete")
    with pytest.raises(ValueError):
        cookbook_config.export_catalogs(output, recipes, tmp_path)
    assert {path: path.read_bytes() for path in output.rglob("*.json")} == before


def test_duplicate_ids_and_empty_recipe_directory(tmp_path):
    recipes, _, _ = fixture_recipe(tmp_path)
    (recipes / "example.yml").write_text((recipes / "example.yaml").read_text())
    with pytest.raises(ValueError, match="duplicate recipe ID"):
        cookbook_config.load_recipes(recipes, tmp_path)
    (recipes / "example.yml").unlink()
    (recipes / "example.yaml").unlink()
    with pytest.raises(ValueError, match="No serving recipe"):
        cookbook_config.load_recipes(recipes, tmp_path)


def test_exports_require_no_http_schema_or_model_download(monkeypatch, tmp_path):
    import fastvideo.registry as registry
    from fastvideo.entrypoints.openai.protocol import VideoGenerationRequest
    def forbidden(*args, **kwargs):
        pytest.fail("Exporter used HTTP schema generation or downloaded checkpoint metadata")
    monkeypatch.setattr(VideoGenerationRequest, "model_json_schema", forbidden)
    monkeypatch.setattr(registry, "maybe_download_model_index", forbidden)
    recipes, _, _ = fixture_recipe(tmp_path)
    cookbook_config.build_catalogs(recipes, tmp_path)


def test_local_schema_refs_annotations_null_and_nonmutation():
    source = {"$defs": {"Field": {"anyOf": [{"type": "integer"}, {"type": "null"}], "default": None}},
              "properties": {"value": {"$ref": "#/$defs/Field", "description": "An optional value"}}}
    before = copy.deepcopy(source)
    result = cookbook_config._inline_references(source, source)
    assert result["properties"]["value"] == {
        "anyOf": [{"type": "integer"}, {"type": "null"}], "default": None, "description": "An optional value"}
    assert source == before
    assert cookbook_config._prepare_schema(result)["properties"]["value"]["default"] is None
    for schema in [{"$ref": "https://example.com/schema"}, {"$defs": {"X": {"$ref": "#/$defs/X"}}, "$ref": "#/$defs/X"}]:
        with pytest.raises(ValueError):
            cookbook_config._inline_references(schema, schema)


def test_literal_schema_data_is_not_rewritten():
    field = {"type": "object", "additionalProperties": True, "default": {"$ref": "literal", "type": "integer"}}
    assert cookbook_config._prepare_schema(field) == field
    assert cookbook_config._inline_references(field, field) == field


def test_default_generated_files_are_ignored():
    paths = [cookbook_config.OUTPUT_DIR / "index.json", cookbook_config.OUTPUT_DIR / "recipes/example/cuda-rest.json"]
    result = subprocess.run(["git", "check-ignore", "--no-index", *map(str, paths)], cwd=cookbook_config.ROOT,
                            check=True, capture_output=True, text=True)
    assert set(result.stdout.splitlines()) == set(map(str, paths))


def test_markdown_guide_is_a_page_reference_not_another_renderer(tmp_path):
    guide = tmp_path / "docs/cookbook/guides/example.md"
    guide.parent.mkdir(parents=True, exist_ok=True)
    markdown = '# Example "guide"\n\n```bash\nfastvideo serve --config config.yaml\n```\n\n| A | B |\n|---|---|\n| 1 | 2 |\n'
    guide.write_text(markdown)
    recipes, baseline, _ = fixture_recipe(tmp_path, guide="docs/cookbook/guides/example.md")
    site = tmp_path / "site"
    output = site / "assets/cookbook-config"
    cookbook_config.export_catalogs(output, recipes, tmp_path)
    catalog = json.loads((output / "recipes/example/cuda-rest.json").read_text())
    assert catalog["guide"] == {"url": "../../cookbook/guides/example/"}
    assert catalog["base_config"] == baseline
    assert guide.read_text() == markdown


@pytest.mark.parametrize("guide", ["../private.md", "/tmp/private.md", "docs/missing.md", "examples/serving/config.yaml"])
def test_guides_are_existing_markdown_files_inside_docs(tmp_path, guide):
    recipes, _, _ = fixture_recipe(tmp_path, guide=guide)
    with pytest.raises(ValueError, match="guide must"):
        cookbook_config.build_catalogs(recipes, tmp_path)


def test_index_markdown_guide_uses_its_directory_url(tmp_path):
    guide = tmp_path / "docs/cookbook/guides/index.md"
    guide.parent.mkdir(parents=True)
    guide.write_text("# Guides\n")
    assert cookbook_config._guide_link("docs/cookbook/guides/index.md", tmp_path) == {"url": "../../cookbook/guides/"}


def test_import_does_not_initialize_fastvideo():
    result = subprocess.run(["python3", "-S", "-c", "import docs.cookbook_config, sys; assert 'fastvideo' not in sys.modules"],
                            cwd=cookbook_config.ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_hidden_unsafe_integer_is_rejected_before_browser_rounding(tmp_path):
    recipes, baseline, config = fixture_recipe(tmp_path)
    baseline["default_request"]["sampling"]["seed"] = 2**63 - 1
    config.write_text(yaml.safe_dump(baseline))
    with pytest.raises(ValueError, match="browser-safe range"):
        cookbook_config.build_catalogs(recipes, tmp_path)


def test_namespaces_expand_all_leaves_in_manifest_then_declaration_order(tmp_path):
    recipes, baseline, _ = fixture_recipe(tmp_path, controls=[
        "default_request.sampling.num_frames", "server", "generator.engine.offload",
        "generator.engine.compile.enabled",
    ])
    result = cookbook_config.build_catalogs(recipes, tmp_path)[0]
    assert [control["path"] for control in result["controls"]] == [
        "default_request.sampling.num_frames",
        "server.host", "server.port", "server.output_dir", "server.served_model_name",
        "generator.engine.offload.dit", "generator.engine.offload.dit_layerwise",
        "generator.engine.offload.text_encoder", "generator.engine.offload.image_encoder",
        "generator.engine.offload.vae", "generator.engine.offload.pin_cpu_memory",
        "generator.engine.offload.lazy_module_load", "generator.engine.compile.enabled",
    ]
    assert result["base_config"] == baseline
    # Expansion selects controls, never projects or expands the native baseline.
    assert result["base_config"]["generator"]["pipeline"]["experimental"] == baseline["generator"]["pipeline"]["experimental"]
    lazy = next(control for control in result["controls"] if control["path"].endswith("lazy_module_load"))
    assert lazy["schema"]["anyOf"] == [{"type": "boolean"}, {"type": "null"}]


@pytest.mark.parametrize("selector, offending", [
    ("generator.engine", "generator.engine.compile.extras"),
    ("generator.engine.compile", "generator.engine.compile.extras"),
    ("default_request.output", "default_request.output.output_path"),
    ("generator.engine.quantization", "generator.engine.quantization"),
])
def test_namespace_rejects_any_unsupported_descendant_instead_of_filtering(tmp_path, selector, offending):
    recipes, _, _ = fixture_recipe(tmp_path, controls=[selector])
    with pytest.raises(ValueError) as failure:
        cookbook_config.build_catalogs(recipes, tmp_path)
    message = str(failure.value)
    assert f"Control selector '{selector}'" in message
    assert f"unsupported field '{offending}'" in message
    assert "use narrower selectors" in message


@pytest.mark.parametrize("selectors", [
    ["server", "server.port"], ["server.port", "server"],
    ["generator.engine.offload", "generator.engine.offload.vae"],
])
def test_overlapping_selectors_are_rejected_in_both_orders(tmp_path, selectors):
    recipes, _, _ = fixture_recipe(tmp_path, controls=selectors)
    with pytest.raises(ValueError, match="Overlapping control selector.*already enabled"):
        cookbook_config.build_catalogs(recipes, tmp_path)


def test_arrays_stay_leaves_and_empty_or_unknown_namespaces_fail():
    schema = {"properties": {"server": {"type": "object", "properties": {
        "values": {"type": "array", "items": {"type": "string"}},
    }}}}
    before = copy.deepcopy(schema)
    result = cookbook_config._expand_controls(schema, ["server"])
    assert result == [{"path": "server.values", "schema": {"type": "array", "items": {"type": "string"}}}]
    result[0]["schema"]["items"]["type"] = "integer"
    assert schema == before
    with pytest.raises(ValueError, match="No public field metadata"):
        cookbook_config._expand_controls(schema, ["server.unknown"])
    schema["properties"]["server"]["properties"] = {}
    with pytest.raises(ValueError, match="use narrower selectors"):
        cookbook_config._expand_controls(schema, ["server"])


def test_explicit_empty_controls_remains_empty(tmp_path):
    recipes, baseline, _ = fixture_recipe(tmp_path, controls=[])
    result = cookbook_config.build_catalogs(recipes, tmp_path)[0]
    assert result["controls"] == []
    assert result["base_config"] == baseline


def test_controls_are_required_instead_of_exposing_fields_implicitly(tmp_path):
    recipes, _, _ = fixture_recipe(tmp_path)
    manifest_path = recipes / "example.yaml"
    manifest = yaml.safe_load(manifest_path.read_text())
    del manifest["deployments"]["cuda-rest"]["controls"]
    manifest_path.write_text(yaml.safe_dump(manifest))
    with pytest.raises(ValueError, match="expected.*controls"):
        cookbook_config.load_recipes(recipes, tmp_path)


def test_gpu_and_parallelism_namespace_are_editable_without_changing_baseline(tmp_path):
    from fastvideo.api.schema import EngineConfig

    recipes, baseline, _ = fixture_recipe(tmp_path, controls=[
        "generator.engine.num_gpus", "generator.engine.parallelism", "server",
    ])
    result = cookbook_config.build_catalogs(recipes, tmp_path)[0]
    fields = {control["path"]: control["schema"] for control in result["controls"]}
    assert list(fields) == [
        "generator.engine.num_gpus", "generator.engine.parallelism.tp_size",
        "generator.engine.parallelism.sp_size", "generator.engine.parallelism.hsdp_replicate_dim",
        "generator.engine.parallelism.hsdp_shard_dim", "generator.engine.parallelism.dist_timeout",
        "server.host", "server.port", "server.output_dir", "server.served_model_name",
    ]
    assert fields["generator.engine.num_gpus"]["type"] == "integer"
    assert fields["generator.engine.parallelism.tp_size"]["default"] == -1
    assert fields["generator.engine.parallelism.sp_size"]["default"] == -1
    assert fields["generator.engine.parallelism.dist_timeout"]["anyOf"][1] == {"type": "null"}
    engine = EngineConfig()
    assert result["runtime"]["topology_defaults"] == {
        "num_gpus": engine.num_gpus,
        "tp_size": engine.parallelism.tp_size,
        "sp_size": engine.parallelism.sp_size,
        "hsdp_replicate_dim": engine.parallelism.hsdp_replicate_dim,
        "hsdp_shard_dim": engine.parallelism.hsdp_shard_dim,
    }
    assert result["base_config"] == baseline
    assert result["base_config"]["generator"]["pipeline"]["experimental"] == (
        baseline["generator"]["pipeline"]["experimental"])


def fixture_mlx_deployment(tmp_path):
    recipes, _, _ = fixture_recipe(tmp_path)
    manifest_path = recipes / "example.yaml"
    manifest = yaml.safe_load(manifest_path.read_text())
    baseline = yaml.safe_load((cookbook_config.ROOT / "examples/serving/mlx_fasth3_8step.yaml").read_text())
    config = tmp_path / "examples/serving/mlx.yaml"
    config.write_text(yaml.safe_dump(baseline))
    manifest["deployments"] = {"mlx-rest": {
        "runtime": "fastvideo-mlx-rest", "workload": "t2v", "config": "examples/serving/mlx.yaml",
        "controls": ["server", "generator.model_root", "generator.mlx_checkpoint", "generator.vae_dtype",
                     "generator.vsa", "generator.vsa_sparsity", "generator.vsa_tile_size"],
    }}
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False))
    return recipes, baseline, config


def test_mlx_controls_reuse_native_types_and_preserve_checkpoint_paths(tmp_path, monkeypatch):
    from fastvideo.entrypoints.openai import mlx_server
    import fastvideo.registry as registry

    def forbidden(*args, **kwargs):
        pytest.fail("Metadata export must not load models, query the Hub, or borrow HTTP schemas")

    monkeypatch.setattr(mlx_server, "MLXH3Generator", forbidden)
    monkeypatch.setattr(mlx_server.VideoGenerationRequest, "model_json_schema", forbidden)
    monkeypatch.setattr(registry, "maybe_download_model_index", forbidden)
    recipes, baseline, config = fixture_mlx_deployment(tmp_path)
    output = tmp_path / "must-not-be-created"
    baseline["server"]["output_dir"] = str(output)
    del baseline["server"]["served_model_name"]
    config.write_text(yaml.safe_dump(baseline))
    result = cookbook_config.build_catalogs(recipes, tmp_path)[0]
    assert not output.exists()
    assert result["base_config"] == baseline
    assert result["base_config"]["generator"]["mlx_checkpoint"] == "./FastH3-8-Step-V2-MLX/int8"
    assert result["runtime"]["server_defaults"]["served_model_name"] == "fasth3"
    assert "served_model_name" not in result["base_config"]["server"]
    assert result["runtime"]["backend"] == "mlx"
    assert result["runtime"]["interface"] == "rest"
    assert result["runtime"]["hardware_label"] == "Apple Silicon"
    assert result["runtime"]["launch_argv"] == [
        "python", "-m", "fastvideo.entrypoints.openai.mlx_server", "--config", "config.yaml"]
    fields = {item["path"]: item["schema"] for item in result["controls"]}
    assert fields["generator.vae_dtype"]["enum"] == ["fp32", "fp16", "bf16"]
    assert fields["generator.vsa_sparsity"]["minimum"] == 0
    assert fields["generator.vsa_sparsity"]["exclusiveMaximum"] == 1
    assert fields["generator.vsa_tile_size"]["enum"] == [64, 256]
    assert fields["server.port"]["minimum"] == 1
    assert fields["server.port"]["maximum"] == 65535
    assert fields["generator.vsa"]["type"] == "boolean"


@pytest.mark.parametrize("field, value, message", [
    ("num_inference_steps", 3, "sigma points"),
    ("guidance_scale", 2, "guidance_scale=1"),
    ("seed", -1, "seed must be"),
])
def test_mlx_baseline_uses_native_serving_admission(tmp_path, field, value, message):
    recipes, baseline, config = fixture_mlx_deployment(tmp_path)
    baseline["default_request"]["sampling"][field] = value
    config.write_text(yaml.safe_dump(baseline))
    with pytest.raises(ValueError, match=message):
        cookbook_config.build_catalogs(recipes, tmp_path)


@pytest.mark.parametrize("path", ["generator.model_path", "generator.engine.offload.vae",
                                  "default_request.sampling.num_frames", "runtime"])
def test_mlx_does_not_advertise_cuda_or_opaque_request_fields(tmp_path, path):
    recipes, _, _ = fixture_mlx_deployment(tmp_path)
    manifest_path = recipes / "example.yaml"
    manifest = yaml.safe_load(manifest_path.read_text())
    manifest["deployments"]["mlx-rest"]["controls"] = [path]
    manifest_path.write_text(yaml.safe_dump(manifest))
    with pytest.raises(ValueError, match="control|metadata"):
        cookbook_config.build_catalogs(recipes, tmp_path)


def test_deployments_share_identity_but_not_baselines_or_controls(tmp_path):
    recipes, baseline, _ = fixture_mlx_deployment(tmp_path)
    cuda_path = tmp_path / "examples/serving/cuda.yaml"
    cuda_path.write_text((cookbook_config.ROOT / "examples/serving/openai_fasth3_8step.yaml").read_text())
    manifest_path = recipes / "example.yaml"
    manifest = yaml.safe_load(manifest_path.read_text())
    manifest["summary"] = "Model overview"
    guide = tmp_path / "docs/guide.md"
    guide.parent.mkdir(exist_ok=True)
    guide.write_text("# Model guide\n")
    manifest["guide"] = "docs/guide.md"
    manifest["deployments"]["cuda-rest"] = {
        "label": "CUDA workstation", "runtime": "fastvideo-cuda-rest", "workload": "t2v",
        "config": "examples/serving/cuda.yaml", "controls": ["server.port"],
        "env": {"CUDA_VISIBLE_DEVICES": "0"}, "summary": "CUDA-specific summary",
    }
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False))
    catalogs = cookbook_config.build_catalogs(recipes, tmp_path)
    assert [item["id"] for item in catalogs] == ["example/mlx-rest", "example/cuda-rest"]
    assert len({item["model"]["id"] for item in catalogs}) == 1
    assert catalogs[0]["summary"] == "Model overview"
    assert catalogs[1]["summary"] == "CUDA-specific summary"
    assert catalogs[0]["guide"] == catalogs[1]["guide"] == {"url": "../../guide/"}
    assert catalogs[0]["env"] == {}
    assert catalogs[1]["env"] == {"CUDA_VISIBLE_DEVICES": "0"}
    assert catalogs[0]["base_config"] == baseline
    assert len(catalogs[1]["controls"]) == 1
    assert catalogs[1]["deployment"]["label"] == "CUDA workstation"
    output = tmp_path / "output"
    index = cookbook_config.export_catalogs(output, recipes, tmp_path)
    assert [row["id"] for row in index["models"][0]["deployments"]] == [
        "example/mlx-rest", "example/cuda-rest"]
    assert index["models"][0]["deployments"][1]["label"] == "CUDA workstation"
    assert all((output / row["catalog_url"]).is_file() for row in index["models"][0]["deployments"])
    del manifest["deployments"]["mlx-rest"]
    manifest_path.write_text(yaml.safe_dump(manifest))
    cookbook_config.export_catalogs(output, recipes, tmp_path)
    assert not (output / "recipes/example/mlx-rest.json").exists()
    assert (output / "recipes/example/cuda-rest.json").exists()


def test_one_model_manifest_rejects_deployments_for_different_models(tmp_path):
    recipes, _, _ = fixture_mlx_deployment(tmp_path)
    manifest_path = recipes / "example.yaml"
    manifest = yaml.safe_load(manifest_path.read_text())
    manifest["deployments"]["cuda-rest"] = {
        "runtime": "fastvideo-cuda-rest", "workload": "t2v", "config": "examples/serving/config.yaml",
        "controls": [],
    }
    manifest_path.write_text(yaml.safe_dump(manifest))
    with pytest.raises(ValueError, match="same model_path"):
        cookbook_config.build_catalogs(recipes, tmp_path)


def test_model_identity_cannot_be_split_across_manifests(tmp_path):
    recipes, _, _ = fixture_recipe(tmp_path)
    (recipes / "another.yaml").write_text((recipes / "example.yaml").read_text())
    with pytest.raises(ValueError, match="duplicate model identity.*add deployments"):
        cookbook_config.build_catalogs(recipes, tmp_path)


def test_streaming_runtime_reuses_native_config_without_importing_engine(tmp_path, monkeypatch):
    import builtins

    recipes, _, config_path = fixture_recipe(tmp_path, runtime="fastvideo-cuda-streaming",
                                             controls=["server.host", "server.port", "default_request.sampling.num_frames"])
    baseline = yaml.safe_load((cookbook_config.ROOT / "examples/serving/streaming_demo.yaml").read_text())
    config_path.write_text(yaml.safe_dump(baseline))
    original_import = builtins.__import__

    def no_streaming_runtime(name, *args, **kwargs):
        if name.startswith("fastvideo.entrypoints.streaming"):
            pytest.fail("The docs exporter must not import streaming runtime or GPU dependencies")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_streaming_runtime)
    result = cookbook_config.build_catalogs(recipes, tmp_path)[0]
    assert result["base_config"] == baseline
    assert result["base_config"]["streaming"]["stream_mode"] == "av_fmp4"
    assert result["base_config"]["streaming"]["pool"]["conditioning_num_frames"] == 9
    assert result["runtime"]["backend"] == "cuda"
    assert result["runtime"]["interface"] == "websocket"
    assert result["runtime"]["client_guide_url"] == "../../design/server_contracts/streaming/"
    assert result["runtime"]["launch_argv"] == ["fastvideo", "serve", "--config", "config.yaml"]


def test_streaming_deployment_requires_native_streaming_block(tmp_path):
    recipes, _, _ = fixture_recipe(tmp_path, runtime="fastvideo-cuda-streaming")
    with pytest.raises(ValueError, match="requires a streaming block"):
        cookbook_config.build_catalogs(recipes, tmp_path)


@pytest.mark.parametrize("path", ["server", "server.served_model_name", "server.output_dir"])
def test_streaming_does_not_expose_ignored_rest_server_controls(tmp_path, path):
    recipes, _, config_path = fixture_recipe(tmp_path, runtime="fastvideo-cuda-streaming", controls=[path])
    config_path.write_text((cookbook_config.ROOT / "examples/serving/streaming_demo.yaml").read_text())
    with pytest.raises(ValueError, match="Unsupported streaming control: server"):
        cookbook_config.build_catalogs(recipes, tmp_path)


def test_stale_catalog_pruning_preserves_unowned_files_and_symlink_targets(tmp_path):
    recipes, _, _ = fixture_recipe(tmp_path)
    output = tmp_path / "output"
    cookbook_config.export_catalogs(output, recipes, tmp_path)
    stale = output / "recipes/example/removed.json"
    stale.write_text("{}")
    marker = output / "recipes/example/keep.custom.json"
    marker.write_text("{}")
    outside = tmp_path / "external"
    outside.mkdir()
    external_catalog = outside / "cuda-rest.json"
    external_catalog.write_text("{}")
    (output / "recipes/external").symlink_to(outside, target_is_directory=True)
    cookbook_config.export_catalogs(output, recipes, tmp_path)
    assert not stale.exists()
    assert marker.exists()
    assert external_catalog.exists()
