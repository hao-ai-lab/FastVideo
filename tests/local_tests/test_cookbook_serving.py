"""Check serving snippets against their executable sources, without model imports."""

import ast
import copy
import json
from pathlib import Path
from html.parser import HTMLParser

import pytest
import yaml

from docs.generate_examples import COOKBOOK_DATA, Example, cookbook_serving_profile, validate_cookbook

ROOT = Path(__file__).resolve().parents[2]


def serving_recipe():
    recipes = json.loads(COOKBOOK_DATA.read_text())["recipes"]
    return next(recipe for recipe in recipes if recipe["id"] == "fasth3-preview-cuda")


def test_cookbook_serving_does_not_inherit_local_benchmark():
    recipe = serving_recipe()
    assert recipe["hardware"]["evidence"] == "validated"
    profile = cookbook_serving_profile(recipe)
    assert profile["hardware"] == {"platform": "cuda", "gpu_count": 4, "evidence": "source-configured"}
    assert profile["model"] == "fasth3"
    assert profile["sampling"]["num_frames"] == 124
    assert "--server.host 127.0.0.1" in profile["command"]
    assert profile["playground_url"] == "http://127.0.0.1:8000/playground/"
    for client in profile["clients"].values():
        assert client["code"] == (ROOT / client["source"]).read_text()
    validate_cookbook()


@pytest.mark.parametrize(
    "config_path",
    [
        "examples/inference/basic/basic_fasth3_spark.yaml",
        "examples/inference/basic/basic_fasth3_spark_pair.yaml",
        "examples/serving/openai_fasth3_spark.yaml",
    ],
)
def test_spark_configs_use_lazy_load_not_sequential(config_path):
    cfg = yaml.safe_load((ROOT / config_path).read_text())
    offload = cfg["generator"]["engine"]["offload"]
    assert offload["lazy_module_load"] is True
    experimental = cfg["generator"]["pipeline"]["experimental"]
    assert "h3_sequential_load" not in experimental


def test_spark_preview_is_a_runtime_with_one_or_two_devices():
    recipes = json.loads(COOKBOOK_DATA.read_text())["recipes"]
    one = next(item for item in recipes if item["id"] == "fasth3-preview-spark")
    pair = next(item for item in recipes if item["id"] == "fasth3-spark-pair")
    assert one["group"] == pair["group"] == "fasth3-preview"
    assert one["hardware"]["device"] == pair["hardware"]["device"] == "spark"
    assert one["hardware"]["gpu_count"] == 1
    assert pair["hardware"]["gpu_count"] == 2
    assert "serving" in one
    assert "serving" not in pair
    profile = cookbook_serving_profile(one)
    assert profile["hardware"] == {
        "platform": "cuda",
        "device": "spark",
        "gpu_count": 1,
        "evidence": "source-configured",
    }
    assert profile["command"].startswith("FASTVIDEO_VSA_SM100A=0 FASTVIDEO_FA4=0")
    assert "openai_fasth3_spark.yaml" in profile["command"]
    assert "--server.host 127.0.0.1" in profile["command"]


def test_mlx_serving_profile_has_native_launcher_and_no_invented_memory():
    recipes = json.loads(COOKBOOK_DATA.read_text())["recipes"]
    recipe = next(item for item in recipes if item["id"] == "fasth3-preview-mlx")
    profile = cookbook_serving_profile(recipe)
    assert profile["hardware"] == {"platform": "mlx", "evidence": "source-configured"}
    assert profile["command"] == (
        "python -m fastvideo.entrypoints.openai.mlx_server --config examples/serving/mlx_fasth3.yaml"
    )
    assert profile["prepare"] in recipe["command"]
    assert profile["sampling"]["num_inference_steps"] == 5
    for client in profile["clients"].values():
        assert client["code"] == (ROOT / client["source"]).read_text()


def test_fasth3_8step_cuda_and_mlx_share_the_openai_client():
    recipes = json.loads(COOKBOOK_DATA.read_text())["recipes"]
    cuda = next(item for item in recipes if item["id"] == "fasth3-8step-v2-cuda")
    mlx = next(item for item in recipes if item["id"] == "fasth3-8step-v2-mlx")
    assert cuda["group"] == mlx["group"] == "fasth3-8step-v2"
    cuda_profile = cookbook_serving_profile(cuda)
    mlx_profile = cookbook_serving_profile(mlx)
    assert cuda_profile["model"] == mlx_profile["model"] == "fasth3"
    assert cuda_profile["sampling"]["num_inference_steps"] == 9
    assert mlx_profile["sampling"]["num_inference_steps"] == 9
    assert "openai_fasth3_8step.yaml" in cuda_profile["command"]
    assert mlx_profile["command"] == (
        "python -m fastvideo.entrypoints.openai.mlx_server --config examples/serving/mlx_fasth3_8step.yaml"
    )
    assert mlx_profile["prepare"] in mlx["command"]
    assert "--include-vsa" in mlx_profile["prepare"]
    preview = cookbook_serving_profile(next(item for item in recipes if item["id"] == "fasth3-preview-cuda"))
    assert cuda_profile["clients"]["python"]["code"] == preview["clients"]["python"]["code"]


def test_mlx_preview_and_8step_yaml_map_sigma_points_to_forwards():
    from fastvideo.entrypoints.openai.mlx_server import (EIGHT_STEP_MODEL, PREVIEW_MODEL, load_config, mlx_num_steps,
                                                         validate_mlx_video_request)
    from fastvideo.entrypoints.openai.protocol import VideoGenerationRequest
    from fastvideo.registry import get_preset_selection

    assert mlx_num_steps(5, model_path=PREVIEW_MODEL) == 4
    assert mlx_num_steps(None, model_path=PREVIEW_MODEL) == 4
    assert mlx_num_steps(9, model_path=EIGHT_STEP_MODEL) == 8
    with pytest.raises(ValueError, match="5 sigma points"):
        mlx_num_steps(9, model_path=PREVIEW_MODEL)
    with pytest.raises(ValueError, match="9 sigma points"):
        mlx_num_steps(5, model_path=EIGHT_STEP_MODEL)
    validate_mlx_video_request(VideoGenerationRequest(prompt="a fox", num_inference_steps=5), model_path=PREVIEW_MODEL)
    validate_mlx_video_request(VideoGenerationRequest(prompt="a fox", num_inference_steps=9),
                               model_path=EIGHT_STEP_MODEL)
    with pytest.raises(ValueError, match="9 sigma points"):
        validate_mlx_video_request(VideoGenerationRequest(prompt="a fox", num_inference_steps=5),
                                   model_path=EIGHT_STEP_MODEL)
    preview = load_config(str(ROOT / "examples/serving/mlx_fasth3.yaml"))
    eight = load_config(str(ROOT / "examples/serving/mlx_fasth3_8step.yaml"))
    assert preview.generator.vsa is False
    assert eight.generator.vsa is True
    assert eight.generator.vsa_sparsity == 0.8
    assert eight.default_request["sampling"]["num_inference_steps"] == 9
    assert get_preset_selection(EIGHT_STEP_MODEL) == ("minimax_h3_t2va", "minimax_h3")


@pytest.mark.parametrize("field,value", [("model", "wrong-checkpoint"), ("hardware", {"platform": "mlx"})])
def test_cookbook_rejects_mismatched_serving_recipe(field, value):
    recipe = copy.deepcopy(serving_recipe())
    recipe[field] = value
    with pytest.raises(ValueError):
        cookbook_serving_profile(recipe)


def test_cookbook_rejects_config_outside_serving_examples():
    recipe = serving_recipe()
    recipe["serving"]["source"] = "pyproject.toml"
    with pytest.raises(ValueError, match="examples/serving"):
        cookbook_serving_profile(recipe)


def test_example_docs_exclude_installed_client_dependencies(tmp_path):
    (tmp_path / "README.md").write_text("# Example")
    (tmp_path / "client.mjs").write_text("// example")
    for dirname in ["node_modules", ".venv", "__pycache__"]:
        dependency = tmp_path / dirname
        dependency.mkdir()
        (dependency / "README.md").write_text("Not example documentation")
    assert Example(tmp_path).other_files == [tmp_path / "client.mjs"]


@pytest.mark.parametrize(
    ("recipe_id", "config_name", "fps", "env_prefix"),
    [
        ("fastwan21-t2v", "openai_fastwan21_1_3b.yaml", 16, "FASTVIDEO_ATTENTION_BACKEND=VIDEO_SPARSE_ATTN "),
        ("wan22-t2v", "openai_wan22_t2v_a14b.yaml", 16, ""),
        ("wan22-ti2v", "openai_wan22_ti2v_5b.yaml", 24, ""),
    ],
)
def test_wan_text_recipes_publish_server_profiles(recipe_id, config_name, fps, env_prefix):
    recipes = json.loads(COOKBOOK_DATA.read_text())["recipes"]
    recipe = next(item for item in recipes if item["id"] == recipe_id)
    assert "audio" not in recipe["serving"]
    profile = cookbook_serving_profile(recipe)
    # The playground router rejects non-H3 servers, so Wan pages must not link to it.
    assert profile["playground_url"] is None
    assert profile["audio"] is False
    assert profile["command"].startswith(f"{env_prefix}fastvideo serve --config examples/serving/{config_name}")
    assert "--server.host 127.0.0.1" in profile["command"]
    assert profile["sampling"]["fps"] == fps
    if recipe_id == "wan22-ti2v":
        assert recipe["serving"]["task"] == "Text to video"
        assert any("text prompt only" in item for item in recipe["serving"]["limitations"])


def test_h3_serving_recipes_advertise_audio_and_playground():
    recipes = json.loads(COOKBOOK_DATA.read_text())["recipes"]
    h3 = [item for item in recipes if item["family"] == "minimax_h3" and "serving" in item]
    assert h3
    for recipe in h3:
        profile = cookbook_serving_profile(recipe)
        assert recipe["serving"]["audio"] is True, recipe["id"]
        assert profile["audio"] is True, recipe["id"]
        assert profile["playground_url"] == "http://127.0.0.1:8000/playground/", recipe["id"]


def test_serving_profile_reports_explicit_compile_setting():
    recipes = {item["id"]: item for item in json.loads(COOKBOOK_DATA.read_text())["recipes"]}
    assert cookbook_serving_profile(recipes["fasth3-preview-cuda"])["compile_enabled"] is False
    assert cookbook_serving_profile(recipes["fastwan21-t2v"])["compile_enabled"] is True
    assert cookbook_serving_profile(recipes["wan22-t2v"])["compile_enabled"] is None


@pytest.mark.parametrize(
    ("config_name", "preset_name"),
    [
        ("openai_wan22_t2v_a14b.yaml", "WAN_2_2_T2V_A14B"),
        ("openai_wan22_ti2v_5b.yaml", "WAN_2_2_TI2V_5B"),
    ],
)
def test_wan_serving_sampling_matches_registered_preset(config_name, preset_name):
    # Read the literal defaults from source so the test needs no model imports (torch).
    tree = ast.parse((ROOT / "fastvideo/pipelines/basic/wan/presets.py").read_text())
    call = next(
        node.value for node in ast.walk(tree)
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)
        and any(isinstance(target, ast.Name) and target.id == preset_name for target in node.targets))
    defaults_node = next(keyword.value for keyword in call.keywords if keyword.arg == "defaults")
    defaults = {
        key.value: value.value
        for key, value in zip(defaults_node.keys, defaults_node.values)
        if isinstance(key, ast.Constant) and isinstance(value, ast.Constant)
    }
    assert defaults
    config = yaml.safe_load((ROOT / "examples/serving" / config_name).read_text())
    for key, value in config["default_request"]["sampling"].items():
        if key in defaults:
            assert defaults[key] == value, f"{config_name} {key} drifted from {preset_name}"


def test_wan21_i2v_stays_python_only():
    recipes = json.loads(COOKBOOK_DATA.read_text())["recipes"]
    recipe = next(item for item in recipes if item["id"] == "wan21-i2v")
    assert "serving" not in recipe


def test_h3_command_blocks_have_unique_copy_targets():
    class CodeBlocks(HTMLParser):
        def __init__(self):
            super().__init__()
            self.ids = []

        def handle_starttag(self, tag, attrs):
            if tag == "pre":
                block_id = dict(attrs).get("id")
                if block_id:
                    self.ids.append(block_id)

    parser = CodeBlocks()
    parser.feed((ROOT / "docs/cookbook/minimax-h3.md").read_text())
    assert parser.ids == [
        "cookbook-local-command",
        "cookbook-server-install",
        "cookbook-server-prepare",
        "cookbook-server-command",
        "cookbook-health-command",
        "cookbook-client-install",
        "cookbook-client-code",
    ]
    assert len(set(parser.ids)) == len(parser.ids)
