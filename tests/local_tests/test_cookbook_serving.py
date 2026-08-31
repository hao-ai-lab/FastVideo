"""Check serving snippets against their executable sources, without model imports."""

import copy
import json
from pathlib import Path
from html.parser import HTMLParser

import pytest

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


def test_h3_command_blocks_have_unique_copy_targets():
    class CodeBlocks(HTMLParser):
        def __init__(self):
            super().__init__()
            self.ids = []

        def handle_starttag(self, tag, attrs):
            if tag == "pre":
                self.ids.append(dict(attrs).get("id"))

    parser = CodeBlocks()
    parser.feed((ROOT / "docs/cookbook/minimax-h3.md").read_text())
    assert len(parser.ids) == 7
    assert all(parser.ids)
    assert len(set(parser.ids)) == len(parser.ids)
