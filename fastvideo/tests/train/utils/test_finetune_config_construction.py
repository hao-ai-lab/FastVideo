# SPDX-License-Identifier: Apache-2.0
"""CPU-only construction check for every shipped fine-tuning recipe.

Parametrizes over ``examples/train/configs/fine_tuning/**/*.yaml`` and
asserts each loads through :func:`load_run_config` with a resolved
pipeline config and an importable-shaped student target — the
construction proof (parse-green is not construct-green) and a
regression net for every family's recipes.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from fastvideo.train.utils.config import load_run_config

_REPO_ROOT = Path(__file__).resolve().parents[4]
_RECIPE_DIR = _REPO_ROOT / "examples" / "train" / "configs" / "fine_tuning"

_RECIPES = sorted(_RECIPE_DIR.glob("**/*.yaml"))


def _recipe_id(path: Path) -> str:
    return str(path.relative_to(_RECIPE_DIR))


@pytest.mark.parametrize("recipe", _RECIPES, ids=_recipe_id)
def test_fine_tuning_recipe_constructs(recipe: Path) -> None:
    cfg = load_run_config(str(recipe))

    assert cfg.training.pipeline_config is not None, (
        f"{_recipe_id(recipe)} did not resolve a pipeline config; "
        "check its `pipeline:` key and init_from registry match")

    target = cfg.models["student"]["_target_"]
    assert isinstance(target, str) and target and "." in target, (
        f"{_recipe_id(recipe)} student _target_ is not importable-shaped: "
        f"{target!r}")


def test_recipe_glob_found_recipes() -> None:
    # Guard against the glob silently matching nothing after a move.
    assert len(_RECIPES) >= 20
