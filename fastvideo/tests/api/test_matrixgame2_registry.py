# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest

from fastvideo.configs.pipelines.matrixgame2 import MatrixGame2I2V480PConfig
from fastvideo.registry import get_pipeline_config_cls_from_name


@pytest.mark.parametrize(
    "model_id",
    [
        "mignonjia/mg_zelda",
        "mignonjia/mg_zelda_longlive",
        "mignonjia/mg_bidirectional_zelda",
    ],
)
def test_mignonjia_zelda_models_resolve_to_matrixgame2_config(model_id: str) -> None:
    assert get_pipeline_config_cls_from_name(model_id) is MatrixGame2I2V480PConfig
