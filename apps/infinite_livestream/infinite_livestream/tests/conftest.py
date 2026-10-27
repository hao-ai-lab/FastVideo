"""CPU fixtures with placeholder checkpoint paths; no model weights are loaded."""

from dataclasses import replace

import pytest

from infinite_livestream.config import Config, REQUIRED_COMPONENTS


@pytest.fixture()
def app_config(tmp_path, monkeypatch):
    weights = tmp_path / "weights"
    weights.mkdir()
    (weights / "modular_model_index.json").write_text("{}")
    for component in REQUIRED_COMPONENTS:
        (weights / component).mkdir()
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    return replace(
        Config.load(["--weights", str(weights)]),
        idle_queue_target=0,
        hls_dir=str(tmp_path / "hls"),
    )
