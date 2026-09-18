"""Reading the config file and the environment.

This is the app's entry surface: everything downstream takes a `Config`, and a
mistake here is a deployment that starts with settings nobody asked for. The
split matters too, so it is asserted rather than assumed: settings come from
the YAML, secrets come from the environment, and neither leaks into the other.
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest

from infinite_livestream.config import Config, PresetError, load_model_config, load_preset


@pytest.fixture()
def workspace(tmp_path, monkeypatch):
    """A config file, a fillers directory, and the two required secrets."""
    fillers = tmp_path / "fillers"
    fillers.mkdir()
    (fillers / "fillers.json").write_text(
        json.dumps({"style": "house style", "idle_prompts": ["a lighthouse", "a seagull"]}))
    config = tmp_path / "infinite_livestream.yaml"
    config.write_text(
        textwrap.dedent(f"""
        inference:
          aspect: "16:9"
          clip_seconds: 14.375
        runtime:
          num_gpus: 4
        upsampler:
          model: my-model
          base_url: https://example.invalid/v1
          max_chunks: 3
          viewer_free_style: false
        moderation:
          enabled: false
        director:
          idle_queue_target: 2
          chat_cooldown_s: 7
          chat_command: "!go"
          fillers: {fillers}
        output:
          hls_dir: /tmp/hls-under-test
          video_bitrate_k: 1234
        web:
          host: 127.0.0.1
          port: 9999
        """))
    monkeypatch.setenv("LIVESTREAM_WEIGHTS_PATH", str(tmp_path / "weights"))
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.delenv("MODERATION_API_KEY", raising=False)
    return config


def test_every_block_reaches_the_config(workspace) -> None:
    config = Config.load(["--config", str(workspace)])
    assert config.openai_model == "my-model"
    assert config.openai_base_url == "https://example.invalid/v1"
    assert config.max_chunks == 3
    assert config.viewer_free_style is False
    assert config.moderation_enabled is False
    assert config.idle_queue_target == 2
    assert config.chat_cooldown_s == 7
    assert config.chat_command == "!go"
    assert config.hls_dir == "/tmp/hls-under-test"
    assert config.video_bitrate_k == 1234
    assert config.web_host == "127.0.0.1"
    assert config.web_port == 9999
    assert config.style == "house style"
    assert config.idle_prompts == ("a lighthouse", "a seagull")


def test_secrets_come_only_from_the_environment(workspace) -> None:
    """A key in a version-controlled file is a key that leaks."""
    assert "OPENAI_API_KEY" not in workspace.read_text()
    config = Config.load(["--config", str(workspace)])
    assert config.openai_api_key == "sk-test"
    # Moderation falls back to the upsampling credentials, which is right when
    # one endpoint serves both.
    assert config.moderation_api_key == "sk-test"


def test_cli_overrides_win(workspace, tmp_path) -> None:
    config = Config.load(["--config", str(workspace), "--port", "4321", "--weights", str(tmp_path / "elsewhere")])
    assert config.web_port == 4321
    assert config.weights_path.name == "elsewhere"


def test_a_missing_key_stops_startup(workspace, monkeypatch) -> None:
    """Rewriting runs for the idle filler too, so there is no useful run without it."""
    monkeypatch.delenv("OPENAI_API_KEY")
    with pytest.raises(SystemExit, match="OPENAI_API_KEY"):
        Config.load(["--config", str(workspace)])


def test_missing_weights_stops_startup(workspace, monkeypatch) -> None:
    monkeypatch.delenv("LIVESTREAM_WEIGHTS_PATH")
    with pytest.raises(SystemExit, match="LIVESTREAM_WEIGHTS_PATH"):
        Config.load(["--config", str(workspace)])


def test_a_missing_config_file_is_named(tmp_path) -> None:
    with pytest.raises(SystemExit, match="config not found"):
        Config.load(["--config", str(tmp_path / "nope.yaml")])


def test_defaults_apply_when_a_block_is_absent(tmp_path, monkeypatch) -> None:
    """An operator writing a minimal file must still get a working deployment."""
    config_file = tmp_path / "minimal.yaml"
    config_file.write_text("inference:\n  aspect: \"16:9\"\nruntime:\n  num_gpus: 4\n")
    monkeypatch.setenv("LIVESTREAM_WEIGHTS_PATH", str(tmp_path))
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    config = Config.load(["--config", str(config_file)])
    assert config.web_port == 8081
    assert config.chat_cooldown_s == 10
    assert config.idle_prompts, "the shipped fillers should be used when none is named"


def test_model_config_reads_the_same_file(workspace) -> None:
    model = load_model_config(workspace)
    assert model.aspect == "16:9"
    assert model.clip_frames == 345
    assert model.runtime["num_gpus"] == 4


def test_a_fillers_directory_without_the_file_is_named(tmp_path) -> None:
    with pytest.raises(PresetError, match="fillers.json"):
        load_preset(tmp_path)


def test_the_playlist_default_is_not_the_working_directory(tmp_path, monkeypatch) -> None:
    """A relative default would scatter .ts segments wherever the server started."""
    config_file = tmp_path / "minimal.yaml"
    config_file.write_text("inference:\n  aspect: \"16:9\"\nruntime:\n  num_gpus: 4\n")
    monkeypatch.setenv("LIVESTREAM_WEIGHTS_PATH", str(tmp_path))
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    # This default is read at import time. A subprocess avoids replacing the
    # parent's config classes or leaving its default tied to this fixture.
    result = subprocess.run(
        [sys.executable, "-c", "from infinite_livestream.config import Config; print(Config.load().hls_dir)",
         "--config", str(config_file)],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        check=True,
    )
    hls_dir = result.stdout.strip()
    assert hls_dir.startswith(str(tmp_path / "state")), hls_dir
