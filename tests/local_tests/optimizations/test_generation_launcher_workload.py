# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the workload-driven generation launcher parser."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
_LAUNCHER = os.path.join(
    _REPO_ROOT,
    "examples",
    "inference",
    "optimizations",
    "generation_launcher.py",
)


def _load_launcher():
    # Avoid the test directory shadowing imports; load the script by path.
    sys.path = [p for p in sys.path if os.path.abspath(p) != _HERE]
    if _REPO_ROOT not in sys.path:
        sys.path.insert(0, _REPO_ROOT)
    spec = importlib.util.spec_from_file_location(
        "generation_launcher_under_test", _LAUNCHER
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


launcher = _load_launcher()


def _minimal_workload(**overrides):
    payload = {
        "schema_version": 1,
        "workload_id": "unit-t2v",
        "model": {"model_id": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"},
        "task": "t2v",
        "prompt": "a raccoon in sunflowers",
        "sampling": {
            "height": 480,
            "width": 832,
            "num_frames": 49,
            "num_inference_steps": 4,
            "guidance_scale": 5.0,
            "seed": 1024,
            "dtype": "bfloat16",
        },
        "runtime": {
            "num_gpus": 1,
            "text_encoder_cpu_offload": True,
        },
        "measurement": {"warmups": 1, "runs": 2, "save_frames": True},
        "mode_env": {
            "native": {"FASTVIDEO_WAN_FUSIONS": "0"},
            "optimized": {"FASTVIDEO_WAN_FUSIONS": "1"},
        },
    }
    payload.update(overrides)
    return payload


def test_load_workload_dict_roundtrip(tmp_path: Path):
    path = tmp_path / "w.json"
    path.write_text(json.dumps(_minimal_workload()), encoding="utf-8")
    loaded = launcher.load_workload_dict(path)
    assert loaded["workload_id"] == "unit-t2v"
    request = launcher.build_request(loaded, base_dir=tmp_path)
    assert request["sampling"]["height"] == 480
    assert "dtype" not in request["sampling"]
    model_id, kwargs = launcher.build_generator_kwargs(loaded)
    assert model_id.endswith("1.3B-Diffusers")
    assert kwargs["num_gpus"] == 1
    assert kwargs["text_encoder_cpu_offload"] is True


def test_prompt_file(tmp_path: Path):
    prompt = tmp_path / "p.txt"
    prompt.write_text("from file\n", encoding="utf-8")
    payload = _minimal_workload()
    del payload["prompt"]
    payload["prompt_file"] = "p.txt"
    path = tmp_path / "w.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    loaded = launcher.load_workload_dict(path)
    assert launcher.resolve_prompt(loaded, base_dir=tmp_path) == "from file"


def test_rejects_bad_schema(tmp_path: Path):
    path = tmp_path / "w.json"
    path.write_text(
        json.dumps(_minimal_workload(schema_version=99)), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="schema_version"):
        launcher.load_workload_dict(path)


def test_dry_run_cli(tmp_path: Path, capsys):
    path = tmp_path / "w.json"
    path.write_text(json.dumps(_minimal_workload()), encoding="utf-8")
    code = launcher.main(
        [
            "--workload",
            str(path),
            "--mode",
            "native",
            "--output-dir",
            str(tmp_path / "out"),
            "--dry-run",
        ]
    )
    assert code == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["workload_id"] == "unit-t2v"
    assert plan["mode"] == "native"
    assert plan["mode_env"]["FASTVIDEO_WAN_FUSIONS"] == "0"
