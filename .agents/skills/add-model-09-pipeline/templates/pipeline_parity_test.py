# SPDX-License-Identifier: Apache-2.0
"""Pipeline parity scaffold for TODO_MODEL_FAMILY.

Copy this file to
`tests/local_tests/pipelines/test_<family>_pipeline_parity.py` and replace every
TODO before treating it as an executable scaffold.

The filled test should compare denoised latents, decoded media, audio waveform,
or another concrete output from the official pipeline against FastVideo. A
successful generation without tensor/media comparison is not parity.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import pytest
import torch
from torch.testing import assert_close

_REPO_ROOT = Path(__file__).resolve().parents[3]
_MODEL_FAMILY = "TODO_MODEL_FAMILY"
_OFFICIAL_REF_ENV = "TODO_OFFICIAL_REF_PATH"
_OFFICIAL_REF_DEFAULT = _REPO_ROOT / "TODO_OFFICIAL_REF_DIR"
_FASTVIDEO_MODEL_ENV = "TODO_FASTVIDEO_MODEL_PATH"
_FASTVIDEO_MODEL_DEFAULT = _REPO_ROOT / "converted_weights" / _MODEL_FAMILY


def _path_from_env(env_name: str, default: Path) -> Path:
    return Path(os.getenv(env_name, str(default))).expanduser()


def _add_official_to_path() -> Path:
    official_path = _path_from_env(_OFFICIAL_REF_ENV, _OFFICIAL_REF_DEFAULT)
    if not official_path.exists():
        pytest.skip(f"Official reference not found at {official_path}")
    if str(official_path) not in sys.path:
        sys.path.insert(0, str(official_path))
    return official_path


def _log_tensor_stats(label: str, tensor: torch.Tensor) -> None:
    value = tensor.detach().float()
    print(
        f"[{_MODEL_FAMILY} PIPELINE] {label}: shape={tuple(tensor.shape)} "
        f"dtype={tensor.dtype} device={tensor.device} "
        f"min={value.min().item():.6f} max={value.max().item():.6f} "
        f"mean={value.mean().item():.6f} std={value.std().item():.6f}"
    )


def _extract_tensor(output: Any, key: str) -> torch.Tensor:
    if isinstance(output, dict):
        value = output.get(key)
    else:
        value = getattr(output, key, None)
    if value is None:
        raise AssertionError(f"Pipeline output did not contain {key!r}")
    if not torch.is_tensor(value):
        try:
            import numpy as np
            value = torch.from_numpy(np.asarray(value))
        except Exception as exc:  # pragma: no cover - scaffold guard
            raise AssertionError(f"Could not convert {key!r} to tensor") from exc
    return value.detach().float().cpu()


def _run_official_pipeline(
    official_path: Path,
    params: dict[str, Any],
    device: torch.device,
) -> Any:
    del official_path, params, device
    pytest.skip(
        "TODO: import the official pipeline/factory, load official weights, "
        "run with params, and return the comparison target."
    )


def _run_fastvideo_pipeline(model_path: Path, params: dict[str, Any]) -> Any:
    from fastvideo import VideoGenerator

    generator = VideoGenerator.from_pretrained(
        str(model_path),
        num_gpus=1,
        use_fsdp_inference=False,
        dit_cpu_offload=False,
        vae_cpu_offload=False,
        text_encoder_cpu_offload=False,
    )
    try:
        return generator.generate_video(
            prompt=params["prompt"],
            negative_prompt=params.get("negative_prompt"),
            output_path=f"outputs_{_MODEL_FAMILY}/pipeline_parity",
            save_video=False,
            height=params.get("height"),
            width=params.get("width"),
            num_frames=params.get("num_frames"),
            fps=params.get("fps"),
            num_inference_steps=params["num_inference_steps"],
            guidance_scale=params.get("guidance_scale"),
            seed=params["seed"],
        )
    finally:
        generator.shutdown()


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="TODO_MODEL_FAMILY pipeline parity requires CUDA.",
)
def test_todo_model_family_pipeline_official_parity() -> None:
    official_path = _add_official_to_path()
    fastvideo_model_path = _path_from_env(
        _FASTVIDEO_MODEL_ENV,
        _FASTVIDEO_MODEL_DEFAULT,
    )
    if not fastvideo_model_path.exists():
        pytest.skip(f"FastVideo model path not found at {fastvideo_model_path}")

    device = torch.device("cuda:0")
    params = {
        "prompt": "TODO: stable parity prompt",
        "negative_prompt": "",
        "height": 64,
        "width": 64,
        "num_frames": 9,
        "fps": 8,
        "num_inference_steps": 4,
        "guidance_scale": 1.0,
        "seed": 0,
    }

    official_output = _run_official_pipeline(official_path, params, device)
    fastvideo_output = _run_fastvideo_pipeline(fastvideo_model_path, params)

    comparison_key = "TODO_COMPARISON_KEY"
    official_tensor = _extract_tensor(official_output, comparison_key)
    fastvideo_tensor = _extract_tensor(fastvideo_output, comparison_key)

    _log_tensor_stats("official", official_tensor)
    _log_tensor_stats("fastvideo", fastvideo_tensor)
    assert official_tensor.shape == fastvideo_tensor.shape

    diff = (official_tensor - fastvideo_tensor).abs()
    print(
        f"diff max={diff.max().item():.6f} "
        f"mean={diff.mean().item():.6f} median={diff.median().item():.6f}"
    )
    assert_close(fastvideo_tensor, official_tensor, atol=1e-2, rtol=1e-2)
