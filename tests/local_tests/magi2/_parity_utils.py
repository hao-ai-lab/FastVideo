# SPDX-License-Identifier: Apache-2.0
"""Shared utilities for MAGI-2 component parity scaffolds."""
from __future__ import annotations

import importlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
OFFICIAL_REF_DIR = Path(
    os.getenv(
        "MAGI2_OFFICIAL_REF_DIR",
        "/mnt/weka/shrd/wm/junda/fv-hub/MAGI-2-preview",
    )
)
LOCAL_WEIGHTS_DIR = Path(
    os.getenv("MAGI2_LOCAL_WEIGHTS_DIR", REPO_ROOT / "official_weights" / "magi2")
)


def require_path(path: Path, purpose: str) -> Path:
    """Return a required local path or skip with its concrete purpose."""
    if not path.exists():
        pytest.skip(f"Missing {purpose}: {path}")
    return path


def require_complete_safetensor_index(component_dir: Path) -> Path:
    """Require every shard named by a component's safetensors index."""
    index_path = require_path(
        component_dir / "model.safetensors.index.json",
        "safetensors index",
    )
    with index_path.open(encoding="utf-8") as index_file:
        weight_index = json.load(index_file)
    shard_names = sorted(set(weight_index["weight_map"].values()))
    missing_shards = [name for name in shard_names if not (component_dir / name).is_file()]
    if missing_shards:
        pytest.skip(
            f"Incomplete weights in {component_dir}; missing shards: "
            f"{', '.join(missing_shards)}"
        )
    return component_dir


def import_official_module(module_name: str):
    """Import one module from the user-specified official MAGI-2 checkout."""
    require_path(OFFICIAL_REF_DIR, "official MAGI-2 checkout")
    reference_path = str(OFFICIAL_REF_DIR)
    if reference_path not in sys.path:
        sys.path.insert(0, reference_path)
    try:
        official_module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name == module_name or module_name.startswith(f"{exc.name}."):
            raise
        pytest.skip(
            f"Official module {module_name} requires missing dependency {exc.name!r}"
        )
    module_path = Path(official_module.__file__).resolve()
    if not module_path.is_relative_to(OFFICIAL_REF_DIR.resolve()):
        raise RuntimeError(
            f"Imported {module_name} from {module_path}, expected {OFFICIAL_REF_DIR}"
        )
    return official_module


def assert_tensor_exact(
    fastvideo_tensor: torch.Tensor,
    official_tensor: torch.Tensor,
    tensor_name: str,
) -> None:
    """Require identical tensor metadata and identical values."""
    assert fastvideo_tensor.shape == official_tensor.shape, tensor_name
    assert fastvideo_tensor.dtype == official_tensor.dtype, tensor_name
    assert fastvideo_tensor.stride() == official_tensor.stride(), tensor_name
    assert torch.equal(fastvideo_tensor, official_tensor), tensor_name


def assert_array_exact(
    fastvideo_array: np.ndarray,
    official_array: np.ndarray,
    array_name: str,
) -> None:
    """Require identical NumPy array metadata and identical values."""
    assert fastvideo_array.shape == official_array.shape, array_name
    assert fastvideo_array.dtype == official_array.dtype, array_name
    assert fastvideo_array.strides == official_array.strides, array_name
    assert np.array_equal(fastvideo_array, official_array), array_name
