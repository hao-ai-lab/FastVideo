# SPDX-License-Identifier: Apache-2.0
"""CPU-only contracts for the Slurm SSIM task scheduler."""

from pathlib import Path
import sys

import pytest

SSIM_DIR = Path(__file__).resolve().parents[1] / "ssim"
sys.path.insert(0, str(SSIM_DIR))

from ci_runner import (  # noqa: E402
    discover_tasks,
    extract_model_ids,
    extract_required_gpus,
    task_master_port,
    visible_gpu_ids,
)


def test_discovery_splits_model_parameters_and_preserves_gpu_requirements(tmp_path: Path) -> None:
    (tmp_path / "test_alpha.py").write_text(
        """REQUIRED_GPUS = 2
ALPHA_MODEL_TO_PARAMS = {"model/b": {}, "model/a": {}}
FULL_QUALITY_ALPHA_MODEL_TO_PARAMS = {"model/b": {}, "model/a": {}}
""",
        encoding="utf-8",
    )
    (tmp_path / "test_helpers.py").write_text("def test_helper(): pass\n", encoding="utf-8")

    tasks = discover_tasks(tmp_path)

    assert [(task.name, task.required_gpus) for task in tasks] == [
        ("test_alpha.py::model/a", 2),
        ("test_alpha.py::model/b", 2),
        ("test_helpers.py", 1),
    ]


def test_ast_helpers_do_not_import_test_modules(tmp_path: Path) -> None:
    path = tmp_path / "test_guard.py"
    path.write_text(
        """raise RuntimeError("must not import")
REQUIRED_GPUS = 3
GUARD_MODEL_TO_PARAMS = {"model/id": {}}
""",
        encoding="utf-8",
    )

    assert extract_required_gpus(path) == 3
    assert extract_model_ids(path) == ["model/id"]


def test_discovery_rejects_a_task_larger_than_the_slurm_tray(tmp_path: Path) -> None:
    (tmp_path / "test_too_large.py").write_text("REQUIRED_GPUS = 5\n", encoding="utf-8")

    with pytest.raises(ValueError, match="supported range is 1-4"):
        discover_tasks(tmp_path)


def test_discovery_can_select_exact_test_basenames(tmp_path: Path) -> None:
    (tmp_path / "test_alpha.py").write_text("REQUIRED_GPUS = 1\n", encoding="utf-8")
    (tmp_path / "test_beta.py").write_text("REQUIRED_GPUS = 2\n", encoding="utf-8")

    tasks = discover_tasks(tmp_path, ["test_beta.py"])

    assert [(task.name, task.required_gpus) for task in tasks] == [("test_beta.py", 2)]


def test_discovery_rejects_missing_or_unsafe_test_file_selections(tmp_path: Path) -> None:
    (tmp_path / "test_alpha.py").write_text("REQUIRED_GPUS = 1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="do not exist"):
        discover_tasks(tmp_path, ["test_missing.py"])
    with pytest.raises(ValueError, match="Invalid SSIM test file"):
        discover_tasks(tmp_path, ["../test_alpha.py"])


def test_visible_gpu_ids_preserve_the_slurm_lease(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3,2")

    assert visible_gpu_ids() == ["2", "3"]


def test_parallel_tasks_get_offsets_in_the_runner_assigned_port_range() -> None:
    assert task_master_port(0, "31200") == "31200"
    assert task_master_port(17, "31200") == "31217"

    with pytest.raises(ValueError, match="100-port range"):
        task_master_port(100, "31200")

    with pytest.raises(ValueError, match="outside 1-65535"):
        task_master_port(17, "65520")
