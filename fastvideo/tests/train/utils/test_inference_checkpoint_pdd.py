# SPDX-License-Identifier: Apache-2.0
"""Inference checkpoint export for models with widened (PDD) parameters."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import torch.distributed.checkpoint as dcp
from safetensors.torch import load_file, save_file

from fastvideo.train.utils.inference_checkpoint import (
    InferenceCheckpointExportError,
    export_inference_checkpoint_from_dcp,
    validate_complete_inference_checkpoint,
)

_GRID = 3


class _WidenedModule(torch.nn.Module):
    """A live module whose ``proj.*`` parameters are ``_GRID`` times the base width."""

    def __init__(self, *, declare_hooks: bool = True) -> None:
        super().__init__()
        self.proj_weight = torch.nn.Parameter(torch.zeros(2 * _GRID, 2))
        self.reverse_param_names_mapping = {
            "proj.weight": ("disk.proj.weight", None, None),
            "other": ("disk.other", None, None),
        }
        if not declare_hooks:
            return
        self.inference_export_expected_shape = self._expected_shape  # type: ignore[method-assign]
        self.inference_config_overrides = lambda: {"pdd_steps": _GRID}  # type: ignore[method-assign]

    @staticmethod
    def _expected_shape(output_key: str, base_shape: tuple[int, ...]) -> tuple[int, ...]:
        if output_key.startswith("disk.proj."):
            return (base_shape[0] * _GRID, *base_shape[1:])
        return base_shape


@pytest.fixture
def base_model_dir(tmp_path: Path) -> Path:
    base = tmp_path / "base"
    transformer = base / "transformer"
    transformer.mkdir(parents=True)
    (transformer / "config.json").write_text(json.dumps({"_class_name": "FakeTransformer", "hidden": 2}),
                                             encoding="utf-8")
    (base / "modular_model_index.json").write_text(json.dumps({"_class_name": "FakePipeline"}), encoding="utf-8")
    tensors = {"disk.proj.weight": torch.zeros(2, 2), "disk.other": torch.ones(3)}
    filename = "diffusion_pytorch_model-00001-of-00001.safetensors"
    save_file(tensors, transformer / filename)
    index = {
        "metadata": {
            "total_size": sum(t.numel() * t.element_size() for t in tensors.values())
        },
        "weight_map": {key: filename for key in tensors},
    }
    (transformer / "diffusion_pytorch_model.safetensors.index.json").write_text(json.dumps(index), encoding="utf-8")
    return base


def _save_dcp(root: Path, tensors: dict[str, torch.Tensor]) -> Path:
    checkpoint = root / "staging"
    dcp_dir = checkpoint / "dcp"
    dcp.save({f"roles.student.transformer.{key}": value.clone() for key, value in tensors.items()},
             checkpoint_id=str(dcp_dir))
    return checkpoint


def test_export_widens_declared_shapes_and_records_config_overrides(tmp_path: Path, base_model_dir: Path) -> None:
    widened = torch.arange(2 * _GRID * 2, dtype=torch.float32).reshape(2 * _GRID, 2)
    source = _save_dcp(tmp_path, {"proj.weight": widened, "other": torch.ones(3)})
    module = _WidenedModule()

    exported = export_inference_checkpoint_from_dcp(
        dcp_checkpoint=source,
        output_dir=tmp_path / "run",
        step=5,
        module=module,
        base_model_dir=base_model_dir,
        dtype=torch.float32,
    )

    transformer_dir = exported / "transformer"
    tensors: dict[str, torch.Tensor] = {}
    for shard in sorted(transformer_dir.glob("*.safetensors")):
        tensors.update(load_file(shard))
    assert torch.equal(tensors["disk.proj.weight"], widened)
    assert torch.equal(tensors["disk.other"], torch.ones(3))
    config = json.loads((transformer_dir / "config.json").read_text(encoding="utf-8"))
    assert config["pdd_steps"] == _GRID
    assert config["_class_name"] == "FakeTransformer"
    assert config["hidden"] == 2
    metadata = json.loads((exported / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["config_overrides"] == {"pdd_steps": _GRID}

    # Validation needs the same shape contract; the plain contract rejects it.
    assert validate_complete_inference_checkpoint(exported, step=5,
                                                  expected_shape_override=module._expected_shape) == exported
    assert validate_complete_inference_checkpoint(
        exported,
        step=5,
        expected_shape_override=module._expected_shape,
        expected_config_overrides={"pdd_steps": _GRID},
    ) == exported
    with pytest.raises(InferenceCheckpointExportError, match="base transformer shape"):
        validate_complete_inference_checkpoint(exported, step=5)
    # Retrying with the hooks is idempotent.
    assert export_inference_checkpoint_from_dcp(
        dcp_checkpoint=source,
        output_dir=tmp_path / "run",
        step=5,
        module=module,
        base_model_dir=base_model_dir,
        dtype=torch.float32,
    ) == exported

    config.pop("pdd_steps")
    (transformer_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(InferenceCheckpointExportError, match="config overrides do not match"):
        export_inference_checkpoint_from_dcp(
            dcp_checkpoint=source,
            output_dir=tmp_path / "run",
            step=5,
            module=module,
            base_model_dir=base_model_dir,
            dtype=torch.float32,
        )


def test_export_without_hooks_still_rejects_widened_tensors(tmp_path: Path, base_model_dir: Path) -> None:
    source = _save_dcp(tmp_path, {"proj.weight": torch.zeros(2 * _GRID, 2), "other": torch.ones(3)})
    with pytest.raises(InferenceCheckpointExportError, match="base transformer shape"):
        export_inference_checkpoint_from_dcp(
            dcp_checkpoint=source,
            output_dir=tmp_path / "run",
            step=1,
            module=_WidenedModule(declare_hooks=False),
            base_model_dir=base_model_dir,
            dtype=torch.float32,
        )
    assert not (tmp_path / "run" / "inference" / "checkpoint-1").exists()


def test_export_from_read_only_warm_start_writes_config_overrides(tmp_path: Path, base_model_dir: Path) -> None:
    # A protected warm start (read-only files and directories) must still export:
    # the copied component config previously kept mode 0444 and the override write failed.
    source = _save_dcp(tmp_path, {"proj.weight": torch.zeros(2 * _GRID, 2), "other": torch.ones(3)})
    paths = sorted(base_model_dir.rglob("*"), key=lambda path: len(path.parts), reverse=True)
    for path in paths:
        path.chmod(0o555 if path.is_dir() else 0o444)
    base_model_dir.chmod(0o555)
    try:
        exported = export_inference_checkpoint_from_dcp(
            dcp_checkpoint=source,
            output_dir=tmp_path / "run",
            step=0,
            module=_WidenedModule(),
            base_model_dir=base_model_dir,
            dtype=torch.float32,
        )
        config_path = exported / "transformer" / "config.json"
        assert json.loads(config_path.read_text(encoding="utf-8"))["pdd_steps"] == _GRID
        assert config_path.stat().st_mode & 0o200
        assert (base_model_dir / "transformer" / "config.json").stat().st_mode & 0o777 == 0o444
    finally:
        base_model_dir.chmod(0o755)
        for path in reversed(paths):
            path.chmod(0o755 if path.is_dir() else 0o644)


def test_config_override_does_not_mutate_hard_linked_base(tmp_path):
    import os
    from fastvideo.train.utils.inference_checkpoint import _apply_config_overrides

    base = tmp_path / 'base.json'
    exported = tmp_path / 'export.json'
    base.write_text('{"pdd_steps": 1}\n')
    os.link(base, exported)
    _apply_config_overrides(exported, {"pdd_steps": 32})
    assert json.loads(base.read_text()) == {"pdd_steps": 1}
    assert json.loads(exported.read_text()) == {"pdd_steps": 32}


@pytest.mark.parametrize('mismatch', ['dtype', 'config', 'role', 'module'])
def test_manager_rejects_incompatible_existing_export(tmp_path, base_model_dir, mismatch):
    from fastvideo.train.utils.checkpoint import CheckpointConfig, CheckpointManager

    source = _save_dcp(tmp_path, {'proj.weight': torch.ones(2 * _GRID, 2), 'other': torch.ones(3)})
    module = _WidenedModule()
    export_inference_checkpoint_from_dcp(
        dcp_checkpoint=source, output_dir=tmp_path / 'run', step=5, module=module,
        base_model_dir=base_model_dir, dtype=torch.float32,
    )
    if mismatch == 'config':
        module.inference_config_overrides = lambda: {'pdd_steps': _GRID, 'new_flag': True}

    class Method:
        def inference_checkpoint_modules(self, role):
            return {'transformer_ref' if mismatch == 'module' else 'transformer': module}

        def inference_checkpoint_base_model_path(self, role):
            return str(base_model_dir)

    manager = CheckpointManager(
        method=Method(), dataloader=None, output_dir=str(tmp_path / 'run'),
        config=CheckpointConfig(save_steps=0, keep_last=0,
            inference_dtype='bfloat16' if mismatch == 'dtype' else 'float32',
            inference_role='critic' if mismatch == 'role' else 'student'),
    )
    with pytest.raises(RuntimeError, match='Existing inference checkpoint failed validation'):
        manager.save_inference(5)
