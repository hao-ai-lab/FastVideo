# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for bounded-memory modular inference checkpoint export."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper
from safetensors.torch import load_file, save_file

from fastvideo.layers.linear import ReplicatedLinear
import fastvideo.train.utils.inference_checkpoint as inference_checkpoint
from fastvideo.train.utils.inference_checkpoint import (
    InferenceCheckpointExportError,
    UnsupportedMergedReverseMappingError,
    export_inference_checkpoint,
    export_inference_checkpoint_from_dcp,
    validate_complete_inference_checkpoint,
)
from fastvideo.train.utils.lora import enable_lora_training, finalize_lora_training


class _LiveModule(torch.nn.Module):

    def __init__(self, reverse_mapping: dict | None = None) -> None:
        super().__init__()
        self.master = torch.nn.Parameter(
            torch.tensor([1.25, -2.5], dtype=torch.float32)
        )
        self.reverse_param_names_mapping = reverse_mapping or {}


class _TinyLoraModule(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.config = type("Config", (), {
            "arch_config": type("ArchConfig", (), {"exclude_lora_layers": []})()
        })()
        self.to_q = ReplicatedLinear(4, 4, bias=False)
        self.reverse_param_names_mapping: dict = {}

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.to_q(value)[0]


@pytest.fixture
def base_model_dir(tmp_path: Path) -> Path:
    base = tmp_path / "base"
    transformer = base / "transformer"
    transformer.mkdir(parents=True)
    (transformer / "config.json").write_text(
        json.dumps({"_class_name": "FakeTransformer"}),
        encoding="utf-8",
    )
    (base / "modular_model_index.json").write_text(
        json.dumps({"_class_name": "FakePipeline"}),
        encoding="utf-8",
    )
    vae = base / "vae"
    vae.mkdir()
    (vae / "config.json").write_text("{}", encoding="utf-8")
    (base / ".cache").mkdir()
    return base


def _write_base_transformer_weights(base: Path, tensors: dict[str, torch.Tensor]) -> None:
    """Write one indexed base shard to define the exact export contract."""
    transformer = base / "transformer"
    filename = "diffusion_pytorch_model-00001-of-00001.safetensors"
    save_file(tensors, transformer / filename)
    index = {
        "metadata": {
            "total_size": sum(tensor.numel() * tensor.element_size() for tensor in tensors.values())
        },
        "weight_map": {key: filename for key in tensors},
    }
    (transformer / "diffusion_pytorch_model.safetensors.index.json").write_text(
        json.dumps(index, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _save_model_only_dcp(
    root: Path,
    tensors: dict[str, torch.Tensor],
    *,
    role: str = "student",
    module_name: str = "transformer",
) -> Path:
    checkpoint = root / "temporary-model-checkpoint"
    dcp_dir = checkpoint / "dcp"
    state = {
        f"roles.{role}.{module_name}.{key}": value.clone()
        for key, value in tensors.items()
    }
    dcp.save(state, checkpoint_id=str(dcp_dir))
    assert (dcp_dir / ".metadata").is_file()
    return checkpoint


def _load_exported_tensors(transformer_dir: Path) -> dict[str, torch.Tensor]:
    tensors: dict[str, torch.Tensor] = {}
    for shard in sorted(transformer_dir.glob("*.safetensors")):
        tensors.update(load_file(shard))
    return tensors


def test_export_casts_maps_shards_and_publishes_complete_layout(
    tmp_path: Path,
    base_model_dir: Path,
) -> None:
    _write_base_transformer_weights(
        base_model_dir,
        {
            "disk.proj.weight": torch.empty(2, 2),
            "disk.position_ids": torch.empty(2, dtype=torch.int64),
        },
    )
    source = _save_model_only_dcp(
        tmp_path,
        {
            "proj.weight": torch.arange(4, dtype=torch.float32).reshape(2, 2),
            "block.attn.to_gate_compress.weight": torch.tensor(
                [0.25, 0.5, 0.75, 1.0], dtype=torch.float32
            ),
            "position_ids": torch.tensor([3, 7], dtype=torch.int64),
        },
    )
    module = _LiveModule(
        {
            "proj.weight": ("disk.proj.weight", None, None),
            "position_ids": ("disk.position_ids", None, None),
        }
    )
    master_before = module.master.detach().clone()

    result = export_inference_checkpoint_from_dcp(
        dcp_checkpoint=source,
        output_dir=tmp_path / "run",
        step=100,
        module=module,
        base_model_dir=base_model_dir,
        dtype="bfloat16",
        max_shard_size_bytes=16,
    )

    assert result == (tmp_path / "run" / "inference" / "checkpoint-100").resolve()
    assert (result / ".complete").read_text(encoding="utf-8") == "complete\n"
    assert (result / "metadata.json").is_file()
    assert (result / "modular_model_index.json").is_symlink()
    assert (result / "vae").is_symlink()
    assert not (result / ".cache").exists()
    assert (result / "transformer" / "config.json").is_file()
    assert not (result / "transformer" / "config.json").is_symlink()
    assert not (result / "transformer" / "base-weights.safetensors").exists()

    exported = _load_exported_tensors(result / "transformer")
    assert set(exported) == {
        "disk.proj.weight",
        "disk.position_ids",
        "block.attn.to_gate_compress.weight",
    }
    assert exported["disk.proj.weight"].dtype == torch.bfloat16
    assert exported["block.attn.to_gate_compress.weight"].dtype == torch.bfloat16
    assert exported["disk.position_ids"].dtype == torch.int64
    torch.testing.assert_close(
        exported["disk.proj.weight"],
        torch.arange(4, dtype=torch.bfloat16).reshape(2, 2),
    )
    assert torch.equal(exported["disk.position_ids"], torch.tensor([3, 7]))
    assert module.master.dtype == torch.float32
    assert torch.equal(module.master.detach(), master_before)

    metadata = json.loads((result / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["kind"] == "inference"
    assert metadata["step"] == 100
    assert metadata["role"] == "student"
    assert metadata["module"] == "transformer"
    assert metadata["dtype"] == "bfloat16"
    assert metadata["tensor_count"] == 3
    assert metadata["total_size"] == 32
    assert metadata["shard_count"] == 3
    assert all(size <= 16 for size in metadata["shard_sizes"])

    index = json.loads(
        (
            result
            / "transformer"
            / "diffusion_pytorch_model.safetensors.index.json"
        ).read_text(encoding="utf-8")
    )
    assert index["metadata"]["total_size"] == 32
    assert set(index["weight_map"]) == set(exported)
    assert len(set(index["weight_map"].values())) == 3


@pytest.mark.parametrize("mapping_key", ["to_q.base_layer.weight", "to_q.weight"])
def test_export_merges_checkpoint_wrapped_lora_into_native_weight(
    tmp_path: Path,
    base_model_dir: Path,
    mapping_key: str,
) -> None:
    torch.manual_seed(13)
    training_model = _TinyLoraModule()
    assert enable_lora_training(
        training_model,
        lora_rank=2,
        lora_alpha=4,
        lora_target_modules=["to_q"],
    ) == 1
    assert finalize_lora_training(training_model) == 1
    with torch.no_grad():
        training_model.to_q.lora_A.copy_(torch.arange(8, dtype=torch.float32).reshape(2, 4) / 11)
        training_model.to_q.lora_B.copy_(torch.arange(8, dtype=torch.float32).reshape(4, 2) / 17)
    expected_weight = (
        training_model.to_q.base_layer.weight.detach()
        + 2.0 * training_model.to_q.lora_B.detach() @ training_model.to_q.lora_A.detach()
    )

    checkpointed = checkpoint_wrapper(training_model, preserve_rng_state=False)
    checkpointed.reverse_param_names_mapping = {
        mapping_key: ("disk.to_q.weight", None, None)
    }
    assert any("_checkpoint_wrapped_module" in name for name, _ in checkpointed.named_modules())
    assert not any("_checkpoint_wrapped_module" in key for key in checkpointed.state_dict())
    source = _save_model_only_dcp(tmp_path, dict(checkpointed.state_dict()))
    _write_base_transformer_weights(
        base_model_dir, {"disk.to_q.weight": torch.empty(4, 4)}
    )

    result = export_inference_checkpoint_from_dcp(
        dcp_checkpoint=source,
        output_dir=tmp_path / "lora-run",
        step=0,
        module=checkpointed,
        base_model_dir=base_model_dir,
        dtype="float32",
        max_shard_size_bytes=1024,
    )

    exported = _load_exported_tensors(result / "transformer")
    assert set(exported) == {"disk.to_q.weight"}
    torch.testing.assert_close(exported["disk.to_q.weight"], expected_weight)
    native = _TinyLoraModule()
    incompatible = native.load_state_dict(
        {"to_q.weight": exported["disk.to_q.weight"]}, strict=True
    )
    assert incompatible.missing_keys == []
    assert incompatible.unexpected_keys == []
    value = torch.arange(12, dtype=torch.float32).reshape(3, 4) / 7
    torch.testing.assert_close(native(value), value @ expected_weight.T)
    metadata = json.loads((result / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["lora"] == "merged"


def test_export_is_atomic_on_failure_and_retry_succeeds(
    tmp_path: Path,
    base_model_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_base_transformer_weights(
        base_model_dir,
        {
            "a": torch.empty(4),
            "b": torch.empty(4),
        },
    )
    source = _save_model_only_dcp(
        tmp_path,
        {
            "a": torch.ones(4, dtype=torch.float32),
            "b": torch.ones(4, dtype=torch.float32),
        },
    )
    module = _LiveModule()
    real_save_file = inference_checkpoint.save_file
    calls = 0

    def fail_after_first_write(tensors, filename, metadata=None):
        nonlocal calls
        calls += 1
        real_save_file(tensors, filename, metadata=metadata)
        if calls == 1:
            raise OSError("injected shard write failure")

    monkeypatch.setattr(inference_checkpoint, "save_file", fail_after_first_write)
    kwargs = {
        "dcp_checkpoint": source,
        "output_dir": tmp_path / "run",
        "step": 7,
        "module": module,
        "base_model_dir": base_model_dir,
        "dtype": torch.bfloat16,
        "max_shard_size_bytes": 8,
    }
    with pytest.raises(OSError, match="injected"):
        export_inference_checkpoint_from_dcp(**kwargs)

    inference_root = tmp_path / "run" / "inference"
    assert not (inference_root / "checkpoint-7").exists()
    assert list(inference_root.glob(".checkpoint-7.tmp-*")) == []

    monkeypatch.setattr(inference_checkpoint, "save_file", real_save_file)
    result = export_inference_checkpoint_from_dcp(**kwargs)
    assert (result / ".complete").is_file()
    with pytest.raises(InferenceCheckpointExportError, match="does not match requested dtype"):
        export_inference_checkpoint_from_dcp(**{**kwargs, "dtype": torch.float32})

    # Completed outputs are immutable and retries are idempotent, even after
    # the temporary DCP has been retired by the caller.
    for child in (source / "dcp").iterdir():
        child.unlink()
    (source / "dcp").rmdir()
    source.rmdir()
    assert export_inference_checkpoint_from_dcp(**kwargs) == result


def test_export_rejects_merged_reverse_mapping(
    tmp_path: Path,
    base_model_dir: Path,
) -> None:
    _write_base_transformer_weights(base_model_dir, {"q.weight": torch.empty(2, 2)})
    source = _save_model_only_dcp(
        tmp_path,
        {"fused_qkv.weight": torch.ones(6, 2)},
    )
    module = _LiveModule(
        {"fused_qkv.weight": ("q.weight", 0, 3)}
    )

    with pytest.raises(UnsupportedMergedReverseMappingError, match="model-specific split"):
        export_inference_checkpoint_from_dcp(
            dcp_checkpoint=source,
            output_dir=tmp_path / "run",
            step=3,
            module=module,
            base_model_dir=base_model_dir,
            max_shard_size_bytes=1024,
        )
    assert not (tmp_path / "run" / "inference" / "checkpoint-3").exists()


def test_export_refuses_incomplete_source_or_destination(
    tmp_path: Path,
    base_model_dir: Path,
) -> None:
    source = tmp_path / "incomplete-source" / "dcp"
    source.mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="missing .metadata"):
        export_inference_checkpoint_from_dcp(
            dcp_checkpoint=source,
            output_dir=tmp_path / "run",
            step=1,
            module=_LiveModule(),
            base_model_dir=base_model_dir,
        )

    complete_source = _save_model_only_dcp(
        tmp_path / "other",
        {"weight": torch.ones(1)},
    )
    incomplete_destination = tmp_path / "run" / "inference" / "checkpoint-2"
    incomplete_destination.mkdir(parents=True)
    (incomplete_destination / "metadata.json").write_text("{}", encoding="utf-8")
    with pytest.raises(InferenceCheckpointExportError, match="Refusing to overwrite incomplete"):
        export_inference_checkpoint_from_dcp(
            dcp_checkpoint=complete_source,
            output_dir=tmp_path / "run",
            step=2,
            module=_LiveModule(),
            base_model_dir=base_model_dir,
        )


def test_export_rejects_unknown_unmapped_tensor(
    tmp_path: Path,
    base_model_dir: Path,
) -> None:
    _write_base_transformer_weights(base_model_dir, {"weight": torch.empty(2)})
    source = _save_model_only_dcp(
        tmp_path,
        {
            "weight": torch.ones(2),
            "surprise.weight": torch.ones(2),
        },
    )

    with pytest.raises(InferenceCheckpointExportError, match="unknown base key"):
        export_inference_checkpoint_from_dcp(
            dcp_checkpoint=source,
            output_dir=tmp_path / "run",
            step=4,
            module=_LiveModule(),
            base_model_dir=base_model_dir,
        )


def test_export_rejects_missing_base_tensor(
    tmp_path: Path,
    base_model_dir: Path,
) -> None:
    _write_base_transformer_weights(
        base_model_dir,
        {
            "weight": torch.empty(2),
            "bias": torch.empty(2),
        },
    )
    source = _save_model_only_dcp(tmp_path, {"weight": torch.ones(2)})

    with pytest.raises(InferenceCheckpointExportError, match="missing 1 base transformer tensors"):
        export_inference_checkpoint_from_dcp(
            dcp_checkpoint=source,
            output_dir=tmp_path / "run",
            step=5,
            module=_LiveModule(),
            base_model_dir=base_model_dir,
        )


def test_export_rejects_base_shape_mismatch(
    tmp_path: Path,
    base_model_dir: Path,
) -> None:
    _write_base_transformer_weights(base_model_dir, {"weight": torch.empty(3)})
    source = _save_model_only_dcp(tmp_path, {"weight": torch.ones(2)})

    with pytest.raises(InferenceCheckpointExportError, match="shape .* != base transformer shape"):
        export_inference_checkpoint_from_dcp(
            dcp_checkpoint=source,
            output_dir=tmp_path / "run",
            step=6,
            module=_LiveModule(),
            base_model_dir=base_model_dir,
        )


def test_export_rejects_base_index_header_mismatch(
    tmp_path: Path,
    base_model_dir: Path,
) -> None:
    _write_base_transformer_weights(base_model_dir, {"weight": torch.empty(2)})
    index_path = base_model_dir / "transformer" / "diffusion_pytorch_model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index["weight_map"]["ghost"] = next(iter(index["weight_map"].values()))
    index_path.write_text(json.dumps(index), encoding="utf-8")
    source = _save_model_only_dcp(tmp_path, {"weight": torch.ones(2)})

    with pytest.raises(InferenceCheckpointExportError, match="index/header mismatch"):
        export_inference_checkpoint_from_dcp(
            dcp_checkpoint=source,
            output_dir=tmp_path / "run",
            step=7,
            module=_LiveModule(),
            base_model_dir=base_model_dir,
        )


def test_completed_checkpoint_validator_rejects_missing_and_corrupt_shards(
    tmp_path: Path,
    base_model_dir: Path,
) -> None:
    _write_base_transformer_weights(base_model_dir, {"weight": torch.empty(2)})
    source = _save_model_only_dcp(tmp_path, {"weight": torch.ones(2)})

    missing = export_inference_checkpoint_from_dcp(
        dcp_checkpoint=source,
        output_dir=tmp_path / "missing-run",
        step=8,
        module=_LiveModule(),
        base_model_dir=base_model_dir,
    )
    missing_shard = next((missing / "transformer").glob("*.safetensors"))
    missing_shard.unlink()
    with pytest.raises(InferenceCheckpointExportError, match="shards differ from its index"):
        validate_complete_inference_checkpoint(missing, step=8)

    corrupt = export_inference_checkpoint_from_dcp(
        dcp_checkpoint=source,
        output_dir=tmp_path / "corrupt-run",
        step=9,
        module=_LiveModule(),
        base_model_dir=base_model_dir,
    )
    corrupt_shard = next((corrupt / "transformer").glob("*.safetensors"))
    corrupt_shard.write_bytes(b"corrupt")
    with pytest.raises(InferenceCheckpointExportError, match="Cannot read inference checkpoint shard"):
        validate_complete_inference_checkpoint(corrupt, step=9)


def test_checkpoint_manager_adapter_uses_exact_target(
    tmp_path: Path,
    base_model_dir: Path,
) -> None:
    _write_base_transformer_weights(base_model_dir, {"weight": torch.empty(2)})
    source = _save_model_only_dcp(tmp_path, {"weight": torch.ones(2)})
    module = _LiveModule()
    target = tmp_path / "run" / "inference" / "checkpoint-9"

    result = export_inference_checkpoint(
        dcp_dir=source / "dcp",
        output_dir=target,
        base_model_path=base_model_dir,
        role="student",
        modules={"transformer": module},
        dtype="bfloat16",
        step=9,
        raw_config={"not": "serialized"},
    )

    assert result == target.resolve()
    metadata = json.loads((result / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["config"] == {"not": "serialized"}
    assert "source_dcp" not in metadata

    for child in (source / "dcp").iterdir():
        child.unlink()
    (source / "dcp").rmdir()
    source.rmdir()
    assert export_inference_checkpoint(
        dcp_dir=source / "dcp",
        output_dir=target,
        base_model_path=base_model_dir,
        role="student",
        modules={"transformer": module},
        dtype="bfloat16",
        step=9,
        raw_config={"not": "serialized"},
    ) == result

    with pytest.raises(ValueError, match="CheckpointManager output_dir"):
        export_inference_checkpoint(
            dcp_dir=source / "dcp",
            output_dir=tmp_path / "wrong-target",
            base_model_path=base_model_dir,
            role="student",
            modules={"transformer": module},
            dtype="bfloat16",
            step=9,
        )

    with pytest.raises(InferenceCheckpointExportError, match="exactly one module"):
        export_inference_checkpoint(
            dcp_dir=source / "dcp",
            output_dir=tmp_path / "run" / "inference" / "checkpoint-10",
            base_model_path=base_model_dir,
            role="student",
            modules={},
            dtype="bfloat16",
            step=10,
        )


def test_export_preserves_dmd_schedule_and_linked_base_configs(tmp_path, base_model_dir):
    for component, shift in (("scheduler", 12.0), ("audio_scheduler", 3.0)):
        directory = base_model_dir / component
        directory.mkdir()
        (directory / "scheduler_config.json").write_text(json.dumps({"shift": shift, "_class_name": "Scheduler"}))
    module = _LiveModule()
    _write_base_transformer_weights(base_model_dir, {"master": module.master.detach()})
    checkpoint = _save_model_only_dcp(tmp_path, {"master": module.master.detach()})
    overrides = {
        "scheduler/scheduler_config.json": {"shift": 10.0},
        "audio_scheduler/scheduler_config.json": {"shift": 3.0},
        "fastvideo_inference.json": {
            "schema_version": "fasth3-inference-contract-v1",
            "dmd_denoising_steps": [999, 874, 749, 624, 500, 375, 250, 125],
            "num_inference_steps": 9,
            "transformer_forwards": 8,
            "video_scheduler_shift": 10.0,
            "audio_scheduler_shift": 3.0,
        },
    }
    kwargs = dict(dcp_checkpoint=checkpoint, output_dir=tmp_path / "run", step=3,
                  module=module, base_model_dir=base_model_dir, json_overrides=overrides)
    exported = export_inference_checkpoint_from_dcp(**kwargs)
    assert json.loads((base_model_dir / "scheduler/scheduler_config.json").read_text())["shift"] == 12.0
    assert json.loads((exported / "scheduler/scheduler_config.json").read_text()) == {
        "shift": 10.0, "_class_name": "Scheduler"}
    assert json.loads((exported / "fastvideo_inference.json").read_text()) == overrides["fastvideo_inference.json"]
    assert export_inference_checkpoint_from_dcp(**kwargs) == exported
    (exported / "fastvideo_inference.json").write_text("{}")
    with pytest.raises(InferenceCheckpointExportError, match="metadata mismatch"):
        export_inference_checkpoint_from_dcp(**kwargs)


def test_json_overrides_replace_hardlinked_scheduler_config(tmp_path):
    import os
    source = tmp_path / "source.json"
    source.write_text('{"shift":12}')
    export = tmp_path / "export"
    (export / "scheduler").mkdir(parents=True)
    os.link(source, export / "scheduler/scheduler_config.json")
    inference_checkpoint.apply_export_json_overrides(export, {"scheduler/scheduler_config.json": {"shift": 10}})
    assert json.loads(source.read_text()) == {"shift": 12}
    assert json.loads((export / "scheduler/scheduler_config.json").read_text()) == {"shift": 10}



def test_json_overrides_reject_dangling_base_config_without_creating_it(tmp_path):
    base = tmp_path / "base"
    (base / "scheduler").mkdir(parents=True)
    missing = base / "missing.json"
    (base / "scheduler/scheduler_config.json").symlink_to(missing)
    export = tmp_path / "export"
    export.mkdir()
    (export / "scheduler").symlink_to(base / "scheduler", target_is_directory=True)
    with pytest.raises(InferenceCheckpointExportError, match="Invalid exported component config"):
        inference_checkpoint.apply_export_json_overrides(export, {"scheduler/scheduler_config.json": {"shift": 10}})
    assert not missing.exists()
    assert (base / "scheduler/scheduler_config.json").is_symlink()
