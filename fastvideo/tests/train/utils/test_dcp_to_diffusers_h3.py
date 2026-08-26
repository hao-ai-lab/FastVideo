# SPDX-License-Identifier: Apache-2.0
"""H3 checkpoint export contracts for full tuning and merged LoRA."""

import json
from pathlib import Path
from types import SimpleNamespace

from safetensors.torch import load_file
import torch

from fastvideo.layers.linear import ReplicatedLinear
from fastvideo.pipelines.basic.minimax_h3.minimax_h3_pipeline import MiniMaxH3Ref2VAModularPipeline
from fastvideo.train.entrypoint.dcp_to_diffusers import (
    _native_export_state_dict,
    _save_role_pretrained,
    _strict_reload_verify,
)
from fastvideo.train.utils.lora import enable_lora_training, finalize_lora_training


class _ArchConfig:
    exclude_lora_layers: list[str] = []


class _Config:
    arch_config = _ArchConfig()


class _TinyTransformer(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.config = _Config()
        self.to_q = ReplicatedLinear(4, 4, bias=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.to_q(value)[0]


def test_lora_export_merges_to_native_strict_loadable_state() -> None:
    torch.manual_seed(3)
    wrapped = _TinyTransformer()
    enable_lora_training(
        wrapped,
        lora_rank=2,
        lora_alpha=4,
        lora_target_modules=["to_q"],
    )
    finalize_lora_training(wrapped)
    with torch.no_grad():
        wrapped.to_q.lora_A.copy_(torch.arange(8, dtype=torch.float32).reshape(2, 4) / 10)
        wrapped.to_q.lora_B.copy_(torch.arange(8, dtype=torch.float32).reshape(4, 2) / 20)

    native_state, reverse_aliases, merged = _native_export_state_dict(wrapped, wrapped.state_dict())

    assert merged is True
    assert set(native_state) == {"to_q.weight"}
    assert reverse_aliases == {"to_q.weight": "to_q.base_layer.weight"}
    assert not any("base_layer" in key or "lora_" in key for key in native_state)

    native = _TinyTransformer()
    incompatible = native.load_state_dict(native_state, strict=True)
    assert incompatible.missing_keys == []
    assert incompatible.unexpected_keys == []
    value = torch.arange(12, dtype=torch.float32).reshape(3, 4) / 7
    torch.testing.assert_close(native(value), wrapped(value), rtol=1e-5, atol=1e-5)


def test_ref2va_export_replaces_only_transformer_ref(
    tmp_path: Path,
    monkeypatch,
) -> None:
    base = tmp_path / "base"
    (base / "transformer").mkdir(parents=True)
    (base / "transformer_ref").mkdir()
    (base / "model_index.json").write_text("{}", encoding="utf-8")
    t2va_sentinel = b"t2va-base-must-remain"
    (base / "transformer" / "model.safetensors").write_bytes(t2va_sentinel)
    (base / "transformer_ref" / "model.safetensors").write_bytes(b"ref-base")

    transformer = torch.nn.Linear(2, 2, bias=False)
    transformer.reverse_param_names_mapping = {}
    model = SimpleNamespace(
        transformer=transformer,
        transformer_module_type="transformer_ref",
    )
    monkeypatch.setattr("fastvideo.utils.maybe_download_model", lambda _path: str(base))
    monkeypatch.setattr(
        "torch.distributed.checkpoint.state_dict.get_model_state_dict",
        lambda module, options: module.state_dict(),
    )

    output = tmp_path / "export"
    result = _save_role_pretrained(
        role="student",
        base_model_path=str(base),
        output_dir=str(output),
        model=model,
    )

    assert result == str(output.resolve())
    assert (output / "transformer" / "model.safetensors").read_bytes() == t2va_sentinel
    exported_ref = load_file(str(output / "transformer_ref" / "model.safetensors"))
    torch.testing.assert_close(exported_ref["weight"], transformer.weight)
    metadata = json.loads((output / "fastvideo_training_export.json").read_text(encoding="utf-8"))
    assert metadata == {
        "format_version": 1,
        "lora": "none",
        "role": "student",
        "transformer_component": "transformer_ref",
    }
    assert MiniMaxH3Ref2VAModularPipeline._extra_config_module_map["transformer"] == "transformer_ref"


def test_strict_reload_uses_physical_component(monkeypatch) -> None:
    loaded: dict[str, str] = {}

    def _load_module_from_path(**kwargs):
        loaded.update(kwargs)
        return torch.nn.Identity()

    monkeypatch.setattr("fastvideo.train.utils.moduleloader.load_module_from_path", _load_module_from_path)
    training_config = object()

    _strict_reload_verify(
        output_dir="/tmp/export",
        training_config=training_config,
        module_type="transformer_ref",
    )

    assert loaded["model_path"] == "/tmp/export"
    assert loaded["module_type"] == "transformer_ref"
    assert loaded["training_config"] is training_config
