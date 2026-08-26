# SPDX-License-Identifier: Apache-2.0
"""H3 checkpoint export contracts for full tuning and merged LoRA."""

import json
from pathlib import Path
from types import SimpleNamespace

from diffusers import ConfigMixin, ModelMixin
from diffusers.configuration_utils import register_to_config
from safetensors.torch import load_file
import torch
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper

from fastvideo.layers.linear import ReplicatedLinear
from fastvideo.models.loader.weight_utils import safetensors_weights_iterator
from fastvideo.pipelines.basic.minimax_h3.minimax_h3_pipeline import MiniMaxH3Ref2VAModularPipeline
from fastvideo.train.entrypoint.dcp_to_diffusers import (
    _native_export_state_dict,
    _save_diffusers_safetensors,
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


class _TinyDiffusersTransformer(ModelMixin, ConfigMixin):

    @register_to_config
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(2, 2))


class _TinyShardedDiffusersTransformer(ModelMixin, ConfigMixin):

    @register_to_config
    def __init__(self) -> None:
        super().__init__()
        self.first = torch.nn.Parameter(torch.zeros(8))
        self.second = torch.nn.Parameter(torch.zeros(8))


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


def test_checkpoint_wrapped_lora_export_uses_native_prefix_and_matches_forward() -> None:
    torch.manual_seed(13)
    training_model = _TinyTransformer()
    enable_lora_training(
        training_model,
        lora_rank=2,
        lora_alpha=4,
        lora_target_modules=["to_q"],
    )
    finalize_lora_training(training_model)
    with torch.no_grad():
        training_model.to_q.lora_A.copy_(torch.arange(8, dtype=torch.float32).reshape(2, 4) / 11)
        training_model.to_q.lora_B.copy_(torch.arange(8, dtype=torch.float32).reshape(4, 2) / 17)

    checkpointed = checkpoint_wrapper(training_model, preserve_rng_state=False)
    assert any("_checkpoint_wrapped_module" in name for name, _ in checkpointed.named_modules())
    assert not any("_checkpoint_wrapped_module" in key for key in checkpointed.state_dict())

    native_state, reverse_aliases, merged = _native_export_state_dict(checkpointed, checkpointed.state_dict())

    assert merged is True
    assert set(native_state) == {"to_q.weight"}
    assert reverse_aliases == {"to_q.weight": "to_q.base_layer.weight"}
    native = _TinyTransformer()
    native.load_state_dict(native_state, strict=True)
    value = torch.arange(12, dtype=torch.float32).reshape(3, 4) / 7
    torch.testing.assert_close(native(value), checkpointed(value), rtol=1e-5, atol=1e-5)


def test_large_export_uses_diffusers_shards_and_fresh_index(tmp_path: Path) -> None:
    state = {
        "first": torch.arange(8, dtype=torch.float32),
        "second": torch.arange(8, dtype=torch.float32) + 10,
    }
    expected = {name: value.clone() for name, value in state.items()}
    _TinyShardedDiffusersTransformer().save_config(tmp_path)

    written = _save_diffusers_safetensors(state, tmp_path, max_shard_size=40)

    assert state == {}
    assert written == [
        "diffusion_pytorch_model-00001-of-00002.safetensors",
        "diffusion_pytorch_model-00002-of-00002.safetensors",
        "diffusion_pytorch_model.safetensors.index.json",
    ]
    index = json.loads((tmp_path / "diffusion_pytorch_model.safetensors.index.json").read_text(encoding="utf-8"))
    assert index["weight_map"] == {
        "first": "diffusion_pytorch_model-00001-of-00002.safetensors",
        "second": "diffusion_pytorch_model-00002-of-00002.safetensors",
    }
    loaded = {}
    for path in sorted(tmp_path.glob("diffusion_pytorch_model-*.safetensors")):
        loaded.update(load_file(str(path)))
    assert set(loaded) == set(expected)
    for name in expected:
        torch.testing.assert_close(loaded[name], expected[name])
    diffusers_model = _TinyShardedDiffusersTransformer.from_pretrained(tmp_path)
    fastvideo_model = _TinyShardedDiffusersTransformer()
    fastvideo_state = dict(
        safetensors_weights_iterator(
            [str(path) for path in sorted(tmp_path.glob("diffusion_pytorch_model-*.safetensors"))],
            to_cpu=True,
            broadcast=False,
        ))
    fastvideo_model.load_state_dict(fastvideo_state, strict=True)
    for name in expected:
        torch.testing.assert_close(getattr(diffusers_model, name), expected[name])
        torch.testing.assert_close(getattr(fastvideo_model, name), expected[name])


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
    (base / "transformer_ref" / "diffusion_pytorch_model-00001-of-00001.safetensors").write_bytes(b"stale")
    (base / "transformer_ref" / "diffusion_pytorch_model.safetensors.index.json").write_text(
        '{"weight_map":{"stale":"diffusion_pytorch_model-00001-of-00001.safetensors"}}',
        encoding="utf-8",
    )
    _TinyDiffusersTransformer().save_config(base / "transformer_ref")

    transformer = torch.nn.Linear(2, 2, bias=False)
    transformer.reverse_param_names_mapping = {}
    model = SimpleNamespace(
        transformer=transformer,
        transformer_module_type="transformer_ref",
        export_diffusers_weight_layout=True,
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
    component_dir = output / "transformer_ref"
    weight_path = component_dir / "diffusion_pytorch_model.safetensors"
    assert weight_path.is_file()
    assert not (component_dir / "model.safetensors").exists()
    assert not (component_dir / "diffusion_pytorch_model.safetensors.index.json").exists()
    assert not list(component_dir.glob("diffusion_pytorch_model-*.safetensors"))
    exported_ref = load_file(str(weight_path))
    torch.testing.assert_close(exported_ref["weight"], transformer.weight)

    # The canonical artifact loads through both the public Diffusers contract
    # and FastVideo's production safetensors iterator.
    diffusers_model = _TinyDiffusersTransformer.from_pretrained(component_dir)
    torch.testing.assert_close(diffusers_model.weight, transformer.weight)
    fastvideo_state = dict(
        safetensors_weights_iterator(
            [str(weight_path)],
            to_cpu=True,
            broadcast=False,
        ))
    fastvideo_model = torch.nn.Linear(2, 2, bias=False)
    fastvideo_model.load_state_dict(fastvideo_state, strict=True)
    torch.testing.assert_close(fastvideo_model.weight, transformer.weight)
    metadata = json.loads((output / "fastvideo_training_export.json").read_text(encoding="utf-8"))
    assert metadata == {
        "format_version": 1,
        "lora": "none",
        "role": "student",
        "transformer_component": "transformer_ref",
    }
    assert MiniMaxH3Ref2VAModularPipeline._extra_config_module_map["transformer"] == "transformer_ref"


def test_non_opted_in_model_preserves_legacy_export_filename(tmp_path: Path, monkeypatch) -> None:
    base = tmp_path / "base"
    (base / "transformer").mkdir(parents=True)
    (base / "model_index.json").write_text("{}", encoding="utf-8")
    (base / "transformer" / "model.safetensors").write_bytes(b"base")
    transformer = torch.nn.Linear(2, 2, bias=False)
    transformer.reverse_param_names_mapping = {}
    model = SimpleNamespace(transformer=transformer, transformer_module_type="transformer")
    monkeypatch.setattr("fastvideo.utils.maybe_download_model", lambda _path: str(base))
    monkeypatch.setattr(
        "torch.distributed.checkpoint.state_dict.get_model_state_dict",
        lambda module, options: module.state_dict(),
    )

    output = tmp_path / "legacy-export"
    _save_role_pretrained(
        role="student",
        base_model_path=str(base),
        output_dir=str(output),
        model=model,
    )

    weight_path = output / "transformer" / "model.safetensors"
    assert weight_path.is_file()
    assert not (output / "transformer" / "diffusion_pytorch_model.safetensors").exists()
    torch.testing.assert_close(load_file(str(weight_path))["weight"], transformer.weight)


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
