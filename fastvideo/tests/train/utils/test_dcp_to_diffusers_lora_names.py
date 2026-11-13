# SPDX-License-Identifier: Apache-2.0
"""Merged-LoRA exports write HF parameter names whether LoRA was enabled before or after loading."""

from pathlib import Path
from types import SimpleNamespace

import pytest
from safetensors.torch import load_file
import torch

from fastvideo.layers.linear import ReplicatedLinear
from fastvideo.models.loader.fsdp_load import load_model_from_full_model_state_dict
from fastvideo.train.entrypoint.dcp_to_diffusers import _save_role_pretrained
from fastvideo.train.utils.lora import enable_lora_training, finalize_lora_training


class _ArchConfig:
    exclude_lora_layers: list[str] = []


class _Config:
    arch_config = _ArchConfig()


class _Block(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.to_q = ReplicatedLinear(4, 4, bias=False)


class _Transformer(torch.nn.Module):
    """Internal name ``block.to_q`` for the HF name ``block.attn.to_q``, like Wan's self-attention."""

    def __init__(self) -> None:
        super().__init__()
        self.config = _Config()
        self.block = _Block()


def _export(tmp_path: Path, monkeypatch, transformer: torch.nn.Module) -> dict[str, torch.Tensor]:
    base = tmp_path / "base"
    (base / "transformer").mkdir(parents=True)
    (base / "model_index.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr("fastvideo.utils.maybe_download_model", lambda _path: str(base))
    monkeypatch.setattr(
        "torch.distributed.checkpoint.state_dict.get_model_state_dict",
        lambda module, options: module.state_dict(),
    )
    output = tmp_path / "export"
    _save_role_pretrained(
        role="student",
        base_model_path=str(base),
        output_dir=str(output),
        model=SimpleNamespace(transformer=transformer, transformer_module_type="transformer"),
    )
    return load_file(str(output / "transformer" / "model.safetensors"))


def _loaded_with_lora(*, lora_before_load: bool) -> _Transformer:
    with torch.device("meta"):
        transformer = _Transformer()
    if lora_before_load:
        enable_lora_training(transformer, lora_rank=2, lora_alpha=2, lora_target_modules=["to_q"],
                             prepare_for_fsdp=True)
    load_model_from_full_model_state_dict(
        transformer,
        iter([("block.attn.to_q.weight", torch.arange(16, dtype=torch.float32).reshape(4, 4) / 10)]),
        torch.device("cpu"),
        torch.float32,
        strict=True,
        param_names_mapping=lambda name: (name.replace("block.attn.to_q.", "block.to_q."), None, None),
    )
    if not lora_before_load:
        enable_lora_training(transformer, lora_rank=2, lora_alpha=2, lora_target_modules=["to_q"])
    finalize_lora_training(transformer)
    return transformer


@pytest.mark.parametrize("lora_before_load", [False, True], ids=["lora-enabled-after-load", "lora-enabled-before-load"])
def test_merged_lora_export_uses_hf_names(tmp_path: Path, monkeypatch, lora_before_load: bool) -> None:
    transformer = _loaded_with_lora(lora_before_load=lora_before_load)
    wrapper = transformer.block.to_q
    with torch.no_grad():
        wrapper.lora_A.copy_(torch.arange(8, dtype=torch.float32).reshape(2, 4) / 10)
        wrapper.lora_B.fill_(0.5)
    expected = wrapper.base_layer.weight.detach() + wrapper.lora_B.detach() @ wrapper.lora_A.detach()

    exported = _export(tmp_path, monkeypatch, transformer)

    assert set(exported) == {"block.attn.to_q.weight"}
    torch.testing.assert_close(exported["block.attn.to_q.weight"], expected)
