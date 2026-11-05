# SPDX-License-Identifier: Apache-2.0
"""Merged-LoRA exports write HF parameter names whether LoRA was enabled before or after loading."""

from pathlib import Path
from types import SimpleNamespace

import pytest
from safetensors.torch import load_file
import torch

from fastvideo.layers.linear import ReplicatedLinear
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


@pytest.mark.parametrize("recorded_name", ["block.to_q.weight", "block.to_q.base_layer.weight"],
                         ids=["lora-enabled-after-load", "lora-enabled-before-load"])
def test_merged_lora_export_uses_hf_names(tmp_path: Path, monkeypatch, recorded_name: str) -> None:
    torch.manual_seed(0)
    transformer = _Transformer()
    # The loader records internal -> HF names under whatever the parameter was called at load time.
    transformer.reverse_param_names_mapping = {recorded_name: ("block.attn.to_q.weight", None, None)}
    enable_lora_training(transformer, lora_rank=2, lora_alpha=2, lora_target_modules=["to_q"])
    finalize_lora_training(transformer)
    wrapper = transformer.block.to_q
    with torch.no_grad():
        wrapper.lora_B.fill_(0.5)
    # rank 2, alpha 2: the merged weight is base + B @ A.
    expected = wrapper.base_layer.weight.detach() + wrapper.lora_B.detach() @ wrapper.lora_A.detach()

    exported = _export(tmp_path, monkeypatch, transformer)

    assert set(exported) == {"block.attn.to_q.weight"}
    torch.testing.assert_close(exported["block.attn.to_q.weight"], expected)
