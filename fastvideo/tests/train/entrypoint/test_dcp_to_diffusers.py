# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from types import SimpleNamespace

import torch
from safetensors.torch import load_file, save_file

from fastvideo.train.entrypoint.dcp_to_diffusers import _save_role_pretrained


def test_save_role_pretrained_splits_merged_parameters(tmp_path) -> None:
    base = tmp_path / "base"
    transformer_dir = base / "transformer"
    transformer_dir.mkdir(parents=True)
    (base / "model_index.json").write_text("{}", encoding="utf-8")
    (transformer_dir / "config.json").write_text(
        json.dumps({"_class_name": "FakeTransformer"}),
        encoding="utf-8",
    )
    save_file(
        {"stale.weight": torch.zeros(1)},
        str(transformer_dir / "stale.safetensors"),
    )

    transformer = torch.nn.Module()
    transformer.to_qkv = torch.nn.Linear(4, 6, bias=True)
    with torch.no_grad():
        transformer.to_qkv.weight.copy_(torch.arange(24).reshape(6, 4))
        transformer.to_qkv.bias.copy_(torch.arange(6))
    transformer.reverse_param_names_mapping = {
        "to_qkv.weight": [
            ("to_q.weight", 0, 3, 1),
            ("to_k.weight", 1, 3, 2),
            ("to_v.weight", 2, 3, 3),
        ],
        "to_qkv.bias": [
            ("to_q.bias", 0, 3, 1),
            ("to_k.bias", 1, 3, 2),
            ("to_v.bias", 2, 3, 3),
        ],
    }

    output = tmp_path / "export"
    _save_role_pretrained(
        role="student",
        base_model_path=str(base),
        output_dir=str(output),
        model=SimpleNamespace(transformer=transformer),
    )

    exported = load_file(str(output / "transformer" / "model.safetensors"))
    assert set(exported) == {
        "to_q.weight",
        "to_q.bias",
        "to_k.weight",
        "to_k.bias",
        "to_v.weight",
        "to_v.bias",
    }
    torch.testing.assert_close(
        exported["to_q.weight"],
        transformer.to_qkv.weight[:1],
    )
    torch.testing.assert_close(
        exported["to_k.weight"],
        transformer.to_qkv.weight[1:3],
    )
    torch.testing.assert_close(
        exported["to_v.weight"],
        transformer.to_qkv.weight[3:],
    )
    torch.testing.assert_close(exported["to_q.bias"], transformer.to_qkv.bias[:1])
    torch.testing.assert_close(exported["to_k.bias"], transformer.to_qkv.bias[1:3])
    torch.testing.assert_close(exported["to_v.bias"], transformer.to_qkv.bias[3:])
    assert not (output / "transformer" / "stale.safetensors").exists()
