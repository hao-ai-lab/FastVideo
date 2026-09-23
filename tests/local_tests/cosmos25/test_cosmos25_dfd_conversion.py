# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import torch
import torch.distributed.checkpoint as dcp
from safetensors import safe_open
from safetensors.torch import save_file

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = REPO_ROOT / "scripts/checkpoint_conversion/cosmos25_dfd_to_diffusers.py"


def _load_converter():
    spec = importlib.util.spec_from_file_location("cosmos25_dfd_converter", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _base_package(root: Path) -> Path:
    _write_json(
        root / "model_index.json",
        {
            "_class_name": "Cosmos2_5Pipeline",
            "transformer": ["diffusers", "Cosmos25Transformer3DModel"],
            "scheduler": ["diffusers", "FlowUniPCMultistepScheduler"],
            "vae": ["diffusers", "AutoencoderKL"],
            "text_encoder": ["transformers", "AutoModel"],
            "tokenizer": ["transformers", "AutoTokenizer"],
            "safety_checker": [None, None],
        },
    )
    _write_json(
        root / "transformer/config.json",
        {
            "_class_name": "Cosmos25Transformer3DModel",
            "rope_enable_fps_modulation": False,
            "use_crossattn_projection": True,
        },
    )
    save_file({"old.weight": torch.ones(1)}, str(root / "transformer/diffusion_pytorch_model.safetensors"))
    for component in ("vae", "text_encoder", "tokenizer"):
        (root / component).mkdir(parents=True)
    return root


def _write_dcp(root: Path) -> Path:
    state = {
        "transformer.blocks.0.weight": torch.arange(6, dtype=torch.float32).reshape(2, 3),
        "transformer._checkpoint_wrapped_module.final_layer.bias": torch.ones(2),
        "transformer.logvar_linear.weight": torch.ones(1),
        "transformer.pos_embedder.seq": torch.ones(1),
    }
    dcp.save(state, checkpoint_id=root)
    return root


def test_normalize_dfd_student_keys_and_skip_training_state():
    converter = _load_converter()
    state = {
        "transformer.blocks.0.weight": torch.ones(1),
        "net.transformer.final_layer.bias": torch.ones(1),
        "module.net.x_embedder.proj.1.weight": torch.ones(1),
        "transformer.logvar_linear.weight": torch.ones(1),
        "transformer.pos_embedder.seq": torch.ones(1),
    }

    actual = converter.normalize_dfd_student_state_dict(state)

    assert set(actual) == {
        "net.blocks.0.weight",
        "net.final_layer.bias",
        "net.x_embedder.proj.1.weight",
    }
    assert all(tensor.dtype == torch.bfloat16 for tensor in actual.values())


def test_load_dfd_dcp_state_dict(tmp_path):
    converter = _load_converter()
    checkpoint = _write_dcp(tmp_path / "0000040.net_model")

    actual = converter.load_dfd_dcp_state_dict(checkpoint)

    assert "transformer.blocks.0.weight" in actual
    torch.testing.assert_close(
        actual["transformer.blocks.0.weight"],
        torch.arange(6, dtype=torch.float32).reshape(2, 3),
    )


def test_convert_dfd_package(tmp_path):
    converter = _load_converter()
    checkpoint = _write_dcp(tmp_path / "0000040.net_model")
    base = _base_package(tmp_path / "base")
    output = tmp_path / "converted"

    report = converter.convert_checkpoint(checkpoint, base, output)

    assert report["student_tensors"] == 2
    model_index = json.loads((output / "model_index.json").read_text())
    assert model_index["distillation_method"] == "dfd"
    assert model_index["scheduler"] == ["diffusers", "Cosmos25DFDScheduler"]
    transformer_config = json.loads((output / "transformer/config.json").read_text())
    assert transformer_config["rope_enable_fps_modulation"] is True
    scheduler_config = json.loads((output / "scheduler/scheduler_config.json").read_text())
    assert scheduler_config["_class_name"] == "Cosmos25DFDScheduler"

    with safe_open(
        str(output / "transformer/diffusion_pytorch_model.safetensors"),
        framework="pt",
        device="cpu",
    ) as handle:
        assert set(handle.keys()) == {
            "net.blocks.0.weight",
            "net.final_layer.bias",
        }


def test_conversion_refuses_nonempty_output(tmp_path):
    converter = _load_converter()
    checkpoint = _write_dcp(tmp_path / "0000040.net_model")
    base = _base_package(tmp_path / "base")
    output = tmp_path / "converted"
    output.mkdir()
    (output / "keep.txt").write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError, match="Pass --overwrite"):
        converter.convert_checkpoint(checkpoint, base, output)
