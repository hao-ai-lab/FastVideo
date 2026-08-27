# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pytest
import torch
from safetensors import safe_open

from scripts.checkpoint_conversion.cosmos25_distilled_to_diffusers import (
    ConversionError,
    convert_checkpoint,
    extract_student_state_dict,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _make_base_model(root: Path) -> None:
    _write_json(
        root / "model_index.json",
        {
            "_class_name": "Cosmos2_5Pipeline",
            "_diffusers_version": "0.37.0.dev0",
            "scheduler": ["diffusers", "FlowUniPCMultistepScheduler"],
            "text_encoder": ["transformers", "Reason1Model"],
            "tokenizer": ["transformers", "AutoTokenizer"],
            "transformer": ["diffusers", "Cosmos25Transformer3DModel"],
            "vae": ["diffusers", "AutoencoderKLWan"],
        },
    )
    _write_json(root / "transformer/config.json", {"_class_name": "Cosmos25Transformer3DModel"})
    (root / "transformer/base.safetensors").write_bytes(b"old weights")
    for component in ("vae", "text_encoder", "tokenizer"):
        (root / component).mkdir(parents=True)
    _write_json(root / "scheduler/scheduler_config.json", {"_class_name": "FlowUniPCMultistepScheduler"})


@pytest.mark.parametrize("nested_key", [None, "state_dict", "model", "ema"])
def test_extracts_only_native_student_tensors(nested_key: str | None) -> None:
    state = {
        "net.layer.weight": torch.ones(2, 3),
        "net.layer.bias": torch.zeros(2),
        "net_teacher.layer.weight": torch.full((2, 3), 2.0),
        "optimizer.step": torch.tensor(4),
        "metadata": "ignored",
    }
    checkpoint = state if nested_key is None else {nested_key: state, "iteration": 12}
    result = extract_student_state_dict(checkpoint)

    assert set(result) == {"net.layer.weight", "net.layer.bias"}
    assert all(tensor.dtype == torch.bfloat16 for tensor in result.values())
    assert all(tensor.is_contiguous() for tensor in result.values())


def test_rejects_checkpoint_without_student() -> None:
    with pytest.raises(ConversionError, match=r"net\.\*"):
        extract_student_state_dict({"net_teacher.weight": torch.ones(1)})


def test_builds_isolated_distilled_package(tmp_path: Path) -> None:
    base = tmp_path / "base"
    _make_base_model(base)
    checkpoint_path = tmp_path / "distilled.pt"
    torch.save(
        {
            "model": {
                "net.layer.weight": torch.arange(6, dtype=torch.float32).reshape(2, 3),
                "net_teacher.layer.weight": torch.ones(2, 3),
            }
        },
        checkpoint_path,
    )

    dst = tmp_path / "converted"
    report = convert_checkpoint(checkpoint_path, base, dst)

    assert report == {"student_tensors": 1, "student_parameters": 6}
    model_index = json.loads((dst / "model_index.json").read_text(encoding="utf-8"))
    assert model_index["is_distilled"] is True
    assert model_index["scheduler"] == ["diffusers", "Cosmos25DistilledScheduler"]
    scheduler = json.loads((dst / "scheduler/scheduler_config.json").read_text(encoding="utf-8"))
    assert scheduler["_class_name"] == "Cosmos25DistilledScheduler"
    assert not (dst / "transformer/base.safetensors").exists()
    assert (dst / "vae").is_dir()
    assert (dst / "text_encoder").is_dir()
    assert (dst / "tokenizer").is_dir()

    with safe_open(
        str(dst / "transformer/diffusion_pytorch_model.safetensors"),
        framework="pt",
        device="cpu",
    ) as handle:
        assert set(handle.keys()) == {"net.layer.weight"}
        assert handle.get_tensor("net.layer.weight").dtype == torch.bfloat16


def test_does_not_overwrite_existing_output_by_default(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "distilled.pt"
    torch.save({"net.weight": torch.ones(1)}, checkpoint_path)
    base = tmp_path / "base"
    _make_base_model(base)
    dst = tmp_path / "converted"
    dst.mkdir()
    (dst / "keep.txt").write_text("user data", encoding="utf-8")

    with pytest.raises(FileExistsError, match="--overwrite"):
        convert_checkpoint(checkpoint_path, base, dst)
    assert (dst / "keep.txt").read_text(encoding="utf-8") == "user data"
