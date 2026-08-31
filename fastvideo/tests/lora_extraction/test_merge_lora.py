"""Coverage for merging the non-factorized half of an adapter into base weights."""

from __future__ import annotations

import json
import logging
from pathlib import Path
import sys

import pytest
from safetensors.torch import save_file
import torch

from fastvideo.configs.models.dits.wanvideo import WanVideoConfig
from fastvideo.models.loader.utils import get_param_names_mapping

_REPO_ROOT = Path(__file__).parents[3]
sys.path.insert(0, str(_REPO_ROOT / "scripts" / "lora_extraction"))

import extract_lora  # noqa: E402
import merge_lora  # noqa: E402


def _base_state() -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(11)
    return {
        "blocks.0.linear.weight": torch.randn(9, 7, generator=generator),
        "audio_proj_in.weight": torch.randn(8, 6, generator=generator),
        "time_embedder.linear.bias": torch.randn(8, generator=generator),
        "scale_param": torch.randn(4, generator=generator),
    }


def _adapter(base: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(12)
    return {
        "blocks.0.linear.lora_A.weight": torch.randn(2, 7, generator=generator),
        "blocks.0.linear.lora_B.weight": torch.randn(9, 2, generator=generator),
        "audio_proj_in.diff": torch.randn(8, 6, generator=generator),
        "time_embedder.linear.diff_b": torch.randn(8, generator=generator),
        "scale_param.diff_param": torch.randn(4, generator=generator),
        "new_proj.set_weight": torch.randn(3, 3, generator=generator),
    }


def test_group_dense_keys_splits_additive_and_replacement() -> None:
    additive, replacement, unrecognized = merge_lora.group_dense_keys(_adapter(_base_state()))

    assert set(additive) == {"audio_proj_in.weight", "time_embedder.linear.bias", "scale_param"}
    assert set(replacement) == {"new_proj.weight"}
    assert unrecognized == []


def test_group_dense_keys_reports_unrecognized_suffixes() -> None:
    _, _, unrecognized = merge_lora.group_dense_keys({"mystery.tensor": torch.zeros(2)})
    assert unrecognized == ["mystery.tensor"]


def test_merge_applies_dense_tensors_alongside_lora() -> None:
    """--exact-tensor-pattern keeps tensors as dense deltas; the merge must still apply them."""
    base = _base_state()
    adapter = _adapter(base)

    merged = merge_lora.merge_lora_into_base(base, adapter)

    expected_lora = base["blocks.0.linear.weight"] + adapter["blocks.0.linear.lora_B.weight"] @ adapter[
        "blocks.0.linear.lora_A.weight"]
    assert torch.allclose(merged["blocks.0.linear.weight"], expected_lora, atol=1e-6)
    assert torch.allclose(merged["audio_proj_in.weight"],
                          base["audio_proj_in.weight"] + adapter["audio_proj_in.diff"],
                          atol=1e-6)
    assert torch.allclose(merged["time_embedder.linear.bias"],
                          base["time_embedder.linear.bias"] + adapter["time_embedder.linear.diff_b"],
                          atol=1e-6)
    assert torch.allclose(merged["scale_param"], base["scale_param"] + adapter["scale_param.diff_param"], atol=1e-6)
    assert torch.equal(merged["new_proj.weight"], adapter["new_proj.set_weight"])
    # the caller's state dict must not be mutated
    assert torch.equal(base["audio_proj_in.weight"], _base_state()["audio_proj_in.weight"])


def _write_transformer(root: Path, state: dict[str, torch.Tensor]) -> None:
    transformer = root / "transformer"
    transformer.mkdir(parents=True)
    shard = "diffusion_pytorch_model-00001-of-00001.safetensors"
    save_file(state, transformer / shard)
    index = {
        "metadata": {
            "total_size": sum(tensor.numel() * tensor.element_size() for tensor in state.values())
        },
        "weight_map": {key: shard for key in state},
    }
    (transformer / extract_lora.INDEX_FILENAME).write_text(json.dumps(index), encoding="utf-8")


def test_indexed_wan_extraction_merges_into_fastvideo_namespace(tmp_path: Path) -> None:
    generator = torch.Generator().manual_seed(21)
    base_hf = {
        "blocks.0.attn1.to_q.weight": torch.randn(5, 4, generator=generator),
        "condition_embedder.time_proj.weight": torch.randn(3, 4, generator=generator),
    }
    finetuned_hf = {name: tensor.clone() for name, tensor in base_hf.items()}
    finetuned_hf["blocks.0.attn1.to_q.weight"] += torch.randn(
        5, 2, generator=generator) @ torch.randn(2, 4, generator=generator)
    finetuned_hf["condition_embedder.time_proj.weight"] += 0.25
    base_dir = tmp_path / "base"
    finetuned_dir = tmp_path / "finetuned"
    output = tmp_path / "adapter.safetensors"
    _write_transformer(base_dir, base_hf)
    _write_transformer(finetuned_dir, finetuned_hf)
    extract_lora.extract_lora_adapter(
        base=str(base_dir),
        ft=str(finetuned_dir),
        out=str(output),
        rank=2,
        min_delta=0.0,
        load_mode="indexed",
        exact_tensor_patterns=(r"^condition_embedder\.time_proj\.weight$", ),
    )

    base_custom = {
        "blocks.0.to_q.weight": base_hf["blocks.0.attn1.to_q.weight"],
        "condition_embedder.time_modulation.linear.weight": base_hf["condition_embedder.time_proj.weight"],
    }
    config = WanVideoConfig()
    merged = merge_lora.merge_lora_into_base(
        base_custom,
        merge_lora.load_adapter(str(output)),
        lora_param_names_mapping=get_param_names_mapping(config.lora_param_names_mapping),
        param_names_mapping=get_param_names_mapping(config.param_names_mapping),
    )

    torch.testing.assert_close(merged["blocks.0.to_q.weight"], finetuned_hf["blocks.0.attn1.to_q.weight"])
    torch.testing.assert_close(merged["condition_embedder.time_modulation.linear.weight"],
                               finetuned_hf["condition_embedder.time_proj.weight"])


def test_merge_accumulates_multiple_sources_into_a_fused_parameter() -> None:
    base = {"fused.weight": torch.zeros(6, 4)}
    adapter = {
        "q.lora_A.weight": torch.ones(1, 4),
        "q.lora_B.weight": torch.ones(3, 1),
        "k.lora_A.weight": torch.full((1, 4), 2.0),
        "k.lora_B.weight": torch.ones(3, 1),
    }

    def mapping(name: str):
        return "fused.weight", 0 if name.startswith("q.") else 1, 2

    merged = merge_lora.merge_lora_into_base(base, adapter, param_names_mapping=mapping)
    torch.testing.assert_close(merged["fused.weight"][:3], torch.ones(3, 4))
    torch.testing.assert_close(merged["fused.weight"][3:], torch.full((3, 4), 2.0))


def test_merge_is_strict_about_unapplied_keys() -> None:
    with pytest.raises(ValueError, match="unapplied"):
        merge_lora.merge_lora_into_base(_base_state(), {"missing.diff": torch.zeros(2)})


def test_merge_warns_instead_of_silently_dropping_unknown_keys(caplog) -> None:
    base = _base_state()
    adapter = {"mystery.tensor": torch.zeros(2), "shape_mismatch.diff": torch.zeros(1, 1)}

    with caplog.at_level(logging.WARNING, logger=merge_lora.LOG.name):
        merge_lora.merge_lora_into_base(base, adapter, strict=False)

    messages = " ".join(record.getMessage() for record in caplog.records)
    assert "unrecognized adapter key" in messages
    assert "mystery.tensor" in messages
    assert "shape_mismatch.weight" in messages
