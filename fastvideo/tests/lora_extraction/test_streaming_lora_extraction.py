"""Unit coverage for streaming and GPU-capable LoRA extraction."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest
from safetensors import safe_open
from safetensors.torch import load_file, save_file
import torch

_REPO_ROOT = Path(__file__).parents[3]
sys.path.insert(0, str(_REPO_ROOT / "scripts" / "lora_extraction"))

import extract_lora  # noqa: E402


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


def _toy_states() -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    generator = torch.Generator().manual_seed(7)
    base = {
        "blocks.0.linear.weight": torch.randn(9, 7, generator=generator),
        "blocks.0.norm.weight": torch.randn(7, generator=generator),
        "context.weight": torch.randn(8, 6, generator=generator),
        "unchanged.bias": torch.randn(8, generator=generator),
    }
    finetuned = {key: value.clone() for key, value in base.items()}
    finetuned["blocks.0.linear.weight"] += torch.randn(9, 2, generator=generator) @ torch.randn(
        2, 7, generator=generator)
    finetuned["blocks.0.norm.weight"] += 0.125
    finetuned["context.weight"] += torch.randn(8, 6, generator=generator) * 0.01
    finetuned["blocks.0.attn.to_gate_compress.weight"] = torch.randn(
        9, 7, generator=generator).to(torch.bfloat16)
    return base, finetuned


def test_streaming_extraction_emits_lora_diff_and_replacement(tmp_path: Path) -> None:
    base, finetuned = _toy_states()
    base_dir = tmp_path / "base"
    finetuned_dir = tmp_path / "finetuned"
    output = tmp_path / "adapter.safetensors"
    _write_transformer(base_dir, base)
    _write_transformer(finetuned_dir, finetuned)

    result = extract_lora.extract_lora_adapter(
        base=str(base_dir),
        ft=str(finetuned_dir),
        out=str(output),
        rank=2,
        min_delta=0.0,
        load_mode="indexed",
        device="cpu",
        svd_method="exact",
        factor_dtype="float16",
        dense_dtype="float32",
        replacement_dtype="source",
        exact_tensor_patterns=(r"^context\.weight$", ),
    )

    assert result == output
    adapter = load_file(output)
    torch.testing.assert_close(
        adapter["blocks.0.linear.lora_B.weight"].float()
        @ adapter["blocks.0.linear.lora_A.weight"].float(),
        finetuned["blocks.0.linear.weight"] - base["blocks.0.linear.weight"],
        atol=6e-3,
        rtol=6e-3,
    )
    assert adapter["blocks.0.norm.diff"].dtype == torch.float32
    assert adapter["context.diff"].dtype == torch.float32
    assert adapter["blocks.0.attn.to_gate_compress.set_weight"].dtype == torch.bfloat16
    assert not any("unchanged" in key for key in adapter)
    assert not any(key.endswith((".lora_rank", ".lora_alpha")) for key in adapter)

    report = json.loads(output.with_suffix(".safetensors.report.json").read_text())
    assert report["counts"] == {"diff": 2, "lora": 1, "set_weight": 1, "unchanged": 1}
    with safe_open(output, framework="pt") as handle:
        assert handle.metadata()["svd_method"] == "exact"
        assert handle.metadata()["factor_dtype"] == "float16"


def test_randomized_extraction_is_seeded_and_reports_residual(tmp_path: Path) -> None:
    base, finetuned = _toy_states()
    base_dir = tmp_path / "base"
    finetuned_dir = tmp_path / "finetuned"
    _write_transformer(base_dir, base)
    _write_transformer(finetuned_dir, finetuned)

    outputs = []
    for index in range(2):
        output = tmp_path / f"adapter-{index}.safetensors"
        extract_lora.extract_lora_adapter(
            base=str(base_dir),
            ft=str(finetuned_dir),
            out=str(output),
            rank=2,
            min_delta=0.0,
            load_mode="indexed",
            device="cpu",
            svd_method="randomized",
            randomized_q=4,
            niter=2,
            seed=123,
            factor_dtype="float32",
            dense_dtype="float32",
            exact_tensor_patterns=(r"^context\.weight$", ),
        )
        outputs.append(load_file(output))

    assert outputs[0].keys() == outputs[1].keys()
    for key in outputs[0]:
        assert torch.equal(outputs[0][key], outputs[1][key]), key
    report = json.loads((tmp_path / "adapter-0.safetensors.report.json").read_text())
    assert 0.0 <= report["factorized_weighted_relative_residual"] <= 1.0
    layer = report["layers"]["blocks.0.linear.weight"]
    assert layer["method"] == "randomized-q4-niter2"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_randomized_factorization_runs_on_gpu() -> None:
    generator = torch.Generator().manual_seed(11)
    delta = torch.randn(32, 24, generator=generator, dtype=torch.float32).cuda()
    exact_a, exact_b, _, _ = extract_lora._factorize_delta(
        delta,
        rank=4,
        full_rank=False,
        method="exact",
        randomized_q=None,
        oversample=4,
        niter=2,
        seed=7,
    )
    random_a, random_b, _, method = extract_lora._factorize_delta(
        delta,
        rank=4,
        full_rank=False,
        method="randomized",
        randomized_q=12,
        oversample=4,
        niter=4,
        seed=7,
    )
    exact_error = torch.linalg.vector_norm(delta - exact_b @ exact_a)
    randomized_error = torch.linalg.vector_norm(delta - random_b @ random_a)
    assert randomized_error <= exact_error * 1.02
    assert method == "randomized-q12-niter4"
    assert random_a.is_cuda and random_b.is_cuda


def test_hub_resolution_downloads_only_transformer(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot"
    (snapshot / "transformer").mkdir(parents=True)
    calls: list[dict[str, object]] = []

    def fake_snapshot_download(**kwargs: object) -> str:
        calls.append(kwargs)
        return str(snapshot)

    monkeypatch.setattr(extract_lora, "snapshot_download", fake_snapshot_download)
    assert extract_lora._resolve_transformer_dir("org/model", "revision") == snapshot / "transformer"
    assert calls == [{
        "repo_id": "org/model",
        "revision": "revision",
        "allow_patterns": ["transformer/*"],
    }]
