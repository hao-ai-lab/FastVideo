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


def _legacy_cpu_adapter(
    base: dict[str, torch.Tensor],
    finetuned: dict[str, torch.Tensor],
    rank: int,
    min_delta: float,
) -> dict[str, torch.Tensor]:
    """Reproduce the pre-streaming extractor for CPU parity coverage."""
    adapter: dict[str, torch.Tensor] = {}
    low_rank_keys: set[str] = set()
    for key in sorted(finetuned):
        if not extract_lora.is_extractable_weight(key) or key not in base:
            continue
        base_weight = base[key].detach().cpu().float().contiguous()
        finetuned_weight = finetuned[key].detach().cpu().float().contiguous()
        if base_weight.shape != finetuned_weight.shape:
            continue
        delta = (finetuned_weight - base_weight).contiguous()
        if float(delta.abs().mean()) < min_delta:
            continue
        try:
            u, singular_values, vh = torch.linalg.svd(delta, full_matrices=False)
        except RuntimeError:
            continue
        chosen_rank = min(rank, singular_values.numel())
        sqrt_s = singular_values[:chosen_rank].float().sqrt()
        lora_b = (u[:, :chosen_rank].float() * sqrt_s.unsqueeze(0)).contiguous()
        lora_a = (vh[:chosen_rank].mT.float() * sqrt_s.unsqueeze(0)).mT.contiguous()
        module_name = key.removesuffix(".weight")
        adapter[f"{module_name}.lora_A.weight"] = lora_a
        adapter[f"{module_name}.lora_B.weight"] = lora_b
        low_rank_keys.add(key)

    adapter.update(extract_lora.build_dense_payload(base, finetuned, low_rank_keys, min_delta))
    return adapter


def _reconstructed_delta(adapter: dict[str, torch.Tensor], module_name: str) -> torch.Tensor:
    return adapter[f"{module_name}.lora_B.weight"].float() @ adapter[f"{module_name}.lora_A.weight"].float()


def test_exact_cpu_extraction_matches_legacy_algorithm(tmp_path: Path) -> None:
    base, finetuned = _toy_states()
    base_dir = tmp_path / "base"
    finetuned_dir = tmp_path / "finetuned"
    output = tmp_path / "adapter.safetensors"
    _write_transformer(base_dir, base)
    _write_transformer(finetuned_dir, finetuned)

    extract_lora.extract_lora_adapter(
        base=str(base_dir),
        ft=str(finetuned_dir),
        out=str(output),
        rank=2,
        min_delta=1e-8,
        load_mode="indexed",
        device="cpu",
        svd_method="exact",
        factor_dtype="float32",
        dense_dtype="source",
        replacement_dtype="source",
    )

    actual = load_file(output)
    expected = _legacy_cpu_adapter(base, finetuned, rank=2, min_delta=1e-8)
    assert actual.keys() == expected.keys()
    for key in actual:
        assert torch.equal(actual[key], expected[key]), key


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


def test_standalone_parameters_use_generic_dense_suffixes(tmp_path: Path) -> None:
    base = {"blocks.0.scale_shift_table": torch.ones(4)}
    finetuned = {
        "blocks.0.scale_shift_table": torch.full((4, ), 1.25),
        "blocks.0.extra_table": torch.arange(4, dtype=torch.float32),
    }
    base_dir = tmp_path / "base"
    finetuned_dir = tmp_path / "finetuned"
    output = tmp_path / "adapter.safetensors"
    _write_transformer(base_dir, base)
    _write_transformer(finetuned_dir, finetuned)

    extract_lora.extract_lora_adapter(
        base=str(base_dir),
        ft=str(finetuned_dir),
        out=str(output),
        rank=2,
        min_delta=0.0,
        load_mode="indexed",
        dense_dtype="float32",
        replacement_dtype="source",
    )

    adapter = load_file(output)
    torch.testing.assert_close(adapter["blocks.0.scale_shift_table.diff_param"], torch.full((4, ), 0.25))
    torch.testing.assert_close(adapter["blocks.0.extra_table.set_param"], finetuned["blocks.0.extra_table"])


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
def test_exact_cpu_and_gpu_extraction_agree(tmp_path: Path) -> None:
    base, finetuned = _toy_states()
    base_dir = tmp_path / "base"
    finetuned_dir = tmp_path / "finetuned"
    _write_transformer(base_dir, base)
    _write_transformer(finetuned_dir, finetuned)

    adapters = {}
    reports = {}
    for label, device in (("cpu", "cpu"), ("gpu", "cuda:0")):
        output = tmp_path / f"adapter-{label}.safetensors"
        extract_lora.extract_lora_adapter(
            base=str(base_dir),
            ft=str(finetuned_dir),
            out=str(output),
            rank=2,
            min_delta=1e-8,
            load_mode="indexed",
            device=device,
            svd_method="exact",
            factor_dtype="float32",
            dense_dtype="float32",
            replacement_dtype="source",
        )
        adapters[label] = load_file(output)
        reports[label] = json.loads(output.with_suffix(".safetensors.report.json").read_text())

    assert adapters["cpu"].keys() == adapters["gpu"].keys()
    for key in adapters["cpu"]:
        if ".lora_" not in key:
            torch.testing.assert_close(adapters["cpu"][key], adapters["gpu"][key], atol=0, rtol=0)

    for module_name in ("blocks.0.linear", "context"):
        cpu_delta = _reconstructed_delta(adapters["cpu"], module_name)
        gpu_delta = _reconstructed_delta(adapters["gpu"], module_name)
        torch.testing.assert_close(cpu_delta, gpu_delta, atol=2e-5, rtol=2e-4)

    cpu_residual = reports["cpu"]["factorized_weighted_relative_residual"]
    gpu_residual = reports["gpu"]["factorized_weighted_relative_residual"]
    # The aggregate residual is close to zero and therefore sensitive to
    # backend-level singular-value rounding even when reconstructed deltas agree.
    assert gpu_residual == pytest.approx(cpu_residual, abs=2e-4)


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


def _toy_checkpoints(tmp_path: Path) -> tuple[Path, Path]:
    base, finetuned = _toy_states()
    base_dir = tmp_path / "base"
    finetuned_dir = tmp_path / "finetuned"
    _write_transformer(base_dir, base)
    _write_transformer(finetuned_dir, finetuned)
    return base_dir, finetuned_dir


def _extract(base_dir: Path, finetuned_dir: Path, output: Path, **overrides: object) -> Path:
    kwargs: dict[str, object] = {
        "rank": 2,
        "min_delta": 1e-8,
        "load_mode": "indexed",
        "device": "cpu",
        "svd_method": "exact",
    }
    kwargs.update(overrides)
    return extract_lora.extract_lora_adapter(base=str(base_dir), ft=str(finetuned_dir), out=str(output), **kwargs)


def test_work_dir_leaves_unrelated_files_alone(tmp_path: Path) -> None:
    """--work-dir names where to put scratch, so its other contents must survive."""
    base_dir, finetuned_dir = _toy_checkpoints(tmp_path)
    work_dir = tmp_path / "shared_scratch"
    (work_dir / "nested").mkdir(parents=True)
    keeper = work_dir / "unrelated.txt"
    keeper.write_text("do not delete me", encoding="utf-8")

    _extract(base_dir, finetuned_dir, tmp_path / "adapter.safetensors", work_dir=str(work_dir))

    assert keeper.read_text(encoding="utf-8") == "do not delete me"
    assert (work_dir / "nested").is_dir()
    assert not (work_dir / extract_lora.WORK_SUBDIR).exists()


def test_work_dir_scratch_is_confined_to_a_subdirectory(tmp_path: Path) -> None:
    base_dir, finetuned_dir = _toy_checkpoints(tmp_path)
    work_dir = tmp_path / "shared_scratch"

    _extract(base_dir,
             finetuned_dir,
             tmp_path / "adapter.safetensors",
             work_dir=str(work_dir),
             keep_work_dir=True)

    assert (work_dir / extract_lora.WORK_SUBDIR / "manifest.json").is_file()


def test_unmatched_exact_tensor_pattern_is_rejected(tmp_path: Path) -> None:
    """A pattern that matches nothing silently factorizes the tensors it was meant to keep."""
    base_dir, finetuned_dir = _toy_checkpoints(tmp_path)

    with pytest.raises(ValueError, match="matched no tensor"):
        _extract(base_dir,
                 finetuned_dir,
                 tmp_path / "adapter.safetensors",
                 exact_tensor_patterns=[r"^context\\.weight$"])

    _extract(base_dir,
             finetuned_dir,
             tmp_path / "adapter.safetensors",
             exact_tensor_patterns=[r"^context\.weight$"])
    assert "context.diff" in load_file(tmp_path / "adapter.safetensors")


def test_rerun_with_a_different_config_reuses_the_work_dir(tmp_path: Path) -> None:
    """A leftover work dir must not make a fresh (non-resume) rerun fail."""
    base_dir, finetuned_dir = _toy_checkpoints(tmp_path)
    output = tmp_path / "adapter.safetensors"

    _extract(base_dir, finetuned_dir, output, rank=2, keep_work_dir=True)
    work_dir = output.parent / f".{output.name}.work"
    assert (work_dir / "manifest.json").is_file()

    _extract(base_dir, finetuned_dir, output, rank=4)
    assert output.is_file()


def test_resume_still_rejects_a_mismatched_config(tmp_path: Path) -> None:
    base_dir, finetuned_dir = _toy_checkpoints(tmp_path)
    output = tmp_path / "adapter.safetensors"

    _extract(base_dir, finetuned_dir, output, rank=2, keep_work_dir=True)
    with pytest.raises(ValueError, match="Resume configuration does not match"):
        _extract(base_dir, finetuned_dir, output, rank=4, resume=True)


def test_non_safetensors_output_is_rejected(tmp_path: Path) -> None:
    """The writer is always safetensors, so a .pt name would be a mislabeled file."""
    base_dir, finetuned_dir = _toy_checkpoints(tmp_path)

    with pytest.raises(ValueError, match="must end in .safetensors"):
        _extract(base_dir, finetuned_dir, tmp_path / "adapter.pt")
