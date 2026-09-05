# SPDX-License-Identifier: Apache-2.0
"""Transactional output contracts for MiniMax H3 Ref2VA preprocessing."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from fastvideo.pipelines.preprocess import preprocess_minimax_h3_ref2va as preprocessing


def test_nonempty_output_requires_explicit_replacement(tmp_path: Path) -> None:
    output_dir = tmp_path / "dataset"
    output_dir.mkdir()
    (output_dir / "data_00000.parquet").write_bytes(b"old")

    with pytest.raises(FileExistsError, match="--replace-existing"):
        preprocessing._validate_output_destination(output_dir, replace_existing=False)


def test_sample_failure_preserves_previous_dataset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("fixture", encoding="utf-8")
    model_path = tmp_path / "model"
    model_path.mkdir()
    output_dir = tmp_path / "dataset"
    output_dir.mkdir()
    old_shard = output_dir / "data_00000.parquet"
    old_shard.write_bytes(b"validated-old-dataset")
    (output_dir / "keep.txt").write_text("unrelated", encoding="utf-8")
    samples = [SimpleNamespace(sample_id="first"), SimpleNamespace(sample_id="second")]

    monkeypatch.setattr(preprocessing, "load_minimax_h3_ref2va_raw_samples", lambda _path: samples)
    monkeypatch.setattr(preprocessing, "_init_single_process_distributed", lambda: None)
    monkeypatch.setattr(
        preprocessing,
        "verify_model_config_and_directory",
        lambda _path: {
            name: ["diffusers", "Fixture"]
            for name in ("vae", "audio_vae", "tokenizer", "processor", "text_encoder", "transformer_ref")
        },
    )
    monkeypatch.setattr(preprocessing, "_build_fastvideo_args", lambda _path: (object(), (1, 2, 2)))

    def _build_record(sample, **_kwargs):
        if sample.sample_id == "second":
            raise RuntimeError("synthetic sample failure")
        return {"id": sample.sample_id}

    def _write_record(_record, staging_dir, index):
        path = staging_dir / f"data_{index:05d}.parquet"
        path.write_bytes(b"partial-new-dataset")
        return path

    monkeypatch.setattr(preprocessing, "_build_record", _build_record)
    monkeypatch.setattr(preprocessing, "_write_record", _write_record)

    with pytest.raises(RuntimeError, match="synthetic sample failure"):
        preprocessing.preprocess(
            manifest_path=manifest,
            model_path=model_path,
            output_dir=output_dir,
            replace_existing=True,
        )

    assert old_shard.read_bytes() == b"validated-old-dataset"
    assert (output_dir / "keep.txt").read_text(encoding="utf-8") == "unrelated"
    assert not list(tmp_path.glob(".dataset.staging-*"))
    assert not list(tmp_path.glob(".dataset.backup-*"))


def test_validated_staging_swap_retains_previous_dataset(tmp_path: Path) -> None:
    output_dir = tmp_path / "dataset"
    output_dir.mkdir()
    (output_dir / "data_00000.parquet").write_bytes(b"old")
    staging_dir = preprocessing._new_staging_directory(output_dir)
    (staging_dir / "data_00000.parquet").write_bytes(b"new")

    backup_dir = preprocessing._promote_staged_dataset(
        staging_dir,
        output_dir,
        replace_existing=True,
    )

    assert backup_dir is not None
    assert (output_dir / "data_00000.parquet").read_bytes() == b"new"
    assert (backup_dir / "data_00000.parquet").read_bytes() == b"old"


def test_failed_promotion_restores_previous_dataset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir = tmp_path / "dataset"
    output_dir.mkdir()
    (output_dir / "data_00000.parquet").write_bytes(b"old")
    staging_dir = preprocessing._new_staging_directory(output_dir)
    (staging_dir / "data_00000.parquet").write_bytes(b"new")
    real_replace = preprocessing.os.replace

    def _fail_staging_promotion(source, destination):
        if Path(source) == staging_dir and Path(destination) == output_dir:
            raise OSError("synthetic promotion failure")
        return real_replace(source, destination)

    monkeypatch.setattr(preprocessing.os, "replace", _fail_staging_promotion)

    with pytest.raises(OSError, match="synthetic promotion failure"):
        preprocessing._promote_staged_dataset(
            staging_dir,
            output_dir,
            replace_existing=True,
        )

    assert (output_dir / "data_00000.parquet").read_bytes() == b"old"
    assert (staging_dir / "data_00000.parquet").read_bytes() == b"new"
    assert not list(tmp_path.glob(".dataset.backup-*"))
