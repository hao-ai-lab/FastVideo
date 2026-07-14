# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

import pytest

from fastvideo.train.entrypoint.dcp_to_diffusers import (
    _export_consolidated_ema, )


def _make_base_model(path: Path) -> None:
    transformer = path / "transformer"
    transformer.mkdir(parents=True)
    (path / "model_index.json").write_text("{}", encoding="utf-8")
    (transformer / "config.json").write_text("{}", encoding="utf-8")
    (transformer / "old.safetensors").write_bytes(b"old")
    (transformer / "model.safetensors.index.json").write_text(
        "{}",
        encoding="utf-8",
    )


def test_export_consolidated_ema_replaces_transformer_weights(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = tmp_path / "base"
    _make_base_model(base)
    ema_path = tmp_path / "student.safetensors"
    ema_path.write_bytes(b"exact-ema")
    monkeypatch.setattr(
        "fastvideo.utils.maybe_download_model",
        lambda _: str(base),
    )

    output = tmp_path / "output"
    result = _export_consolidated_ema(
        ema_path=str(ema_path),
        base_model_path="unused",
        output_dir=str(output),
    )

    assert result == str(output.resolve())
    assert (output / "model_index.json").is_file()
    assert (output / "transformer" / "config.json").is_file()
    assert (output / "transformer" / "model.safetensors").read_bytes() == b"exact-ema"
    assert not (output / "transformer" / "old.safetensors").exists()
    assert not (output / "transformer" / "model.safetensors.index.json").exists()


def test_export_consolidated_ema_rejects_legacy_checkpoint(tmp_path: Path, ) -> None:
    with pytest.raises(FileNotFoundError, match="Legacy checkpoints"):
        _export_consolidated_ema(
            ema_path=str(tmp_path / "missing.safetensors"),
            base_model_path="unused",
            output_dir=str(tmp_path / "output"),
        )
