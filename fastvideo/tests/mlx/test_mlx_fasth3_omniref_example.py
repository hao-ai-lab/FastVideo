# SPDX-License-Identifier: Apache-2.0
"""The MLX OmniRef example's composition and argument contracts.

The example builds the model directory the runtime reads, so a wrong link or a
mis-assembled contract fails long before a weight loads. These are the pure
parts: which components come from the export, which from the base snapshot, and
what the parser rejects.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

try:
    from examples.inference.basic import mlx_fasth3_omniref as example
except ImportError:  # pragma: no cover - an editable install of another checkout can shadow examples/
    import importlib.util

    _path = Path(__file__).resolve().parents[3] / "examples" / "inference" / "basic" / "mlx_fasth3_omniref.py"
    _spec = importlib.util.spec_from_file_location("mlx_fasth3_omniref", _path)
    assert _spec is not None and _spec.loader is not None
    example = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(example)

CONTRACT = {
    "model_type": "ref2va",
    "pdd_steps": 120,
    "num_inference_steps": 8,
    "transformer_forwards": 8,
    "attention_backend": "VIDEO_SPARSE_ATTN_H3",
    "vsa_sparsity": 0.9,
    "vsa_tile_size": 128,
    "video_scheduler_shift": 12.0,
    "audio_scheduler_shift": 3.0,
    "dmd_denoising_steps": [0, 30, 60, 90, 120],
}


def _export(tmp_path: Path) -> Path:
    export = tmp_path / "export"
    (export / "transformer_ref").mkdir(parents=True)
    (export / "transformer_ref" / "config.json").write_text("{}")
    (export / "scheduler").mkdir()
    (export / "audio_scheduler").mkdir()
    (export / example.CONTRACT).write_text(json.dumps(CONTRACT))
    return export


def _base(tmp_path: Path) -> Path:
    base = tmp_path / "base"
    for name in example.BASE_COMPONENTS:
        (base / name).mkdir(parents=True)
    return base


def test_compose_links_export_and_base_components(tmp_path: Path) -> None:
    composed = example.compose_model_dir(_export(tmp_path), _base(tmp_path), tmp_path / "composed")

    for name in (*example.EXPORT_COMPONENTS, example.CONTRACT, *example.BASE_COMPONENTS):
        assert (composed / name).is_symlink(), name
    assert (composed / "transformer_ref").resolve() == (tmp_path / "export" / "transformer_ref").resolve()


def test_compose_is_idempotent(tmp_path: Path) -> None:
    export, base = _export(tmp_path), _base(tmp_path)
    composed = example.compose_model_dir(export, base, tmp_path / "composed")
    example.compose_model_dir(export, base, composed)
    assert (composed / "vae").is_symlink()


def test_compose_rejects_a_directory_that_is_not_an_export(tmp_path: Path) -> None:
    base = _base(tmp_path)
    with pytest.raises(FileNotFoundError, match="fastvideo_inference.json"):
        example.compose_model_dir(base, base, tmp_path / "composed")


def test_load_contract_rejects_a_non_ref2va_export(tmp_path: Path) -> None:
    export = _export(tmp_path)
    (export / example.CONTRACT).write_text(json.dumps({"model_type": "t2va"}))
    with pytest.raises(ValueError, match="not a Ref2VA contract"):
        example.load_contract(export)


def test_parser_requires_a_visual_reference(capsys: pytest.CaptureFixture[str]) -> None:
    args = example.parse_args([
        "--model-root", "/m",
        "--mlx-checkpoint", "/c",
        "--prompt", "p",
        "--image", "a.png",
    ])
    assert args.references == [("image", "a.png")]

    with pytest.raises(SystemExit):
        example.parse_args([
            "--model-root", "/m",
            "--mlx-checkpoint", "/c",
            "--prompt", "p",
            "--audio", "a.wav",
        ])
    assert "at least one image or video" in capsys.readouterr().err


def test_parser_keeps_command_line_reference_order() -> None:
    args = example.parse_args([
        "--video", "dance.mp4",
        "--image", "outfit.png",
        "--audio", "voice.wav",
        "--model-root", "/m",
        "--mlx-checkpoint", "/c",
        "--prompt", "p",
    ])
    assert [kind for kind, _ in args.references] == ["video", "image", "audio"]


def test_parser_needs_a_model_root_or_the_pieces_to_build_one(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        example.parse_args([
            "--mlx-checkpoint", "/c",
            "--prompt", "p",
            "--image", "a.png",
        ])
    assert "give --model-root" in capsys.readouterr().err
    args = example.parse_args([
        "--export", "/e",
        "--base-model-path", "/b",
        "--mlx-checkpoint", "/c",
        "--prompt", "p",
        "--image", "a.png",
    ])
    assert args.composed_dir == Path("outputs/fasth3_omniref_mlx_model")


def test_compose_only_needs_no_checkpoint_prompt_or_reference(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    export, base = _export(tmp_path), _base(tmp_path)
    composed = tmp_path / "composed"
    args = example.parse_args(["--compose-only", "--export", str(export), "--base-model-path", str(base),
                               "--composed-dir", str(composed)])
    example.run(args)
    assert (composed / "transformer_ref").is_symlink()
    assert str(composed) in capsys.readouterr().out


def test_compose_only_still_needs_the_export_and_base(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        example.parse_args(["--compose-only", "--export", "/e"])
    assert "--compose-only needs" in capsys.readouterr().err


def test_generation_still_requires_a_checkpoint_and_prompt(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        example.parse_args(["--model-root", "/m", "--image", "a.png"])
    assert "--mlx-checkpoint and --prompt are required" in capsys.readouterr().err


def test_reference_max_pixels_flag_resolves() -> None:
    resolve_reference_max_pixels = example.resolve_reference_max_pixels
    assert resolve_reference_max_pixels(None, 832, 480) is None
    assert resolve_reference_max_pixels("match", 832, 480) == 832 * 480
    assert resolve_reference_max_pixels("1000000", 832, 480) == 1_000_000
    with pytest.raises(ValueError, match="positive"):
        resolve_reference_max_pixels("0", 832, 480)


def test_quant_gemm_flag_defaults_to_auto_and_rejects_unknown(capsys: pytest.CaptureFixture[str]) -> None:
    base = ["--model-root", "/m", "--mlx-checkpoint", "/c", "--prompt", "p", "--image", "a.png"]
    assert example.parse_args(base).quant_gemm == "auto"
    for mode in ("auto", "dequant", "qmm"):
        assert example.parse_args([*base, "--quant-gemm", mode]).quant_gemm == mode
    with pytest.raises(SystemExit):
        example.parse_args([*base, "--quant-gemm", "bogus"])
