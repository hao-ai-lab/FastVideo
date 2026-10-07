import json
import subprocess
from pathlib import Path

from fastvideo.eval.v2a import run_av_benchmark
from fastvideo.eval.v2a import _av_benchmark_runner as runner


def _make_gt_cache(path: Path) -> None:
    path.mkdir()
    for name in runner._BASE_CACHE_FILES:
        (path / name).touch()


def test_run_av_benchmark_uses_selected_python(monkeypatch, tmp_path):
    audio_dir = tmp_path / "audio"
    gt_cache = tmp_path / "gt"
    audio_dir.mkdir()
    gt_cache.mkdir()
    selected_python = tmp_path / "av-benchmark-venv" / "python"
    captured = {}

    def fake_subprocess_run(command, check):
        captured["command"] = command
        captured["check"] = check
        output = Path(command[command.index("--output") + 1])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("{}", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", fake_subprocess_run)

    output = run_av_benchmark(
        audio_dir=audio_dir,
        gt_cache=gt_cache,
        python_executable=selected_python,
        batch_size=8,
        num_workers=3,
        skip_clap=True,
    )

    command = captured["command"]
    assert command[0] == str(selected_python)
    assert command[1].endswith("_av_benchmark_runner.py")
    assert command[command.index("--batch-size") + 1] == "8"
    assert command[command.index("--num-workers") + 1] == "3"
    assert "--skip-clap" in command
    assert captured["check"] is True
    assert output == audio_dir.parent / "av_benchmark_results.json"


def test_official_runner_extracts_and_serializes_metrics(monkeypatch, tmp_path):
    audio_dir = tmp_path / "audio"
    gt_cache = tmp_path / "gt"
    prediction_cache = tmp_path / "pred-cache"
    output = tmp_path / "results.json"
    audio_dir.mkdir()
    (audio_dir / "sample.wav").touch()
    _make_gt_cache(gt_cache)
    calls = {}

    def fake_extract(**kwargs):
        calls["extract"] = kwargs
        prediction_cache.mkdir(parents=True, exist_ok=True)
        for name in runner._required_prediction_files(skip_video_related=False, skip_clap=True):
            (prediction_cache / name).touch()

    def fake_evaluate(**kwargs):
        calls["evaluate"] = kwargs
        return {
            "FD-PASST": 12.5,
            "KL-PASST-softmax": 0.25,
            "DeSync": 0.4,
        }

    monkeypatch.setattr(runner, "_load_backend", lambda: (fake_extract, fake_evaluate))

    payload = runner.run_benchmark(
        audio_dir=audio_dir,
        gt_cache=gt_cache,
        prediction_cache=prediction_cache,
        output=output,
        device="cuda",
        batch_size=4,
        num_workers=2,
        audio_length=8.0,
        recompute=False,
        skip_video_related=False,
        skip_clap=True,
    )

    assert calls["extract"]["audio_path"] == audio_dir.resolve()
    assert calls["extract"]["batch_size"] == 4
    assert calls["extract"]["num_workers"] == 2
    assert calls["evaluate"]["gt_audio_cache"] == gt_cache.resolve()
    assert payload["num_generated_audio"] == 1
    assert payload["extraction_ran"] is True
    assert payload["metrics"]["FD-PASST"] == 12.5
    assert json.loads(output.read_text(encoding="utf-8"))["metrics"]["DeSync"] == 0.4


def test_official_runner_reuses_complete_prediction_cache(monkeypatch, tmp_path):
    audio_dir = tmp_path / "audio"
    gt_cache = tmp_path / "gt"
    prediction_cache = tmp_path / "pred-cache"
    output = tmp_path / "results.json"
    audio_dir.mkdir()
    prediction_cache.mkdir()
    (audio_dir / "sample.flac").touch()
    _make_gt_cache(gt_cache)
    for name in runner._required_prediction_files(skip_video_related=True, skip_clap=True):
        (prediction_cache / name).touch()

    def fail_extract(**_kwargs):
        raise AssertionError("complete prediction cache should be reused")

    monkeypatch.setattr(runner, "_load_backend", lambda: (fail_extract, lambda **_kwargs: {"FD-PASST": 1.0}))

    payload = runner.run_benchmark(
        audio_dir=audio_dir,
        gt_cache=gt_cache,
        prediction_cache=prediction_cache,
        output=output,
        device="cuda",
        batch_size=4,
        num_workers=0,
        audio_length=8.0,
        recompute=False,
        skip_video_related=True,
        skip_clap=True,
    )

    assert payload["extraction_ran"] is False
    assert payload["metrics"] == {"FD-PASST": 1.0}


def test_prediction_keys_align_only_unique_leading_underscore_matches(tmp_path):
    import torch

    gt_cache = tmp_path / "gt"
    prediction_cache = tmp_path / "pred"
    gt_cache.mkdir()
    prediction_cache.mkdir()
    torch.save({"_123": torch.tensor(1), "normal": torch.tensor(2)}, gt_cache / "passt_logits.pth")
    torch.save(
        {"123": torch.tensor(3), "normal": torch.tensor(4), "unknown": torch.tensor(5)},
        prediction_cache / "passt_logits.pth",
    )

    stats = runner._align_prediction_cache_keys(
        gt_cache=gt_cache,
        prediction_cache=prediction_cache,
        required_files=("passt_logits.pth",),
    )

    aligned = torch.load(prediction_cache / "passt_logits.pth", weights_only=True)
    assert set(aligned) == {"_123", "normal", "unknown"}
    assert stats == {
        "exact_before_alignment": 1,
        "remapped": 1,
        "ambiguous": 0,
        "unmatched_after_alignment": 1,
    }
