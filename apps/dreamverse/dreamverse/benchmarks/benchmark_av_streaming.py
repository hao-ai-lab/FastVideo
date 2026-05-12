"""Benchmark the AV streaming hot-path used by dreamverse-server.

Measures wall-time, encoded byte volume, and chunk count for
``av_streaming.stream_fmp4`` over synthetic frames + audio at
production resolution. Sweeps codecs (default: ``libx264`` and
``h264_nvenc`` if the active ffmpeg supports it) and ffmpeg presets so
the deploy can pick a configuration that achieves a >=1.0 realtime
ratio (5.04s of generated video produced in <=5.04s wall-time).

Usage::

    python -m apps.dreamverse.server.benchmarks.benchmark_av_streaming
    python -m apps.dreamverse.server.benchmarks.benchmark_av_streaming \\
        --frames 121 --width 1920 --height 1088 --runs 3 \\
        --codecs libx264 h264_nvenc --x264-preset ultrafast \\
        --nvenc-preset p1
    FASTVIDEO_FFMPEG_BIN=$HOME/opt/ffmpeg-native/bin/ffmpeg \\
        python -m apps.dreamverse.server.benchmarks.benchmark_av_streaming

Skips ``h264_nvenc`` automatically if the binary lacks the encoder.
This is the regression guard documented in
`.agents/memory/dreamverse-integration/decisions-log.md` D-21.
"""
from __future__ import annotations

import argparse
import os
import shutil
import statistics
import subprocess
import sys
import time
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_SERVER_ROOT = os.path.dirname(_HERE)
if _SERVER_ROOT not in sys.path:
    sys.path.insert(0, _SERVER_ROOT)

from dreamverse.av_streaming import stream_fmp4  # noqa: E402


@dataclass
class BenchResult:
    codec: str
    preset: str
    runs: int
    wall_ms_min: float
    wall_ms_median: float
    wall_ms_p95: float
    wall_ms_max: float
    bytes_median: int
    chunks_median: float
    realtime_ratio_median: float
    error: str | None = None


def _make_synthetic_frames(num: int, width: int, height: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 255, size=(height, width, 3), dtype=np.uint8)
    out: list[np.ndarray] = []
    for i in range(num):
        f = base.copy()
        f[:, :, 0] = (f[:, :, 0].astype(np.int32) + i * 2) % 256
        out.append(f)
    return out


def _make_synthetic_audio(num_frames: int, fps: int, sample_rate: int, seed: int) -> torch.Tensor:
    duration_s = num_frames / fps
    samples = int(round(duration_s * sample_rate))
    rng = np.random.default_rng(seed + 1)
    audio = (rng.uniform(-0.1, 0.1, (2, samples))).astype(np.float32)
    return torch.from_numpy(audio)


def _ffmpeg_supports(codec: str, ffmpeg_bin: str) -> bool:
    try:
        out = subprocess.run(
            [ffmpeg_bin, "-hide_banner", "-encoders"],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        ).stdout
    except Exception:
        return False
    needle = f" {codec} "
    return any(needle in line for line in out.splitlines())


def _run_one(frames: list[np.ndarray], audio: torch.Tensor, sample_rate: int, codec: str,
             preset: str) -> tuple[float, int, int, str | None]:
    timings: dict = {}
    chunks: list = []

    def _publish(event):
        chunks.append(event)

    prev_codec = os.environ.get("FASTVIDEO_VIDEO_CODEC")
    prev_preset = os.environ.get("FASTVIDEO_X264_PRESET")
    prev_nvenc_preset = os.environ.get("FASTVIDEO_NVENC_PRESET")
    os.environ["FASTVIDEO_VIDEO_CODEC"] = codec
    if codec.endswith("_nvenc"):
        os.environ["FASTVIDEO_NVENC_PRESET"] = preset
    else:
        os.environ["FASTVIDEO_X264_PRESET"] = preset

    t0 = time.perf_counter()
    try:
        ok, err = stream_fmp4(
            frames=frames,
            audio=audio,
            audio_sample_rate=sample_rate,
            stream_id="bench",
            timings=timings,
            head_trim_frames=0,
            head_trim_audio_frames=0,
            shared_buffer=None,
            shared_buffer_bytes=0,
            publish=_publish,
            log_prefix="[bench]",
        )
    finally:
        if prev_codec is None:
            os.environ.pop("FASTVIDEO_VIDEO_CODEC", None)
        else:
            os.environ["FASTVIDEO_VIDEO_CODEC"] = prev_codec
        if prev_preset is None:
            os.environ.pop("FASTVIDEO_X264_PRESET", None)
        else:
            os.environ["FASTVIDEO_X264_PRESET"] = prev_preset
        if prev_nvenc_preset is None:
            os.environ.pop("FASTVIDEO_NVENC_PRESET", None)
        else:
            os.environ["FASTVIDEO_NVENC_PRESET"] = prev_nvenc_preset
    wall_ms = (time.perf_counter() - t0) * 1000.0
    if not ok:
        return wall_ms, 0, 0, err or "stream_fmp4 returned False"
    total_bytes = int(timings.get("av_stream_bytes", 0))
    return wall_ms, total_bytes, len(chunks), None


def benchmark(codecs: Iterable[str], runs: int, frames_n: int, width: int, height: int, fps: int, sample_rate: int,
              x264_preset: str, nvenc_preset: str, seed: int, ffmpeg_bin: str) -> list[BenchResult]:
    print(f"[bench] ffmpeg_bin={ffmpeg_bin}")
    print(f"[bench] frames={frames_n} {width}x{height} fps={fps} "
          f"audio_sr={sample_rate} runs/codec={runs}")
    frames = _make_synthetic_frames(frames_n, width, height, seed)
    audio = _make_synthetic_audio(frames_n, fps, sample_rate, seed)
    playable_s = frames_n / fps
    print(f"[bench] playable={playable_s:.3f}s "
          f"(realtime_ratio = playable / wall_time; >= 1.0 means no "
          f"buffer drain)")

    results: list[BenchResult] = []
    for codec in codecs:
        preset = nvenc_preset if codec.endswith("_nvenc") else x264_preset
        if not _ffmpeg_supports(codec, ffmpeg_bin):
            results.append(
                BenchResult(codec=codec,
                            preset=preset,
                            runs=0,
                            wall_ms_min=0,
                            wall_ms_median=0,
                            wall_ms_p95=0,
                            wall_ms_max=0,
                            bytes_median=0,
                            chunks_median=0,
                            realtime_ratio_median=0,
                            error=f"{codec} not in ffmpeg"))
            continue
        walls: list[float] = []
        sizes: list[int] = []
        chunkcounts: list[int] = []
        last_err: str | None = None
        for run in range(runs):
            wall_ms, total_bytes, chunk_count, err = _run_one(frames, audio, sample_rate, codec, preset)
            print(f"[bench] codec={codec:12s} preset={preset:9s} "
                  f"run={run + 1}/{runs}  wall={wall_ms:7.1f}ms  "
                  f"bytes={total_bytes:>9d}  chunks={chunk_count:>3d}  "
                  f"realtime={playable_s / (wall_ms / 1000.0):5.2f}x"
                  f"{'  ERR=' + err if err else ''}")
            if err is not None:
                last_err = err
                continue
            walls.append(wall_ms)
            sizes.append(total_bytes)
            chunkcounts.append(chunk_count)
        if not walls:
            results.append(
                BenchResult(codec=codec,
                            preset=preset,
                            runs=0,
                            wall_ms_min=0,
                            wall_ms_median=0,
                            wall_ms_p95=0,
                            wall_ms_max=0,
                            bytes_median=0,
                            chunks_median=0,
                            realtime_ratio_median=0,
                            error=last_err or "all runs failed"))
            continue
        walls_sorted = sorted(walls)
        p95_idx = max(0, int(round(0.95 * (len(walls_sorted) - 1))))
        wall_med = statistics.median(walls)
        results.append(
            BenchResult(
                codec=codec,
                preset=preset,
                runs=len(walls),
                wall_ms_min=min(walls),
                wall_ms_median=wall_med,
                wall_ms_p95=walls_sorted[p95_idx],
                wall_ms_max=max(walls),
                bytes_median=int(statistics.median(sizes)),
                chunks_median=statistics.median(chunkcounts),
                realtime_ratio_median=playable_s / (wall_med / 1000.0),
            ))
    return results


def _print_summary(results: list[BenchResult]) -> None:
    print()
    print("=== summary ===")
    header = (f"{'codec':14s} {'preset':10s} {'runs':>4s} "
              f"{'wall_med_ms':>11s} {'wall_p95_ms':>11s} "
              f"{'bytes_med':>10s} {'realtime':>8s}  notes")
    print(header)
    print("-" * len(header))
    for r in results:
        if r.error is not None:
            print(f"{r.codec:14s} {r.preset:10s} {r.runs:>4d} "
                  f"{'-':>11s} {'-':>11s} {'-':>10s} {'-':>8s}  "
                  f"ERR: {r.error}")
            continue
        print(f"{r.codec:14s} {r.preset:10s} {r.runs:>4d} "
              f"{r.wall_ms_median:>11.1f} {r.wall_ms_p95:>11.1f} "
              f"{r.bytes_median:>10d} {r.realtime_ratio_median:>7.2f}x  "
              f"{'OK' if r.realtime_ratio_median >= 1.0 else 'BUFFER DRAINS'}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--frames", type=int, default=121, help="num frames (default: 121, matches NUM_FRAMES)")
    p.add_argument("--width", type=int, default=1920)
    p.add_argument("--height", type=int, default=1088)
    p.add_argument("--fps", type=int, default=24)
    p.add_argument("--sample-rate", type=int, default=24000)
    p.add_argument("--runs", type=int, default=3, help="runs per codec (default: 3 — 1 warmup + 2 timed in median)")
    p.add_argument("--codecs",
                   nargs="+",
                   default=["libx264", "h264_nvenc"],
                   help="codecs to benchmark; missing ones are skipped")
    p.add_argument("--x264-preset", default="ultrafast")
    p.add_argument("--nvenc-preset", default="p1")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    ffmpeg_bin = shutil.which(os.getenv("FASTVIDEO_FFMPEG_BIN", "ffmpeg"))
    if ffmpeg_bin is None:
        print("ffmpeg not found", file=sys.stderr)
        return 2

    results = benchmark(
        codecs=args.codecs,
        runs=args.runs,
        frames_n=args.frames,
        width=args.width,
        height=args.height,
        fps=args.fps,
        sample_rate=args.sample_rate,
        x264_preset=args.x264_preset,
        nvenc_preset=args.nvenc_preset,
        seed=args.seed,
        ffmpeg_bin=ffmpeg_bin,
    )
    _print_summary(results)
    any_below = any(r.error is None and r.realtime_ratio_median < 1.0 for r in results)
    return 1 if any_below else 0


if __name__ == "__main__":
    raise SystemExit(main())
