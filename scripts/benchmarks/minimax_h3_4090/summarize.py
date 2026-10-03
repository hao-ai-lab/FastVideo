# SPDX-License-Identifier: Apache-2.0
"""Join benchmark results with per-request stage logs, excluding warmup runs."""

import argparse
import json
import re
import statistics
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("log", type=Path)
    args = parser.parse_args()
    results = json.loads(args.results.read_text())
    completed = []
    stages = {}
    peaks = []
    for line in args.log.read_text().splitlines():
        timing = re.search(r"\[(\w+)_stage\|[^\]]+\] Execution completed in ([\d.]+) ms", line)
        if timing:
            stages[timing[1]] = float(timing[2]) / 1000
        peak = re.search(r"Memory peak_allocated=([\d.]+) GiB", line)
        if peak:
            peaks.append(float(peak[1]))
        if line.startswith("RUN "):
            run = json.loads(line[4:])
            run["stage_s"] = stages
            run["peak_gpu_allocated_gib"] = max(peaks) if peaks else None
            completed.append(run)
            stages = {}
            peaks = []
    if len(completed) != len(results["runs"]):
        raise ValueError("Log and results.json have different completed run counts")
    timed = [run for run in completed if not run["warmup"]]
    if len(timed) < 2:
        raise ValueError("At least two timed runs are required for a baseline summary")
    for run in timed:
        if not {"denoising", "video_decoding", "audio_decoding"}.issubset(run["stage_s"]):
            raise ValueError("Missing stage timings for a completed run")
    summary = {
        "name": results["name"],
        "sampling": results["sampling"],
        "source_commit": results["source_commit"],
        "timed_runs": len(timed),
        "median_e2e_s": statistics.median(run["wall_s"] for run in timed),
        "median_stage_s": {name: statistics.median(run["stage_s"][name] for run in timed)
                           for name in ("conditioning", "denoising", "video_decoding", "audio_decoding")},
        "peak_gpu_allocated_gib": max(run["peak_gpu_allocated_gib"] for run in timed),
        "peak_host_anon_gib": max(run["peak_host_anon_gib"] for run in timed),
        "peak_host_cgroup_gib": max(run["peak_host_cgroup_gib"] for run in timed),
        "notes": "Stage times include deferred component loading. Host peaks are pod-wide samples every 100 ms.",
        "runs": timed,
    }
    args.results.with_name(f"{args.results.stem}-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
