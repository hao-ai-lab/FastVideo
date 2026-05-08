"""Multiprocess realtime stress test for LTX2 streaming websocket service."""
# pyright: reportArgumentType=false, reportOptionalMemberAccess=false

from __future__ import annotations

import argparse
import asyncio
from collections import defaultdict
from datetime import datetime, timezone
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time
import traceback
from typing import Any
import uuid

import pytest

pytestmark = pytest.mark.gpu

try:
    import websockets
except ModuleNotFoundError:
    websockets = None  # type: ignore[assignment]


DEFAULT_PRESET_FILE = (
    Path(__file__).resolve().parents[2]
    / "web"
    / "prompts"
    / "selected_ltx2_continuation_story_presets.json"
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def iso_from_epoch(epoch_s: float) -> str:
    return datetime.fromtimestamp(epoch_s, tz=timezone.utc).isoformat()


def iso_to_epoch(iso_ts: str | None) -> float | None:
    if not iso_ts:
        return None
    try:
        return datetime.fromisoformat(iso_ts).timestamp()
    except ValueError:
        return None


def safe_percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (len(sorted_values) - 1) * (percentile / 100.0)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return sorted_values[lower]
    fraction = rank - lower
    return (
        sorted_values[lower]
        + (sorted_values[upper] - sorted_values[lower]) * fraction
    )


def summarize_series(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "min": None,
            "p50": None,
            "p95": None,
            "p99": None,
            "max": None,
            "avg": None,
        }
    return {
        "count": len(values),
        "min": min(values),
        "p50": safe_percentile(values, 50),
        "p95": safe_percentile(values, 95),
        "p99": safe_percentile(values, 99),
        "max": max(values),
        "avg": sum(values) / len(values),
    }


def parse_int(value: object) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed


def format_num(value: float | int | None, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, int):
        return str(value)
    return f"{value:.{digits}f}"


def load_curated_prompts(
    preset_file: Path,
    preset_id: str | None,
    curated_limit: int,
) -> tuple[str, list[str], int]:
    if not preset_file.is_file():
        raise ValueError(f"Preset file not found: {preset_file}")
    try:
        payload = json.loads(preset_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in preset file: {preset_file}") from exc

    if not isinstance(payload, list) or len(payload) == 0:
        raise ValueError("Preset file must contain a non-empty JSON array.")

    selected: dict[str, Any] | None = None
    if preset_id:
        for item in payload:
            if not isinstance(item, dict):
                continue
            if str(item.get("id", "")).strip() == preset_id:
                selected = item
                break
        if selected is None:
            raise ValueError(f"Preset id not found: {preset_id}")
    else:
        for item in payload:
            if isinstance(item, dict):
                selected = item
                break
        if selected is None:
            raise ValueError("No valid preset object found in preset file.")

    selected_id = str(selected.get("id", "")).strip() or "unknown_preset"
    raw_prompts = selected.get("segment_prompts", [])
    if not isinstance(raw_prompts, list):
        raise ValueError(
            f"Preset {selected_id} has invalid segment_prompts (must be list)."
        )

    prompts = [
        str(prompt).strip()
        for prompt in raw_prompts
        if isinstance(prompt, str) and str(prompt).strip()
    ]
    if not prompts:
        raise ValueError(f"Preset {selected_id} has no non-empty prompts.")

    limited = prompts[:curated_limit]
    if not limited:
        raise ValueError(
            f"curated_limit={curated_limit} produced no prompts for preset "
            f"{selected_id}."
        )
    return selected_id, limited, len(prompts)


async def run_single_session(
    *,
    worker_id: int,
    worker_session_idx: int,
    config: dict[str, Any],
) -> dict[str, Any]:
    session_id = f"w{worker_id}_u{worker_session_idx}_{uuid.uuid4().hex[:8]}"
    process_id = os.getpid()
    url = str(config["url"])
    session_timeout_s = float(config["session_timeout_s"])
    post_complete_wait_s = float(config["post_complete_wait_s"])
    connect_timeout_s = float(config["connect_timeout_s"])
    curated_prompts = list(config["curated_prompts"])
    preset_id = str(config["preset_id"])

    session_start_monotonic = time.monotonic()
    session_start_epoch = time.time()
    session_data: dict[str, Any] = {
        "session_id": session_id,
        "process_id": process_id,
        "worker_id": worker_id,
        "status": "failed",
        "error": None,
        "preset_id": preset_id,
        "curated_prompt_count": len(curated_prompts),
        "connect_start_ts_utc": iso_from_epoch(session_start_epoch),
        "connect_finish_ts_utc": None,
        "session_init_sent_ts_utc": None,
        "gpu_assigned_ts_utc": None,
        "target_segment_complete_ts_utc": None,
        "leave_sent_ts_utc": None,
        "close_ts_utc": None,
        "duration_ms": None,
        "queue_wait_ms": None,
        "initial_total_segments": None,
        "segments_started": 0,
        "segments_completed": 0,
        "media_segments_completed": 0,
        "total_chunks": 0,
        "total_chunk_bytes": 0,
        "first_chunk_finish_ts_utc": None,
        "last_chunk_finish_ts_utc": None,
        "first_media_segment_complete_ts_utc": None,
        "first_chunk_before_first_media_complete": None,
        "session_goodput_mbps": None,
        "chunks": [],
    }

    connect_finish_monotonic: float | None = None
    current_segment_idx: int | None = None
    initial_total_segments: int | None = None
    first_chunk_finish_epoch: float | None = None
    first_media_segment_complete_epoch: float | None = None
    last_chunk_finish_epoch: float | None = None
    last_chunk_finish_monotonic: float | None = None

    try:
        async with websockets.connect(
            url,
            max_size=None,
            ping_interval=None,
            open_timeout=connect_timeout_s,
            close_timeout=2.0,
        ) as ws:
            connect_finish_monotonic = time.monotonic()
            session_data["connect_finish_ts_utc"] = utc_now_iso()

            init_payload = {
                "type": "session_init_v2",
                "preset_id": preset_id,
                "curated_prompts": curated_prompts,
                "enhancement_enabled": False,
                "auto_extension_enabled": False,
                "loop_generation_enabled": False,
            }
            await ws.send(json.dumps(init_payload))
            session_data["session_init_sent_ts_utc"] = utc_now_iso()

            while True:
                elapsed_s = time.monotonic() - session_start_monotonic
                timeout_remaining = session_timeout_s - elapsed_s
                if timeout_remaining <= 0:
                    session_data["status"] = "timeout"
                    session_data["error"] = (
                        f"Session timed out after {session_timeout_s:.1f}s."
                    )
                    break

                recv_start_epoch = time.time()
                recv_start_monotonic = time.monotonic()
                recv_start_iso = iso_from_epoch(recv_start_epoch)

                try:
                    message = await asyncio.wait_for(
                        ws.recv(),
                        timeout=timeout_remaining,
                    )
                except asyncio.TimeoutError:
                    session_data["status"] = "timeout"
                    session_data["error"] = (
                        "Timed out waiting for websocket message."
                    )
                    break
                except Exception as exc:
                    session_data["status"] = "failed"
                    session_data["error"] = f"WebSocket receive failed: {exc}"
                    break

                recv_finish_epoch = time.time()
                recv_finish_monotonic = time.monotonic()
                recv_finish_iso = iso_from_epoch(recv_finish_epoch)

                if isinstance(message, bytes):
                    session_data["total_chunks"] += 1
                    session_data["total_chunk_bytes"] += len(message)

                    if first_chunk_finish_epoch is None:
                        first_chunk_finish_epoch = recv_finish_epoch
                        session_data["first_chunk_finish_ts_utc"] = recv_finish_iso

                    chunk_gap_ms: float | None = None
                    if last_chunk_finish_monotonic is not None:
                        chunk_gap_ms = (
                            recv_finish_monotonic - last_chunk_finish_monotonic
                        ) * 1000.0

                    session_data["chunks"].append(
                        {
                            "segment_idx": current_segment_idx,
                            "chunk_idx": session_data["total_chunks"],
                            "size_bytes": len(message),
                            "chunk_start_ts_utc": recv_start_iso,
                            "chunk_finish_ts_utc": recv_finish_iso,
                            "chunk_gap_ms": chunk_gap_ms,
                        }
                    )
                    last_chunk_finish_monotonic = recv_finish_monotonic
                    last_chunk_finish_epoch = recv_finish_epoch
                    session_data["last_chunk_finish_ts_utc"] = recv_finish_iso
                    continue

                if not isinstance(message, str):
                    continue

                try:
                    data = json.loads(message)
                except json.JSONDecodeError as exc:
                    session_data["status"] = "protocol_error"
                    session_data["error"] = f"Invalid JSON message: {exc}"
                    break

                msg_type = data.get("type")
                if msg_type == "gpu_assigned":
                    session_data["gpu_assigned_ts_utc"] = recv_finish_iso
                    if connect_finish_monotonic is not None:
                        session_data["queue_wait_ms"] = (
                            recv_finish_monotonic - connect_finish_monotonic
                        ) * 1000.0
                elif msg_type == "ltx2_stream_start":
                    if initial_total_segments is None:
                        parsed_total = parse_int(data.get("total_segments"))
                        if parsed_total is not None and parsed_total > 0:
                            initial_total_segments = parsed_total
                            session_data["initial_total_segments"] = parsed_total
                elif msg_type == "ltx2_segment_start":
                    parsed_idx = parse_int(data.get("segment_idx"))
                    current_segment_idx = parsed_idx
                    session_data["segments_started"] += 1
                elif msg_type == "media_segment_complete":
                    session_data["media_segments_completed"] += 1
                    if first_media_segment_complete_epoch is None:
                        first_media_segment_complete_epoch = recv_finish_epoch
                        session_data[
                            "first_media_segment_complete_ts_utc"
                        ] = recv_finish_iso
                elif msg_type == "ltx2_segment_complete":
                    session_data["segments_completed"] += 1
                    seg_idx = parse_int(data.get("segment_idx"))
                    if (
                        initial_total_segments is not None
                        and seg_idx is not None
                        and seg_idx >= initial_total_segments
                    ):
                        session_data[
                            "target_segment_complete_ts_utc"
                        ] = recv_finish_iso
                        await asyncio.sleep(post_complete_wait_s)
                        session_data["leave_sent_ts_utc"] = utc_now_iso()
                        try:
                            await ws.send(json.dumps({"type": "leave"}))
                        except Exception:
                            pass
                        session_data["status"] = "success"
                        break
                elif msg_type == "session_timeout":
                    session_data["status"] = "timeout"
                    session_data["error"] = str(
                        data.get("message") or "Backend session timeout"
                    )
                    break
                elif msg_type == "error":
                    session_data["status"] = "failed"
                    session_data["error"] = str(
                        data.get("message") or "Backend error message"
                    )
                    break

            if session_data["status"] == "failed" and session_data["error"] is None:
                session_data["error"] = "Session ended without success."
    except Exception as exc:
        session_data["status"] = "failed"
        session_data["error"] = f"WebSocket connect/run failed: {exc}"

    if (
        first_chunk_finish_epoch is not None
        and last_chunk_finish_epoch is not None
        and session_data["total_chunk_bytes"] > 0
    ):
        duration_s = last_chunk_finish_epoch - first_chunk_finish_epoch
        if duration_s > 0:
            session_data["session_goodput_mbps"] = (
                session_data["total_chunk_bytes"] * 8.0 / duration_s / 1_000_000.0
            )

    if (
        first_chunk_finish_epoch is not None
        and first_media_segment_complete_epoch is not None
    ):
        session_data["first_chunk_before_first_media_complete"] = (
            first_chunk_finish_epoch < first_media_segment_complete_epoch
        )

    session_data["close_ts_utc"] = utc_now_iso()
    session_data["duration_ms"] = (
        time.monotonic() - session_start_monotonic
    ) * 1000.0
    return session_data


async def run_worker_sessions(
    *,
    worker_id: int,
    session_count: int,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    tasks = [
        asyncio.create_task(
            run_single_session(
                worker_id=worker_id,
                worker_session_idx=idx,
                config=config,
            )
        )
        for idx in range(session_count)
    ]
    if not tasks:
        return []
    return await asyncio.gather(*tasks)


def worker_entry(
    worker_id: int,
    session_count: int,
    config: dict[str, Any],
    start_event: Any,
    ready_queue: Any,
    result_queue: Any,
) -> None:
    try:
        ready_queue.put({"worker_id": worker_id, "status": "ready"})
        start_event.wait()
        sessions = asyncio.run(
            run_worker_sessions(
                worker_id=worker_id,
                session_count=session_count,
                config=config,
            )
        )
        result_queue.put(
            {
                "worker_id": worker_id,
                "status": "ok",
                "sessions": sessions,
            }
        )
    except Exception as exc:
        result_queue.put(
            {
                "worker_id": worker_id,
                "status": "error",
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )


def build_summary(
    *,
    sessions: list[dict[str, Any]],
    chunk_gap_threshold_ms: float,
) -> dict[str, Any]:
    status_counts: dict[str, int] = defaultdict(int)
    chunk_gaps: list[float] = []
    queue_waits: list[float] = []
    session_goodputs: list[float] = []
    progressive_eligible = 0
    progressive_success = 0

    all_chunk_finish_epochs: list[float] = []
    bucket_bytes: dict[int, int] = defaultdict(int)
    total_chunk_bytes = 0

    for session in sessions:
        status = str(session.get("status") or "unknown")
        status_counts[status] += 1

        queue_wait_ms = session.get("queue_wait_ms")
        if isinstance(queue_wait_ms, (int, float)):
            queue_waits.append(float(queue_wait_ms))

        session_goodput = session.get("session_goodput_mbps")
        if isinstance(session_goodput, (int, float)):
            session_goodputs.append(float(session_goodput))

        progressive_value = session.get("first_chunk_before_first_media_complete")
        if isinstance(progressive_value, bool):
            progressive_eligible += 1
            if progressive_value:
                progressive_success += 1

        for chunk in session.get("chunks", []):
            gap = chunk.get("chunk_gap_ms")
            if isinstance(gap, (int, float)):
                chunk_gaps.append(float(gap))

            size_bytes = int(chunk.get("size_bytes") or 0)
            finish_epoch = iso_to_epoch(chunk.get("chunk_finish_ts_utc"))
            if finish_epoch is None or size_bytes <= 0:
                continue
            total_chunk_bytes += size_bytes
            all_chunk_finish_epochs.append(finish_epoch)
            bucket_bytes[int(finish_epoch)] += size_bytes

    chunk_gap_stats = summarize_series(chunk_gaps)
    queue_wait_stats = summarize_series(queue_waits)
    session_goodput_stats = summarize_series(session_goodputs)

    global_goodput_mbps: float | None = None
    if len(all_chunk_finish_epochs) >= 2 and total_chunk_bytes > 0:
        duration_s = max(all_chunk_finish_epochs) - min(all_chunk_finish_epochs)
        if duration_s > 0:
            global_goodput_mbps = (
                total_chunk_bytes * 8.0 / duration_s / 1_000_000.0
            )

    bucket_throughputs_mbps = [
        (bytes_count * 8.0) / 1_000_000.0
        for _, bytes_count in sorted(bucket_bytes.items())
    ]
    bucket_stats = summarize_series(bucket_throughputs_mbps)

    chunk_gap_threshold_breaches = [
        value for value in chunk_gaps if value >= chunk_gap_threshold_ms
    ]
    non_success = len(sessions) - status_counts.get("success", 0)

    fail_reasons: list[str] = []
    if non_success > 0:
        fail_reasons.append(
            f"{non_success} session(s) did not complete successfully."
        )
    if not chunk_gaps:
        fail_reasons.append("No chunk gap data collected.")
    if chunk_gap_threshold_breaches:
        fail_reasons.append(
            f"{len(chunk_gap_threshold_breaches)} chunk gap(s) were >= "
            f"{chunk_gap_threshold_ms:.0f}ms."
        )

    passed = len(fail_reasons) == 0
    progressive_ratio = None
    if progressive_eligible > 0:
        progressive_ratio = progressive_success / progressive_eligible

    return {
        "passed": passed,
        "fail_reasons": fail_reasons,
        "sessions": {
            "total": len(sessions),
            "success": status_counts.get("success", 0),
            "failed": status_counts.get("failed", 0),
            "timeout": status_counts.get("timeout", 0),
            "protocol_error": status_counts.get("protocol_error", 0),
            "other": (
                len(sessions)
                - (
                    status_counts.get("success", 0)
                    + status_counts.get("failed", 0)
                    + status_counts.get("timeout", 0)
                    + status_counts.get("protocol_error", 0)
                )
            ),
        },
        "chunk_gap_ms": {
            **chunk_gap_stats,
            "threshold_ms": chunk_gap_threshold_ms,
            "breach_count": len(chunk_gap_threshold_breaches),
        },
        "queue_wait_ms": queue_wait_stats,
        "progressive_streaming": {
            "eligible_sessions": progressive_eligible,
            "success_sessions": progressive_success,
            "ratio": progressive_ratio,
        },
        "bandwidth_mbps": {
            "per_session": session_goodput_stats,
            "global_goodput_mbps": global_goodput_mbps,
            "bucketed_1s": {
                "count": bucket_stats.get("count"),
                "avg_mbps": bucket_stats.get("avg"),
                "peak_mbps": bucket_stats.get("max"),
            },
        },
    }


def print_summary(
    *,
    run_info: dict[str, Any],
    summary: dict[str, Any],
) -> None:
    sessions = summary["sessions"]
    chunk_gap = summary["chunk_gap_ms"]
    queue_wait = summary["queue_wait_ms"]
    progressive = summary["progressive_streaming"]
    bandwidth = summary["bandwidth_mbps"]
    per_session_bw = bandwidth["per_session"]
    bucket_bw = bandwidth["bucketed_1s"]

    print("=== LTX2 Realtime Stress Test Summary ===")
    print(
        "Run: "
        f"url={run_info['url']} clients={run_info['clients']} "
        f"processes={run_info['processes']} "
        f"preset={run_info['preset_id']} "
        f"curated_limit={run_info['curated_limit']}"
    )
    print(
        "Sessions: "
        f"total={sessions['total']} success={sessions['success']} "
        f"failed={sessions['failed']} timeout={sessions['timeout']} "
        f"protocol_error={sessions['protocol_error']}"
    )
    print(
        "Chunk gap ms: "
        f"min={format_num(chunk_gap['min'])} "
        f"p50={format_num(chunk_gap['p50'])} "
        f"p95={format_num(chunk_gap['p95'])} "
        f"p99={format_num(chunk_gap['p99'])} "
        f"max={format_num(chunk_gap['max'])} "
        f"threshold={format_num(chunk_gap['threshold_ms'])} "
        f"breaches={chunk_gap['breach_count']}"
    )
    print(
        "Queue wait ms: "
        f"min={format_num(queue_wait['min'])} "
        f"p50={format_num(queue_wait['p50'])} "
        f"p95={format_num(queue_wait['p95'])} "
        f"max={format_num(queue_wait['max'])}"
    )
    ratio = progressive["ratio"]
    ratio_text = "n/a" if ratio is None else f"{ratio * 100:.2f}%"
    print(
        "Progressive streaming: "
        f"{progressive['success_sessions']}/"
        f"{progressive['eligible_sessions']} ({ratio_text})"
    )
    print(
        "Bandwidth Mbps: "
        f"per_session_avg={format_num(per_session_bw['avg'])} "
        f"per_session_p95={format_num(per_session_bw['p95'])} "
        f"global={format_num(bandwidth['global_goodput_mbps'])} "
        f"bucket_avg={format_num(bucket_bw['avg_mbps'])} "
        f"bucket_peak={format_num(bucket_bw['peak_mbps'])}"
    )
    print(f"VERDICT: {'PASS' if summary['passed'] else 'FAIL'}")
    if summary["fail_reasons"]:
        print("Fail reasons:")
        for reason in summary["fail_reasons"]:
            print(f"- {reason}")


def distribute_sessions(total_clients: int, process_count: int) -> list[int]:
    base = total_clients // process_count
    remainder = total_clients % process_count
    counts = []
    for idx in range(process_count):
        count = base + (1 if idx < remainder else 0)
        counts.append(count)
    return counts


def run_stress(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    if websockets is None:
        raise RuntimeError(
            "Missing dependency: websockets. Install it before running this "
            "stress test."
        )

    preset_file = Path(args.preset_file).expanduser().resolve()
    selected_preset_id, curated_prompts, total_prompt_count = load_curated_prompts(
        preset_file=preset_file,
        preset_id=args.preset_id,
        curated_limit=args.curated_limit,
    )

    process_count = args.processes
    if process_count is None:
        process_count = min(args.clients, os.cpu_count() or 1)
    process_count = max(1, min(process_count, args.clients))

    mp_ctx = mp.get_context("spawn")
    start_event = mp_ctx.Event()
    ready_queue = mp_ctx.Queue()
    result_queue = mp_ctx.Queue()

    worker_config = {
        "url": args.url,
        "preset_id": selected_preset_id,
        "curated_prompts": curated_prompts,
        "connect_timeout_s": args.connect_timeout_s,
        "session_timeout_s": args.session_timeout_s,
        "post_complete_wait_s": args.post_complete_wait_s,
    }

    session_counts = distribute_sessions(args.clients, process_count)
    run_start_epoch = time.time()
    run_start_monotonic = time.monotonic()
    run_start_iso = iso_from_epoch(run_start_epoch)

    processes: list[mp.Process] = []
    for worker_id, session_count in enumerate(session_counts):
        proc = mp_ctx.Process(
            target=worker_entry,
            args=(
                worker_id,
                session_count,
                worker_config,
                start_event,
                ready_queue,
                result_queue,
            ),
        )
        proc.start()
        processes.append(proc)

    try:
        ready_workers = 0
        ready_deadline = time.monotonic() + 60.0
        while ready_workers < len(processes):
            timeout_s = max(0.1, ready_deadline - time.monotonic())
            if timeout_s <= 0:
                raise RuntimeError("Timed out waiting for workers to become ready.")
            msg = ready_queue.get(timeout=timeout_s)
            if msg.get("status") == "ready":
                ready_workers += 1

        start_event.set()

        result_deadline = (
            time.monotonic()
            + args.connect_timeout_s
            + args.session_timeout_s
            + args.post_complete_wait_s
            + 180.0
        )
        worker_results: list[dict[str, Any]] = []
        while len(worker_results) < len(processes):
            timeout_s = max(0.1, result_deadline - time.monotonic())
            if timeout_s <= 0:
                break
            try:
                result = result_queue.get(timeout=timeout_s)
            except Exception:
                break
            worker_results.append(result)

        for proc in processes:
            proc.join(timeout=5.0)
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=2.0)

        sessions: list[dict[str, Any]] = []
        worker_errors: list[dict[str, Any]] = []
        for result in worker_results:
            if result.get("status") == "ok":
                sessions.extend(result.get("sessions", []))
            else:
                worker_errors.append(
                    {
                        "worker_id": result.get("worker_id"),
                        "error": result.get("error"),
                        "traceback": result.get("traceback"),
                    }
                )

        received_workers = {result.get("worker_id") for result in worker_results}
        expected_workers = set(range(len(processes)))
        missing_workers = sorted(expected_workers - received_workers)
        for worker_id in missing_workers:
            worker_errors.append(
                {
                    "worker_id": worker_id,
                    "error": "No worker result received.",
                }
            )

        run_end_epoch = time.time()
        run_end_iso = iso_from_epoch(run_end_epoch)
        run_duration_ms = (time.monotonic() - run_start_monotonic) * 1000.0

        summary = build_summary(
            sessions=sessions,
            chunk_gap_threshold_ms=args.chunk_gap_threshold_ms,
        )

        if worker_errors:
            summary["passed"] = False
            summary["fail_reasons"] = list(summary["fail_reasons"]) + [
                f"{len(worker_errors)} worker error(s) occurred."
            ]

        output_payload = {
            "run_info": {
                "url": args.url,
                "clients": args.clients,
                "processes": process_count,
                "preset_file": str(preset_file),
                "preset_id": selected_preset_id,
                "curated_limit": args.curated_limit,
                "selected_prompt_count": len(curated_prompts),
                "preset_total_prompt_count": total_prompt_count,
                "chunk_gap_threshold_ms": args.chunk_gap_threshold_ms,
                "connect_timeout_s": args.connect_timeout_s,
                "session_timeout_s": args.session_timeout_s,
                "post_complete_wait_s": args.post_complete_wait_s,
                "run_start_ts_utc": run_start_iso,
                "run_end_ts_utc": run_end_iso,
                "run_duration_ms": run_duration_ms,
            },
            "summary": summary,
            "full_data": {
                "sessions": sessions,
                "worker_errors": worker_errors,
            },
        }
        exit_code = 0 if summary["passed"] else 1
        return output_payload, exit_code
    finally:
        for proc in processes:
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=1.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multiprocess realtime stress test for LTX2 streaming.",
    )
    parser.add_argument(
        "-u",
        "--url",
        required=True,
        help="WebSocket URL, e.g. wss://your-domain/ws",
    )
    parser.add_argument(
        "-c",
        "--clients",
        type=int,
        required=True,
        help="Total concurrent virtual users.",
    )
    parser.add_argument(
        "--processes",
        type=int,
        default=None,
        help="Worker process count (default: min(clients, cpu_count)).",
    )
    parser.add_argument(
        "--preset-file",
        default=str(DEFAULT_PRESET_FILE),
        help="Path to curated presets JSON file.",
    )
    parser.add_argument(
        "--preset-id",
        default=None,
        help="Preset id to use (default: first preset in file).",
    )
    parser.add_argument(
        "--curated-limit",
        type=int,
        default=6,
        help="Number of curated prompts to send from selected preset.",
    )
    parser.add_argument(
        "--chunk-gap-threshold-ms",
        type=float,
        default=5000.0,
        help="Fail if any chunk gap is >= this value.",
    )
    parser.add_argument(
        "--post-complete-wait-s",
        type=float,
        default=5.0,
        help="Seconds to wait after target segment completion before leave.",
    )
    parser.add_argument(
        "--connect-timeout-s",
        type=float,
        default=20.0,
        help="WebSocket connect timeout in seconds.",
    )
    parser.add_argument(
        "--session-timeout-s",
        type=float,
        default=180.0,
        help="Max session runtime per virtual user in seconds.",
    )
    parser.add_argument(
        "-o",
        "--output-json",
        required=True,
        help="Required output JSON path (summary + full data).",
    )
    args = parser.parse_args()

    if args.clients <= 0:
        parser.error("--clients must be > 0")
    if args.processes is not None and args.processes <= 0:
        parser.error("--processes must be > 0")
    if args.curated_limit <= 0:
        parser.error("--curated-limit must be > 0")
    if args.chunk_gap_threshold_ms <= 0:
        parser.error("--chunk-gap-threshold-ms must be > 0")
    if args.post_complete_wait_s < 0:
        parser.error("--post-complete-wait-s must be >= 0")
    if args.connect_timeout_s <= 0:
        parser.error("--connect-timeout-s must be > 0")
    if args.session_timeout_s <= 0:
        parser.error("--session-timeout-s must be > 0")
    return args


def main() -> int:
    try:
        args = parse_args()
        print("Starting LTX2 realtime stress test with the following parameters:")
        for arg, value in vars(args).items():
            print(f"  {arg}: {value}")
        output_payload, exit_code = run_stress(args)

        output_path = Path(args.output_json).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(output_payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        print_summary(
            run_info=output_payload["run_info"],
            summary=output_payload["summary"],
        )
        return exit_code
    except SystemExit:
        raise
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
