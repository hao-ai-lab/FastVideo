# pyright: reportAttributeAccessIssue=false
from __future__ import annotations

import asyncio
import multiprocessing as mp
from types import SimpleNamespace

import pytest


import dreamverse.gpu_pool as gpu_pool


def _child_consume_and_exit(cmd_q, resp_q):
    """Top-level so the spawn context can pickle it.

    Signals startup by putting "READY" on resp_q (so the parent can
    wait out spawn-import latency separately from the actual test
    assertion), then consumes one command from cmd_q and exits without
    putting anything else on resp_q.  Simulates a worker that dies
    mid-command — e.g. SIGQUIT'd after a fatal pipeline error.
    """
    resp_q.put("READY")
    cmd_q.get()


def test_get_available_gpus_defaults_to_single_detected_gpu(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("FASTVIDEO_GPU_COUNT", raising=False)

    def fake_run(*args, **kwargs):
        del args, kwargs
        return SimpleNamespace(returncode=0, stdout="0\n1\n2\n")

    monkeypatch.setattr(gpu_pool.subprocess, "run", fake_run)

    assert gpu_pool.get_available_gpus() == [0]


def test_get_available_gpus_respects_explicit_gpu_count(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setenv("FASTVIDEO_GPU_COUNT", "2")

    def fake_run(*args, **kwargs):
        del args, kwargs
        return SimpleNamespace(returncode=0, stdout="0\n1\n2\n")

    monkeypatch.setattr(gpu_pool.subprocess, "run", fake_run)

    assert gpu_pool.get_available_gpus() == [0, 1]


def test_get_available_gpus_can_use_all_visible_devices(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,5,7")
    monkeypatch.setenv("FASTVIDEO_GPU_COUNT", "all")

    assert gpu_pool.get_available_gpus() == [3, 5, 7]


def test_get_available_gpus_defaults_to_first_visible_device(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,5,7")
    monkeypatch.delenv("FASTVIDEO_GPU_COUNT", raising=False)

    assert gpu_pool.get_available_gpus() == [3]


def test_get_available_gpus_rejects_invalid_gpu_count(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setenv("FASTVIDEO_GPU_COUNT", "zero")

    with pytest.raises(RuntimeError, match="Invalid FASTVIDEO_GPU_COUNT"):
        gpu_pool.get_available_gpus()


def test_send_command_raises_on_worker_death():
    """A worker that consumes a command and exits without replying must
    surface as RuntimeError via sentinel detection, not after the long
    queue timeout.

    The child signals startup with a "READY" message; we drain that
    first so the wait_for budget bounds only the sentinel-detection
    time, not spawn-import latency.
    """
    ctx = mp.get_context("spawn")
    cmd_q = ctx.Queue()
    resp_q = ctx.Queue()

    proc = ctx.Process(
        target=_child_consume_and_exit, args=(cmd_q, resp_q)
    )
    proc.start()

    # Wait for the spawn child to fully boot.  Allow generous time —
    # this isn't what we're measuring.
    ready = resp_q.get(timeout=30.0)
    assert ready == "READY"

    async def runner():
        slot = gpu_pool.GPUSlot(gpu_id=0, cuda_device="0")
        slot.process = proc
        slot.command_queue = cmd_q
        slot.response_queue = resp_q

        # Now the child is blocked in cmd_q.get().  Sending the cmd
        # makes it exit ~immediately; sentinel must fire well before
        # the queue timeout.
        await asyncio.wait_for(
            slot._send_command(
                gpu_pool.Command(gpu_pool.CommandType.SHUTDOWN),
                timeout=10.0,
            ),
            timeout=5.0,
        )

    try:
        with pytest.raises(RuntimeError, match="worker died"):
            asyncio.run(runner())
    finally:
        proc.join(timeout=5)
        cmd_q.close()
        resp_q.close()
