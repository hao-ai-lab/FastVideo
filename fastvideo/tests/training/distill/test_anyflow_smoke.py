# SPDX-License-Identifier: Apache-2.0
"""GPU smoke test for AnyFlow pretrain + on-policy.

Mirrors ``test_distill_dmd.py`` — runs the new YAML-driven training
entrypoint via ``torchrun`` for two iterations to verify end-to-end
wiring (model load, optimizer build, forward, backward, step, save).

CPU-only environments are skipped; the test is intended to fire on the
Buildkite ``/test distillation`` lane and on local boxes with at least
2 H100/H200 GPUs.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[4]
PRETRAIN_YAML = (
    REPO_ROOT
    / "examples/train/configs/distribution_matching/wan/anyflow_pretrain_t2v.yaml")
ONPOLICY_YAML = (
    REPO_ROOT
    / "examples/train/configs/distribution_matching/wan/anyflow_onpolicy_t2v.yaml")

NUM_NODES = "1"
NUM_GPUS_PER_NODE = "2"


def _have_enough_gpus() -> bool:
    """Return True iff at least 2 CUDA devices are visible. The smoke test
    needs HSDP/FSDP with a non-trivial world size; single-GPU bring-up
    races against the new framework's distributed barriers."""
    try:
        import torch
    except Exception:
        return False
    if not torch.cuda.is_available():
        return False
    return torch.cuda.device_count() >= 2


pytestmark = pytest.mark.skipif(
    not _have_enough_gpus(),
    reason="AnyFlow smoke test requires >= 2 CUDA devices")


def _run_torchrun(config_path: Path, *, output_dir: Path) -> None:
    if not config_path.exists():
        pytest.fail(f"YAML config missing: {config_path}")

    env = os.environ.copy()
    env.setdefault("MASTER_ADDR", "127.0.0.1")
    env.setdefault("MASTER_PORT", "29551")
    env.setdefault("WANDB_MODE", "offline")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")

    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nnodes", NUM_NODES,
        "--nproc_per_node", NUM_GPUS_PER_NODE,
        "--master_port", env["MASTER_PORT"],
        "-m", "fastvideo.train.entrypoint.train",
        "--config", str(config_path),
        "--training.loop.max_train_steps", "2",
        "--training.checkpoint.output_dir", str(output_dir),
        "--training.distributed.num_gpus", NUM_GPUS_PER_NODE,
        "--training.distributed.hsdp_shard_dim", NUM_GPUS_PER_NODE,
        "--training.data.train_batch_size", "1",
    ]
    process = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if process.stdout:
        print("STDOUT:", process.stdout)
    if process.stderr:
        print("STDERR:", process.stderr)
    if process.returncode != 0:
        raise subprocess.CalledProcessError(
            process.returncode, cmd, process.stdout, process.stderr)


def test_anyflow_pretrain_smoke(tmp_path: Path) -> None:
    """Two-iteration pretrain — exercises (t, r) sampling, central-difference
    target, scale balance, optimizer step, and checkpoint save path."""
    _run_torchrun(PRETRAIN_YAML, output_dir=tmp_path / "pretrain")


def test_anyflow_onpolicy_smoke(tmp_path: Path) -> None:
    """Two-iteration on-policy DMD — exercises the multi-step Euler-flow
    rollout, grad-step broadcast, DMD2 alternating updates."""
    # Override init_from to the public Wan2.1-T2V-1.3B-Diffusers checkpoint
    # for the smoke run; param_names_mapping handles the (non-existent)
    # delta_embedder rename as a no-op.
    env = os.environ.copy()
    env.setdefault("MASTER_ADDR", "127.0.0.1")
    env.setdefault("MASTER_PORT", "29552")
    env.setdefault("WANDB_MODE", "offline")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")

    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nnodes", NUM_NODES,
        "--nproc_per_node", NUM_GPUS_PER_NODE,
        "--master_port", env["MASTER_PORT"],
        "-m", "fastvideo.train.entrypoint.train",
        "--config", str(ONPOLICY_YAML),
        "--training.loop.max_train_steps", "2",
        "--training.checkpoint.output_dir", str(tmp_path / "onpolicy"),
        "--training.distributed.num_gpus", NUM_GPUS_PER_NODE,
        "--training.distributed.hsdp_shard_dim", NUM_GPUS_PER_NODE,
        "--training.data.train_batch_size", "1",
        "--models.student.init_from", "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        "--method.student_sample_steps", "2",
    ]
    process = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if process.stdout:
        print("STDOUT:", process.stdout)
    if process.stderr:
        print("STDERR:", process.stderr)
    if process.returncode != 0:
        raise subprocess.CalledProcessError(
            process.returncode, cmd, process.stdout, process.stderr)
