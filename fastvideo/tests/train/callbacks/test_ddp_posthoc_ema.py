# SPDX-License-Identifier: Apache-2.0
"""CPU-only lifecycle test for official MMAudio DDP PostHocEMA."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from fastvideo.train.callbacks.ddp_posthoc_ema import DDPPostHocEMACallback
from fastvideo.train.utils.distributed_strategy import wrap_module_ddp


def test_official_ddp_posthoc_ema_snapshots_and_resumes(
    tmp_path: Path,
) -> None:
    pytest.importorskip("nitrous_ema")
    rendezvous = tmp_path / "ema_rendezvous"
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=0,
        world_size=1,
    )
    try:
        transformer = wrap_module_ddp(
            torch.nn.Linear(2, 2, bias=False),
            device=torch.device("cpu"),
            broadcast_buffers=False,
        )
        method = SimpleNamespace(
            student=SimpleNamespace(
                transformer=transformer,
                device=torch.device("cpu"),
            ),
        )
        callback = DDPPostHocEMACallback(
            checkpoint_every=2,
            checkpoint_folder=str(tmp_path / "snapshots"),
        )
        callback.training_config = SimpleNamespace(
            checkpoint=SimpleNamespace(output_dir=str(tmp_path)),
        )
        callback.on_train_start(method)
        callback.on_training_step_end(method, {}, iteration=1)
        with torch.no_grad():
            transformer.module.weight.add_(1)
        callback.on_training_step_end(method, {}, iteration=2)

        snapshot_root = tmp_path / "snapshots"
        assert (snapshot_root / "0.2.pt").is_file()
        assert (snapshot_root / "1.2.pt").is_file()

        restored = DDPPostHocEMACallback(
            checkpoint_every=2,
            checkpoint_folder=str(snapshot_root),
        )
        restored.training_config = callback.training_config
        restored.on_train_start(method)
        restored.load_state_dict(callback.state_dict())
        assert restored._ema is not None
        assert int(restored._ema.step) == 2

        callback.on_train_end(method, iteration=2)
        final = list(snapshot_root.glob(
            "mmaudio_ema_final_sigma_0p05_step_*.pth",
        ))
        assert len(final) == 1
    finally:
        dist.destroy_process_group()
