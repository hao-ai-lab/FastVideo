# SPDX-License-Identifier: Apache-2.0
"""End-to-end FLUX T2I smoke: short denoise + decode from local checkpoint.

Requires ``official_weights/FLUX.1-dev`` (or set ``FLUX_DEV_ROOT``) and CUDA.

Run from repo root::

    pytest tests/local_tests/pipelines/test_flux_dev_pipeline_smoke.py -vs
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
_FLUX_DEV_ROOT = os.environ.get(
    "FLUX_DEV_ROOT",
    str(_REPO_ROOT / "official_weights" / "FLUX.1-dev"),
)

os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", "29519")


def test_flux_dev_pipeline_short_run_finite_output(
        monkeypatch: pytest.MonkeyPatch) -> None:
    import torch

    if not torch.cuda.is_available():
        pytest.skip("FLUX pipeline smoke requires CUDA")
    if not Path(_FLUX_DEV_ROOT, "model_index.json").is_file():
        pytest.skip(
            "official_weights/FLUX.1-dev missing (set FLUX_DEV_ROOT or download "
            "black-forest-labs/FLUX.1-dev)")

    monkeypatch.setenv("FASTVIDEO_ATTENTION_BACKEND", "TORCH_SDPA")

    from fastvideo.configs.pipelines.flux import FluxPipelineConfig
    from fastvideo.distributed import (
        cleanup_dist_env_and_memory,
        maybe_init_distributed_environment_and_model_parallel,
    )
    from fastvideo.fastvideo_args import FastVideoArgs, WorkloadType
    from fastvideo.pipelines.basic.flux.flux_pipeline import FluxPipeline
    from fastvideo.pipelines.pipeline_batch_info import ForwardBatch

    torch.manual_seed(0)
    maybe_init_distributed_environment_and_model_parallel(1, 1)
    try:
        args = FastVideoArgs(
            model_path=_FLUX_DEV_ROOT,
            pipeline_config=FluxPipelineConfig(),
            workload_type=WorkloadType.T2I,
            hsdp_shard_dim=1,
            pin_cpu_memory=False,
            distributed_executor_backend="mp",
        )
        pipeline = FluxPipeline(_FLUX_DEV_ROOT, args)

        batch = ForwardBatch(
            data_type="image",
            prompt="a red circle on white",
            height=256,
            width=256,
            seed=0,
            num_inference_steps=2,
            guidance_scale=3.5,
            use_embedded_guidance=True,
            true_cfg_scale=1.0,
            num_videos_per_prompt=1,
            save_video=False,
        )

        out = pipeline.forward(batch, args)
        assert out.output is not None
        assert out.output.shape[0] >= 1
        assert out.output.shape[1] == 3
        assert torch.isfinite(out.output).all()
    finally:
        cleanup_dist_env_and_memory()
