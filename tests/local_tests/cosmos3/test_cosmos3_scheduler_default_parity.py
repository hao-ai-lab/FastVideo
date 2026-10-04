# SPDX-License-Identifier: Apache-2.0
"""Cosmos3 scheduler default + per-request override parity (Tier A scaffold).

Reference:
  * ``vllm_omni/diffusion/models/cosmos3/pipeline_cosmos3.py:275-307`` —
    initial UniPCMultistepScheduler load (preserves solver_order,
    timestep_spacing, beta_schedule, sigma bounds, flow_shift) and
    one-time override at engine-init if ``od_config.flow_shift`` is set.
  * ``vllm_omni/diffusion/models/cosmos3/pipeline_cosmos3.py:498-512`` —
    ``_set_flow_shift(target_shift)``: rebuild the scheduler via
    ``UniPCMultistepScheduler.from_config(base_config, flow_shift=target)``
    when the requested target differs from the current shift.
  * ``vllm_omni/diffusion/models/cosmos3/pipeline_cosmos3.py:1069-1110`` —
    per-request mode defaults: T2I uses ``shift=3.0``; T2V/I2V use the
    engine-init shift (typically 1.0); ``flow_shift`` may be overridden
    per request via ``sampling_params.extra_args["flow_shift"]``.

The invariant under test: for the same RNG seed and the same number of
inference steps, the scheduler's ``timesteps`` tensor must be identical
whenever the ``flow_shift`` is identical, and must change deterministically
when ``flow_shift`` is overridden via ``_set_flow_shift``.
"""
from __future__ import annotations

import pytest
import torch

pytestmark = [pytest.mark.local]


def test_t2i_default_flow_shift_is_3() -> None:
    """Asserts that T2I requests rebuild the scheduler at ``flow_shift=3.0``.

    Cross-check: pipeline_cosmos3.py:1073-1080 sets
    ``default_flow_shift = 3.0`` for T2I, and
    pipeline_cosmos3.py:1110 calls ``self._set_flow_shift(flow_shift_target)``
    which rebuilds the scheduler via ``UniPCMultistepScheduler.from_config(
    base_config, flow_shift=3.0)``.
    """
    try:
        from fastvideo.pipelines.basic.cosmos3.cosmos3_pipeline import (  # type: ignore
            Cosmos3OmniDiffusersPipeline,
        )
    except ImportError:
        pytest.skip("FastVideo Cosmos3 pipeline not yet implemented (Phase 2b)")

    pipeline = Cosmos3OmniDiffusersPipeline.__new__(Cosmos3OmniDiffusersPipeline)
    if not hasattr(pipeline, "_set_flow_shift") or not hasattr(pipeline, "scheduler"):
        pytest.skip("FastVideo Cosmos3 scheduler/_set_flow_shift not yet wired")

    pipeline._set_flow_shift(3.0)
    assert float(pipeline.scheduler.config.flow_shift) == 3.0


def test_t2v_default_flow_shift_is_engine_init() -> None:
    """Asserts T2V/I2V use the engine-init shift (e.g. 1.0), NOT a fixed default.

    Cross-check: pipeline_cosmos3.py:1091 sets
    ``default_flow_shift = self._engine_init_flow_shift`` for T2V/I2V
    (NOT ``None`` — passing ``None`` would leak a prior T2I rebuild
    forward).
    """
    try:
        from fastvideo.pipelines.basic.cosmos3.cosmos3_pipeline import (  # type: ignore
            Cosmos3OmniDiffusersPipeline,
        )
    except ImportError:
        pytest.skip("FastVideo Cosmos3 pipeline not yet implemented (Phase 2b)")

    pipeline = Cosmos3OmniDiffusersPipeline.__new__(Cosmos3OmniDiffusersPipeline)
    if not hasattr(pipeline, "_engine_init_flow_shift") or not hasattr(pipeline, "_set_flow_shift"):
        pytest.skip("FastVideo Cosmos3 _engine_init_flow_shift not yet wired")

    init_shift = float(pipeline._engine_init_flow_shift)
    pipeline._set_flow_shift(init_shift)
    assert float(pipeline.scheduler.config.flow_shift) == init_shift


def test_scheduler_timesteps_deterministic_under_seed() -> None:
    """Asserts that ``scheduler.set_timesteps(N)`` is deterministic given the
    same N and the same flow_shift.

    The UniPC scheduler's timestep sequence does not depend on a torch
    seed (it's a closed-form function of N + scheduler config), so
    invoking ``set_timesteps`` twice should produce identical tensors.
    """
    try:
        from fastvideo.pipelines.basic.cosmos3.cosmos3_pipeline import (  # type: ignore
            Cosmos3OmniDiffusersPipeline,
        )
    except ImportError:
        pytest.skip("FastVideo Cosmos3 pipeline not yet implemented (Phase 2b)")

    pipeline = Cosmos3OmniDiffusersPipeline.__new__(Cosmos3OmniDiffusersPipeline)
    if not hasattr(pipeline, "scheduler") or not hasattr(pipeline, "_set_flow_shift"):
        pytest.skip("FastVideo Cosmos3 scheduler not yet wired")

    pipeline._set_flow_shift(3.0)
    pipeline.scheduler.set_timesteps(35, device=torch.device("cpu"))
    seq_a = pipeline.scheduler.timesteps.clone()

    pipeline.scheduler.set_timesteps(35, device=torch.device("cpu"))
    seq_b = pipeline.scheduler.timesteps.clone()

    torch.testing.assert_close(seq_a, seq_b)
