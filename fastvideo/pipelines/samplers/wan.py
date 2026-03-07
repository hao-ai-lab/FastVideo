# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.models.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler, )
from fastvideo.models.schedulers.scheduling_flow_unipc_multistep import (
    FlowUniPCMultistepScheduler, )
from fastvideo.pipelines.samplers.base import SamplerKind, normalize_sampler_kind


def get_wan_sampler_kind(fastvideo_args: FastVideoArgs) -> SamplerKind:
    raw = getattr(fastvideo_args.pipeline_config, "sampler_kind", None)
    return normalize_sampler_kind(raw, where="pipeline_config.sampler_kind")


def build_wan_scheduler(fastvideo_args: FastVideoArgs, kind: SamplerKind):
    shift = fastvideo_args.pipeline_config.flow_shift
    if kind == "sde":
        return FlowMatchEulerDiscreteScheduler(shift=shift)

    ode_solver_raw = getattr(fastvideo_args.pipeline_config, "ode_solver",
                             "unipc")
    ode_solver = str(ode_solver_raw).strip().lower(
    ) if ode_solver_raw is not None else "unipc"
    if ode_solver in {"unipc", "unipc_multistep", "multistep"}:
        return FlowUniPCMultistepScheduler(shift=shift)
    if ode_solver in {"euler", "flowmatch", "flowmatch_euler"}:
        return FlowMatchEulerDiscreteScheduler(shift=shift)

    raise ValueError("Unknown pipeline_config.ode_solver for wan pipelines: "
                     f"{ode_solver_raw!r} (expected 'unipc' or 'euler').")


def wan_use_btchw_layout(kind: SamplerKind) -> bool:
    return kind == "sde"
