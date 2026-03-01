# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.models.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)
from fastvideo.models.schedulers.scheduling_flow_unipc_multistep import (
    FlowUniPCMultistepScheduler,
)
from fastvideo.pipelines.samplers.base import SamplerKind, normalize_sampler_kind


def get_wan_sampler_kind(fastvideo_args: FastVideoArgs) -> SamplerKind:
    raw = getattr(fastvideo_args.pipeline_config, "sampler_kind", None)
    return normalize_sampler_kind(raw, where="pipeline_config.sampler_kind")


def build_wan_scheduler(fastvideo_args: FastVideoArgs, kind: SamplerKind):
    shift = fastvideo_args.pipeline_config.flow_shift
    if kind == "sde":
        return FlowMatchEulerDiscreteScheduler(shift=shift)
    return FlowUniPCMultistepScheduler(shift=shift)


def wan_use_btchw_layout(kind: SamplerKind) -> bool:
    return kind == "sde"

