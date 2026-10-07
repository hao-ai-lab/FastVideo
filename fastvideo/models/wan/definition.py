# SPDX-License-Identifier: Apache-2.0
"""Data-only Wan variant definitions consumed by the shared registry.

Config names reference ``pipeline_config.py``; preset names reference
``fastvideo/pipelines/basic/wan/presets.py``. Neither values nor factories are
duplicated here. Checkpoint manifests and explicit pipeline overrides still
select the executable pipeline; these definitions do not pin or replace them.
"""

from dataclasses import dataclass
from typing import Literal

# Dense DMD indexes the complete training-noise table. This is independent of
# the configurable inference flow_shift and must not become a shared scheduler
# instance: timestep preparation mutates its scheduler.
DMD_TRAINING_NOISE_SHIFT = 8.0


@dataclass(frozen=True)
class WanModelDefinition:
    pipeline_config: str
    preset: str
    sampling: Literal["unipc", "dmd", "causal_dmd"]
    hf_model_paths: tuple[str, ...]
    workload_types: tuple[Literal["t2v", "i2v"], ...]
    match_any: tuple[str, ...] = ()
    match_all: tuple[str, ...] = ()
    exclude: tuple[str, ...] = ()

    def matches(self, path_or_class: str) -> bool:
        """Preserve legacy substring detectors, including their broad matches."""
        value = path_or_class.lower()
        return (any(token in value for token in self.match_any) and all(token in value for token in self.match_all)
                and not any(token in value for token in self.exclude))


# The two groups preserve the registry's existing first-match ordering around
# DreamX. Keeping this boundary avoids changing local/custom-checkpoint
# resolution when both a Wan and DreamX detector match. Within each group,
# declaration order is registration order.
WAN_MODEL_DEFINITION_GROUPS = (
    (
        WanModelDefinition(
            pipeline_config="WanT2V480PConfig",
            preset="wan_t2v_1_3b",
            sampling="unipc",
            hf_model_paths=("Wan-AI/Wan2.1-T2V-1.3B-Diffusers", ),
            workload_types=("t2v", ),
            match_any=("wanpipeline", ),
        ),
        WanModelDefinition(
            pipeline_config="WanT2V720PConfig",
            preset="wan_t2v_14b",
            sampling="unipc",
            hf_model_paths=("Wan-AI/Wan2.1-T2V-14B-Diffusers", "FastVideo/Wan2.1-VSA-T2V-14B-720P-Diffusers"),
            workload_types=("t2v", ),
        ),
        WanModelDefinition(
            pipeline_config="WanI2V480PConfig",
            preset="wan_i2v_14b_480p",
            sampling="unipc",
            hf_model_paths=("Wan-AI/Wan2.1-I2V-14B-480P-Diffusers", ),
            workload_types=("i2v", ),
            match_any=("wanimagetovideo", ),
        ),
        WanModelDefinition(
            pipeline_config="WanI2V720PConfig",
            preset="wan_i2v_14b_720p",
            sampling="unipc",
            hf_model_paths=("Wan-AI/Wan2.1-I2V-14B-720P-Diffusers", ),
            workload_types=("i2v", ),
        ),
        WanModelDefinition(
            pipeline_config="WanI2V480PConfig",
            preset="wan_fun_1_3b_inp",
            sampling="unipc",
            hf_model_paths=("weizhou03/Wan2.1-Fun-1.3B-InP-Diffusers", ),
            workload_types=("i2v", ),
        ),
        WanModelDefinition(
            pipeline_config="WANV2VConfig",
            preset="wan_fun_1_3b_control",
            sampling="unipc",
            hf_model_paths=("IRMChen/Wan2.1-Fun-1.3B-Control-Diffusers", ),
            workload_types=(),
        ),
        WanModelDefinition(
            pipeline_config="FastWan2_1_T2V_480P_Config",
            preset="fast_wan_t2v_480p",
            sampling="dmd",
            hf_model_paths=("FastVideo/FastWan2.1-T2V-1.3B-Diffusers", "FastVideo/FastWan2.1-T2V-14B-480P-Diffusers"),
            workload_types=("t2v", ),
            match_any=("wandmdpipeline", ),
        ),
        WanModelDefinition(
            pipeline_config="Wan2_2_TI2V_5B_Config",
            preset="wan_2_2_ti2v_5b",
            sampling="unipc",
            hf_model_paths=("Wan-AI/Wan2.2-TI2V-5B-Diffusers", ),
            workload_types=("t2v", "i2v"),
        ),
    ),
    (
        WanModelDefinition(
            pipeline_config="FastWan2_2_TI2V_5B_Config",
            preset="fast_wan_2_2_ti2v_5b",
            sampling="dmd",
            hf_model_paths=("FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers",
                            "FastVideo/FastWan2.2-TI2V-5B-Diffusers"),
            workload_types=("t2v", "i2v"),
        ),
        WanModelDefinition(
            pipeline_config="LucyEditDevConfig",
            preset="lucy_edit_dev",
            sampling="unipc",
            hf_model_paths=("decart-ai/Lucy-Edit-Dev", "decart-ai/Lucy-Edit-1.1-Dev"),
            workload_types=(),
            match_any=("lucy-edit", ),
        ),
        WanModelDefinition(
            pipeline_config="Wan2_2_T2V_A14B_Config",
            preset="wan_2_2_t2v_a14b",
            sampling="unipc",
            hf_model_paths=("Wan-AI/Wan2.2-T2V-A14B-Diffusers", ),
            workload_types=("t2v", ),
        ),
        WanModelDefinition(
            pipeline_config="Wan2_2_I2V_A14B_Config",
            preset="wan_2_2_i2v_a14b",
            sampling="unipc",
            hf_model_paths=("Wan-AI/Wan2.2-I2V-A14B-Diffusers", ),
            workload_types=("i2v", ),
        ),
        WanModelDefinition(
            pipeline_config="SelfForcingWanT2V480PConfig",
            preset="sf_wan_t2v_1_3b",
            sampling="causal_dmd",
            hf_model_paths=("wlsaidhi/SFWan2.1-T2V-1.3B-Diffusers", ),
            workload_types=("t2v", ),
            match_any=("wancausaldmdpipeline", ),
        ),
        WanModelDefinition(
            pipeline_config="SelfForcingWan2_2_T2V480PConfig",
            preset="sf_wan_2_2_t2v_a14b",
            sampling="causal_dmd",
            hf_model_paths=("rand0nmr/SFWan2.2-T2V-A14B-Diffusers", ),
            workload_types=("t2v", ),
            match_any=("sfwan2.2", "sfwan2_2"),
            exclude=("i2v", ),
        ),
        WanModelDefinition(
            pipeline_config="SelfForcingWan2_2_T2V480PConfig",
            preset="sf_wan_2_2_i2v_a14b",
            sampling="causal_dmd",
            hf_model_paths=("FastVideo/SFWan2.2-I2V-A14B-Preview-Diffusers", ),
            workload_types=("i2v", ),
            match_any=("sfwan2.2", "sfwan2_2"),
            match_all=("i2v", ),
        ),
    ),
)

WAN_MODEL_DEFINITIONS = tuple(definition for group in WAN_MODEL_DEFINITION_GROUPS for definition in group)
