# SPDX-License-Identifier: Apache-2.0
"""Wan model family pipeline presets.

Each preset is a named inference preset that declares the user-facing
stage topology, default sampling values, and which per-stage overrides
are allowed.  Presets are registered explicitly from
:func:`fastvideo.registry._register_presets`.
"""
from fastvideo.api.presets import InferencePreset, PresetStageSpec

# -------------------------------------------------------------------
# Shared negative prompts
# -------------------------------------------------------------------

_NEGATIVE_PROMPT_EN = ("Bright tones, overexposed, static, blurred details, subtitles,"
                       " style, works, paintings, images, static, overall gray, worst"
                       " quality, low quality, JPEG compression residue, ugly,"
                       " incomplete, extra fingers, poorly drawn hands, poorly drawn"
                       " faces, deformed, disfigured, misshapen limbs, fused fingers,"
                       " still picture, messy background, three legs, many people in"
                       " the background, walking backwards")

_NEGATIVE_PROMPT_CN = ("色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，"
                       "静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，"
                       "多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，"
                       "形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，"
                       "背景人很多，倒着走")

# -------------------------------------------------------------------
# Shared stage specs
# -------------------------------------------------------------------

_DENOISE_STAGE = PresetStageSpec(
    name="denoise",
    kind="denoising",
    description="Main denoising pass",
    allowed_overrides=frozenset({
        "num_inference_steps",
        "guidance_scale",
    }),
)

# -------------------------------------------------------------------
# Wan 2.1 T2V presets
# -------------------------------------------------------------------

WAN_T2V_1_3B = InferencePreset(
    name="wan_t2v_1_3b",
    version=1,
    model_family="wan",
    description="Wan 2.1 T2V 1.3B at 480p",
    workload_type="t2v",
    stage_schemas=(_DENOISE_STAGE, ),
    defaults={
        "height": 480,
        "width": 832,
        "num_frames": 81,
        "fps": 16,
        "guidance_scale": 3.0,
        "num_inference_steps": 50,
        "negative_prompt": _NEGATIVE_PROMPT_EN,
    },
)

WAN_T2V_14B = InferencePreset(
    name="wan_t2v_14b",
    version=1,
    model_family="wan",
    description="Wan 2.1 T2V 14B at 720p",
    workload_type="t2v",
    stage_schemas=(_DENOISE_STAGE, ),
    defaults={
        "height": 720,
        "width": 1280,
        "num_frames": 81,
        "fps": 16,
        "guidance_scale": 5.0,
        "num_inference_steps": 50,
        "negative_prompt": _NEGATIVE_PROMPT_EN,
    },
)

# -------------------------------------------------------------------
# Wan 2.1 I2V presets
# -------------------------------------------------------------------

WAN_I2V_14B_480P = InferencePreset(
    name="wan_i2v_14b_480p",
    version=1,
    model_family="wan",
    description="Wan 2.1 I2V 14B at 480p",
    workload_type="i2v",
    stage_schemas=(_DENOISE_STAGE, ),
    defaults={
        "height": 480,
        "width": 832,
        "num_frames": 81,
        "fps": 16,
        "guidance_scale": 5.0,
        "num_inference_steps": 40,
        "negative_prompt": _NEGATIVE_PROMPT_EN,
    },
)

WAN_I2V_14B_720P = InferencePreset(
    name="wan_i2v_14b_720p",
    version=1,
    model_family="wan",
    description="Wan 2.1 I2V 14B at 720p",
    workload_type="i2v",
    stage_schemas=(_DENOISE_STAGE, ),
    defaults={
        "height": 720,
        "width": 1280,
        "num_frames": 81,
        "fps": 16,
        "guidance_scale": 5.0,
        "num_inference_steps": 40,
        "negative_prompt": _NEGATIVE_PROMPT_EN,
    },
)

# -------------------------------------------------------------------
# Wan 2.2 presets
# -------------------------------------------------------------------

_DENOISE_STAGE_WAN22 = PresetStageSpec(
    name="denoise",
    kind="denoising",
    description="Wan 2.2 two-guidance-scale denoising",
    allowed_overrides=frozenset({
        "num_inference_steps",
        "guidance_scale",
        "guidance_scale_2",
        "boundary_ratio",
    }),
)

WAN_2_2_T2V_A14B = InferencePreset(
    name="wan_2_2_t2v_a14b",
    version=1,
    model_family="wan",
    description="Wan 2.2 T2V A14B with dual guidance scales",
    workload_type="t2v",
    stage_schemas=(_DENOISE_STAGE_WAN22, ),
    defaults={
        "guidance_scale": 4.0,
        "guidance_scale_2": 3.0,
        "num_inference_steps": 40,
        "fps": 16,
        "negative_prompt": _NEGATIVE_PROMPT_CN,
    },
)

WAN_2_2_I2V_A14B = InferencePreset(
    name="wan_2_2_i2v_a14b",
    version=1,
    model_family="wan",
    description="Wan 2.2 I2V A14B with dual guidance scales",
    workload_type="i2v",
    stage_schemas=(_DENOISE_STAGE_WAN22, ),
    defaults={
        "guidance_scale": 3.5,
        "guidance_scale_2": 3.5,
        "num_inference_steps": 40,
        "fps": 16,
        "negative_prompt": _NEGATIVE_PROMPT_CN,
    },
)

# -------------------------------------------------------------------
# Wan 2.1 Fun / Control presets
# -------------------------------------------------------------------

WAN_FUN_1_3B_INP = InferencePreset(
    name="wan_fun_1_3b_inp",
    version=1,
    model_family="wan",
    description="Wan 2.1 Fun 1.3B InP (image-to-video inpainting)",
    workload_type="i2v",
    stage_schemas=(_DENOISE_STAGE, ),
    defaults={
        "height": 480,
        "width": 832,
        "num_frames": 81,
        "fps": 16,
        "guidance_scale": 6.0,
        "num_inference_steps": 50,
        "negative_prompt": _NEGATIVE_PROMPT_CN,
    },
)

WAN_FUN_1_3B_CONTROL = InferencePreset(
    name="wan_fun_1_3b_control",
    version=1,
    model_family="wan",
    description="Wan 2.1 Fun 1.3B Control (V2V)",
    stage_schemas=(_DENOISE_STAGE, ),
    defaults={
        "height": 832,
        "width": 480,
        "num_frames": 49,
        "fps": 16,
        "guidance_scale": 6.0,
        "negative_prompt": _NEGATIVE_PROMPT_EN,
    },
)

# -------------------------------------------------------------------
# FastWan (DMD) presets
# -------------------------------------------------------------------

FAST_WAN_T2V_480P = InferencePreset(
    name="fast_wan_t2v_480p",
    version=1,
    model_family="wan",
    description="FastWan 2.1 T2V DMD at 480p (3-step)",
    workload_type="t2v",
    stage_schemas=(_DENOISE_STAGE, ),
    defaults={
        "height": 448,
        "width": 832,
        "num_frames": 61,
        "fps": 16,
        "guidance_scale": 3.0,
        "num_inference_steps": 3,
        "negative_prompt": _NEGATIVE_PROMPT_EN,
    },
)

# -------------------------------------------------------------------
# Wan 2.2 TI2V 5B presets
# -------------------------------------------------------------------

WAN_2_2_TI2V_5B = InferencePreset(
    name="wan_2_2_ti2v_5b",
    version=1,
    model_family="wan",
    description="Wan 2.2 TI2V 5B",
    workload_type="t2v",
    stage_schemas=(_DENOISE_STAGE, ),
    defaults={
        "height": 704,
        "width": 1280,
        "num_frames": 121,
        "fps": 24,
        "guidance_scale": 5.0,
        "num_inference_steps": 50,
        "negative_prompt": _NEGATIVE_PROMPT_CN,
    },
)

FAST_WAN_2_2_TI2V_5B = InferencePreset(
    name="fast_wan_2_2_ti2v_5b",
    version=1,
    model_family="wan",
    description="FastWan 2.2 TI2V 5B DMD",
    workload_type="t2v",
    stage_schemas=(_DENOISE_STAGE, ),
    defaults={
        "height": 704,
        "width": 1280,
        "num_frames": 121,
        "fps": 24,
        "guidance_scale": 5.0,
        "num_inference_steps": 50,
        "negative_prompt": _NEGATIVE_PROMPT_CN,
    },
)

# -------------------------------------------------------------------
# Self-Forcing (causal) presets
# -------------------------------------------------------------------

SF_WAN_T2V_1_3B = InferencePreset(
    name="sf_wan_t2v_1_3b",
    version=1,
    model_family="wan",
    description="Self-Forcing Wan 2.1 T2V 1.3B (causal)",
    workload_type="t2v",
    stage_schemas=(_DENOISE_STAGE, ),
    defaults={
        "height": 480,
        "width": 832,
        "num_frames": 81,
        "fps": 16,
        "guidance_scale": 6.0,
        "num_inference_steps": 50,
        "negative_prompt": _NEGATIVE_PROMPT_CN,
    },
)

SF_WAN_2_2_T2V_A14B = InferencePreset(
    name="sf_wan_2_2_t2v_a14b",
    version=1,
    model_family="wan",
    description="Self-Forcing Wan 2.2 T2V A14B (causal)",
    workload_type="t2v",
    stage_schemas=(_DENOISE_STAGE_WAN22, ),
    defaults={
        "height": 448,
        "width": 832,
        "num_frames": 81,
        "fps": 16,
        "guidance_scale": 4.0,
        "guidance_scale_2": 3.0,
        "num_inference_steps": 8,
        "negative_prompt": _NEGATIVE_PROMPT_CN,
    },
)

SF_WAN_2_2_I2V_A14B = InferencePreset(
    name="sf_wan_2_2_i2v_a14b",
    version=1,
    model_family="wan",
    description="Self-Forcing Wan 2.2 I2V A14B (causal)",
    workload_type="i2v",
    stage_schemas=(_DENOISE_STAGE_WAN22, ),
    defaults={
        "height": 448,
        "width": 832,
        "num_frames": 81,
        "fps": 16,
        "guidance_scale": 4.0,
        "guidance_scale_2": 3.0,
        "num_inference_steps": 8,
        "negative_prompt": _NEGATIVE_PROMPT_CN,
    },
)

# Collect all presets for bulk registration.
ALL_PRESETS = (
    WAN_T2V_1_3B,
    WAN_T2V_14B,
    WAN_I2V_14B_480P,
    WAN_I2V_14B_720P,
    WAN_2_2_T2V_A14B,
    WAN_2_2_I2V_A14B,
    WAN_FUN_1_3B_INP,
    WAN_FUN_1_3B_CONTROL,
    FAST_WAN_T2V_480P,
    WAN_2_2_TI2V_5B,
    FAST_WAN_2_2_TI2V_5B,
    SF_WAN_T2V_1_3B,
    SF_WAN_2_2_T2V_A14B,
    SF_WAN_2_2_I2V_A14B,
)
