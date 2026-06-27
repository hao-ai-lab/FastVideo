# SPDX-License-Identifier: Apache-2.0
"""Kandinsky-5 model family pipeline presets."""

from fastvideo.api.presets import InferencePreset, PresetStageSpec

_NEGATIVE_PROMPT = ("Static, 2D cartoon, cartoon, 2d animation, paintings, images, worst quality, low quality, ugly, "
                    "deformed, walking backwards")

_DENOISE_STAGE = PresetStageSpec(
    name="denoise",
    kind="denoising",
    description="Main Kandinsky-5 denoising pass",
    allowed_overrides=frozenset({
        "num_inference_steps",
        "guidance_scale",
    }),
)

KANDINSKY5_T2V_LITE_5S = InferencePreset(
    name="kandinsky5_t2v_lite_5s",
    version=1,
    model_family="kandinsky5",
    description="Kandinsky-5.0 Lite T2V 5s",
    workload_type="t2v",
    stage_schemas=(_DENOISE_STAGE, ),
    defaults={
        "height": 512,
        "width": 768,
        "num_frames": 121,
        "fps": 24,
        "guidance_scale": 5.0,
        "num_inference_steps": 50,
        "negative_prompt": _NEGATIVE_PROMPT,
    },
)

KANDINSKY5_T2V_LITE_DISTILLED_5S = InferencePreset(
    name="kandinsky5_t2v_lite_distilled_5s",
    version=1,
    model_family="kandinsky5",
    description="Kandinsky-5.0 Lite T2V Distilled 5s",
    workload_type="t2v",
    stage_schemas=(_DENOISE_STAGE, ),
    defaults={
        "height": 512,
        "width": 768,
        "num_frames": 121,
        "fps": 24,
        "guidance_scale": 1.0,
        "num_inference_steps": 16,
        "negative_prompt": _NEGATIVE_PROMPT,
    },
)

KANDINSKY5_T2V_PRO_5S = InferencePreset(
    name="kandinsky5_t2v_pro_5s",
    version=1,
    model_family="kandinsky5",
    description="Kandinsky-5.0 Pro T2V 5s",
    workload_type="t2v",
    stage_schemas=(_DENOISE_STAGE, ),
    defaults={
        "height": 512,
        "width": 768,
        "num_frames": 121,
        "fps": 24,
        "guidance_scale": 5.0,
        "num_inference_steps": 50,
        "negative_prompt": _NEGATIVE_PROMPT,
    },
)

KANDINSKY5_T2V_PRO_DISTILLED_5S = InferencePreset(
    name="kandinsky5_t2v_pro_distilled_5s",
    version=1,
    model_family="kandinsky5",
    description="Kandinsky-5.0 Pro T2V Distilled 5s",
    workload_type="t2v",
    stage_schemas=(_DENOISE_STAGE, ),
    defaults={
        "height": 512,
        "width": 768,
        "num_frames": 121,
        "fps": 24,
        "guidance_scale": 1.0,
        "num_inference_steps": 16,
        "negative_prompt": _NEGATIVE_PROMPT,
    },
)

KANDINSKY5_I2V_LITE_5S = InferencePreset(
    name="kandinsky5_i2v_lite_5s",
    version=1,
    model_family="kandinsky5",
    description="Kandinsky-5.0 Lite I2V 5s",
    workload_type="i2v",
    stage_schemas=(_DENOISE_STAGE, ),
    defaults={
        "height": 512,
        "width": 768,
        "num_frames": 121,
        "fps": 24,
        "guidance_scale": 5.0,
        "num_inference_steps": 50,
        "negative_prompt": _NEGATIVE_PROMPT,
    },
)

KANDINSKY5_I2V_PRO_5S = InferencePreset(
    name="kandinsky5_i2v_pro_5s",
    version=1,
    model_family="kandinsky5",
    description="Kandinsky-5.0 Pro I2V 5s",
    workload_type="i2v",
    stage_schemas=(_DENOISE_STAGE, ),
    defaults={
        "height": 512,
        "width": 768,
        "num_frames": 121,
        "fps": 24,
        "guidance_scale": 5.0,
        "num_inference_steps": 50,
        "negative_prompt": _NEGATIVE_PROMPT,
    },
)

KANDINSKY5_I2V_PRO_DISTILLED_5S = InferencePreset(
    name="kandinsky5_i2v_pro_distilled_5s",
    version=1,
    model_family="kandinsky5",
    description="Kandinsky-5.0 Pro I2V Distilled 5s",
    workload_type="i2v",
    stage_schemas=(_DENOISE_STAGE, ),
    defaults={
        "height": 512,
        "width": 768,
        "num_frames": 121,
        "fps": 24,
        "guidance_scale": 1.0,
        "num_inference_steps": 16,
        "negative_prompt": _NEGATIVE_PROMPT,
    },
)

ALL_PRESETS = (KANDINSKY5_T2V_LITE_5S, KANDINSKY5_T2V_LITE_DISTILLED_5S, KANDINSKY5_T2V_PRO_5S, KANDINSKY5_T2V_PRO_DISTILLED_5S, KANDINSKY5_I2V_LITE_5S, KANDINSKY5_I2V_PRO_5S, KANDINSKY5_I2V_PRO_DISTILLED_5S)
