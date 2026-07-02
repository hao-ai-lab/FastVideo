# SPDX-License-Identifier: Apache-2.0
"""DreamX-World model family pipeline presets."""

from fastvideo.api.presets import InferencePreset, PresetStageSpec

_NEGATIVE_PROMPT_CN = ("色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，"
                       "静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，"
                       "多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，"
                       "形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，"
                       "背景人很多，倒着走")

_DENOISE_STAGE = PresetStageSpec(
    name="denoise",
    kind="denoising",
    description="DreamX-World camera-conditioned denoising pass",
    allowed_overrides=frozenset({
        "num_inference_steps",
        "guidance_scale",
    }),
)

DREAMX_WORLD_5B_CAM = InferencePreset(
    name="dreamx_world_5b_cam",
    version=1,
    model_family="dreamx_world",
    description="DreamX-World 5B camera-control video generation",
    workload_type="i2v",
    stage_schemas=(_DENOISE_STAGE, ),
    defaults={
        "height": 480,
        "width": 832,
        "num_frames": 161,
        "fps": 16,
        "guidance_scale": 5.0,
        "num_inference_steps": 30,
        "negative_prompt": _NEGATIVE_PROMPT_CN,
    },
)

DREAMX_WORLD_5B_AR = InferencePreset(
    name="dreamx_world_5b_ar",
    version=1,
    model_family="dreamx_world",
    description="DreamX-World 5B autoregressive camera-control generation",
    workload_type="i2v",
    stage_schemas=(_DENOISE_STAGE, ),
    defaults={
        "height": 704,
        "width": 1280,
        "num_frames": 1005,
        "fps": 16,
        "guidance_scale": 1.0,
        "num_inference_steps": 4,
        "negative_prompt": _NEGATIVE_PROMPT_CN,
    },
)

ALL_PRESETS = (DREAMX_WORLD_5B_CAM, DREAMX_WORLD_5B_AR)
