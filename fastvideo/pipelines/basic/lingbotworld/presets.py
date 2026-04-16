# SPDX-License-Identifier: Apache-2.0
"""LingBotWorld model family pipeline presets."""
from fastvideo.api.presets import InferencePreset, PresetStageSpec

_DENOISE_STAGE = PresetStageSpec(
    name="denoise",
    kind="denoising",
    description="Dual-guidance denoising pass",
    allowed_overrides=frozenset({
        "num_inference_steps",
        "guidance_scale",
        "guidance_scale_2",
        "boundary_ratio",
    }),
)

LINGBOTWORLD_I2V = InferencePreset(
    name="lingbotworld_i2v",
    version=1,
    model_family="lingbotworld",
    description="LingBot-World I2V with dual guidance",
    workload_type="i2v",
    stage_schemas=(_DENOISE_STAGE, ),
    defaults={
        "guidance_scale":
        5.0,
        "guidance_scale_2":
        5.0,
        "num_inference_steps":
        70,
        "fps":
        16,
        "boundary_ratio":
        0.947,
        "negative_prompt": ("画面突变，色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，"
                            "最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，"
                            "畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走，"
                            "镜头晃动，画面闪烁，模糊，噪点，水印，签名，文字，变形，扭曲，液化，不合逻辑的结构，卡顿，"
                            "PPT幻灯片感，过暗，欠曝，低对比度，霓虹灯光感，过度锐化，3D渲染感，人物，行人，游客，身体，"
                            "皮肤，肢体，面部特征，汽车，电线"),
    },
)

ALL_PRESETS = (LINGBOTWORLD_I2V, )
