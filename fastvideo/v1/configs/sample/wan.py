# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field

from fastvideo.v1.configs.sample.base import SamplingParam
from fastvideo.v1.configs.sample.teacache import WanTeaCacheParams


@dataclass
class WanT2V_1_3B_SamplingParam(SamplingParam):
    # Video parameters
    height: int = 480
    width: int = 832
    num_frames: int = 81
    fps: int = 16

    # Denoising stage
    guidance_scale: float = 3.0
    negative_prompt: str = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"
    num_inference_steps: int = 50

    teacache_params: WanTeaCacheParams = field(
        default_factory=lambda: WanTeaCacheParams(
            teacache_thresh=0.08,
            ret_steps_coeffs=[
                -5.21862437e+04, 9.23041404e+03, -5.28275948e+02,
                1.36987616e+01, -4.99875664e-02
            ],
            non_ret_steps_coeffs=[
                2.39676752e+03, -1.31110545e+03, 2.01331979e+02,
                -8.29855975e+00, 1.37887774e-01
            ]))


@dataclass
class WanT2V_14B_SamplingParam(SamplingParam):
    # Video parameters
    height: int = 720
    width: int = 1280
    num_frames: int = 81
    fps: int = 16

    # Denoising stage
    guidance_scale: float = 5.0
    negative_prompt: str = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"
    num_inference_steps: int = 50

    teacache_params: WanTeaCacheParams = field(
        default_factory=lambda: WanTeaCacheParams(
            teacache_thresh=0.20,
            use_ret_steps=False,
            ret_steps_coeffs=[
                -3.03318725e+05, 4.90537029e+04, -2.65530556e+03,
                5.87365115e+01, -3.15583525e-01
            ],
            non_ret_steps_coeffs=[
                -5784.54975374, 5449.50911966, -1811.16591783, 256.27178429,
                -13.02252404
            ]))


@dataclass
class WanI2V_14B_480P_SamplingParam(WanT2V_1_3B_SamplingParam):
    # Denoising stage
    guidance_scale: float = 5.0
    num_inference_steps: int = 40

    teacache_params: WanTeaCacheParams = field(
        default_factory=lambda: WanTeaCacheParams(
            teacache_thresh=0.26,
            ret_steps_coeffs=[
                -3.03318725e+05, 4.90537029e+04, -2.65530556e+03,
                5.87365115e+01, -3.15583525e-01
            ],
            non_ret_steps_coeffs=[
                -5784.54975374, 5449.50911966, -1811.16591783, 256.27178429,
                -13.02252404
            ]))


@dataclass
class WanI2V_14B_720P_SamplingParam(WanT2V_14B_SamplingParam):
    # Denoising stage
    guidance_scale: float = 5.0
    num_inference_steps: int = 40

    teacache_params: WanTeaCacheParams = field(
        default_factory=lambda: WanTeaCacheParams(
            teacache_thresh=0.3,
            ret_steps_coeffs=[
                -3.03318725e+05, 4.90537029e+04, -2.65530556e+03,
                5.87365115e+01, -3.15583525e-01
            ],
            non_ret_steps_coeffs=[
                -5784.54975374, 5449.50911966, -1811.16591783, 256.27178429,
                -13.02252404
            ]))
