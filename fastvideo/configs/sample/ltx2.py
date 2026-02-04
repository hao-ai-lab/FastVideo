# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass

from fastvideo.configs.sample.base import SamplingParam


@dataclass
class LTX2BaseSamplingParam(SamplingParam):
    """Default sampling parameters for LTX-2 base one-stage T2V.

    Values follow the official LTX-2 one-stage defaults.
    """

    seed: int = 10
    num_frames: int = 121
    height: int = 512
    width: int = 768
    fps: int = 24
    num_inference_steps: int = 40
    guidance_scale: float = 3.0
    # Copied/following official LTX-2 DEFAULT_NEGATIVE_PROMPT.
    negative_prompt: str = (
        "blurry, out of focus, overexposed, underexposed, low contrast, "
        "washed out colors, excessive noise, grainy texture, poor lighting, "
        "flickering, motion blur, distorted proportions, unnatural skin "
        "tones, deformed facial features, asymmetrical face, missing facial "
        "features, extra limbs, disfigured hands, wrong hand count, "
        "artifacts around text, inconsistent perspective, camera shake, "
        "incorrect depth of field, background too sharp, background clutter, "
        "distracting reflections, harsh shadows, inconsistent lighting "
        "direction, color banding, cartoonish rendering, 3D CGI look, "
        "unrealistic materials, uncanny valley effect, incorrect ethnicity, "
        "wrong gender, exaggerated expressions, wrong gaze direction, "
        "mismatched lip sync, silent or muted audio, distorted voice, "
        "robotic voice, echo, background noise, off-sync audio, incorrect "
        "dialogue, added dialogue, repetitive speech, jittery movement, "
        "awkward pauses, incorrect timing, unnatural transitions, "
        "inconsistent framing, tilted camera, flat lighting, inconsistent "
        "tone, cinematic oversaturation, stylized filters, or AI artifacts.")


@dataclass
class LTX2DistilledSamplingParam(SamplingParam):
    """Default sampling parameters for LTX-2 distilled one-stage T2V."""

    seed: int = 10
    num_frames: int = 121
    height: int = 1024
    width: int = 1536
    fps: int = 24
    num_inference_steps: int = 8
    guidance_scale: float = 1.0
    # No default negative_prompt for distilled models
    negative_prompt: str = ""


# Backward compatibility alias.
LTX2SamplingParam = LTX2DistilledSamplingParam
