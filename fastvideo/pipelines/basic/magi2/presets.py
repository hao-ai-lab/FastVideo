# SPDX-License-Identifier: Apache-2.0
"""Inference preset for the published MAGI-2 Preview 1080p profile."""

from fastvideo.api.presets import InferencePreset, PresetStageSpec

MAGI2_NEGATIVE_PROMPT = (
    "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, "
    "overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly "
    "drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy "
    "background, three legs, many people in the background, walking backwards, low quality, worst quality, poor "
    "quality, noise, background noise, hiss, hum, buzz, crackle, static, compression artifacts, MP3 artifacts, "
    "digital clipping, distortion, muffled, muddy, unclear, echo, reverb, room echo, over-reverberated, hollow "
    "sound, distant, washed out, harsh, shrill, piercing, grating, tinny, thin sound, boomy, bass-heavy, flat EQ, "
    "over-compressed, abrupt cut, jarring transition, sudden silence, looping artifact, music, instrumental, "
    "sirens, alarms, crowd noise, unrelated sound effects, chaotic, disorganized, messy, cheap sound, emotionless, "
    "flat delivery, deadpan, lifeless, apathetic, robotic, mechanical, monotone, flat intonation, undynamic, boring, "
    "reading from a script, AI voice, synthetic, text-to-speech, TTS, insincere, fake emotion, exaggerated, overly "
    "dramatic, melodramatic, cheesy, cringey, hesitant, unconfident, tired, weak voice, stuttering, stammering, "
    "mumbling, slurred speech, mispronounced, bad articulation, lisp, vocal fry, creaky voice, mouth clicks, lip "
    "smacks, wet mouth sounds, heavy breathing, audible inhales, plosives, p-pops, coughing, clearing throat, "
    "sneezing, speaking too fast, rushed, speaking too slow, dragged out, unnatural pauses, awkward silence, choppy, "
    "disconnected, multiple speakers, two voices, background talking, out of tune, off-key, autotune artifacts")

_PREVIEW_STAGE = PresetStageSpec(
    name="preview",
    kind="denoising",
    description="Joint video and audio preview denoising pass.",
    allowed_overrides=frozenset({"num_inference_steps"}),
)

_REFINER_STAGE = PresetStageSpec(
    name="refiner",
    kind="refinement",
    description="Spatial-temporal 1080p latent refinement pass.",
    allowed_overrides=frozenset({"num_inference_steps_sr"}),
)

MAGI2_PREVIEW_1080P = InferencePreset(
    name="magi2_preview_1080p",
    version=1,
    model_family="magi2",
    description="MAGI-2 Preview text- or image-conditioned 10-second 1080p video with stereo audio.",
    workload_type=None,
    stage_schemas=(_PREVIEW_STAGE, _REFINER_STAGE),
    defaults={
        "seed": 42,
        "height": 1088,
        "width": 1920,
        "num_frames": 249,
        "fps": 25,
        "num_inference_steps": 100,
        "num_inference_steps_sr": 5,
        "guidance_scale": 5.0,
        "negative_prompt": MAGI2_NEGATIVE_PROMPT,
    },
)

ALL_PRESETS = (MAGI2_PREVIEW_1080P, )

__all__ = ["ALL_PRESETS", "MAGI2_NEGATIVE_PROMPT", "MAGI2_PREVIEW_1080P"]
