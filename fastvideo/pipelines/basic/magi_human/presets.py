# SPDX-License-Identifier: Apache-2.0
"""Presets for the daVinci-MagiHuman pipelines."""
from fastvideo.api.presets import InferencePreset, PresetStageSpec

# Keep this in sync with upstream MagiEvaluator.negative_prompt
# (daVinci-MagiHuman/inference/pipeline/video_generate.py:222-224): the
# video, audio-quality, and speech-delivery blocks all condition CFG.
_MAGI_HUMAN_NEGATIVE_PROMPT = ("Bright tones, overexposed, static, blurred details, subtitles, style, works, "
                               "paintings, images, static, overall gray, worst quality, low quality, JPEG "
                               "compression residue, ugly, incomplete, extra fingers, poorly drawn hands, "
                               "poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, "
                               "still picture, messy background, three legs, many people in the background, "
                               "walking backwards, low quality, worst quality, poor quality, noise, background "
                               "noise, hiss, hum, buzz, crackle, static, compression artifacts, MP3 artifacts, "
                               "digital clipping, distortion, muffled, muddy, unclear, echo, reverb, room echo, "
                               "over-reverberated, hollow sound, distant, washed out, harsh, shrill, piercing, "
                               "grating, tinny, thin sound, boomy, bass-heavy, flat EQ, over-compressed, "
                               "abrupt cut, jarring transition, sudden silence, looping artifact, music, "
                               "instrumental, sirens, alarms, crowd noise, unrelated sound effects, chaotic, "
                               "disorganized, messy, cheap sound, emotionless, flat delivery, deadpan, lifeless, "
                               "apathetic, robotic, mechanical, monotone, flat intonation, undynamic, boring, "
                               "reading from a script, AI voice, synthetic, text-to-speech, TTS, insincere, "
                               "fake emotion, exaggerated, overly dramatic, melodramatic, cheesy, cringey, "
                               "hesitant, unconfident, tired, weak voice, stuttering, stammering, mumbling, "
                               "slurred speech, mispronounced, bad articulation, lisp, vocal fry, creaky voice, "
                               "mouth clicks, lip smacks, wet mouth sounds, heavy breathing, audible inhales, "
                               "plosives, p-pops, coughing, clearing throat, sneezing, speaking too fast, rushed, "
                               "speaking too slow, dragged out, unnatural pauses, awkward silence, choppy, "
                               "disjointed, multiple speakers, two voices, background talking, out of tune, "
                               "off-key, autotune artifacts")

_DENOISE_STAGE = PresetStageSpec(
    name="denoise",
    kind="denoising",
    description="Joint video+audio UniPC flow-matching denoise pass.",
    allowed_overrides=frozenset({
        "num_inference_steps",
        "guidance_scale",
    }),
)

MAGI_HUMAN_BASE = InferencePreset(
    name="magi_human_base",
    version=1,
    model_family="magi_human",
    description=("daVinci-MagiHuman base text-to-AV at 256x480, 4s @ 25 fps. "
                 "Produces an mp4 with muxed audio + video. workload_type "
                 "is `t2v` because the framework enum has no `t2av` variant yet."),
    workload_type="t2v",
    stage_schemas=(_DENOISE_STAGE, ),
    defaults={
        "seed": 42,
        "height": 256,
        # Upstream pipeline.py:61-64 defaults br_width=480, br_height=272,
        # and video_generate.py:254-261 snaps height to 256 while width stays
        # 480, so the rendered default is 256x480.
        "width": 480,
        # num_frames is derived by the pipeline as `seconds*fps + 1`; we
        # surface it here for APIs that expect a concrete default.
        "num_frames": 101,
        "fps": 25,
        "guidance_scale": 5.0,  # used as video_txt_guidance_scale
        "num_inference_steps": 32,
        "negative_prompt": _MAGI_HUMAN_NEGATIVE_PROMPT,
    },
)

MAGI_HUMAN_DISTILL = InferencePreset(
    name="magi_human_distill",
    version=1,
    model_family="magi_human",
    description=("daVinci-MagiHuman DMD-2 distilled text-to-AV at 256x480, 4s @ "
                 "25 fps. 8-step inference, no classifier-free guidance. Produces "
                 "an mp4 with muxed audio + video."),
    workload_type="t2v",
    stage_schemas=(_DENOISE_STAGE, ),
    defaults={
        "seed": 42,
        "height": 256,
        "width": 480,
        "num_frames": 101,
        "fps": 25,
        # DMD: cfg=1 at the pipeline level. guidance_scale is kept at 1.0
        # for interop; the DenoisingStage ignores it when cfg_number=1.
        "guidance_scale": 1.0,
        "num_inference_steps": 8,
        "negative_prompt": _MAGI_HUMAN_NEGATIVE_PROMPT,
    },
)

MAGI_HUMAN_BASE_TI2V = InferencePreset(
    name="magi_human_base_ti2v",
    version=1,
    model_family="magi_human",
    description=("daVinci-MagiHuman base text+image-to-AV at 256x480, 4s @ 25 fps. "
                 "The reference image is VAE-encoded and pinned to the first "
                 "video latent frame at each denoise step."),
    workload_type="i2v",
    stage_schemas=(_DENOISE_STAGE, ),
    defaults={
        "seed": 42,
        "height": 256,
        "width": 480,
        "num_frames": 101,
        "fps": 25,
        "guidance_scale": 5.0,
        "num_inference_steps": 32,
        "negative_prompt": _MAGI_HUMAN_NEGATIVE_PROMPT,
    },
)

MAGI_HUMAN_DISTILL_TI2V = InferencePreset(
    name="magi_human_distill_ti2v",
    version=1,
    model_family="magi_human",
    description=("daVinci-MagiHuman DMD-2 distilled text+image-to-AV at 256x480, "
                 "4s @ 25 fps. 8-step inference, no CFG."),
    workload_type="i2v",
    stage_schemas=(_DENOISE_STAGE, ),
    defaults={
        "seed": 42,
        "height": 256,
        "width": 480,
        "num_frames": 101,
        "fps": 25,
        "guidance_scale": 1.0,
        "num_inference_steps": 8,
        "negative_prompt": _MAGI_HUMAN_NEGATIVE_PROMPT,
    },
)

MAGI_HUMAN_SR_540P = InferencePreset(
    name="magi_human_sr_540p",
    version=1,
    model_family="magi_human",
    description=("daVinci-MagiHuman two-stage base + SR-540p text-to-AV. "
                 "Base pass runs at 256x480; SR pass refines to upstream's "
                 "aligned 512x896 output with muxed audio."),
    workload_type="t2v",
    stage_schemas=(_DENOISE_STAGE, ),
    defaults={
        "seed": 42,
        "height": 256,
        "width": 480,
        "num_frames": 101,
        "fps": 25,
        "guidance_scale": 5.0,
        "num_inference_steps": 32,
        "negative_prompt": _MAGI_HUMAN_NEGATIVE_PROMPT,
    },
)

MAGI_HUMAN_SR_540P_TI2V = InferencePreset(
    name="magi_human_sr_540p_ti2v",
    version=1,
    model_family="magi_human",
    description=("daVinci-MagiHuman two-stage base + SR-540p text+image-to-AV. "
                 "The reference image is encoded at base resolution and then "
                 "re-encoded at SR resolution before the SR denoise pass."),
    workload_type="i2v",
    stage_schemas=(_DENOISE_STAGE, ),
    defaults={
        "seed": 42,
        "height": 256,
        "width": 480,
        "num_frames": 101,
        "fps": 25,
        "guidance_scale": 5.0,
        "num_inference_steps": 32,
        "negative_prompt": _MAGI_HUMAN_NEGATIVE_PROMPT,
    },
)

MAGI_HUMAN_SR_1080P = InferencePreset(
    name="magi_human_sr_1080p",
    version=1,
    model_family="magi_human",
    description=("daVinci-MagiHuman two-stage base + SR-1080p text-to-AV. "
                 "The SR DiT uses upstream local-window attention in 32 of "
                 "40 layers and refines to 1080p-class output."),
    workload_type="t2v",
    stage_schemas=(_DENOISE_STAGE, ),
    defaults={
        "seed": 42,
        "height": 256,
        "width": 480,
        "num_frames": 101,
        "fps": 25,
        "guidance_scale": 5.0,
        "num_inference_steps": 32,
        "negative_prompt": _MAGI_HUMAN_NEGATIVE_PROMPT,
    },
)

MAGI_HUMAN_SR_1080P_TI2V = InferencePreset(
    name="magi_human_sr_1080p_ti2v",
    version=1,
    model_family="magi_human",
    description=("daVinci-MagiHuman two-stage base + SR-1080p text+image-to-AV. "
                 "The SR DiT uses upstream local-window attention in 32 of "
                 "40 layers; the reference image is re-encoded at SR resolution."),
    workload_type="i2v",
    stage_schemas=(_DENOISE_STAGE, ),
    defaults={
        "seed": 42,
        "height": 256,
        "width": 480,
        "num_frames": 101,
        "fps": 25,
        "guidance_scale": 5.0,
        "num_inference_steps": 32,
        "negative_prompt": _MAGI_HUMAN_NEGATIVE_PROMPT,
    },
)

ALL_PRESETS = (
    MAGI_HUMAN_BASE,
    MAGI_HUMAN_DISTILL,
    MAGI_HUMAN_BASE_TI2V,
    MAGI_HUMAN_DISTILL_TI2V,
    MAGI_HUMAN_SR_540P,
    MAGI_HUMAN_SR_540P_TI2V,
    MAGI_HUMAN_SR_1080P,
    MAGI_HUMAN_SR_1080P_TI2V,
)
