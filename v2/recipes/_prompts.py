"""Default negative prompts per model family — copied verbatim from fastvideo's pipeline presets
(`fastvideo/pipelines/basic/{wan,ltx2}/presets.py`). These are stable recipe DATA, not model code,
so v2 keeps its own copy rather than importing private symbols from fastvideo's pipelines package."""
from __future__ import annotations

# Wan family — English (Wan2.1 T2V) and Chinese (Wan2.2 / self-forcing) negative prompts.
WAN_NEG_EN = ("Bright tones, overexposed, static, blurred details, subtitles,"
              " style, works, paintings, images, static, overall gray, worst"
              " quality, low quality, JPEG compression residue, ugly,"
              " incomplete, extra fingers, poorly drawn hands, poorly drawn"
              " faces, deformed, disfigured, misshapen limbs, fused fingers,"
              " still picture, messy background, three legs, many people in"
              " the background, walking backwards")

WAN_NEG_CN = ("色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，"
              "静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，"
              "多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，"
              "形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，"
              "背景人很多，倒着走")

# Cosmos-Predict2 family negative prompt (verbatim from the Cosmos2 pipeline preset).
COSMOS_NEG = ("The video captures a series of frames showing ugly scenes, static with no motion, motion"
              " blur, over-saturation, shaky footage, low resolution, grainy texture, pixelated images,"
              " poorly lit areas, underexposed and overexposed scenes, poor color balance, washed out"
              " colors, choppy sequences, jerky movements, low frame rate, artifacting, color banding,"
              " unnatural transitions, outdated special effects, fake elements, unconvincing visuals,"
              " poorly edited content, jump cuts, visual noise, and flickering. Overall, the video is of"
              " poor quality.")

# LTX-2 family negative prompt (base / 2.3); the distilled few-step presets use "" instead.
LTX2_NEG = ("blurry, out of focus, overexposed, underexposed, low contrast, "
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
