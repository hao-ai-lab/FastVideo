# SPDX-License-Identifier: Apache-2.0
"""Wan2.2-Animate-14B: animate a character image with someone else's performance.

The reference image fixes who the character is; a preprocessed skeleton video
drives the body and a preprocessed face-crop video drives the expression. The
driving inputs are the artifacts the official preprocessing pipeline emits
(https://github.com/Wan-Video/Wan2.2, ``wan/modules/animate/preprocess``):
``src_pose.mp4``, ``src_face.mp4``, and the aligned reference ``src_ref.png``
(replace mode adds ``src_bg.mp4`` + ``src_mask.mp4`` and typically the
relighting LoRA -- see scripts/checkpoint_conversion/wan_animate_relight_lora.py).

CFG is off by default (guidance 1.0): the prompt is non-core for this model.
"""
from fastvideo import VideoGenerator

OUTPUT_PATH = "video_samples_wan_animate"

# Outputs of the official preprocessing run on your reference image + driving video.
REF_IMAGE_PATH = "preprocessed/src_ref.png"
POSE_VIDEO_PATH = "preprocessed/src_pose.mp4"
FACE_VIDEO_PATH = "preprocessed/src_face.mp4"


def main():
    generator = VideoGenerator.from_pretrained(
        "Wan-AI/Wan2.2-Animate-14B-Diffusers",
        num_gpus=1,
        use_fsdp_inference=False,
        dit_cpu_offload=True,
        vae_cpu_offload=False,
        text_encoder_cpu_offload=True,
        pin_cpu_memory=True,
    )

    generator.generate_video(
        "视频中的人在做动作",  # the official default prompt; non-core for this model
        image_path=REF_IMAGE_PATH,
        pose_video_path=POSE_VIDEO_PATH,
        face_video_path=FACE_VIDEO_PATH,
        output_path=OUTPUT_PATH,
        save_video=True,
        height=720,
        width=1280,
        num_frames=77,
        fps=30,
        guidance_scale=1.0,
        num_inference_steps=20,
    )


if __name__ == "__main__":
    main()
