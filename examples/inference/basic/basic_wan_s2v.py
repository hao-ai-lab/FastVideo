# SPDX-License-Identifier: Apache-2.0
"""Wan2.2-S2V-14B: animate a reference image in sync with a speech track.

The audio drives the motion -- lip sync, head movement, and gesture rhythm all
come from the waveform, while the reference image fixes who the subject is and
the prompt sets the scene and camera.
"""
from fastvideo import VideoGenerator

OUTPUT_PATH = "video_samples_wan_s2v"


def main():
    generator = VideoGenerator.from_pretrained(
        "FastVideo/Wan2.2-S2V-14B-Diffusers",
        num_gpus=1,
        use_fsdp_inference=False,
        dit_cpu_offload=True,
        vae_cpu_offload=False,
        text_encoder_cpu_offload=True,
        pin_cpu_memory=True,
    )

    prompt = ("A woman in a recording studio singing into a condenser microphone, "
              "warm key light from the left, shallow depth of field, medium close-up, "
              "subtle head movement in time with the music.")

    generator.generate_video(
        prompt,
        image_path="https://raw.githubusercontent.com/Wan-Video/Wan2.2/main/examples/i2v_input.JPG",
        audio_path="https://raw.githubusercontent.com/Wan-Video/Wan2.2/main/examples/talk.wav",
        output_path=OUTPUT_PATH,
        save_video=True,
        height=480,
        width=832,
        num_frames=80,
        fps=16,
        guidance_scale=5.0,
        num_inference_steps=40,
    )


if __name__ == "__main__":
    main()
