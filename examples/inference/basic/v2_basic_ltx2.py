"""v2 port of basic_ltx2.py — LTX-2 base (single-stage) through the v2 VideoGenerator.

Same convenience API as upstream; only delta is importing VideoGenerator from v2. LTX-2 base is the
single-stage (non-distilled) model: the v2 single-stage card (build_ltx2_base_card) runs a request-driven
many-step flow-match at FULL latent res (no distilled base/refine split, no spatial upsampler), reusing
the LTX-2 DiT/VAE/Gemma adapters. The SAME single-stage card also serves LTX-2.3-Distilled (which is also
single-stage) — just pass fewer num_inference_steps for the few-step distilled schedule.

NOTE: modest res/frames here — upstream defaults to 1088x1920x121, which on an 18.88B base is very slow;
raise them for full quality. v2 bring-up: single-GPU, resident, SDPA.
"""
from v2 import VideoGenerator

PROMPT = ("A warm sunny backyard, cinematic close-up of two people talking; the camera slowly pans right "
          "to reveal a grandfather in the garden wearing enormous butterfly wings, flapping his arms like "
          "he is trying to take off. Deadpan, absurd, quietly tragic.")


def main() -> None:
    generator = VideoGenerator.from_pretrained("Davids048/LTX2-Base-Diffusers", num_gpus=1)
    video = generator.generate_video(
        prompt=PROMPT, output_path="v2_video_samples_ltx2_base", output_video_name="ltx2_base_backyard",
        save_video=True, num_frames=25, height=512, width=768, num_inference_steps=30)
    print(f"Output: {video.video_path}")
    generator.shutdown()


if __name__ == "__main__":
    main()
