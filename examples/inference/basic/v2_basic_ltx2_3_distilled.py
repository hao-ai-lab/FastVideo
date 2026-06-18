"""v2 port of basic_ltx2_3_distilled.py — LTX-2.3 Distilled (single-stage) through the v2 VideoGenerator.

LTX-2.3-Distilled is single-stage (no spatial upsampler, unlike LTX-2.0 distilled's two-stage), so the
architecture-driven dispatch (v2/video_generator.py:_select_builders, via has_spatial_upsampler=False)
routes it to the SAME single-stage card as LTX-2 base (build_ltx2_base_card). Being distilled, it wants
FEW steps — pass a small num_inference_steps for the few-step schedule.

NOTE: the v2 single-stage loop currently builds a request-driven *linspace* flow-match schedule
(base_flow_sigmas); the distilled-tuned few-step schedule is a follow-up. GPU re-verify of this example
is pending the env (the box was rescheduled aarch64->x86 mid-session — see V2_PORTING_STATUS.md).
"""
from v2 import VideoGenerator

PROMPT = "a hummingbird hovering beside a red flower, wings blurred, macro, cinematic, highly detailed"


def main() -> None:
    generator = VideoGenerator.from_pretrained("FastVideo/LTX-2.3-Distilled-Diffusers", num_gpus=1)
    video = generator.generate_video(
        prompt=PROMPT, output_path="v2_video_samples_ltx2_3", output_video_name="ltx2_3_hummingbird",
        save_video=True, num_frames=9, height=512, width=768, num_inference_steps=8)
    print(f"Output: {video.video_path}")
    generator.shutdown()


if __name__ == "__main__":
    main()
