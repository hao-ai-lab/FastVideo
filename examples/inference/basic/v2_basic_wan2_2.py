"""v2 port of basic_wan2_2.py — Wan2.2-T2V-A14B (MoE) through the v2 VideoGenerator.

Same convenience API as upstream; only delta is importing VideoGenerator from v2. A14B is a 2-expert
MoE: WanTransformer3DModel x2 (in_ch=16, Wan2.1 geometry) with a boundary-timestep switch
(boundary_ratio 0.875) — ported via build_wan22_a14b_card (BoundaryTimestepRouting: transformer =
high-noise expert, transformer_2 = low-noise), reusing the Wan adapters for both experts.

NOTE: upstream runs A14B with num_gpus=2 + dit_cpu_offload=True ("DiT need to be offloaded for MoE").
The v2 bring-up is single-GPU + resident (no offload), so the two 14B experts (~56GB bf16) + UMT5 are
near an 80GB GPU's limit — this example uses reduced res/frames to fit. If it OOMs, the A14B card is
still correct; it just needs the (not-yet-ported) MoE DiT CPU offload. See V2_PORTING_STATUS.md.
"""
from v2 import VideoGenerator

OUTPUT_PATH = "v2_video_samples_wan2_2_14B_t2v"


def main() -> None:
    generator = VideoGenerator.from_pretrained(
        "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
        num_gpus=1,
        use_fsdp_inference=False,
        dit_cpu_offload=False,
        vae_cpu_offload=False,
        text_encoder_cpu_offload=False,
        pin_cpu_memory=False,
    )

    prompt = ("A majestic lion strides across the golden savanna, its powerful frame glistening under "
              "the warm afternoon sun. The tall grass ripples gently in the breeze. Low angle, steady "
              "tracking shot, cinematic.")
    # Reduced res/frames so the two resident 14B experts fit a single 80GB GPU (upstream: 720x1280x81).
    video = generator.generate_video(prompt, output_path=OUTPUT_PATH, output_video_name="wan22_a14b_lion",
                                     save_video=True, num_frames=17, height=480, width=832,
                                     num_inference_steps=20, guidance_scale=5.0)
    print(f"Output: {video.video_path}")


if __name__ == "__main__":
    main()
