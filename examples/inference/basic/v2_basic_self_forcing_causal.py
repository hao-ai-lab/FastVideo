"""v2 port of basic_self_forcing_causal.py — SF-causal Wan2.1 (CausalWanTransformer3DModel) through
the v2 VideoGenerator (chunk_rollout loop).

Same convenience API as upstream; only delta is importing VideoGenerator from v2. NOTE: the v2 causal
loop runs per-chunk few-step (not the upstream kv-cache streaming + SF schedule), so output is coherent
but lower-fidelity (a documented gap). num_frames is set by the card's chunk schedule; height/width
drive the latent geometry.
"""
from v2 import VideoGenerator
from fastvideo.api.sampling_param import SamplingParam

OUTPUT_PATH = "v2_video_samples_causal"


def main() -> None:
    model_name = "wlsaidhi/SFWan2.1-T2V-1.3B-Diffusers"
    generator = VideoGenerator.from_pretrained(
        model_name, num_gpus=1, text_encoder_cpu_offload=False, dit_cpu_offload=False)
    sampling_param = SamplingParam.from_pretrained(model_name)

    prompt = ("A curious raccoon peers through a vibrant field of yellow sunflowers, its eyes wide with "
              "interest. The playful yet serene atmosphere is complemented by soft natural light "
              "filtering through the petals. Mid-shot, warm and cheerful tones.")
    video = generator.generate_video(prompt, output_path=OUTPUT_PATH, output_video_name="causal_raccoon",
                                     save_video=True, sampling_param=sampling_param, height=480, width=832)
    print(f"Output: {video.video_path}")


if __name__ == "__main__":
    main()
