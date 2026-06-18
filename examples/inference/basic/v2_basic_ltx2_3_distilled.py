"""v2 port of basic_ltx2_3_distilled.py — LTX-2.3 Distilled (single-stage, joint A/V) through the v2
VideoGenerator.

Unlike LTX-2.0 distilled (two-stage, video-only), LTX-2.3 is a single-stage *audio+video* model. The
shared registry (v2/registry.py) maps ``FastVideo/LTX-2.3-Distilled-Diffusers`` to its OWN card,
``build_ltx2_3_card`` — distinct from the LTX-2 base/2-stage cards — which wires the 2.3-specific path:
  * SEPARATE video + audio text connectors (the Gemma encoder projects the prompt to two embeddings,
    2048-dim for audio, 4096-dim for video) plus gated attention;
  * a JOINT DiT forward where video and audio latents cross-attend in a single denoise per step;
  * a video VAE decode + an AudioDecoder→Vocoder decode → video frames AND a stereo waveform @24kHz.

Because the model advertises TEXT_TO_VIDEO_SOUND, the VideoGenerator issues a T2VS request by default,
so ``generate_video`` returns BOTH modalities: the mp4 plus a sibling ``.wav`` (and ``result.audio`` /
``result.audio_sample_rate`` in memory). Being distilled, it wants FEW steps (8). GPU-verified on the
rebuilt x86 stack: video (3,33,256,384) + stereo audio (2×61920 @ 24kHz).
"""
from v2 import VideoGenerator

PROMPT = "ocean waves crashing on rocks at sunset, seagulls calling in the distance, cinematic, highly detailed"


def main() -> None:
    generator = VideoGenerator.from_pretrained("FastVideo/LTX-2.3-Distilled-Diffusers", num_gpus=1)
    # audio=None auto-enables sound for this A/V model (pass audio=False to force video-only).
    result = generator.generate_video(
        prompt=PROMPT, output_path="v2_video_samples_ltx2_3", output_video_name="ltx2_3_ocean",
        save_video=True, num_frames=33, height=512, width=768, num_inference_steps=8, seed=1)
    print(f"Video: {result.video_path}")
    audio_path = result.extra.get("audio_path")
    if audio_path:
        print(f"Audio: {audio_path}  ({result.audio_sample_rate} Hz)")
    generator.shutdown()


if __name__ == "__main__":
    main()
