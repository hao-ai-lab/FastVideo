
import numpy as np

from fastvideo import VideoGenerator

# from fastvideo.configs.sample import SamplingParam

OUTPUT_PATH = "video_samples"


def _print_frame_matrix(frames, label: str) -> None:
    if not frames:
        print(f"[{label}] No frames returned")
        return
    frame0 = frames[0]
    if isinstance(frame0, np.ndarray):
        arr = frame0
    else:
        arr = np.array(frame0)

    print(
        f"[{label}] frame0 shape={arr.shape} dtype={arr.dtype} min={arr.min()} max={arr.max()} mean={arr.mean()}"
    )

    if arr.ndim >= 2:
        h = min(4, arr.shape[0])
        w = min(4, arr.shape[1])
        if arr.ndim == 3:
            c = min(3, arr.shape[2])
            print(f"[{label}] frame0 slice (H{h}xW{w}xC{c}):\n{arr[:h, :w, :c]}")
        else:
            print(f"[{label}] frame0 slice (H{h}xW{w}):\n{arr[:h, :w]}")
def main():
    # FastVideo will automatically use the optimal default arguments for the
    # model.
    # If a local path is provided, FastVideo will make a best effort
    # attempt to identify the optimal arguments.
    generator = VideoGenerator.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        # FastVideo will automatically handle distributed setup
        num_gpus=1,
        use_fsdp_inference=False, # set to True if GPU is out of memory
        dit_cpu_offload=False,
        vae_cpu_offload=False,
        text_encoder_cpu_offload=True,
        pin_cpu_memory=True, # set to false if low CPU RAM or hit obscure "CUDA error: Invalid argument"
        # image_encoder_cpu_offload=False,
    )

    prompt = (
    "A cinematic portrait of a fox, 35mm film, soft light, gentle grain."
)
    video = generator.generate_video(
        prompt,
        output_path=OUTPUT_PATH,
        save_video=True,
        return_frames=True,
    )
    if isinstance(video, dict):
        frames = video.get("frames", [])
    else:
        frames = video
    _print_frame_matrix(frames, "prompt1")
    # video = generator.generate_video(prompt, sampling_param=sampling_param, output_path="wan_t2v_videos/")

    # Generate another video with a different prompt, without reloading the
    # model!
    prompt2 = (
        "A majestic lion strides across the golden savanna, its powerful frame "
        "glistening under the warm afternoon sun. The tall grass ripples gently in "
        "the breeze, enhancing the lion's commanding presence. The tone is vibrant, "
        "embodying the raw energy of the wild. Low angle, steady tracking shot, "
        "cinematic.")
    video2 = generator.generate_video(
        prompt2,
        output_path=OUTPUT_PATH,
        save_video=True,
        return_frames=True,
    )
    if isinstance(video2, dict):
        frames2 = video2.get("frames", [])
    else:
        frames2 = video2
    _print_frame_matrix(frames2, "prompt2")


if __name__ == "__main__":
    main()
