
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


# [[[137 134 130]
#   [143 140 137]
#   [165 161 161]
#   [211 207 208]]

#  [[135 135 130]
#   [145 143 141]
#   [172 170 169]
#   [211 211 211]]

#  [[129 128 128]
#   [143 142 143]
#   [167 164 165]
#   [207 206 206]]

#  [[126 125 127]
#   [134 133 134]
#   [150 146 148]
#   [181 178 179]]]
# INFO 02-15 08:23:10 [video_generator.py:363] 
# INFO 02-15 08:23:10 [video_generator.py:363]                       height: 1024
# INFO 02-15 08:23:10 [video_generator.py:363]                        width: 1024
# INFO 02-15 08:23:10 [video_generator.py:363]                 video_length: 1
# INFO 02-15 08:23:10 [video_generator.py:363]                       prompt: A majestic lion strides across the golden savanna, its powerful frame glistening under the warm afternoon sun. The tall grass ripples gently in the breeze, enhancing the lion's commanding presence. The tone is vibrant, embodying the raw energy of the wild. Low angle, steady tracking shot, cinematic.
# INFO 02-15 08:23:10 [video_generator.py:363]                       image_path: None
# INFO 02-15 08:23:10 [video_generator.py:363]                   neg_prompt: 
# INFO 02-15 08:23:10 [video_generator.py:363]                         seed: 1024
# INFO 02-15 08:23:10 [video_generator.py:363]                  infer_steps: 28
# INFO 02-15 08:23:10 [video_generator.py:363]        num_videos_per_prompt: 1
# INFO 02-15 08:23:10 [video_generator.py:363]               guidance_scale: 3.5
# INFO 02-15 08:23:10 [video_generator.py:363]                     n_tokens: 16384
# INFO 02-15 08:23:10 [video_generator.py:363]                   flow_shift: 3.0
# INFO 02-15 08:23:10 [video_generator.py:363]      embedded_guidance_scale: 0.0035
# INFO 02-15 08:23:10 [video_generator.py:363]                   save_video: True
# INFO 02-15 08:23:10 [video_generator.py:363]                   output_path: video_samples/A majestic lion strides across the golden savanna, its powerful frame glistening under the warm afte.mp4
# INFO 02-15 08:23:10 [video_generator.py:363]         
# (Worker pid=2517) INFO 02-15 08:23:10 [composed_pipeline_base.py:417] Running pipeline stages: dict_keys(['input_validation_stage', 'prompt_encoding_stage_primary', 'conditioning_stage', 'timestep_preparation_stage', 'latent_preparation_stage', 'denoising_stage', 'decoding_stage'])
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████| 28/28 [00:56<00:00,  2.03s/it]
# (Worker pid=2517) INFO 02-15 08:24:09 [multiproc_executor.py:627] Worker 0 starting event loop...
# INFO 02-15 08:24:09 [video_generator.py:380] Generated successfully in 58.69 seconds
# INFO 02-15 08:24:09 [video_generator.py:393] Saved video to video_samples/A majestic lion strides across the golden savanna, its powerful frame glistening under the warm afte.mp4
# [prompt2] frame0 shape=(1024, 1024, 3) dtype=uint8 min=0 max=255 mean=122.06377951304118
# [prompt2] frame0 slice (H4xW4xC3):
# [[[144 142 137]
#   [152 150 145]
#   [173 170 168]
#   [213 212 211]]

#  [[144 144 138]
#   [155 154 150]
#   [181 180 176]
#   [216 217 214]]

#  [[137 137 134]
#   [152 151 149]
#   [175 172 171]
#   [212 211 209]]

#  [[131 131 129]
#   [140 139 137]
#   [158 155 153]
#   [188 187 184]]]