import torch
from diffusers import HunyuanVideoPipeline, HunyuanVideoTransformer3DModel, BitsAndBytesConfig
import imageio as iio 
import math
import numpy as np
import io
import time
import argparse
import os

def export_to_video_bytes(fps, frames):
    request = iio.core.Request("<bytes>", mode="w", extension=".mp4")
    pyavobject = iio.plugins.pyav.PyAVPlugin(request)
    if isinstance(frames, np.ndarray):
        frames = (np.array(frames) * 255).astype('uint8')
    else:
        frames = np.array(frames)
    new_bytes = pyavobject.write(frames, codec="libx264", fps=fps)
    out_bytes = io.BytesIO(new_bytes)
    return out_bytes

def export_to_video(frames, path, fps):
    video_bytes = export_to_video_bytes(fps, frames)
    video_bytes.seek(0)
    with open(path, "wb") as f:
        f.write(video_bytes.getbuffer())

def main(args):
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    prompt_template = {
        "template": (
            "<|start_header_cid|>system<|end_header_id|>\n\nDescribe the video by detailing the following aspects: "
            "1. The main content and theme of the video."
            "2. The color, shape, size, texture, quantity, text, and spatial relationships of the contents, including objects, people, and anything else."
            "3. Actions, events, behaviors temporal relationships, physical movement changes of the contents."
            "4. Background environment, light, style, atmosphere, and qualities."
            "5. Camera angles, movements, and transitions used in the video."
            "6. Thematic and aesthetic concepts associated with the scene, i.e. realistic, futuristic, fairy tale, etc<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|>"
        ),
        "crop_start": 95,
    }
    
    
    model_id = args.model_path

    if args.quantization == "nf4":
        quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4", llm_int8_skip_modules=["proj_out", "norm_out"])
        transformer = HunyuanVideoTransformer3DModel.from_pretrained(
            model_id, subfolder="transformer/" ,torch_dtype=torch.bfloat16, quantization_config=quantization_config
        )
    if args.quantization == "int8":
        quantization_config = BitsAndBytesConfig(load_in_8bit=True, llm_int8_skip_modules=["proj_out", "norm_out"])
        transformer = HunyuanVideoTransformer3DModel.from_pretrained(
            model_id, subfolder="transformer/" ,torch_dtype=torch.bfloat16, quantization_config=quantization_config
        )
    elif not args.quantization:
        transformer = HunyuanVideoTransformer3DModel.from_pretrained(
            model_id, subfolder="transformer/" ,torch_dtype=torch.bfloat16
        ).to(device)
    
    print("Max vram for read transofrmer:", round(torch.cuda.max_memory_allocated(device="cuda") / 1024 ** 3, 3), "GiB")
    torch.cuda.reset_max_memory_allocated(device)
    
    if not args.cpu_offload:
        pipe = HunyuanVideoPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16).to(device)
        pipe.transformer = transformer
    else:
        pipe = HunyuanVideoPipeline.from_pretrained(model_id, transformer=transformer, torch_dtype=torch.bfloat16)
    torch.cuda.reset_max_memory_allocated(device)
    pipe.scheduler._shift = args.flow_shift
    pipe.vae.enable_tiling()
    if args.cpu_offload:
        pipe.enable_model_cpu_offload()
    print("Max vram for init pipeline:", round(torch.cuda.max_memory_allocated(device="cuda") / 1024 ** 3, 3), "GiB")
    with open(args.prompt) as f:
        prompts = f.readlines()
    
    generator = torch.Generator("cpu").manual_seed(args.seed)
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    torch.cuda.reset_max_memory_allocated(device)
    for prompt in prompts:
        start_time = time.perf_counter()
        output = pipe(
            prompt=prompt,
            height = args.height,
            width = args.width,
            num_frames = args.num_frames,
            prompt_template=prompt_template,
            num_inference_steps = args.num_inference_steps,
            generator=generator,
        ).frames[0]
        export_to_video(output, os.path.join(args.output_path, f"{prompt[:100]}.mp4"), fps=args.fps)
        print("Time:", round(time.perf_counter() - start_time, 2), "seconds")
        print("Max vram for denoise:", round(torch.cuda.max_memory_allocated(device="cuda") / 1024 ** 3, 3), "GiB")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Basic parameters
    parser.add_argument("--prompt", type=str, help="prompt file for inference")
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--model_path", type=str, default="data/hunyuan")
    parser.add_argument("--output_path", type=str, default="./outputs/video")
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--quantization", type=str, default=None)
    parser.add_argument("--cpu_offload", action="store_true")
    # Additional parameters
    parser.add_argument(
        "--denoise-type",
        type=str,
        default="flow",
        help="Denoise type for noised inputs.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Seed for evaluation.")
    parser.add_argument(
        "--neg_prompt", type=str, default=None, help="Negative prompt for sampling."
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=1.0,
        help="Classifier free guidance scale.",
    )
    parser.add_argument(
        "--embedded_cfg_scale",
        type=float,
        default=6.0,
        help="Embedded classifier free guidance scale.",
    )
    parser.add_argument(
        "--flow_shift", type=int, default=7, help="Flow shift parameter."
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Batch size for inference."
    )
    parser.add_argument(
        "--num_videos",
        type=int,
        default=1,
        help="Number of videos to generate per prompt.",
    )
    parser.add_argument(
        "--load-key",
        type=str,
        default="module",
        help="Key to load the model states. 'module' for the main model, 'ema' for the EMA model.",
    )
    parser.add_argument(
        "--use-cpu-offload",
        action="store_true",
        help="Use CPU offload for the model load.",
    )
    parser.add_argument(
        "--dit-weight",
        type=str,
        default="data/hunyuan/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt",
    )
    parser.add_argument(
        "--reproduce",
        action="store_true",
        help="Enable reproducibility by setting random seeds and deterministic algorithms.",
    )
    parser.add_argument(
        "--disable-autocast",
        action="store_true",
        help="Disable autocast for denoising loop and vae decoding in pipeline sampling.",
    )

    # Flow Matching
    parser.add_argument(
        "--flow-reverse",
        action="store_true",
        help="If reverse, learning/sampling from t=1 -> t=0.",
    )
    parser.add_argument(
        "--flow-solver", type=str, default="euler", help="Solver for flow matching."
    )
    parser.add_argument(
        "--use-linear-quadratic-schedule",
        action="store_true",
        help="Use linear quadratic schedule for flow matching. Following MovieGen (https://ai.meta.com/static-resource/movie-gen-research-paper)",
    )
    parser.add_argument(
        "--linear-schedule-end",
        type=int,
        default=25,
        help="End step for linear quadratic schedule for flow matching.",
    )

    # Model parameters
    parser.add_argument("--model", type=str, default="HYVideo-T/2-cfgdistill")
    parser.add_argument("--latent-channels", type=int, default=16)
    parser.add_argument(
        "--precision", type=str, default="bf16", choices=["fp32", "fp16", "bf16", "fp8"]
    )
    parser.add_argument(
        "--rope-theta", type=int, default=256, help="Theta used in RoPE."
    )

    parser.add_argument("--vae", type=str, default="884-16c-hy")
    parser.add_argument(
        "--vae-precision", type=str, default="fp16", choices=["fp32", "fp16", "bf16"]
    )
    parser.add_argument("--vae-tiling", action="store_true", default=True)

    parser.add_argument("--text-encoder", type=str, default="llm")
    parser.add_argument(
        "--text-encoder-precision",
        type=str,
        default="fp16",
        choices=["fp32", "fp16", "bf16"],
    )
    parser.add_argument("--text-states-dim", type=int, default=4096)
    parser.add_argument("--text-len", type=int, default=256)
    parser.add_argument("--tokenizer", type=str, default="llm")
    parser.add_argument("--prompt-template", type=str, default="dit-llm-encode")
    parser.add_argument(
        "--prompt-template-video", type=str, default="dit-llm-encode-video"
    )
    parser.add_argument("--hidden-state-skip-layer", type=int, default=2)
    parser.add_argument("--apply-final-norm", action="store_true")

    parser.add_argument("--text-encoder-2", type=str, default="clipL")
    parser.add_argument(
        "--text-encoder-precision-2",
        type=str,
        default="fp16",
        choices=["fp32", "fp16", "bf16"],
    )
    parser.add_argument("--text-states-dim-2", type=int, default=768)
    parser.add_argument("--tokenizer-2", type=str, default="clipL")
    parser.add_argument("--text-len-2", type=int, default=77)

    args = parser.parse_args()
    main(args)