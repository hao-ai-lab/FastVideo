# SPDX-License-Identifier: Apache-2.0
import json
import os

import pytest

from fastvideo import VideoGenerator
from fastvideo.v1.logger import init_logger
from fastvideo.v1.tests.utils import compute_video_ssim_torchvision, write_ssim_results
from fastvideo.v1.worker.multiproc_executor import MultiprocExecutor
from diffusers import AutoencoderKLWan, WanPipeline
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler

logger = init_logger(__name__)

# Base parameters for LoRA inference tests
WAN_LORA_PARAMS = {
    "num_gpus": 1,
    "model_path": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    "height": 480,
    "width": 832,
    "num_frames": 81,
    "num_inference_steps": 32,
    "guidance_scale": 5.0,
    "embedded_cfg_scale": 6,
    "flow_shift": 7.0,
    "seed": 1024,
    "sp_size": 2,
    "tp_size": 2,
    "vae_sp": True,
    "fps": 24,
    "neg_prompt": "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
    "text-encoder-precision": ("fp32",)
}

# LoRA configurations for testing
LORA_CONFIGS = [
    {
        "lora_path": "benjamin-paine/steamboat-willie-1.3b",
        "lora_nickname": "steamboat",
        "prompt": "steamboat willie style, golden era animation, close-up of a short fluffy monster kneeling beside a melting red candle. the mood is one of wonder and curiosity, as the monster gazes at the flame with wide eyes and open mouth. Its pose and expression convey a sense of innocence and playfulness, as if it is exploring the world around it for the first time. The use of warm colors and dramatic lighting further enhances the cozy atmosphere of the image.",
        "negative_prompt": "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
    },
    {
        "lora_path": "motimalu/wan-flat-color-1.3b-v2",
        "lora_nickname": "flat_color",
        "prompt": "flat color, no lineart, blending, negative space, artist:[john kafka|ponsuke kaikai|hara id 21|yoneyama mai|fuzichoco], 1girl, sakura miko, pink hair, cowboy shot, white shirt, floral print, off shoulder, outdoors, cherry blossom, tree shade, wariza, looking up, falling petals, half-closed eyes, white sky, clouds, live2d animation, upper body, high quality cinematic video of a woman sitting under a sakura tree. Dreamy and lonely, the camera close-ups on the face of the woman as she turns towards the viewer. The Camera is steady, This is a cowboy shot. The animation is smooth and fluid.",
        "negative_prompt": "bad quality video,色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
    }
]

MODEL_TO_PARAMS = {
    "Wan2.1-T2V-1.3B-Diffusers": WAN_LORA_PARAMS,
}

LORA_SWITCH_CONFIGS = [
    {
        "lora_path": "motimalu/wan-flat-color-1.3b-v2",
        "lora_nickname": "flat_color",
        "prompt": "flat color, no lineart, blending, negative space, artist:[john kafka|ponsuke kaikai|hara id 21|yoneyama mai|fuzichoco], 1girl, sakura miko, pink hair, cowboy shot, white shirt, floral print, off shoulder, outdoors, cherry blossom, tree shade, wariza, looking up, falling petals, half-closed eyes, white sky, clouds, live2d animation, upper body, high quality cinematic video of a woman sitting under a sakura tree. Dreamy and lonely, the camera close-ups on the face of the woman as she turns towards the viewer. The Camera is steady, This is a cowboy shot. The animation is smooth and fluid.",
        "negative_prompt": "bad quality video,色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
    },
    {
        "lora_path": "benjamin-paine/steamboat-willie-1.3b",
        "lora_nickname": "steamboat",
        "prompt": "steamboat willie style, golden era animation, close-up of a short fluffy monster kneeling beside a melting red candle. the mood is one of wonder and curiosity, as the monster gazes at the flame with wide eyes and open mouth. Its pose and expression convey a sense of innocence and playfulness, as if it is exploring the world around it for the first time. The use of warm colors and dramatic lighting further enhances the cozy atmosphere of the image.",
        "negative_prompt": "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
    }
]

@pytest.mark.parametrize("lora_config", LORA_CONFIGS)
@pytest.mark.parametrize("ATTENTION_BACKEND", ["TORCH_SDPA"])
@pytest.mark.parametrize("model_id", list(MODEL_TO_PARAMS.keys()))
def test_lora_switching_similarity(lora_config, ATTENTION_BACKEND, model_id):
    """
    Test that runs LoRA inference with LoRA switching and compares the output
    to reference videos using SSIM.
    """
    os.environ["FASTVIDEO_ATTENTION_BACKEND"] = ATTENTION_BACKEND

    script_dir = os.path.dirname(os.path.abspath(__file__))

    base_output_dir = os.path.join(script_dir, 'generated_videos', model_id)
    output_dir = os.path.join(base_output_dir, ATTENTION_BACKEND, 'lora_switching')
    output_video_name = f"lora_{lora_config['lora_path']}_{lora_config['prompt'][:50]}.mp4"

    os.makedirs(output_dir, exist_ok=True)

    BASE_PARAMS = MODEL_TO_PARAMS[model_id]
    num_inference_steps = BASE_PARAMS["num_inference_steps"]
    prompt = lora_config["prompt"]
    negative_prompt = lora_config["negative_prompt"]
    lora_path = lora_config["lora_path"]
    lora_nickname = lora_config["lora_nickname"]

    init_kwargs = {
        "num_gpus": BASE_PARAMS["num_gpus"],
        "flow_shift": BASE_PARAMS["flow_shift"],
        "sp_size": BASE_PARAMS["sp_size"],
        "tp_size": BASE_PARAMS["tp_size"],
        "lora_path": lora_path,
        "lora_nickname": lora_nickname,
        "use_cpu_offload": True,
    }
    if BASE_PARAMS.get("vae_sp"):
        init_kwargs["vae_sp"] = True
        init_kwargs["vae_tiling"] = True
    if "text-encoder-precision" in BASE_PARAMS:
        init_kwargs["text_encoder_precisions"] = BASE_PARAMS["text-encoder-precision"]

    generation_kwargs = {
        "num_inference_steps": num_inference_steps,
        "output_path": output_dir,
        "height": BASE_PARAMS["height"],
        "width": BASE_PARAMS["width"],
        "num_frames": BASE_PARAMS["num_frames"],
        "guidance_scale": BASE_PARAMS["guidance_scale"],
        "embedded_cfg_scale": BASE_PARAMS["embedded_cfg_scale"],
        "seed": BASE_PARAMS["seed"],
        "fps": BASE_PARAMS["fps"],
        "negative_prompt": negative_prompt,
        "save_video": True,
    }
    generator = VideoGenerator.from_pretrained(model_path=BASE_PARAMS["model_path"], **init_kwargs)
    for lora_config in LORA_SWITCH_CONFIGS:
        lora_nickname = lora_config["lora_nickname"]
        lora_path = lora_config["lora_path"]
        prompt = lora_config["prompt"]
        negative_prompt = lora_config["negative_prompt"]

        generator.set_lora_adapter(lora_nickname=lora_nickname, lora_path=lora_path)
        generation_kwargs["negative_prompt"] = negative_prompt
        output_video_name_switch = f"lora_switch_{lora_nickname}_{prompt[:50]}.mp4"
        generation_kwargs["output_path"] = output_dir
        
        generator.generate_video(prompt, **generation_kwargs)

        if isinstance(generator.executor, MultiprocExecutor):
            generator.executor.shutdown()

        assert os.path.exists(
            output_dir), f"Output video was not generated at {output_dir}"

        reference_folder = os.path.join(script_dir, 'reference_videos', model_id, ATTENTION_BACKEND, 'lora_switching')
        
        if not os.path.exists(reference_folder):
            logger.error("Reference folder missing")
            raise FileNotFoundError(
                f"Reference video folder does not exist: {reference_folder}")

        # Find the matching reference video for the switched LoRA
        reference_video_name = None

        for filename in os.listdir(reference_folder):
            if filename.endswith('.mp4') and lora_nickname in filename and prompt[:50] in filename:
                reference_video_name = filename
                break

        if not reference_video_name:
            logger.error(f"Reference video not found for adapter: {lora_path} with prompt: {prompt[:50]} and backend: {ATTENTION_BACKEND}")
            raise FileNotFoundError(f"Reference video missing for adapter {lora_path}")

        reference_video_path = os.path.join(reference_folder, reference_video_name)
        generated_video_path = os.path.join(output_dir, output_video_name_switch)

        logger.info(
            f"Computing SSIM between {reference_video_path} and {generated_video_path}"
        )
        ssim_values = compute_video_ssim_torchvision(reference_video_path,
                                                    generated_video_path,
                                                    use_ms_ssim=True)

        mean_ssim = ssim_values[0]
        logger.info(f"SSIM mean value: {mean_ssim}")
        logger.info(f"Writing SSIM results to directory: {output_dir}")

        success = write_ssim_results(output_dir, ssim_values, reference_video_path,
                                    generated_video_path, num_inference_steps,
                                    prompt, lora_nickname)

        if not success:
            logger.error("Failed to write SSIM results to file")

        min_acceptable_ssim = 0.95
        assert mean_ssim >= min_acceptable_ssim, f"SSIM value {mean_ssim} is below threshold {min_acceptable_ssim} for adapter {lora_config['lora_path']}" 
