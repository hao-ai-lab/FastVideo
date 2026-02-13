# SPDX-License-Identifier: Apache-2.0
import os
import re

import pytest

from fastvideo import VideoGenerator
from fastvideo.logger import init_logger
from fastvideo.tests.utils import compute_video_ssim_torchvision, write_ssim_results
from diffusers import DiffusionPipeline
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.pipelines import build_pipeline
from fastvideo.models.loader.utils import hf_to_custom_state_dict, get_param_names_mapping
from torch.testing import assert_close
from torch.distributed.tensor import DTensor
import torch
logger = init_logger(__name__)
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "29500"

# Base parameters for LoRA inference tests
WAN_LORA_PARAMS = {
    "num_gpus": 1,
    "model_path": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    "height": 480,
    "width": 832,
    "num_frames": 45,
    "num_inference_steps": 32,
    "guidance_scale": 5.0,
    "flow_shift": 3.0,
    "seed": 42,
    "fps": 24,
    "neg_prompt": "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
    "text-encoder-precision": ("fp32",),
    "dit_cpu_offload": True,
}

# LoRA configurations for testing
LORA_CONFIGS = [
    {
        "lora_path": "benjamin-paine/steamboat-willie-1.3b",
        "lora_nickname": "steamboat",
        "prompt": "steamboat willie style, golden era animation, close-up of a short fluffy monster kneeling beside a melting red candle. the mood is one of wonder and curiosity, as the monster gazes at the flame with wide eyes and open mouth. Its pose and expression convey a sense of innocence and playfulness, as if it is exploring the world around it for the first time. The use of warm colors and dramatic lighting further enhances the cozy atmosphere of the image.",
        "negative_prompt": "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
        "ssim_threshold": 0.79
    },
    {
        "lora_path": "motimalu/wan-flat-color-1.3b-v2",
        "lora_nickname": "flat_color",
        "prompt": "flat color, no lineart, blending, negative space, artist:[john kafka|ponsuke kaikai|hara id 21|yoneyama mai|fuzichoco], 1girl, sakura miko, pink hair, cowboy shot, white shirt, floral print, off shoulder, outdoors, cherry blossom, tree shade, wariza, looking up, falling petals, half-closed eyes, white sky, clouds, live2d animation, upper body, high quality cinematic video of a woman sitting under a sakura tree. Dreamy and lonely, the camera close-ups on the face of the woman as she turns towards the viewer. The Camera is steady, This is a cowboy shot. The animation is smooth and fluid.",
        "negative_prompt": "bad quality video,色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
        "ssim_threshold": 0.79
    }
    # TODO: Add a LoRA with lora_alpha values to test alpha scaling
    # 
    # Context: This change is mainly for an in-progress ticket porting over LongCat-Video,
    # where they used an alpha value that is two times smaller than their rank. This fix
    # ensures that LoRA weights are correctly scaled by the alpha/rank ratio when merged.
    #
    # Issue: Currently, we cannot add a test for LoRA adapters with alpha values because:
    # - The existing public LoRAs for Wan-AI/Wan2.1-T2V-1.3B-Diffusers don't store lora_alpha
    # - No publicly available LoRA for this model includes lora_alpha tensors in their weights
    # - This is why the alpha/rank scaling bug wasn't caught by existing tests
    #
    # The fix has been validated with:
    # - LongCat-Video distilled LoRA (which includes alpha values)
    # - Manual testing shows correct alpha/rank scaling behavior
    # - Backward compatibility confirmed with LoRAs without alpha values
    #
    # Future work:
    # - Add a synthetic LoRA test fixture with alpha values when feasible
    # - Or wait for public Wan LoRAs with alpha to become available
]

MODEL_TO_PARAMS = {
    "Wan-AI/Wan2.1-T2V-1.3B-Diffusers": WAN_LORA_PARAMS,
}

def _sanitize_filename_component(name: str) -> str:
    """Sanitize filename to remove invalid characters (same logic as VideoGenerator)"""
    sanitized = re.sub(r'[\\/:*?"<>|]', '', name)
    sanitized = sanitized.strip().strip('.')
    sanitized = re.sub(r'\s+', ' ', sanitized)
    return sanitized or "video"

@pytest.mark.parametrize("model_id", list(MODEL_TO_PARAMS.keys()))
def test_merge_lora_weights(model_id):
    lora_config = LORA_CONFIGS[0] # test only one
    hf_pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
    hf_pipe.enable_model_cpu_offload()

    lora_nickname = lora_config["lora_nickname"]
    lora_path = lora_config["lora_path"]
    # When layerwise offload is enabled, placeholder tensors cannot be compared directly.
    args = FastVideoArgs.from_kwargs(
        model_path=model_id,
        dit_layerwise_offload=False,
        use_fsdp_inference=True,
        dit_cpu_offload=True,
        dit_precision="bf16",
    )
    pipe = build_pipeline(args)
    pipe.set_lora_adapter(lora_nickname, lora_path)
    custom_transformer = pipe.modules["transformer"]
    custom_state_dict = custom_transformer.state_dict()

    hf_pipe.load_lora_weights(lora_path, adapter_name=lora_nickname)
    for name, layer in hf_pipe.transformer.named_modules():
        if hasattr(layer, "unmerge"):
            layer.unmerge()
            layer.merge(adapter_names=[lora_nickname])

    hf_transformer = hf_pipe.transformer
    param_names_mapping = get_param_names_mapping(custom_transformer.param_names_mapping)
    hf_state_dict, _ = hf_to_custom_state_dict(hf_transformer.state_dict(), param_names_mapping)
    for key in hf_state_dict.keys():
        if "base_layer" not in key:
            continue
        hf_param = hf_state_dict[key]
        custom_param = custom_state_dict[key].to_local() if isinstance(custom_state_dict[key], DTensor) else custom_state_dict[key]
        assert_close(hf_param, custom_param, atol=7e-4, rtol=7e-4)

@pytest.mark.parametrize("ATTENTION_BACKEND", ["TORCH_SDPA"])
@pytest.mark.parametrize("model_id", list(MODEL_TO_PARAMS.keys()))
def test_lora_inference_similarity(ATTENTION_BACKEND, model_id):
    """
    Test that runs LoRA inference with LoRA switching and compares the output
    to reference videos using SSIM.
    """
    os.environ["FASTVIDEO_ATTENTION_BACKEND"] = ATTENTION_BACKEND

    script_dir = os.path.dirname(os.path.abspath(__file__))

    output_dir = os.path.join(script_dir, 'generated_videos', model_id.split('/')[-1], ATTENTION_BACKEND)

    os.makedirs(output_dir, exist_ok=True)

    BASE_PARAMS = MODEL_TO_PARAMS[model_id]
    num_inference_steps = BASE_PARAMS["num_inference_steps"]

    init_kwargs = {
        "num_gpus": BASE_PARAMS["num_gpus"],
        "flow_shift": BASE_PARAMS["flow_shift"],
        "dit_cpu_offload": BASE_PARAMS["dit_cpu_offload"],
    }
    if "text-encoder-precision" in BASE_PARAMS:
        init_kwargs["text_encoder_precisions"] = BASE_PARAMS["text-encoder-precision"]

    generation_kwargs = {
        "num_inference_steps": num_inference_steps,
        "output_path": output_dir,
        "height": BASE_PARAMS["height"],
        "width": BASE_PARAMS["width"],
        "num_frames": BASE_PARAMS["num_frames"],
        "guidance_scale": BASE_PARAMS["guidance_scale"],
        "seed": BASE_PARAMS["seed"],
        "fps": BASE_PARAMS["fps"],
        "save_video": True,
    }
    generator = VideoGenerator.from_pretrained(model_path=BASE_PARAMS["model_path"], **init_kwargs)
    for lora_config in LORA_CONFIGS:
        lora_nickname = lora_config["lora_nickname"]
        lora_path = lora_config["lora_path"]
        prompt = lora_config["prompt"]
        generation_kwargs["negative_prompt"] = lora_config["negative_prompt"]

        generator.set_lora_adapter(lora_nickname=lora_nickname, lora_path=lora_path)
        # Sanitize the filename before adding .mp4 extension to match VideoGenerator's behavior
        output_video_name = f"{lora_path.split('/')[-1]}_{prompt[:50]}"
        output_video_name = _sanitize_filename_component(output_video_name)
        generated_video_path = os.path.join(output_dir, f"{output_video_name}.mp4")
        generation_kwargs["output_path"] = generated_video_path

        generator.generate_video(prompt, **generation_kwargs)

        assert os.path.exists(
            generated_video_path), f"Output video was not generated at {generated_video_path}"

        reference_folder = os.path.join(script_dir, 'L40S_reference_videos', model_id.split('/')[-1], ATTENTION_BACKEND)
        
        if not os.path.exists(reference_folder):
            logger.error("Reference folder missing")
            raise FileNotFoundError(
                f"Reference video folder does not exist: {reference_folder}")

        # Find the matching reference video - try exact match first, then fuzzy match
        # The reference might have different sanitization (e.g., trailing spaces)
        reference_video_name = None
        unsanitized_prefix = f"{lora_path.split('/')[-1]}_{prompt[:50]}"
        
        for filename in os.listdir(reference_folder):
            if not filename.endswith('.mp4'):
                continue
            
            # Try exact match with sanitized name
            if filename.startswith(output_video_name):
                reference_video_name = filename
                break
            
            # Try match with unsanitized prefix (for legacy reference videos)
            # Remove .mp4 and compare the base names after sanitization
            base_filename = filename[:-4]  # Remove .mp4
            if _sanitize_filename_component(base_filename) == output_video_name:
                reference_video_name = filename
                break

        if not reference_video_name:
            logger.error(f"Reference video not found for adapter: {lora_path} with prompt: {prompt[:50]} and backend: {ATTENTION_BACKEND}")
            raise FileNotFoundError(f"Reference video missing for adapter {lora_path}")

        reference_video_path = os.path.join(reference_folder, reference_video_name)

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
                                    prompt)

        if not success:
            logger.error("Failed to write SSIM results to file")

        min_acceptable_ssim = lora_config["ssim_threshold"]
        assert mean_ssim >= min_acceptable_ssim, f"SSIM value {mean_ssim} is below threshold {min_acceptable_ssim} for adapter {lora_config['lora_path']}" 




