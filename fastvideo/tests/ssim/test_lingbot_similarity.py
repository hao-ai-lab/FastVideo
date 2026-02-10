# SPDX-License-Identifier: Apache-2.0
"""
SSIM-based similarity test for LingBotWorld I2V with camera control.

Camera trajectory is loaded from LingBot example npy files
(poses.npy/intrinsics.npy), matching the official script workflow.

Note: num_inference_steps is reduced to 4 for faster CI.
"""

import os

import pytest
import torch

from fastvideo import VideoGenerator
from fastvideo.logger import init_logger
from fastvideo.models.dits.lingbotworld.cam_utils import prepare_camera_embedding
from fastvideo.tests.utils import compute_video_ssim_torchvision, write_ssim_results

logger = init_logger(__name__)


def _find_lingbotworld_examples_root() -> str | None:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(script_dir, "..", "..", ".."))
    candidates = [
        os.path.join(repo_root, "examples", "inference", "basic",
                     "lingbotworld_examples"),
        os.path.join(repo_root, "..", "FastVideo", "examples", "inference",
                     "basic", "lingbotworld_examples"),
    ]
    for candidate in candidates:
        if (os.path.exists(os.path.join(candidate, "00", "poses.npy"))
                and os.path.exists(
                    os.path.join(candidate, "00", "intrinsics.npy"))):
            return os.path.abspath(candidate)
    return None


device_name = torch.cuda.get_device_name()
device_reference_folder_suffix = "_reference_videos"

if "A40" in device_name:
    device_reference_folder = "A40" + device_reference_folder_suffix
elif "L40S" in device_name:
    device_reference_folder = "L40S" + device_reference_folder_suffix
elif "H100" in device_name:
    device_reference_folder = "H100" + device_reference_folder_suffix
else:
    device_reference_folder = None
    logger.warning("Unsupported device for ssim tests: %s", device_name)


LINGBOT_PARAMS = {
    "model_path": "FastVideo/LingBot-World-Base-Cam-Diffusers",
    "num_gpus": 2,
    "height": 256,
    "width": 448,
    "num_frames": 45,  # must be 4k+1
    "num_inference_steps": 4,
    "guidance_scale": 5.0,
    "guidance_scale_2": 5.0,
    "embedded_cfg_scale": 6,
    "flow_shift": 10.0,
    "boundary_ratio": 0.947,
    "seed": 42,
    "fps": 16,
    "spatial_scale": 8,
    "example_case": "00",
    "image_path": (
        "https://raw.githubusercontent.com/Robbyant/lingbot-world/main/"
        "examples/00/image.jpg"
    ),
    "negative_prompt": (
        "画面突变，色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，"
        "最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，"
        "畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走，"
        "镜头晃动，画面闪烁，模糊，噪点，水印，签名，文字，变形，扭曲，液化，不合逻辑的结构，卡顿，"
        "PPT幻灯片感，过暗，欠曝，低对比度，霓虹灯光感，过度锐化，3D渲染感，人物，行人，游客，身体，"
        "皮肤，肢体，面部特征，汽车，电线"
    ),
}

TEST_PROMPTS = [
    "The video presents a soaring journey through a fantasy jungle. The wind "
    "whips past the rider's blue hands gripping the reins, causing the leather "
    "straps to vibrate. The ancient gothic castle approaches steadily, its stone "
    "details becoming clearer against the backdrop of floating islands and "
    "distant waterfalls.",
]


@pytest.mark.parametrize("prompt", TEST_PROMPTS)
@pytest.mark.parametrize("ATTENTION_BACKEND", ["FLASH_ATTN"])
def test_lingbot_i2v_similarity(prompt: str, ATTENTION_BACKEND: str):
    os.environ["FASTVIDEO_ATTENTION_BACKEND"] = ATTENTION_BACKEND

    if device_reference_folder is None:
        pytest.skip(f"Unsupported device for LingBot SSIM test: {device_name}")
    if torch.cuda.device_count() < LINGBOT_PARAMS["num_gpus"]:
        pytest.skip(
            f"LingBot SSIM test requires {LINGBOT_PARAMS['num_gpus']} GPUs, "
            f"but only {torch.cuda.device_count()} detected."
        )

    examples_root = _find_lingbotworld_examples_root()
    if examples_root is None:
        pytest.skip(
            "lingbotworld_examples not found under examples/inference/basic.")

    action_path = os.path.join(examples_root, LINGBOT_PARAMS["example_case"])
    if not (os.path.exists(os.path.join(action_path, "poses.npy"))
            and os.path.exists(os.path.join(action_path, "intrinsics.npy"))):
        pytest.skip(f"Missing camera npy files under {action_path}")

    c2ws_plucker_emb, aligned_num_frames = prepare_camera_embedding(
        action_path=action_path,
        num_frames=LINGBOT_PARAMS["num_frames"],
        height=LINGBOT_PARAMS["height"],
        width=LINGBOT_PARAMS["width"],
        spatial_scale=LINGBOT_PARAMS["spatial_scale"],
    )

    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_id = "LingBot-World-Base-Cam-Diffusers"
    output_dir = os.path.join(script_dir, "generated_videos", model_id,
                              ATTENTION_BACKEND)
    output_video_name = f"{prompt[:100].strip()}.mp4"
    os.makedirs(output_dir, exist_ok=True)

    init_kwargs = {
        "num_gpus": LINGBOT_PARAMS["num_gpus"],
        "flow_shift": LINGBOT_PARAMS["flow_shift"],
        "boundary_ratio": LINGBOT_PARAMS["boundary_ratio"],
        "use_fsdp_inference": True,
        "dit_cpu_offload": True,
        "dit_layerwise_offload": False,
        "text_encoder_cpu_offload": True,
        "vae_cpu_offload": False,
        "pin_cpu_memory": True,
    }
    generation_kwargs = {
        "output_path": output_dir,
        "image_path": LINGBOT_PARAMS["image_path"],
        "height": LINGBOT_PARAMS["height"],
        "width": LINGBOT_PARAMS["width"],
        "num_frames": aligned_num_frames,
        "num_inference_steps": LINGBOT_PARAMS["num_inference_steps"],
        "guidance_scale": LINGBOT_PARAMS["guidance_scale"],
        "guidance_scale_2": LINGBOT_PARAMS["guidance_scale_2"],
        "embedded_cfg_scale": LINGBOT_PARAMS["embedded_cfg_scale"],
        "seed": LINGBOT_PARAMS["seed"],
        "fps": LINGBOT_PARAMS["fps"],
        "negative_prompt": LINGBOT_PARAMS["negative_prompt"],
        "c2ws_plucker_emb": c2ws_plucker_emb,
    }

    generator: VideoGenerator | None = None
    try:
        generator = VideoGenerator.from_pretrained(
            model_path=LINGBOT_PARAMS["model_path"], **init_kwargs)
        generator.generate_video(prompt, **generation_kwargs)
    finally:
        if generator is not None:
            generator.shutdown()

    generated_video_path = os.path.join(output_dir, output_video_name)
    assert os.path.exists(generated_video_path), (
        f"Output video was not generated at {generated_video_path}")

    reference_folder = os.path.join(script_dir, device_reference_folder, model_id,
                                    ATTENTION_BACKEND)
    if not os.path.exists(reference_folder):
        raise FileNotFoundError(
            f"Reference video folder does not exist: {reference_folder}")

    reference_video_name = None
    for filename in os.listdir(reference_folder):
        if filename.endswith(".mp4") and prompt[:100].strip() in filename:
            reference_video_name = filename
            break
    if not reference_video_name:
        raise FileNotFoundError(
            f"Reference video missing for prompt/backend under {reference_folder}"
        )

    reference_video_path = os.path.join(reference_folder, reference_video_name)
    logger.info("Computing SSIM between %s and %s", reference_video_path,
                generated_video_path)
    ssim_values = compute_video_ssim_torchvision(reference_video_path,
                                                 generated_video_path,
                                                 use_ms_ssim=True)
    mean_ssim = ssim_values[0]
    logger.info("SSIM mean value: %s", mean_ssim)

    write_ssim_results(output_dir, ssim_values, reference_video_path,
                       generated_video_path,
                       LINGBOT_PARAMS["num_inference_steps"], prompt)

    min_acceptable_ssim = 0.90
    assert mean_ssim >= min_acceptable_ssim, (
        f"SSIM value {mean_ssim} is below threshold {min_acceptable_ssim} "
        f"for {model_id} with backend {ATTENTION_BACKEND}")
