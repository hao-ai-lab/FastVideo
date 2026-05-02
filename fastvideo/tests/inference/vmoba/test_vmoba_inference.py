# SPDX-License-Identifier: Apache-2.0

import json
import os
import subprocess
import tempfile
from pathlib import Path


def test_inference_vmoba():
    """Test FastVideo VMOBA_ATTN inference pipeline"""

    output_dir = Path("outputs_video/vmoba_1.3B/")
    moba_config = "fastvideo/configs/backend/vmoba/wan_1.3B_77_480_832.json"

    os.environ["FASTVIDEO_ATTENTION_BACKEND"] = "VMOBA_ATTN"

    config = {
        "generator": {
            "model_path": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
            "engine": {
                "num_gpus": 1,
                "parallelism": {
                    "tp_size": 1,
                    "sp_size": 1
                },
                "offload": {
                    "dit": False,
                    "vae": False,
                    "text_encoder": True,
                    "pin_cpu_memory": False,
                },
            },
            "pipeline": {
                "experimental": {
                    "flow_shift": 8.0,
                    "moba_config_path": moba_config,
                },
            },
        },
        "request": {
            "prompt":
            "A majestic lion strides across the golden savanna, "
            "its powerful frame glistening under the warm afternoon "
            "sun. The tall grass ripples gently in the breeze, "
            "enhancing the lion's commanding presence. The tone is "
            "vibrant, embodying the raw energy of the wild. Low "
            "angle, steady tracking shot, cinematic.",
            "negative_prompt":
            "Bright tones, overexposed, static, blurred details, "
            "subtitles, style, works, paintings, images, static, "
            "overall gray, worst quality, low quality, JPEG "
            "compression residue, ugly, incomplete, extra fingers, "
            "poorly drawn hands, poorly drawn faces, deformed, "
            "disfigured, misshapen limbs, fused fingers, still "
            "picture, messy background, three legs, many people in "
            "the background, walking backwards",
            "sampling": {
                "seed": 1024,
                "num_frames": 77,
                "height": 480,
                "width": 832,
                "fps": 16,
                "num_inference_steps": 10,
                "guidance_scale": 6.0,
            },
            "output": {
                "output_path": str(output_dir),
            },
        },
    }

    with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False) as f:
        json.dump(config, f)
        config_path = f.name

    try:
        cmd = ["fastvideo", "generate", "--config", config_path]
        subprocess.run(cmd, check=True)
    finally:
        os.unlink(config_path)

    assert output_dir.exists(), \
        f"Output directory {output_dir} does not exist"

    video_files = list(output_dir.glob("*.mp4"))
    assert len(video_files) > 0, "No video files were generated"

    for video_file in video_files:
        assert video_file.stat().st_size > 0, \
            f"Video file {video_file} is empty"


if __name__ == "__main__":
    test_inference_vmoba()
