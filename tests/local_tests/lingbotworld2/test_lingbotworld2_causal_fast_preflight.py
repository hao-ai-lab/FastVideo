# SPDX-License-Identifier: Apache-2.0
"""Preflight checks for the LingBot World 2 causal-fast FastVideo port."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path("/mnt/weka/shrd/wm/junda/fv-hub/lingbot-world-v2")
FASTVIDEO_ROOT = Path(__file__).resolve().parents[3]
RAW_MODEL_DIR = ROOT / "ckpts" / "lingbot-world-v2-14b-causal-fast"
FASTVIDEO_MODEL_DIR = ROOT / "ckpts" / "lingbot-world-v2-14b-causal-fast-fastvideo"
ACTION_PATH = FASTVIDEO_ROOT / "examples" / "dataset" / "lingbotworld2"


def _source_effective_frames(frame_num: int, pose_count: int, chunk_size: int = 4) -> int:
    """Mirror the released `wan/image2video.py` frame-count truncation."""
    rounded = ((frame_num - 1) // 4) * 4 + 1
    rounded = min(rounded, ((pose_count - 1) // 4) * 4 + 1)
    latent_frames = (rounded - 1) // 4 + 1
    latent_frames = int(latent_frames - (latent_frames % chunk_size))
    return (latent_frames - 1) * 4 + 1


def test_lingbotworld2_bundle_declares_native_components() -> None:
    assert RAW_MODEL_DIR.exists()
    assert FASTVIDEO_MODEL_DIR.exists()
    assert (FASTVIDEO_MODEL_DIR / "model_index.json").exists()
    assert (FASTVIDEO_MODEL_DIR / "transformer" / "config.json").exists()
    assert (FASTVIDEO_MODEL_DIR / "text_encoder" / "config.json").exists()
    assert (FASTVIDEO_MODEL_DIR / "text_encoder" / "pytorch_model.pt").exists()
    assert (FASTVIDEO_MODEL_DIR / "vae" / "config.json").exists()
    assert (FASTVIDEO_MODEL_DIR / "vae" / "Wan2.1_VAE.pth").exists()
    assert (FASTVIDEO_MODEL_DIR / "scheduler" / "scheduler_config.json").exists()

    model_index = json.loads((FASTVIDEO_MODEL_DIR / "model_index.json").read_text(encoding="utf-8"))
    assert model_index["_class_name"] == "LingBotWorld2CausalFastPipeline"
    assert model_index["transformer"][1] == "LingBotWorld2CausalFastTransformer3DModel"
    assert model_index["text_encoder"][1] == "LingBotWorld2T5EncoderModel"
    assert model_index["vae"][1] == "LingBotWorld2WanVAE"


def test_lingbotworld2_source_frame_requests_truncate_as_expected() -> None:
    pose_count = int(np.load(ACTION_PATH / "poses.npy").shape[0])
    effective = {
        frame_num: _source_effective_frames(frame_num, pose_count)
        for frame_num in (5, 9, 17, 33, 65)
    }
    assert effective == {
        5: -3,
        9: -3,
        17: 13,
        33: 29,
        65: 61,
    }


def test_lingbotworld2_request_preserves_action_path() -> None:
    from fastvideo.api.compat import normalize_generation_request, request_to_sampling_param

    request = normalize_generation_request({
        "prompt": "camera path schema check",
        "inputs": {
            "image_path": str(ACTION_PATH / "image.jpg"),
            "action_path": str(ACTION_PATH),
        },
        "sampling": {
            "height": 480,
            "width": 832,
            "num_frames": 17,
            "num_inference_steps": 4,
            "guidance_scale": 1.0,
        },
        "output": {
            "save_video": False,
            "return_frames": False,
        },
    })
    sampling_param = request_to_sampling_param(request, model_path=str(FASTVIDEO_MODEL_DIR))
    assert sampling_param.action_path == str(ACTION_PATH)
    assert sampling_param.image_path == str(ACTION_PATH / "image.jpg")
    assert sampling_param.num_frames == 17
