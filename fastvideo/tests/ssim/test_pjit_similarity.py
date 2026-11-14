# SPDX-License-Identifier: Apache-2.0
"""One-GPU PJiT image regression against a reviewed, device-matched PNG."""

from pathlib import Path

import pytest
import torch

from fastvideo.logger import init_logger
from fastvideo.tests.ssim.inference_similarity_utils import DEVICE_MAPPINGS, run_text_to_video_similarity_test
from fastvideo.tests.ssim.reference_utils import get_cuda_device_name, resolve_device_reference_folder

logger = init_logger(__name__)
REQUIRED_GPUS = 1
MODEL_ID = "Linum-AI/pjit-21k"
PJIT_MODEL_TO_PARAMS = {
    "Linum-AI/pjit-21k": dict(model_path=MODEL_ID, num_gpus=1, height=512, width=512, num_frames=1, fps=1, seed=2,
                              num_inference_steps=4, guidance_scale=15.0),
}
FULL_PARAMS = {MODEL_ID: {**PJIT_MODEL_TO_PARAMS[MODEL_ID], "height": 1216, "width": 864, "num_inference_steps": 50}}
PROMPT = (
    "Close-up portrait of an elderly Middle Eastern woman looking calmly into the lens, her face filling the tall frame. "
    "A cream headscarf and the collar of a dark green cardigan frame her chin. Soft overcast daylight smooths the fine "
    "lines of her skin, and the olive grove behind melts into silvery blur."
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="PJiT SSIM requires a CUDA GPU")
@pytest.mark.parametrize("model_id", list(PJIT_MODEL_TO_PARAMS))
def test_pjit_similarity(model_id):
    folder = resolve_device_reference_folder(DEVICE_MAPPINGS, device_name=get_cuda_device_name(), logger=logger)
    if folder is None:
        pytest.skip("No PJiT reference mapping for this GPU")
    run_text_to_video_similarity_test(
        logger=logger, script_dir=str(Path(__file__).parent), device_reference_folder=folder, prompt=PROMPT,
        attention_backend_name="TORCH_SDPA", model_id=model_id, default_params_map=PJIT_MODEL_TO_PARAMS,
        full_quality_params_map=FULL_PARAMS, min_acceptable_ssim=0.98, media_extension=".png",
        init_kwargs_override=dict(workload_type="t2i", use_fsdp_inference=False, lazy_module_load=True,
                                  revision="251725ad9a0618f0a4ba3cdf4a488d35312be352"),
        generation_kwargs_override=dict(negative_prompt="", save_video=True),
    )
