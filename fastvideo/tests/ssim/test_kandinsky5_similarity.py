# SPDX-License-Identifier: Apache-2.0
"""SSIM regression test for Kandinsky-5.0 Lite text-to-video.

Runs a reduced-size generation (256x384, 21 frames) with a fixed seed and
compares against a device-specific reference video via MS-SSIM. The preset
default is 512x768 x 121 frames; the reduced size keeps CI cheap while the
patch (1, 2, 2) and VAE temporal-4 / spatial-8 constraints stay satisfied
(dims divisible by 16, num_frames % 4 == 1).
"""
import os

import pytest

from fastvideo.logger import init_logger
from fastvideo.pipelines.basic.kandinsky5.presets import KANDINSKY5_T2V_LITE_5S
from fastvideo.tests.ssim.inference_similarity_utils import (
    resolve_inference_device_reference_folder,
    run_text_to_video_similarity_test,
)

logger = init_logger(__name__)

REQUIRED_GPUS = 1

device_reference_folder = resolve_inference_device_reference_folder(logger)

_PRESET_DEFAULTS = KANDINSKY5_T2V_LITE_5S.defaults

KANDINSKY5_T2V_PARAMS = {
    "num_gpus": 1,
    "model_path": "kandinskylab/Kandinsky-5.0-T2V-Lite-sft-5s-Diffusers",
    "height": 256,
    "width": 384,
    "num_frames": 21,
    "num_inference_steps": _PRESET_DEFAULTS["num_inference_steps"],
    "guidance_scale": _PRESET_DEFAULTS["guidance_scale"],
    "seed": 1024,
    "sp_size": 1,
    "tp_size": 1,
    "fps": _PRESET_DEFAULTS["fps"],
    "neg_prompt": _PRESET_DEFAULTS["negative_prompt"],
}

KANDINSKY5_T2V_FULL_QUALITY_PARAMS = {
    **KANDINSKY5_T2V_PARAMS,
    "height": _PRESET_DEFAULTS["height"],
    "width": _PRESET_DEFAULTS["width"],
    # num_frames stays at the inherited reduced 21 even at full quality
    # (preset default: 121).
}

KANDINSKY5_T2V_MODEL_TO_PARAMS = {
    "Kandinsky-5.0-T2V-Lite-sft-5s-Diffusers": KANDINSKY5_T2V_PARAMS,
}
FULL_QUALITY_KANDINSKY5_T2V_MODEL_TO_PARAMS = {
    "Kandinsky-5.0-T2V-Lite-sft-5s-Diffusers": KANDINSKY5_T2V_FULL_QUALITY_PARAMS,
}

KANDINSKY5_T2V_TEST_PROMPTS = [
    "A curious raccoon peers through a vibrant field of yellow sunflowers, its "
    "eyes wide with interest. The playful yet serene atmosphere is complemented "
    "by soft natural light filtering through the petals. Mid-shot, warm and "
    "cheerful tones.",
]


@pytest.mark.parametrize("prompt", KANDINSKY5_T2V_TEST_PROMPTS)
@pytest.mark.parametrize("attention_backend_name", ["FLASH_ATTN"])
@pytest.mark.parametrize("model_id", list(KANDINSKY5_T2V_MODEL_TO_PARAMS.keys()))
def test_kandinsky5_t2v_inference_similarity(
    prompt: str,
    attention_backend_name: str,
    model_id: str,
) -> None:
    # The SSIM lane image bakes FASTVIDEO_FA4=1, but on L40S (sm89) the CLIP
    # encoder's dense MHA inference call routes into the FA4 CuTeDSL JIT,
    # which fails to compile there (nvvm.fmax signature mismatch vs the
    # image's nvidia_cutlass_dsl). Pin FA4 off so reference seeding and CI
    # runs both use the FA2 path with identical numerics. Scoped to this test
    # and restored afterwards: the other SSIM references are seeded with FA4
    # on, so a module-level pin would corrupt every test collected in the
    # same pytest process.
    saved_fa4 = os.environ.get("FASTVIDEO_FA4")
    os.environ["FASTVIDEO_FA4"] = "0"
    try:
        run_text_to_video_similarity_test(
            logger=logger,
            script_dir=os.path.dirname(os.path.abspath(__file__)),
            device_reference_folder=device_reference_folder,
            prompt=prompt,
            attention_backend_name=attention_backend_name,
            model_id=model_id,
            default_params_map=KANDINSKY5_T2V_MODEL_TO_PARAMS,
            full_quality_params_map=FULL_QUALITY_KANDINSKY5_T2V_MODEL_TO_PARAMS,
            min_acceptable_ssim=0.93,
            # Match examples/inference/basic/basic_kandinsky5_t2v.py: FSDP
            # inference is not used for Kandinsky-5, and the Qwen2.5-VL text
            # encoder stays on CPU between encodes.
            init_kwargs_override={
                "use_fsdp_inference": False,
                "text_encoder_cpu_offload": True,
                "pin_cpu_memory": True,
            },
        )
    finally:
        if saved_fa4 is None:
            os.environ.pop("FASTVIDEO_FA4", None)
        else:
            os.environ["FASTVIDEO_FA4"] = saved_fa4
