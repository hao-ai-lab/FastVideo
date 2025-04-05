import os
import pytest
import json
from fastvideo.v1.tests.ssim.compute_ssim import compute_video_ssim_torchvision
from fastvideo.v1.logger import init_logger
from fastvideo.v1.entrypoints.cli.utils import launch_distributed

logger = init_logger(__name__)

# Base parameters from the shell script
BASE_PARAMS = {
    "num_gpus": 2,
    "model_path": "data/FastHunyuan-diffusers",
    "height": 720,
    "width": 1280,
    "num_frames": 45,
    "num_inference_steps": 6,
    "guidance_scale": 1,
    "embedded_cfg_scale": 6,
    "flow_shift": 17,
    "seed": 1024,
    "sp_size": 2,
    "tp_size": 2,
    "vae_sp": True,
    "use_v1_transformer": True,
    "use_v1_vae": True,
    "use_v1_text_encoder": True,
    "fps": 24,
}

TEST_PROMPTS = [
    "Will Smith casually eats noodles, his relaxed demeanor contrasting with the energetic background of a bustling street food market. The scene captures a mix of humor and authenticity. Mid-shot framing, vibrant lighting.",
    "A lone hiker stands atop a towering cliff, silhouetted against the vast horizon. The rugged landscape stretches endlessly beneath, its earthy tones blending into the soft blues of the sky. The scene captures the spirit of exploration and human resilience. High angle, dynamic framing, with soft natural lighting emphasizing the grandeur of nature."
]


def write_ssim_results(output_dir, ssim_values, reference_path, generated_path,
                       num_inference_steps, prompt):
    """
    Write SSIM results to a JSON file in the same directory as the generated videos.
    """
    try:
        logger.info(
            f"Attempting to write SSIM results to directory: {output_dir}")

        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        mean_ssim, min_ssim, max_ssim = ssim_values

        result = {
            "mean_ssim": mean_ssim,
            "min_ssim": min_ssim,
            "max_ssim": max_ssim,
            "reference_video": reference_path,
            "generated_video": generated_path,
            "parameters": {
                "num_inference_steps": num_inference_steps,
                "prompt": prompt
            }
        }

        test_name = f"steps{num_inference_steps}_{prompt[:100]}"
        result_file = os.path.join(output_dir, f"{test_name}_ssim.json")
        logger.info(f"Writing JSON results to: {result_file}")
        with open(result_file, 'w') as f:
            json.dump(result, f, indent=2)

        logger.info(f"SSIM results written to {result_file}")
        return True
    except Exception as e:
        logger.error(f"ERROR writing SSIM results: {str(e)}")
        return False


@pytest.mark.parametrize("num_inference_steps", [6])
@pytest.mark.parametrize("prompt", TEST_PROMPTS)
def test_inference_similarity(num_inference_steps, prompt):
    """
    Test that runs inference with different parameters and compares the output
    to reference videos using SSIM.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))

    base_output_dir = os.path.join(script_dir, 'generated_videos')
    output_dir = os.path.join(base_output_dir,
                              f'num_inference_steps={num_inference_steps}')
    output_video_name = f"{prompt[:100]}.mp4"

    os.makedirs(output_dir, exist_ok=True)

    launch_args = [
        "--num-inference-steps",
        str(num_inference_steps),
        "--prompt",
        prompt,
        "--output-path",
        output_dir,
        "--model-path",
        BASE_PARAMS["model_path"],
        "--height",
        str(BASE_PARAMS["height"]),
        "--width",
        str(BASE_PARAMS["width"]),
        "--num-frames",
        str(BASE_PARAMS["num_frames"]),
        "--guidance-scale",
        str(BASE_PARAMS["guidance_scale"]),
        "--embedded-cfg-scale",
        str(BASE_PARAMS["embedded_cfg_scale"]),
        "--flow-shift",
        str(BASE_PARAMS["flow_shift"]),
        "--seed",
        str(BASE_PARAMS["seed"]),
        "--sp-size",
        str(BASE_PARAMS["sp_size"]),
        "--tp-size",
        str(BASE_PARAMS["tp_size"]),
        "--fps",
        str(BASE_PARAMS["fps"]),
    ]

    if BASE_PARAMS["use_v1_transformer"]:
        launch_args.append("--use-v1-transformer")
    if BASE_PARAMS["use_v1_vae"]:
        launch_args.append("--use-v1-vae")
    if BASE_PARAMS["use_v1_text_encoder"]:
        launch_args.append("--use-v1-text-encoder")
    if BASE_PARAMS["vae_sp"]:
        launch_args.append("--vae-sp")

    launch_distributed(num_gpus=BASE_PARAMS["num_gpus"], args=launch_args)

    assert os.path.exists(
        output_dir), f"Output video was not generated at {output_dir}"

    reference_folder = os.path.join(script_dir, 'reference_videos')
    
    if not os.path.exists(reference_folder):
        logger.error("Reference folder missing")
        raise FileNotFoundError(f"Reference video folder does not exist: {reference_folder}")

    # Find the matching reference video based on the prompt
    reference_video_name = None

    for filename in os.listdir(reference_folder):
        if filename.endswith('.mp4') and prompt[:100] in filename:
            reference_video_name = filename
            break

    if not reference_video_name:
        logger.error(f"Reference video not found for prompt: {prompt}")
        raise FileNotFoundError(f"Reference video missing")

    reference_video_path = os.path.join(reference_folder, reference_video_name)
    generated_video_path = os.path.join(output_dir, output_video_name)

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

    min_acceptable_ssim = 1
    assert mean_ssim >= min_acceptable_ssim, f"SSIM value {mean_ssim} is below threshold {min_acceptable_ssim}"
