# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from contextlib import contextmanager
from logging import Logger
from typing import Iterator

from fastvideo import VideoGenerator
from fastvideo.tests.ssim.reference_utils import (
    build_generated_output_dir,
    build_reference_folder_path,
    get_cuda_device_name,
    resolve_device_reference_folder,
    select_ssim_params,
)
from fastvideo.tests.utils import compute_video_ssim_torchvision, write_ssim_results
from fastvideo.worker.multiproc_executor import MultiprocExecutor

DEVICE_MAPPINGS = (
    ("A40", "A40"),
    ("L40S", "L40S"),
    ("H200", "H200"),
)


@contextmanager
def attention_backend(backend: str) -> Iterator[None]:
    previous = os.environ.get("FASTVIDEO_ATTENTION_BACKEND")
    os.environ["FASTVIDEO_ATTENTION_BACKEND"] = backend
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("FASTVIDEO_ATTENTION_BACKEND", None)
        else:
            os.environ["FASTVIDEO_ATTENTION_BACKEND"] = previous


def shutdown_executor(generator: VideoGenerator | None) -> None:
    if generator is None:
        return
    if isinstance(generator.executor, MultiprocExecutor):
        generator.executor.shutdown()


def resolve_inference_device_reference_folder(logger: Logger) -> str:
    device_name = get_cuda_device_name()
    device_reference_folder = resolve_device_reference_folder(
        DEVICE_MAPPINGS,
        device_name=device_name,
        logger=logger,
    )
    if device_reference_folder is None:
        raise ValueError(f"Unsupported device for ssim tests: {device_name}")
    return device_reference_folder


def _find_reference_video(reference_folder: str, prompt: str) -> str:
    prompt_prefix = prompt[:100].strip()
    for filename in os.listdir(reference_folder):
        if filename.endswith(".mp4") and prompt_prefix in filename:
            return os.path.join(reference_folder, filename)
    raise FileNotFoundError("Reference video missing")


def _assert_similarity(
    *,
    logger: Logger,
    output_dir: str,
    output_video_name: str,
    reference_folder: str,
    prompt: str,
    num_inference_steps: int,
    min_acceptable_ssim: float,
    model_id: str,
    attention_backend_name: str,
) -> None:
    if not os.path.exists(reference_folder):
        logger.error("Reference folder missing")
        raise FileNotFoundError(
            f"Reference video folder does not exist: {reference_folder}"
        )

    try:
        reference_video_path = _find_reference_video(reference_folder, prompt)
    except FileNotFoundError as error:
        logger.error(
            "Reference video not found for prompt: %s with backend: %s",
            prompt,
            attention_backend_name,
        )
        raise error

    generated_video_path = os.path.join(output_dir, output_video_name)

    logger.info(
        "Computing SSIM between %s and %s",
        reference_video_path,
        generated_video_path,
    )
    ssim_values = compute_video_ssim_torchvision(
        reference_video_path,
        generated_video_path,
        use_ms_ssim=True,
    )

    mean_ssim = ssim_values[0]
    logger.info("SSIM mean value: %s", mean_ssim)
    logger.info("Writing SSIM results to directory: %s", output_dir)

    success = write_ssim_results(
        output_dir,
        ssim_values,
        reference_video_path,
        generated_video_path,
        num_inference_steps,
        prompt,
    )
    if not success:
        logger.error("Failed to write SSIM results to file")

    assert mean_ssim >= min_acceptable_ssim, (
        f"SSIM value {mean_ssim} is below threshold {min_acceptable_ssim} "
        f"for {model_id} with backend {attention_backend_name}"
    )


def run_text_to_video_similarity_test(
    *,
    logger: Logger,
    script_dir: str,
    device_reference_folder: str,
    prompt: str,
    attention_backend_name: str,
    model_id: str,
    default_params_map: dict[str, dict[str, object]],
    full_quality_params_map: dict[str, dict[str, object]],
    min_acceptable_ssim: float,
) -> None:
    with attention_backend(attention_backend_name):
        output_dir = build_generated_output_dir(
            script_dir,
            device_reference_folder,
            model_id,
            attention_backend_name,
        )
        output_video_name = f"{prompt[:100].strip()}.mp4"
        os.makedirs(output_dir, exist_ok=True)

        params_map = select_ssim_params(
            default_params_map,
            full_quality_params_map,
        )
        base_params = params_map[model_id]
        num_inference_steps = int(base_params["num_inference_steps"])

        init_kwargs = {
            "num_gpus": base_params["num_gpus"],
            "sp_size": base_params["sp_size"],
            "tp_size": base_params["tp_size"],
            "use_fsdp_inference": True,
            "dit_cpu_offload": False,
            "dit_layerwise_offload": False,
        }
        if "flow_shift" in base_params:
            init_kwargs["flow_shift"] = base_params["flow_shift"]
        if base_params.get("vae_sp"):
            init_kwargs["vae_sp"] = True
            init_kwargs["vae_tiling"] = True
        if "text-encoder-precision" in base_params:
            init_kwargs["text_encoder_precisions"] = base_params[
                "text-encoder-precision"
            ]
        if base_params.get("ltx2_vae_tiling"):
            init_kwargs["ltx2_vae_tiling"] = True
            init_kwargs["ltx2_vae_spatial_tile_size_in_pixels"] = base_params.get(
                "ltx2_vae_spatial_tile_size_in_pixels",
                512,
            )
            init_kwargs["ltx2_vae_spatial_tile_overlap_in_pixels"] = (
                base_params.get(
                    "ltx2_vae_spatial_tile_overlap_in_pixels",
                    64,
                )
            )
            init_kwargs["ltx2_vae_temporal_tile_size_in_frames"] = base_params.get(
                "ltx2_vae_temporal_tile_size_in_frames",
                64,
            )
            init_kwargs["ltx2_vae_temporal_tile_overlap_in_frames"] = (
                base_params.get(
                    "ltx2_vae_temporal_tile_overlap_in_frames",
                    24,
                )
            )

        generation_kwargs = {
            "num_inference_steps": num_inference_steps,
            "output_path": output_dir,
            "height": base_params["height"],
            "width": base_params["width"],
            "num_frames": base_params["num_frames"],
            "guidance_scale": base_params["guidance_scale"],
            "embedded_cfg_scale": base_params["embedded_cfg_scale"],
            "seed": base_params["seed"],
            "fps": base_params["fps"],
        }
        if "neg_prompt" in base_params:
            generation_kwargs["neg_prompt"] = base_params["neg_prompt"]

        generator: VideoGenerator | None = None
        try:
            generator = VideoGenerator.from_pretrained(
                model_path=base_params["model_path"],
                **init_kwargs,
            )
            generator.generate_video(prompt, **generation_kwargs)
        finally:
            shutdown_executor(generator)

    assert os.path.exists(
        output_dir
    ), f"Output video was not generated at {output_dir}"

    reference_folder = build_reference_folder_path(
        script_dir,
        device_reference_folder,
        model_id,
        attention_backend_name,
    )
    _assert_similarity(
        logger=logger,
        output_dir=output_dir,
        output_video_name=output_video_name,
        reference_folder=reference_folder,
        prompt=prompt,
        num_inference_steps=num_inference_steps,
        min_acceptable_ssim=min_acceptable_ssim,
        model_id=model_id,
        attention_backend_name=attention_backend_name,
    )


def run_image_to_video_similarity_test(
    *,
    logger: Logger,
    script_dir: str,
    device_reference_folder: str,
    prompt: str,
    image_path: str,
    attention_backend_name: str,
    model_id: str,
    default_params_map: dict[str, dict[str, object]],
    full_quality_params_map: dict[str, dict[str, object]],
    min_acceptable_ssim: float,
) -> None:
    with attention_backend(attention_backend_name):
        output_dir = build_generated_output_dir(
            script_dir,
            device_reference_folder,
            model_id,
            attention_backend_name,
        )
        output_video_name = f"{prompt[:100].strip()}.mp4"
        os.makedirs(output_dir, exist_ok=True)

        params_map = select_ssim_params(
            default_params_map,
            full_quality_params_map,
        )
        base_params = params_map[model_id]
        num_inference_steps = int(base_params["num_inference_steps"])

        init_kwargs = {
            "num_gpus": base_params["num_gpus"],
            "flow_shift": base_params["flow_shift"],
            "sp_size": base_params["sp_size"],
            "tp_size": base_params["tp_size"],
        }
        if base_params.get("vae_sp"):
            init_kwargs["vae_sp"] = True
            init_kwargs["vae_tiling"] = True
        if "text-encoder-precision" in base_params:
            init_kwargs["text_encoder_precisions"] = base_params[
                "text-encoder-precision"
            ]

        generation_kwargs = {
            "num_inference_steps": num_inference_steps,
            "output_path": output_dir,
            "image_path": image_path,
            "height": base_params["height"],
            "width": base_params["width"],
            "num_frames": base_params["num_frames"],
            "guidance_scale": base_params["guidance_scale"],
            "embedded_cfg_scale": base_params["embedded_cfg_scale"],
            "seed": base_params["seed"],
            "fps": base_params["fps"],
        }
        if "neg_prompt" in base_params:
            generation_kwargs["neg_prompt"] = base_params["neg_prompt"]

        generator: VideoGenerator | None = None
        try:
            generator = VideoGenerator.from_pretrained(
                model_path=base_params["model_path"],
                **init_kwargs,
            )
            generator.generate_video(prompt, **generation_kwargs)
        finally:
            shutdown_executor(generator)

    assert os.path.exists(
        output_dir
    ), f"Output video was not generated at {output_dir}"

    reference_folder = build_reference_folder_path(
        script_dir,
        device_reference_folder,
        model_id,
        attention_backend_name,
    )
    _assert_similarity(
        logger=logger,
        output_dir=output_dir,
        output_video_name=output_video_name,
        reference_folder=reference_folder,
        prompt=prompt,
        num_inference_steps=num_inference_steps,
        min_acceptable_ssim=min_acceptable_ssim,
        model_id=model_id,
        attention_backend_name=attention_backend_name,
    )
