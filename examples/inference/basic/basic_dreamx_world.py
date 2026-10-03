import os

from fastvideo import VideoGenerator

OUTPUT_PATH = os.getenv("DREAMX_WORLD_OUTPUT_PATH", "video_samples_dreamx_world")


def _env_int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def _env_float(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


def main():
    model_name = os.getenv("DREAMX_WORLD_MODEL_DIR", "FastVideo/DreamX-World-5B-Cam-Diffusers")
    generator = VideoGenerator.from_config({
        "model_path": model_name,
        "engine": {
            "num_gpus": 1,
            "use_fsdp_inference": False,
            "offload": {
                "dit": False,
                "vae": True,
                "text_encoder": True,
                "pin_cpu_memory": False,
            },
        },
        "pipeline": {"components": {"override_pipeline_cls_name": "DreamXWorldPipeline"}},
    })

    prompt = os.getenv(
        "DREAMX_WORLD_PROMPT",
        "A cinematic first-person drive through a futuristic coastal city at "
        "sunrise, reflective glass towers, clean streets, soft volumetric light.",
    )
    image_path = os.getenv(
        "DREAMX_WORLD_IMAGE_PATH",
        "https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/wan_i2v_input.JPG",
    )

    request = {
        "prompt": prompt,
        "sampling": {
            "height": _env_int("DREAMX_WORLD_HEIGHT", 480),
            "width": _env_int("DREAMX_WORLD_WIDTH", 832),
            "num_frames": _env_int("DREAMX_WORLD_NUM_FRAMES", 161),
            "num_inference_steps": _env_int("DREAMX_WORLD_STEPS", 30),
            "guidance_scale": _env_float("DREAMX_WORLD_GUIDANCE", 5.0),
        },
        "output": {
            "output_path": OUTPUT_PATH,
            "save_video": os.getenv("DREAMX_WORLD_SAVE_VIDEO", "1") != "0",
        },
        "extensions": {
            "action_list": os.getenv("DREAMX_WORLD_ACTIONS", "w,d,w").split(","),
            "action_speed_list":
            [float(value) for value in os.getenv("DREAMX_WORLD_ACTION_SPEEDS", "4.0,2.0,4.0").split(",")],
        },
    }
    if image_path:
        request["inputs"] = {"image_path": image_path}

    try:
        generator.generate(request)
    finally:
        generator.shutdown()


if __name__ == "__main__":
    main()
