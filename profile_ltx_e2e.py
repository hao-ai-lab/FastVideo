import os
from fastvideo import VideoGenerator
import argparse


prompt = (
    "A curious raccoon peers through a vibrant field of yellow sunflowers, its eyes "
    "wide with interest. The playful yet serene atmosphere is complemented by soft "
    "natural light filtering through the petals. Mid-shot, warm and cheerful tones."
)

output_path = "outputs_video/ltx2_basic/output_ltx2_distilled_t2v.mp4"

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", type=int, default=1)
    args = parser.parse_args()
    os.environ["FASTVIDEO_STAGE_LOGGING"] = "1"

    generator = VideoGenerator.from_pretrained(
        "FastVideo/LTX2-Distilled-Diffusers",
        num_gpus=args.gpus,
        tp_size=1,
        sp_size=args.gpus,
        use_fsdp_inference=False, # set to True if GPU is out of memory
        dit_cpu_offload=False,
        vae_cpu_offload=False,
        text_encoder_cpu_offload=False,
        pin_cpu_memory=True, # set to false if low CPU RAM or hit obscure "CUDA error: Invalid argument"
        dit_layerwise_offload=False,
    )

    for i in range(10):
        result = generator.generate_video(
            prompt,
            output_path=output_path,
            save_video=True,
            negative_prompt="",
            ##########
            num_frames=121,
            fps=24,
            height = 1088, # TODO: Make hw same between LTX2 and HY15
            width = 1920,
            guidance_scale=1.0,
            num_inference_steps=1,
        )
        print(f"==================OUTPUT LOGGING INFO {i=}=====================")
        logging_info = result.get("logging_info") if isinstance(result, dict) else None
        if logging_info is None:
            print("No logging_info returned; enable FASTVIDEO_STAGE_LOGGING=1.")
            return
        stage_names = logging_info.get_execution_order()
        stage_execution_times = [
            logging_info.get_stage_info(stage_name).get("execution_time", 0.0)
            for stage_name in stage_names
        ]
        # for name, exec_time in zip(stage_names, stage_execution_times):
        #     print(f"{name}, {exec_time * 1e3:.5f}")
        print(",".join(stage_names))
        print(",".join([f"{e * 1e3:.5f}" for e in stage_execution_times]))
        print(f"=====================End LOGGING INFO {i=}=====================")


if __name__ == "__main__":
    main()
