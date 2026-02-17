import os
import time
from fastvideo import VideoGenerator
import argparse


prompt = (
    "A warm sunny backyard. The camera starts in a tight cinematic close-up "
    "of a woman and a man in their 30s, facing each other with serious "
    "expressions. The woman, emotional and dramatic, says softly, \"That's "
    "it... Dad's lost it. And we've lost Dad.\" The man exhales, slightly "
    "annoyed: \"Stop being so dramatic, Jess.\" A beat. He glances aside, "
    "then mutters defensively, \"He's just having fun.\" The camera slowly "
    "pans right, revealing the grandfather in the garden wearing enormous "
    "butterfly wings, waving his arms in the air like he's trying to take "
    "off. He shouts, \"Wheeeew!\" as he flaps his wings with full commitment. "
    "The woman covers her face, on the verge of tears. The tone is deadpan, "
    "absurd, and quietly tragic."
)

output_path = "outputs_video/ltx2_basic/output_ltx2_distilled_t2v.mp4"

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--attn", type=str, choices=["", "vsa"], default="")
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args()
    os.environ["FASTVIDEO_STAGE_LOGGING"] = "1"

    if args.compile:
        do_compile = True
    else:
        do_compile = False

    if args.attn == "vsa":
        os.environ["FASTVIDEO_ATTENTION_BACKEND"] = "VIDEO_SPARSE_ATTN"
        os.environ["FASTVIDEO_TORCH_PROFILER_DIR"] = "./vsa-traces-0.7"
    else:
        os.environ["FASTVIDEO_TORCH_PROFILER_DIR"] = "./traces"
    
    if args.trace:
        os.environ["FASTVIDEO_TORCH_PROFILER_RECORD_SHAPES"] = "0"
        os.environ["FASTVIDEO_TORCH_PROFILER_WITH_FLOPS"] = "0"
        os.environ["FASTVIDEO_TORCH_PROFILER_WAIT_STEPS"] = "2"
        os.environ["FASTVIDEO_TORCH_PROFILER_WARMUP_STEPS"] = "2"
        os.environ["FASTVIDEO_TORCH_PROFILER_ACTIVE_STEPS"] = "2"
        os.environ["FASTVIDEO_TORCH_PROFILE_REGIONS"] = "profiler_region_dit_forward"
    else:
        os.environ["FASTVIDEO_TORCH_PROFILER_DIR"] = "" # Disable tracing

    # If the trace dir exist and has files, create a new folder with a postfix which is the timestamp. 
    trace_dir = os.environ.get("FASTVIDEO_TORCH_PROFILER_DIR", "")
    if trace_dir and os.path.isdir(trace_dir):
        with os.scandir(trace_dir) as entries:
            if any(entries):
                os.environ["FASTVIDEO_TORCH_PROFILER_DIR"] = (
                    f"{trace_dir}_{time.strftime('%Y%m%d_%H%M%S')}"
                ) 

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
        enable_torch_compile=do_compile,
        VSA_sparsity=0.7  # inference/validation sparsity
    )

    all_stage_times: list[list[float]] = []  # Each element is one iteration's stage times
    stage_names: list[str] = []
    
    for i in range(10):
        result = generator.generate_video(
            prompt,
            output_path=output_path,
            save_video=False,
            negative_prompt="",
            ##########
            num_frames=121,
            fps=24,
            height = 1088,
            width = 1920,
            guidance_scale=1.0,
            num_inference_steps=8,
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
        all_stage_times.append(stage_execution_times)
        # for name, exec_time in zip(stage_names, stage_execution_times):
        #     print(f"{name}, {exec_time * 1e3:.5f}")
        print(",".join(stage_names))
        print(",".join([f"{e * 1e3:.5f}" for e in stage_execution_times]))
        print(f"=====================End LOGGING INFO {i=}=====================")
    
    # Calculate average time for each stage
    if all_stage_times and stage_names:
        num_stages = len(stage_names)
        
        # Average including all rounds
        avg_with_warmup = [
            sum(times[j] for times in all_stage_times) / len(all_stage_times)
            for j in range(num_stages)
        ]
        
        # Average excluding first round (warmup)
        times_no_warmup = all_stage_times[1:]  # Skip first iteration
        avg_no_warmup = [
            sum(times[j] for times in times_no_warmup) / len(times_no_warmup)
            for j in range(num_stages)
        ] if times_no_warmup else avg_with_warmup
        
        print("\n==================AVERAGE TIMING SUMMARY=====================")
        print("Stage names,", ",".join(stage_names))
        print(f"Avg (all rounds),     {','.join(f'{t * 1e3:.5f}' for t in avg_with_warmup)}")
        print(f"Avg (excl. warmup),   {','.join(f'{t * 1e3:.5f}' for t in avg_no_warmup)}")
        print(f"Total avg (all),      {sum(avg_with_warmup) * 1e3:.5f} ms")
        print(f"Total avg (no warmup),{sum(avg_no_warmup) * 1e3:.5f} ms")
        print("==============================================================")

    


if __name__ == "__main__":
    main()
