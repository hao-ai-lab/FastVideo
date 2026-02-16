from copy import deepcopy
from dataclasses import dataclass
import os
from fastvideo import VideoGenerator
import multiprocessing as mp
import argparse

# Use the prompt from profile_ltx_e2e_vsa.py
PROMPT = (
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

OUTPUT_PATH = "video_samples_vsa"

@dataclass
class VSAConfig:
    name: str = "ltx2_vsa"
    model: str = "FastVideo/LTX2-Distilled-Diffusers"
    num_gpus: int = 8
    sparsity: float = 0.0
    
    # Generation params
    width: int = 1920
    height: int = 1088
    n_frames: int = 121
    fps: int = 24
    num_inference_steps: int = 8
    guidance_scale: float = 1.0

    def get_run_name(self) -> str:
        return f"{self.name}_{self.num_gpus}gpus_sparsity{self.sparsity:.2f}"


def run_benchmark(config: VSAConfig):
    """
    Function to run in a separate process.
    Configures environment and runs the generation loop with timing logging.
    """
    print(f"[{config.get_run_name()}] Starting benchmark process...")
    
    # 1. Set Environment Variables
    os.environ["FASTVIDEO_ATTENTION_BACKEND"] = "VIDEO_SPARSE_ATTN"
    os.environ["FASTVIDEO_STAGE_LOGGING"] = "1"
    
    # 2. Initialize Generator
    # Using settings from profile_ltx_e2e_vsa.py
    generator = VideoGenerator.from_pretrained(
        config.model,
        num_gpus=config.num_gpus,
        tp_size=1,
        sp_size=config.num_gpus, # sp_size = num_gpus as per profile_ltx_e2e_vsa
        use_fsdp_inference=False,
        dit_cpu_offload=False,
        vae_cpu_offload=False,
        text_encoder_cpu_offload=False,
        pin_cpu_memory=True,
        dit_layerwise_offload=False,
        VSA_sparsity=config.sparsity,
    )

    # 3. Generation Loop
    all_stage_times: list[list[float]] = []
    stage_names: list[str] = []
    
    # Ensure output directory exists
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    output_filename = os.path.join(OUTPUT_PATH, f"{config.get_run_name()}.mp4")

    # Run loop (using 10 iterations from profile_ltx_e2e_vsa for stable timing)
    num_iterations = 10
    
    print(f"[{config.get_run_name()}] Warming up and running {num_iterations} iterations...")

    for i in range(num_iterations):
        # We only save video on the last iteration to verify quality, 
        # or disable it to save time/IO like profile_ltx_e2e_vsa. 
        # bench_ltx2 saves every time. Let's save on the last one.
        save_current = (i == num_iterations - 1)
        
        result = generator.generate_video(
            prompt=PROMPT,
            output_path=output_filename if save_current else "null",
            save_video=save_current,
            negative_prompt="",
            num_frames=config.n_frames,
            fps=config.fps,
            height=config.height,
            width=config.width,
            guidance_scale=config.guidance_scale,
            num_inference_steps=config.num_inference_steps,
        )

        # 4. Extract Timing
        logging_info = result.get("logging_info") if isinstance(result, dict) else None
        if logging_info is None:
            print(f"[{config.get_run_name()}] Iter {i}: No logging_info returned.")
            continue
            
        current_stage_names = logging_info.get_execution_order()
        current_stage_times = [
            logging_info.get_stage_info(name).get("execution_time", 0.0)
            for name in current_stage_names
        ]
        
        # update stage names if first run
        if not stage_names:
            stage_names = current_stage_names
            
        all_stage_times.append(current_stage_times)
        
        # Print iteration summary
        total_time = sum(current_stage_times)
        print(f"[{config.get_run_name()}] Iter {i}: Total Distilled Time = {total_time*1000:.2f} ms")


    # 5. Calculate and Print Statistics
    if all_stage_times and stage_names:
        num_stages = len(stage_names)
        
        # Average excluding first round (warmup)
        times_no_warmup = all_stage_times[1:] if len(all_stage_times) > 1 else all_stage_times
        avg_no_warmup = [
            sum(times[j] for times in times_no_warmup) / len(times_no_warmup)
            for j in range(num_stages)
        ]
        
        total_avg_ms = sum(avg_no_warmup) * 1000
        
        print(f"\n========== SUMMARY: {config.get_run_name()} ==========")
        print(f"Sparsity: {config.sparsity}")
        print("-" * 60)
        print(f"{'Stage':<30} | {'Avg Time (ms)':<15}")
        print("-" * 60)
        for name, t in zip(stage_names, avg_no_warmup):
             print(f"{name:<30} | {t*1000:.5f}")
        print("-" * 60)
        print(f"{'Total (excluding warmup)':<30} | {total_avg_ms:.5f}")
        print("======================================================\n")
        
    print(f"[{config.get_run_name()}] Benchmark finished.")


def main():
    # Enforce spawn to avoid CUDA context inheritance
    mp.set_start_method("spawn", force=True)
    
    parser = argparse.ArgumentParser(description="Benchmark VSA Sparsity Levels")
    parser.add_argument("--gpus", type=int, default=8, help="Number of GPUs to use")
    parser.add_argument("--sparsities", type=float, nargs="+", default=[0.0, 0.5, 0.7, 0.8, 0.9], help="List of sparsity levels to test")
    args = parser.parse_args()

    # Define configurations to run
    configs = []
    
    for s in args.sparsities:
        cfg = VSAConfig(
            num_gpus=args.gpus,
            sparsity=s
        )
        configs.append(cfg)

    print(f"Scheduled {len(configs)} benchmarks: {[c.get_run_name() for c in configs]}")

    # Run each config in a separate process
    for cfg in configs:
        p = mp.Process(target=run_benchmark, args=(cfg,))
        p.start()
        p.join()  # Wait for process to finish before starting next one to avoid OOM/Interference

if __name__ == "__main__":
    main()
