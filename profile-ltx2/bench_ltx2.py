from copy import deepcopy
from dataclasses import dataclass
import os
from fastvideo import VideoGenerator
import json
import multiprocessing as mp

from fastvideo.configs.sample.base import SamplingParam

OUTPUT_PATH = "video_samples"


@dataclass
class BenchConfig:
    name: str

    model: str

    attn_str: str
    num_gpus: int
    sp_size: int

    width: int
    height: int
    n_frames: int = 121
    fps: int = 24

    def get_full_qualified_name(self) -> str:
        return (
            f"{self.name}_{self.num_gpus}gpus_sp{self.sp_size}_{self.attn_str}"
        )


def write_to_flag(flag_path: str, content: str):
    with open(flag_path, "w") as f:
        f.write(content)


def benchmark(bench_config: BenchConfig, file_path_prefix: str):
    os.environ["FASTVIDEO_ATTENTION_BACKEND"] = bench_config.attn_str
    os.environ["FASTVIDEO_STAGE_LOGGING"] = "1"
    flag_path = "/tmp/dit_log_flag.txt"
    write_to_flag(flag_path, file_path_prefix + "_0.json")
    sampling_param = SamplingParam.from_pretrained(bench_config.model)
    sampling_param.guidance_scale = 1.0

    generator = VideoGenerator.from_pretrained(
        bench_config.model,
        # FastVideo will automatically handle distributed setup
        num_gpus=bench_config.num_gpus,
        use_fsdp_inference=False,  # set to True if GPU is out of memory
        dit_cpu_offload=False,
        vae_cpu_offload=False,
        text_encoder_cpu_offload=False,
        pin_cpu_memory=True,  # set to false if low CPU RAM or hit obscure "CUDA error: Invalid argument"
        image_encoder_cpu_offload=False,
        dit_layerwise_offload=False,
        sp_size=bench_config.sp_size,
        tp_size=1,
        enable_torch_compile=True,
    )

    prompt = (
        "A curious raccoon peers through a vibrant field of yellow sunflowers, its eyes "
        "wide with interest. The playful yet serene atmosphere is complemented by soft "
        "natural light filtering through the petals. Mid-shot, warm and cheerful tones."
    )
    prompt = (
        "A warm sunny backyard. The camera starts in a tight cinematic close-up "
        "of a woman and a man in their 30s, facing each other with serious "
        "expressions. The woman, emotional and dramatic, says softly, \"That's "
        "it... Dad's lost it. And we've lost Dad.\" The man exhales, slightly "
        'annoyed: "Stop being so dramatic, Jess." A beat. He glances aside, '
        'then mutters defensively, "He\'s just having fun." The camera slowly '
        "pans right, revealing the grandfather in the garden wearing enormous "
        "butterfly wings, waving his arms in the air like he's trying to take "
        'off. He shouts, "Wheeeew!" as he flaps his wings with full commitment. '
        "The woman covers her face, on the verge of tears. The tone is deadpan, "
        "absurd, and quietly tragic."
    )

    video = generator.generate_video(
        prompt,
        output_path=f"{OUTPUT_PATH}/{bench_config.get_full_qualified_name()}_first.mp4",
        sampling_param=sampling_param,
        save_video=True,
        negative_prompt="",
        num_frames=bench_config.n_frames,
        fps=bench_config.fps,
        height=bench_config.height,
        width=bench_config.width,
    )
    write_to_flag(flag_path, file_path_prefix + "_1.json")
    video = generator.generate_video(
        prompt,
        output_path=f"{OUTPUT_PATH}/{bench_config.get_full_qualified_name()}_second.mp4",
        sampling_param=sampling_param,
        save_video=True,
        negative_prompt="",
        num_frames=bench_config.n_frames,
        fps=bench_config.fps,
        height=bench_config.height,
        width=bench_config.width,
    )
    write_to_flag(flag_path, file_path_prefix + "_2.json")
    video = generator.generate_video(
        prompt,
        output_path=f"{OUTPUT_PATH}/{bench_config.get_full_qualified_name()}_third.mp4",
        sampling_param=sampling_param,
        save_video=True,
        negative_prompt="",
        num_frames=bench_config.n_frames,
        fps=bench_config.fps,
        height=bench_config.height,
        width=bench_config.width,
    )


def main():
    # Set multiprocessing start method to 'spawn' to avoid CUDA initialization issues
    mp.set_start_method("spawn", force=True)
    ltx2 = BenchConfig(
        name="ltx2",
        model="FastVideo/LTX2-Distilled-Diffusers",
        # model="Davids048/LTX2-Base-Diffusers",
        attn_str="FLASH_ATTN",
        num_gpus=1,
        sp_size=1,
        width=1920,
        height=1088,
    )
    # atten = "SAGE_ATTN"
    atten = "FLASH_ATTN"
    will_overwrite = False
    # for attn in [atten]:
    for attn in ["FLASH_ATTN"]:
        for bench_config in [ltx2]:
            for gpus in [1]:
                cfg = deepcopy(bench_config)
                cfg.attn_str = attn
                cfg.num_gpus = gpus
                cfg.sp_size = gpus
                file_path_prefix = (
                    f"runs/output/{cfg.get_full_qualified_name()}"
                )
                if will_overwrite:
                    if os.path.exists(file_path_prefix):
                        os.remove(file_path_prefix)
                if os.path.exists(file_path_prefix):
                    print(
                        f"Skip existing benchmark: {cfg.get_full_qualified_name()}"
                    )
                    continue

                # Run benchmark in a subprocess for clean isolation
                process = mp.Process(
                    target=benchmark, args=(cfg, file_path_prefix)
                )
                process.start()
                process.join()  # Wait for the benchmark to complete


if __name__ == "__main__":
    main()