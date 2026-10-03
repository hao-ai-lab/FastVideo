"""Headline e2e benchmark for the FastH3 local release: 480p 5 s and 768p 10 s.

usage: python bench_headline.py <run_name> <model_dir> <num_gpus> <nvfp4_profile|none> [--settings 480p5s,768p10s]

Protocol (RELEASE_PLAN §5): prompts latency-ceramics-005 + latency-harbor-005, one untimed warmup per
setting, then each prompt timed twice; e2e = generate_video wall time (encode + DiT + decode + mp4 write).
Per-stage times come from FASTVIDEO_STAGE_LOGGING. Clips and results.json go to <out>/<run_name>/,
and one W&B run per invocation (group headline-<model tag>) when WANDB_PROJECT is set.
"""
import argparse
import json
import os
import statistics
import time

SETTINGS = {"480p5s": (832, 480, 124), "768p10s": (1344, 768, 243)}  # 17n+5 frames at 24 fps
PROMPT_IDS = ("latency-ceramics-005", "latency-harbor-005")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_name")
    ap.add_argument("model_dir")
    ap.add_argument("num_gpus", type=int)
    ap.add_argument("nvfp4_profile")
    ap.add_argument("--settings", default="480p5s,768p10s")
    ap.add_argument("--prompts", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "headline_prompts.json"))
    ap.add_argument("--out", default=os.environ.get("HEADLINE_OUT", "headline"))
    ap.add_argument("--timed", type=int, default=2)
    ap.add_argument("--sparsity", type=float, default=None, help="default: the checkpoint contract's vsa_sparsity")
    a = ap.parse_args()
    if a.timed < 1:
        ap.error("--timed must be at least 1")

    contract = json.load(open(os.path.join(a.model_dir, "fastvideo_inference.json")))
    steps = contract["dmd_denoising_steps"]
    sparsity = a.sparsity if a.sparsity is not None else float(contract.get("vsa_sparsity", 0.8))
    texts = json.load(open(a.prompts))
    out_dir = os.path.join(a.out, a.run_name)
    os.makedirs(out_dir, exist_ok=True)

    engine = {"num_gpus": a.num_gpus, "use_fsdp_inference": False,
              "parallelism": {"tp_size": 1, "sp_size": a.num_gpus},
              "offload": {"dit": False, "dit_layerwise": False, "text_encoder": False, "vae": False,
                          "pin_cpu_memory": False, "lazy_module_load": False},
              "compile": {"enabled": False, "vae_enabled": os.environ.get("HEADLINE_VAE_COMPILE", "1") == "1"}}
    if os.environ.get("HEADLINE_BACKEND"):
        engine["execution_backend"] = os.environ["HEADLINE_BACKEND"]  # "ray" for multi-node
    if a.nvfp4_profile != "none":
        engine["quantization"] = {"transformer_quant": "NVFP4", "layer_profile": a.nvfp4_profile}
    experimental = {"attention_backend": "VIDEO_SPARSE_ATTN_H3", "VSA_sparsity": sparsity, "VSA_tile_size": 64,
                    "h3_sequential_load": False, "inference_torch_compile": False}
    if os.environ.get("HEADLINE_VAE_PARALLEL") == "1" and a.num_gpus > 1:
        # Decode VAE tiles on every GPU instead of rank 0 only.
        experimental.update(vae_parallel_decode=True, vae_parallel_decode_strategy="gather")
    # Placement overrides for memory-limited GPUs, e.g. on a 32 GB RTX 5090:
    # HEADLINE_ENGINE_JSON='{"offload": {"text_encoder": true, "pin_cpu_memory": true}}'
    # HEADLINE_EXPERIMENTAL_JSON='{"h3_sequential_load": true}'
    for key, value in json.loads(os.environ.get("HEADLINE_ENGINE_JSON", "{}")).items():
        if isinstance(value, dict) and isinstance(engine.get(key), dict):
            engine[key].update(value)
        else:
            engine[key] = value
    experimental.update(json.loads(os.environ.get("HEADLINE_EXPERIMENTAL_JSON", "{}")))
    config = {"model_path": a.model_dir, "engine": engine, "pipeline": {"experimental": experimental}}
    env = {k: v for k, v in os.environ.items() if k.startswith(("FASTVIDEO_", "PYTORCH_CUDA"))}
    run = None
    if os.environ.get("WANDB_PROJECT"):
        import wandb
        run = wandb.init(project=os.environ["WANDB_PROJECT"], entity=os.environ.get("WANDB_ENTITY"),
                         name=a.run_name, group=os.environ.get("HEADLINE_GROUP", "headline"), job_type="benchmark",
                         dir=out_dir, config={"model_dir": a.model_dir, "num_gpus": a.num_gpus,
                                              "nvfp4_profile": a.nvfp4_profile, "dmd_steps": steps,
                                              "vsa_sparsity": sparsity, "settings": a.settings, "engine": engine,
                                              "env": env, "device": os.environ.get("HEADLINE_DEVICE", "")})

    from fastvideo import VideoGenerator
    t0 = time.perf_counter()
    generator = VideoGenerator.from_config(config)
    results = {"run_name": a.run_name, "model_dir": a.model_dir, "num_gpus": a.num_gpus,
               "nvfp4_profile": a.nvfp4_profile, "load_s": round(time.perf_counter() - t0, 1), "env": env,
               "settings": {}}
    try:
        for name in a.settings.split(","):
            width, height, frames = SETTINGS[name]
            runs = []
            plan = [(PROMPT_IDS[0], True)] + [(pid, False) for _ in range(a.timed) for pid in PROMPT_IDS]
            for i, (pid, warmup) in enumerate(plan):
                path = os.path.join(out_dir, f"{name}_{pid}_{'warmup' if warmup else i}.mp4")
                t = time.perf_counter()
                generator.generate_video(prompt=texts[pid], height=height, width=width, num_frames=frames, fps=24,
                                         guidance_scale=1.0, num_inference_steps=len(steps) + 1, seed=1234,
                                         output_path=path, save_video=True)
                wall = round(time.perf_counter() - t, 2)
                runs.append({"prompt": pid, "warmup": warmup, "e2e_s": wall, "path": path})
                print("RUN", name, pid, "warmup" if warmup else "timed", wall, flush=True)
                if run is not None and not warmup:
                    import wandb
                    run.log({f"{name}/e2e_s": wall, f"{name}/{pid}": wandb.Video(path, fps=24, format="mp4")})
            timed = [r["e2e_s"] for r in runs if not r["warmup"]]
            results["settings"][name] = {"width": width, "height": height, "frames": frames,
                                         "e2e_median_s": statistics.median(timed), "e2e_min_s": min(timed), "runs": runs}
            print("SETTING", name, json.dumps(results["settings"][name]), flush=True)
            if run is not None:
                run.summary[f"{name}_e2e_median_s"] = statistics.median(timed)
            json.dump(results, open(os.path.join(out_dir, "results.json"), "w"), indent=1)
    finally:
        generator.shutdown()
    if run is not None:
        run.summary["load_s"] = results["load_s"]
        run.finish()
    print("HEADLINE_DONE", a.run_name, flush=True)


if __name__ == "__main__":
    main()
