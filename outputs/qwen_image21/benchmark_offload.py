"""Time the validated I2I request with the DiT resident during denoising."""
import argparse
import json
import subprocess
import threading
import time
from pathlib import Path


def gpu_metrics(worker):
    import torch

    torch.cuda.synchronize()
    args = worker.fastvideo_args
    return {
        'peak_allocated_mib': torch.cuda.max_memory_allocated() / 2**20,
        'peak_reserved_mib': torch.cuda.max_memory_reserved() / 2**20,
        'effective_dit_layerwise_offload': args.dit_layerwise_offload,
        'effective_dit_cpu_offload': args.dit_cpu_offload,
        'effective_use_fsdp_inference': args.use_fsdp_inference,
    }


def main():
    process_start = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--layerwise', action='store_true')
    parser.add_argument('--output-dir', required=True)
    opts = parser.parse_args()
    out = Path(opts.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parent
    baseline = root / 'parity40/test_qwen_image21_modes_1024_p0'
    spec = json.loads((baseline / 'spec.json').read_text())

    import cloudpickle
    import torch
    from fastvideo import VideoGenerator
    from fastvideo.api import (EngineConfig, GenerationRequest, GeneratorConfig,
                               InputConfig, OffloadConfig, OutputConfig,
                               ParallelismConfig, PipelineSelection, SamplingConfig)

    stop = threading.Event()
    memory_samples = []

    def monitor():
        while not stop.is_set():
            result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used',
                                     '--format=csv,noheader,nounits'], capture_output=True, text=True)
            if result.returncode == 0:
                memory_samples.append({'seconds': time.perf_counter() - process_start,
                                       'used_mib': float(result.stdout.strip().splitlines()[0])})
            stop.wait(.2)

    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    generator = None
    report = {'case': 'i2i', 'height': spec['height'], 'width': spec['width'],
              'steps': spec['steps'], 'seed': spec['seed'], 'layerwise': opts.layerwise,
              'text_encoder_offload': True, 'vae_offload': True, 'lazy_module_load': True,
              'kv_cache_device': 'cpu', 'use_kv_cache': spec['cache'],
              'initial_latents': spec['latents'], 'prompt': spec['prompt'], 'references': spec['refs']}
    try:
        init_start = time.perf_counter()
        generator = VideoGenerator.from_config(GeneratorConfig(
            model_path=spec['model'],
            engine=EngineConfig(num_gpus=1, use_fsdp_inference=False,
                                parallelism=ParallelismConfig(tp_size=1, sp_size=1),
                                offload=OffloadConfig(dit=False, dit_layerwise=opts.layerwise,
                                                      text_encoder=True, vae=True, lazy_module_load=True)),
            pipeline=PipelineSelection(workload_type='i2i')))
        report['generator_initialization_seconds'] = time.perf_counter() - init_start
        noise = torch.load(spec['latents'], map_location='cpu', weights_only=True)
        request = GenerationRequest(
            prompt=spec['prompt'], negative_prompt=spec['negative_prompt'],
            inputs=InputConfig(references=spec['refs'], latents=noise.clone()),
            sampling=SamplingConfig(height=spec['height'], width=spec['width'], num_frames=1, fps=1,
                                    num_inference_steps=spec['steps'], guidance_scale=1,
                                    true_cfg_scale=spec['true_cfg_scale'], seed=spec['seed'],
                                    reference_resolution=spec['reference_resolution'],
                                    use_kv_cache=spec['cache'], sigmas=spec['sigmas']),
            output=OutputConfig(output_path=str(out / 'image.png'), return_frames=True, save_video=True))
        request_start = time.perf_counter()
        result = generator.generate(request)
        report['request_wall_seconds'] = time.perf_counter() - request_start
        report['process_start_to_image_seconds'] = time.perf_counter() - process_start
        report['initialization_plus_request_seconds'] = (report['generator_initialization_seconds']
                                                          + report['request_wall_seconds'])
        report['gpu_metrics'] = generator.executor.collective_rpc(cloudpickle.dumps(gpu_metrics))[0]
        if hasattr(result, 'e2e_latency'):
            report['api_e2e_latency_seconds'] = result.e2e_latency
        if hasattr(result, 'generation_time'):
            report['api_generation_time_seconds'] = result.generation_time
        expected = torch.load(baseline / 'native.pt', map_location='cpu', weights_only=True)['pixels']
        actual = result.samples.detach().float().cpu()
        difference = (actual - expected).abs()
        report['compared_with_validated_output'] = {'exact': torch.equal(actual, expected),
                                                  'mean_abs': difference.mean().item(),
                                                  'max_abs': difference.max().item()}
        report['status'] = 'success'
    except Exception as exc:
        report['status'] = 'failed'
        report['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        if generator is not None:
            generator.shutdown()
        stop.set()
        thread.join(timeout=2)
        report['sampled_peak_gpu_used_mib'] = max((s['used_mib'] for s in memory_samples), default=None)
        (out / 'gpu-memory.json').write_text(json.dumps(memory_samples, indent=2) + '\n')
        (out / 'metrics.json').write_text(json.dumps(report, indent=2) + '\n')
        print('OFFLOAD_BENCHMARK ' + json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
