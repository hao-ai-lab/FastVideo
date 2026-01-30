import json
import time

import torch

from fastvideo import VideoGenerator
from fastvideo.models.vaes.ltx2vae import TilingConfig

MODEL_ID = "FastVideo/LTX2-Distilled-Diffusers"
LATENT_SHAPE = (1, 128, 16, 34, 60)
GPU_COUNTS = [8, 4, 2, 1]
# GPU_COUNTS = [2]


def _time_parallel_tiled_decode(worker, latent_shape):
    import torch.distributed as dist
    from fastvideo.distributed.parallel_state import get_local_torch_device

    device = get_local_torch_device()
    vae = worker.pipeline.modules["vae"]
    vae.eval()
    vae.enable_tiling()

    dtype = next(vae.decoder.parameters()).dtype
    rank = dist.get_rank()

    if rank == 0:
        latent = torch.randn(latent_shape, device=device, dtype=dtype)
    else:
        latent = torch.empty(latent_shape, device=device, dtype=dtype)
    dist.broadcast(latent, src=0)

    tiling_config = TilingConfig.default()

    with torch.inference_mode():
        with torch.autocast(device_type="cuda",
                            dtype=dtype,
                            enabled=torch.cuda.is_available()):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start = time.perf_counter()

            # All ranks participate in parallel_tiled_decode (for distributed decode)
            # but only rank 0 gets results (others yield nothing and return early)
            result_chunks = list(
                vae.parallel_tiled_decode(latent, tiling_config=tiling_config))

            # Only rank 0 has results to concatenate
            if result_chunks:
                output = torch.cat(result_chunks, dim=2)
            else:
                output = None  # Non-rank-0 workers have nothing

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start

    if output is not None:
        del output
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Only rank 0 returns timing (other ranks' results are discarded anyway)
    return {"seconds": elapsed, "rank": rank}


def run():
    results = {}

    for num_gpus in GPU_COUNTS:
        try:
            generator = VideoGenerator.from_pretrained(
                MODEL_ID,
                num_gpus=num_gpus,
                sp_size=num_gpus,
                tp_size=1,
                use_fsdp_inference=False,
                dit_cpu_offload=True,
                vae_cpu_offload=False,
                text_encoder_cpu_offload=True,
                pin_cpu_memory=True,
            )
            try:
                warmup_times = []
                for _ in range(2):
                    rpc_warmup = generator.executor.collective_rpc(
                        _time_parallel_tiled_decode,
                        kwargs={"latent_shape": LATENT_SHAPE},
                    )
                    if rpc_warmup and "seconds" in rpc_warmup[0]:
                        warmup_times.append(rpc_warmup[0]["seconds"])

                rpc_results = generator.executor.collective_rpc(
                    _time_parallel_tiled_decode,
                    kwargs={"latent_shape": LATENT_SHAPE},
                )

                if not rpc_results:
                    results[num_gpus] = {"error": "no results"}
                else:
                    primary = rpc_results[0]
                    if "seconds" in primary:
                        results[num_gpus] = {
                            "warmup_seconds": warmup_times,
                            "measured_seconds": primary["seconds"],
                        }
                    else:
                        results[num_gpus] = {"error": str(primary)}
            finally:
                generator.shutdown()
        except Exception as exc:  # noqa: BLE001
            results[num_gpus] = {"error": str(exc)}

    print(json.dumps({"parallel_tiled_decode": results}, indent=2))


if __name__ == "__main__":
    run()
