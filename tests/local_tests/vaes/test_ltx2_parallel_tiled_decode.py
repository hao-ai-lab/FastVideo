import pytest
import pytest
import torch

from fastvideo import VideoGenerator
from fastvideo.models.vaes.ltx2vae import TilingConfig


def _parallel_vs_tiled_decode(worker, latent_shape, use_bf16=True):
    import torch
    import torch.distributed as dist
    from fastvideo.distributed.parallel_state import get_local_torch_device

    device = get_local_torch_device()

    vae = worker.pipeline.modules["vae"]
    vae.eval()
    vae.enable_tiling()

    dtype = next(vae.decoder.parameters()).dtype

    if dist.get_rank() == 0:
        z = torch.randn(latent_shape, device=device, dtype=dtype)
    else:
        z = torch.empty(latent_shape, device=device, dtype=dtype)
    dist.broadcast(z, src=0)

    # Run parallel decode on all ranks; run tiled decode on rank 0 only
    with torch.inference_mode():
        with torch.autocast(device_type="cuda",
                            dtype=dtype,
                            enabled=torch.cuda.is_available()):
            out_parallel = torch.cat(
                list(
                    vae.parallel_tiled_decode(z,
                                              tiling_config=TilingConfig.default()
                                              )),
                dim=2)
        out_parallel_cpu = out_parallel.float().cpu()
        del out_parallel
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        dist.barrier()

        if dist.get_rank() == 0:
            with torch.autocast(device_type="cuda",
                                dtype=dtype,
                                enabled=torch.cuda.is_available()):
                out_tiled = torch.cat(
                    list(vae.tiled_decode(
                        z, tiling_config=TilingConfig.default())),
                    dim=2)
            out_tiled_cpu = out_tiled.float().cpu()
            del out_tiled
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            torch.testing.assert_close(out_parallel_cpu,
                                       out_tiled_cpu,
                                       rtol=1e-3,
                                       atol=1e-3)

        dist.barrier()

    return {"status": "ok"}


@pytest.mark.cuda
def test_ltx2_parallel_tiled_decode_parity():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for this test")
    if torch.cuda.device_count() < 2:
        pytest.skip("Need at least 2 GPUs for this test")

    generator = VideoGenerator.from_pretrained(
        "FastVideo/LTX2-Distilled-Diffusers",
        num_gpus=2,
        sp_size=2,
        tp_size=1,
        use_fsdp_inference=False,
        dit_cpu_offload=True,
        vae_cpu_offload=False,
        text_encoder_cpu_offload=True,
        pin_cpu_memory=True,
    )

    try:
        # Small latent to avoid OOM while exercising tiling
        latent_shape = (1, 128, 16, 34, 60)
        results = generator.executor.collective_rpc(
            _parallel_vs_tiled_decode,
            kwargs={
                "latent_shape": latent_shape,
                "use_bf16": True,
            },
        )
        assert results, "No results returned from workers"
        assert results[0]["status"] == "ok"
    finally:
        generator.shutdown()
