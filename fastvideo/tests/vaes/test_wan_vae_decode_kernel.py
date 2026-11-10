# SPDX-License-Identifier: Apache-2.0
"""``AutoencoderKLWan.decode`` through fastvideo-kernel's Wan2.2 decoder (``WanVAEConfig.decode_kernel``).

CPU tests: the switch is off unless asked for (PAN2 sampling asks), and a decode it cannot take runs the module's own
decode without loading the kernel. GPU tests (sm_100 / sm_103, the PAN2 VAE's weights in
``FASTVIDEO_TEST_PAN2_MODEL_DIR``): the kernel's decode against the module's fp32 decode, within three times the error
of a bf16 decode (the kernel's own rule); another geometry decoding exactly as before; two ranks decoding on their own
GPUs with no collective.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest
import torch

import fastvideo.envs as envs
from fastvideo.configs.pipelines.pan2 import PAN2PipelineConfig
from fastvideo.models.wan import vae as wan_vae
from fastvideo.models.wan.vae_config import WanVAEArchConfig, WanVAEConfig


def _tiny_vae(decode_kernel: bool) -> wan_vae.AutoencoderKLWan:
    torch.manual_seed(0)
    arch = WanVAEArchConfig(base_dim=8, z_dim=4, dim_mult=(1, 1), num_res_blocks=1, temperal_downsample=(False, ),
                            latents_mean=(0.0, ) * 4, latents_std=(1.0, ) * 4)
    config = WanVAEConfig(arch_config=arch)
    config.load_encoder, config.load_decoder, config.decode_kernel = False, True, decode_kernel
    return wan_vae.AutoencoderKLWan(config).eval()


def test_the_kernel_is_opt_in_and_pan2_sampling_opts_in() -> None:
    assert WanVAEConfig().decode_kernel is False and WanVAEConfig().decode_kernel_parallel is False
    pan2 = PAN2PipelineConfig()
    assert pan2.vae_decode_kernel is True and pan2.vae_decode_kernel_parallel is True and pan2.dit_fused_kernels


def test_a_decode_the_kernel_cannot_take_runs_the_module(monkeypatch) -> None:
    def no_kernel():
        raise AssertionError("a CPU decode must not load the kernel")

    monkeypatch.setattr(wan_vae, "_wan22_vae_decode", no_kernel)
    z = torch.randn(1, 4, 2, 4, 4)
    with torch.no_grad():
        plain = _tiny_vae(False).decode(z)
        routed = _tiny_vae(True).decode(z)
    assert torch.equal(plain, routed)


_ON_KERNEL_GPU = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="needs an sm_100 / sm_103 GPU")


@pytest.fixture
def pan2_vae_dir() -> str:
    root = envs.FASTVIDEO_TEST_PAN2_MODEL_DIR.get()
    if not root:
        pytest.skip("set FASTVIDEO_TEST_PAN2_MODEL_DIR to a PAN2 root with vae/")
    return os.path.join(root, "vae")


def _pan2_vae(vae_dir: str) -> wan_vae.AutoencoderKLWan:
    from fastvideo.models.loader.component_loader import VAELoader

    config = WanVAEConfig()
    config.load_encoder, config.load_decoder = False, True
    loader_args = SimpleNamespace(model_paths={}, vae_cpu_offload=False,
                                  pipeline_config=SimpleNamespace(vae_config=config, vae_precision="fp32"))
    return VAELoader().load(vae_dir, loader_args).to("cuda").eval()


@_ON_KERNEL_GPU
def test_the_kernel_decodes_pan2_latents_within_the_bf16_error(pan2_vae_dir) -> None:
    vae = _pan2_vae(pan2_vae_dir)
    z = torch.randn((1, 48, 3, 44, 78), generator=torch.Generator().manual_seed(3)).cuda()
    with torch.no_grad():
        vae.config.decode_kernel = False
        reference = vae.decode(z)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            bf16 = vae.decode(z)
        vae.config.decode_kernel = True
        kernel = vae.decode(z)
    assert list(vae._kernel_decoders) == [(z.device, None)], "the kernel path was not taken"
    assert kernel.shape == reference.shape == (1, 3, 9, 704, 1248) and kernel.dtype == torch.float32
    assert float(kernel.abs().max()) <= 1.0
    d, d_bf16 = kernel - reference, bf16.float() - reference
    assert float(d.abs().max()) <= 3 * float(d_bf16.abs().max())
    assert float(d.pow(2).mean().sqrt()) <= 3 * float(d_bf16.pow(2).mean().sqrt())


@_ON_KERNEL_GPU
def test_another_geometry_decodes_as_before_on_the_gpu(pan2_vae_dir) -> None:
    vae = _pan2_vae(pan2_vae_dir)
    z = torch.randn((1, 48, 2, 32, 32), generator=torch.Generator().manual_seed(4)).cuda()
    with torch.no_grad():
        vae.config.decode_kernel = False
        plain = vae.decode(z)
        vae.config.decode_kernel = True
        routed = vae.decode(z)
    assert torch.equal(plain, routed) and not vae._kernel_decoders


def _decode_alone(rank: int, store_path: str, vae_dir: str, z: torch.Tensor, out_path: str) -> None:
    import torch.distributed as dist
    from safetensors.torch import load_file

    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", init_method=f"file://{store_path}", rank=rank, world_size=2)
    try:
        weights = load_file(os.path.join(vae_dir, "diffusion_pytorch_model.safetensors"))
        decoder = wan_vae._wan22_vae_decode().WanVAEDecoder(weights, device=f"cuda:{rank}", local=True)
        video = decoder.decode(z)
        torch.save(video.cpu(), f"{out_path}.{rank}")
        dist.barrier(device_ids=[rank])
    finally:
        dist.destroy_process_group()


@_ON_KERNEL_GPU
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs two GPUs")
def test_two_ranks_decode_on_their_own_gpus(tmp_path, pan2_vae_dir) -> None:
    import torch.multiprocessing as mp

    # Build the extension here first, so the two ranks load it rather than both compiling into one directory.
    wan_vae._wan22_vae_decode()._module()
    z = torch.randn((1, 48, 2, 44, 78), generator=torch.Generator().manual_seed(5))
    out = str(tmp_path / "video")
    mp.start_processes(_decode_alone, args=(str(tmp_path / "store"), pan2_vae_dir, z, out), nprocs=2, join=True,
                       start_method="spawn")
    first, second = torch.load(f"{out}.0"), torch.load(f"{out}.1")
    assert first.shape == (1, 3, 5, 704, 1248) and torch.equal(first, second)
