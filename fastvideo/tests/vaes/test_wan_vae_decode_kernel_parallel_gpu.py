# SPDX-License-Identifier: Apache-2.0
"""``AutoencoderKLWan.decode`` through fastvideo-kernel's Wan2.2 decoder split over the sequence-parallel group
(``WanVAEConfig.decode_kernel_parallel``), as PAN2 sampling decodes on several GPUs.

Needs a multi-GPU torchrun launch (real NCCL collectives over the group), sm_100 / sm_103 GPUs and the PAN2 VAE's
weights in ``FASTVIDEO_TEST_PAN2_MODEL_DIR``; skipped otherwise:

    FASTVIDEO_TEST_PAN2_MODEL_DIR=<PAN2 root> torchrun --nproc-per-node=4 -m pytest \\
        fastvideo/tests/vaes/test_wan_vae_decode_kernel_parallel_gpu.py -q

Every rank must return the same video, the group's first rank's, within the kernel's own error rule against the
module's fp32 decode (three times the error of a bf16 decode, as the one-GPU test); a video with fewer latent frames
than GPUs decodes on each GPU alone, and a rank that cannot build the kernel makes the group decode alone without
hanging the others. Over two GPUs the split decode is
bit for bit the video one GPU decodes; over four and eight the bands of rows round differently, at the same error.
"""

import os
from types import SimpleNamespace

import pytest
import torch

import fastvideo.envs as envs
from fastvideo.models.wan.vae_config import WanVAEConfig

_WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "1"))

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
                       reason="needs sm_100 / sm_103 GPUs"),
    pytest.mark.skipif(_WORLD_SIZE < 2, reason="needs a torchrun launch with WORLD_SIZE > 1"),
]


@pytest.fixture(scope="module")
def sp_group():
    from fastvideo.distributed import get_sp_group, maybe_init_distributed_environment_and_model_parallel
    maybe_init_distributed_environment_and_model_parallel(1, _WORLD_SIZE)
    return get_sp_group()


@pytest.fixture(scope="module")
def pan2_vae_dir() -> str:
    root = envs.FASTVIDEO_TEST_PAN2_MODEL_DIR.get()
    if not root:
        pytest.skip("set FASTVIDEO_TEST_PAN2_MODEL_DIR to a PAN2 root with vae/")
    return os.path.join(root, "vae")


def _load_pan2_vae(vae_dir: str):
    from fastvideo.models.loader.component_loader import VAELoader

    config = WanVAEConfig()
    config.load_encoder, config.load_decoder, config.decode_kernel = False, True, True
    loader_args = SimpleNamespace(model_paths={}, vae_cpu_offload=False,
                                  pipeline_config=SimpleNamespace(vae_config=config, vae_precision="fp32"))
    return VAELoader().load(vae_dir, loader_args).to("cuda").eval()


@pytest.fixture(scope="module")
def pan2_vae(pan2_vae_dir):
    return _load_pan2_vae(pan2_vae_dir)


@pytest.mark.parametrize("latent_frames", (1, 2, 13))
@torch.no_grad()
def test_every_rank_returns_the_split_decode(sp_group, pan2_vae, latent_frames: int) -> None:
    z = torch.randn((1, 48, latent_frames, 44, 78), generator=torch.Generator().manual_seed(latent_frames)).cuda()
    pan2_vae.config.decode_kernel = False
    reference = pan2_vae.decode(z)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        bf16 = pan2_vae.decode(z)
    pan2_vae.config.decode_kernel, pan2_vae.config.decode_kernel_parallel = True, False
    alone = pan2_vae.decode(z)
    pan2_vae.config.decode_kernel_parallel = True
    split = pan2_vae.decode(z)
    if latent_frames < sp_group.world_size:  # fewer latent frames than GPUs: each GPU decodes alone
        assert torch.equal(split, alone)
        return
    assert pan2_vae._kernel_decoders.get((z.device, tuple(sp_group.ranks))) is not None, "the split was not taken"
    assert split.shape == reference.shape == (1, 3, 4 * latent_frames - 3, 704, 1248) and split.dtype == torch.float32
    assert torch.equal(pan2_vae.decode(z), split), "a second split decode (the group's decoder reused) differs"

    checksum = split.double().sum().reshape(1)
    low, high = checksum.clone(), checksum.clone()
    torch.distributed.all_reduce(low, op=torch.distributed.ReduceOp.MIN, group=sp_group.device_group)
    torch.distributed.all_reduce(high, op=torch.distributed.ReduceOp.MAX, group=sp_group.device_group)
    assert torch.equal(low, high), "the ranks hold different videos"

    d, d_bf16 = split - reference, bf16.float() - reference
    assert float(d.abs().max()) <= 3 * float(d_bf16.abs().max())
    assert float(d.pow(2).mean().sqrt()) <= 3 * float(d_bf16.pow(2).mean().sqrt())
    if sp_group.world_size == 2:
        assert torch.equal(split, alone)


@torch.no_grad()
def test_a_rank_without_the_kernel_makes_the_group_decode_alone(sp_group, pan2_vae_dir) -> None:
    """Group rank 1 could not build the kernel (here: an earlier failure on that rank alone). Every rank still joins the
    group's one agreement, so none waits on it forever, and they all decode alone: rank 1 with AutoencoderKLWan, the
    others with the kernel on their own GPU."""
    vae = _load_pan2_vae(pan2_vae_dir)
    vae.config.decode_kernel_parallel = True
    if sp_group.rank_in_group == 1:
        vae._kernel_unavailable = "simulated: this rank could not build the kernel"
    z = torch.randn((1, 48, 13, 44, 78), generator=torch.Generator().manual_seed(7)).cuda()
    for _ in range(2):  # the second decode reads the group's cached verdict without communicating
        video = vae.decode(z)
        assert video.shape == (1, 3, 49, 704, 1248) and video.dtype == torch.float32
    assert vae._kernel_decoders[(z.device, tuple(sp_group.ranks))] is None
    assert (vae._kernel_decoders[(z.device, None)] is None) == (sp_group.rank_in_group == 1)
