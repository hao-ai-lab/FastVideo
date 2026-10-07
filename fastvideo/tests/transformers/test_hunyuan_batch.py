# SPDX-License-Identifier: Apache-2.0
"""Regression: HunyuanVideo must support a forward pass with batch size two.

Requires one CUDA GPU; tiny random weights avoid checkpoint downloads.

Run with: python -m pytest fastvideo/tests/transformers/test_hunyuan_batch.py -q
The existing transformer CI lane collects this directory.
"""
import contextlib
import os
import socket

import pytest
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

import fastvideo.envs as envs
from fastvideo.forward_context import set_forward_context
from fastvideo.configs.models.dits.hunyuanvideo import HunyuanVideoArchConfig, HunyuanVideoConfig
from fastvideo.configs.models.dits.hunyuanvideo15 import HunyuanVideo15ArchConfig, HunyuanVideo15Config
from fastvideo.models.dits.hunyuanvideo import HunyuanVideoTransformer3DModel
from fastvideo.models.dits.hunyuanvideo15 import HunyuanVideo15Transformer3DModel

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires one CUDA GPU")


@pytest.fixture
def hunyuan_setup(monkeypatch, request):
    # Environment writes go through the registry helpers (docs/contributing/env_vars.md).
    with contextlib.ExitStack() as stack:
        # Respect the port reserved by the CI runner when sharing a GPU host.
        if "MASTER_PORT" not in os.environ:
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 0))
                port = str(sock.getsockname()[1])
            stack.enter_context(envs.override_external("MASTER_PORT", port))
        if "MASTER_ADDR" not in os.environ:
            stack.enter_context(envs.override_external("MASTER_ADDR", "127.0.0.1"))
        stack.enter_context(envs.FASTVIDEO_ATTENTION_BACKEND.override("TORCH_SDPA"))
        monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
        monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", False)
        request.getfixturevalue("distributed_setup")
        yield


def _initialize(module):
    # ReplicatedLinear uses empty storage; initialize every parameter explicitly.
    with torch.no_grad():
        for name, parameter in module.named_parameters():
            if parameter.ndim > 1:
                torch.nn.init.xavier_uniform_(parameter)
            elif name.endswith("weight"):
                parameter.fill_(1)
            else:
                parameter.zero_()


def test_hunyuan_supports_batch_two(hunyuan_setup):
    config = HunyuanVideoConfig(arch_config=HunyuanVideoArchConfig(
        in_channels=4, out_channels=4, num_attention_heads=4, attention_head_dim=8,
        num_layers=2, num_single_layers=2, num_refiner_layers=1, mlp_ratio=2,
        patch_size=1, patch_size_t=1, rope_axes_dim=(2, 2, 4), text_embed_dim=12,
        pooled_projection_dim=8, guidance_embeds=False))
    model = HunyuanVideoTransformer3DModel(config, {}).to(device="cuda", dtype=torch.float32).eval()
    _initialize(model)

    # Batch size 2, with 6 visual tokens per sample. Before the fix, modulation
    # shaped [2, C] cannot broadcast over hidden states shaped [2, 6, C].
    latents = torch.randn(2, 4, 1, 2, 3, device="cuda")
    text = torch.randn(2, 5, 12, device="cuda")
    pooled_text = torch.randn(2, 8, device="cuda")
    timestep = torch.tensor([10, 20], device="cuda", dtype=torch.float32)

    with torch.no_grad(), set_forward_context(0, None), sdpa_kernel(SDPBackend.MATH):
        output = model(
            hidden_states=latents,
            encoder_hidden_states=[text, pooled_text],
            timestep=timestep,
        )

    assert output.shape == latents.shape
    assert torch.isfinite(output).all()


def test_hunyuan15_supports_batch_two(hunyuan_setup):
    config = HunyuanVideo15Config(arch_config=HunyuanVideo15ArchConfig(
        in_channels=4, out_channels=4, num_attention_heads=4, attention_head_dim=8,
        num_layers=2, num_refiner_layers=1, mlp_ratio=2, patch_size=1, patch_size_t=1,
        rope_axes_dim=(2, 2, 4), text_embed_dim=12, text_embed_2_dim=8, image_embed_dim=8))
    model = HunyuanVideo15Transformer3DModel(config, {}).to(device="cuda", dtype=torch.float32).eval()
    _initialize(model)

    # Batch size 2 and 6 visual tokens also reproduce the separate 1.5 path's
    # modulation broadcast error. Zero image conditioning selects T2V.
    latents = torch.randn(2, 4, 1, 2, 3, device="cuda")
    text = torch.randn(2, 5, 12, device="cuda")
    text2 = torch.randn(2, 3, 8, device="cuda")
    image = torch.zeros(2, 4, 8, device="cuda")
    timestep = torch.tensor([10, 20], device="cuda", dtype=torch.float32)

    with torch.no_grad(), set_forward_context(0, None), sdpa_kernel(SDPBackend.MATH):
        output = model(
            hidden_states=latents,
            encoder_hidden_states=[text, text2],
            encoder_hidden_states_image=[image],
            timestep=timestep,
        )

    assert output.shape == latents.shape
    assert torch.isfinite(output).all()
