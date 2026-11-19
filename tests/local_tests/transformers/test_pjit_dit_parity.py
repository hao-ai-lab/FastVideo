# SPDX-License-Identifier: Apache-2.0
"""Tiny Pyramid-JiT checkpoint parity against Linum's independent implementation.

Put the Linum-AI/pjit-21k checkout (95403933f3727abe947977c75915a53292e1d841)
on PYTHONPATH or set PJIT_REFERENCE_ROOT to its directory. The reference runs on CPU because it constructs fp64/complex128
embeddings. The MPS case compares the same fp32 fixture on Apple hardware.
"""
import os

import pytest
import torch

from fastvideo.configs.models.dits.pjit import PJiTArchConfig, PJiTConfig
from fastvideo.models.dits.pjit import PyramidJiT
from fastvideo.platforms import AttentionBackendEnum


PARITY_SCOPE = "implementation_subcomponent"


@pytest.mark.parametrize("device", ["cpu", pytest.param("mps", marks=pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="MPS unavailable"))])
@pytest.mark.parametrize("bf16", [False, True], ids=["fp32", "bf16-autocast"])
def test_pjit_dit_against_linum(device, bf16, monkeypatch):
    if root := os.environ.get("PJIT_REFERENCE_ROOT"):
        monkeypatch.syspath_prepend(root)
    official = pytest.importorskip("pjit_21k", reason="add the official Linum-AI/pjit-21k checkout to PYTHONPATH")
    from pjit_21k.config import PyramidJiTConfig

    torch.manual_seed(2075)
    arch = dict(dim=48, num_heads=2, num_layers=2, ffn_dim=96, refiner_blocks=1, refiner_ffn_dim=72,
                adaln_rank=8, text_dim=12, text_len=7, freq_dim=16, bottleneck_dim=8, patch=(1, 4, 4),
                rope_max_positions=16)

    official.set_attention_backend("sdpa")
    reference = official.PyramidJiT(PyramidJiTConfig(**arch)).eval()
    config = PJiTConfig(arch_config=PJiTArchConfig(**arch), _resolved_attention_backend=AttentionBackendEnum.TORCH_SDPA)
    # Exercise strict loading and explicit CPU RoPE initialization under the
    # same meta-construction pattern used to load the release.
    with torch.device("meta"):
        candidate = PyramidJiT(config)
    candidate.load_state_dict(reference.state_dict(), strict=True, assign=True)
    candidate = candidate.to(device).eval()
    # Odd image dimensions require patch padding/cropping. Different caption
    # lengths exercise masking independently in positive and negative branches.
    inputs = dict(x_t=torch.randn(2, 3, 1, 9, 14), text=torch.randn(2, 7, 12),
                  text_lens=torch.tensor([6, 2], dtype=torch.int32), t=torch.tensor([1000, 720]))
    check_layers = ("image_refiner.0", "text_refiner.0", "blocks.0", "blocks.1")

    def run(model, target_device):
        activations = {}
        handles = [model.get_submodule(name).register_forward_hook(
            lambda _m, _inputs, output, name=name: activations.update({name: output.detach().cpu()}))
                   for name in check_layers]
        with torch.inference_mode(), torch.autocast(target_device, dtype=torch.bfloat16, enabled=bf16):
            output = model(**{key: x.to(target_device) for key, x in inputs.items()})
        for handle in handles:
            handle.remove()
        activations["output"] = output.cpu()
        return activations

    expected = run(reference, "cpu")
    actual = run(candidate, device)
    assert actual["output"].shape == inputs["x_t"].shape
    for name in actual:
        # BF16 reductions depend on the device kernels. CPU uses the same
        # kernels on both sides and permits a tighter tolerance.
        atol, rtol = (2e-2, 1e-2) if bf16 and device == "mps" else (1e-5, 1e-5)
        torch.testing.assert_close(actual[name], expected[name], atol=atol, rtol=rtol, msg=lambda m: f"{name}: {m}")
