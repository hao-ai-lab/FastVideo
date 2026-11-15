"""The Wan2.2 VAE decoder (fastvideo-kernel/wan22_vae_decode) through its wrapper, against the reference.

Needs an sm_100a or sm_103a GPU (B200 / GB200, B300) and the VAE's weights: WAN22_VAE_WEIGHTS names either a diffusers AutoencoderKLWan
directory (its diffusion_pytorch_model.safetensors) or a .pt file holding the 110 tensors in weight_names() order. The
first test builds the extension (about a minute). The kernel's own checks are wan22_vae_decode/tools/bench.py (nine
timed workloads and six short clips at 1, 2 and 4 GPUs, on real latents); these tests cover the wrapper on one GPU:
the weights it reads from a state dict, and decodes against the fp32 reference, each within three times what the bf16
reference errs by on the same latents (the bench's rule, calibrated per input instead of per clip).
"""

import functools
import os
import sys

import pytest
import torch

PACKAGE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "wan22_vae_decode")
sys.path.insert(0, PACKAGE)
sys.path.insert(0, os.path.join(PACKAGE, "tools"))

WEIGHTS = os.environ.get("WAN22_VAE_WEIGHTS", "")

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
                       reason="the Wan2.2 VAE decoder is built for sm_100a / sm_103a (B200 / GB200, B300)"),
    pytest.mark.skipif(not WEIGHTS or not os.path.exists(WEIGHTS),
                       reason="set WAN22_VAE_WEIGHTS to the VAE's diffusers directory or a weights .pt file"),
]


@functools.lru_cache(maxsize=None)
def _state_dict():
    """name -> tensor of the VAE's parameters, under diffusers' names."""
    from wan22_vae_decode import weight_names

    if os.path.isdir(WEIGHTS):
        from safetensors.torch import load_file

        return load_file(os.path.join(WEIGHTS, "diffusion_pytorch_model.safetensors"))
    tensors = torch.load(WEIGHTS, map_location="cpu", weights_only=True)
    return tensors if isinstance(tensors, dict) else dict(zip(weight_names(), tensors))


@functools.lru_cache(maxsize=None)
def _decoder():
    from wan22_vae_decode import WanVAEDecoder

    return WanVAEDecoder(_state_dict(), device="cuda")


def test_weights_follow_diffusers_names():
    from wan22_vae_decode import decoder_weights, weight_names

    names = weight_names()
    assert len(names) == 110 and names[0] == "post_quant_conv.weight" and names[-1] == "decoder.conv_out.bias"
    weights = decoder_weights(_state_dict(), "cpu")
    assert all(w.dtype == torch.bfloat16 and w.is_contiguous() for w in weights)
    assert all(torch.equal(w, _state_dict()[n].to(torch.bfloat16)) for n, w in zip(names, weights))


@pytest.mark.parametrize("latent_frames", [1, 2, 5])
def test_decode_against_reference(latent_frames):
    import reference
    from wan22_vae_decode import weight_names

    generator = torch.Generator().manual_seed(latent_frames)
    z = torch.randn((1, 48, latent_frames, 44, 78), generator=generator).to("cuda", torch.bfloat16)
    video = _decoder().decode(z)
    assert video.shape == (1, 3, 4 * latent_frames - 3, 704, 1248) and video.dtype == torch.bfloat16

    weights = {n: _state_dict()[n].to("cuda", torch.bfloat16).float() for n in weight_names()}
    ref = reference.decode(z.float(), weights, torch.float32)
    release = reference.decode(z, weights, torch.bfloat16).float()
    d, d_release = video.float() - ref, release - ref
    assert bool(torch.isfinite(video).all()) and float(video.abs().max()) <= 1.0
    assert float(d.abs().max()) <= 3 * float(d_release.abs().max())
    assert float(d.pow(2).mean().sqrt()) <= 3 * float(d_release.pow(2).mean().sqrt())


def test_decodes_are_independent():
    """No state carries from one call to the next: the same latents decode to the same video after another clip."""
    z = torch.randn((1, 48, 2, 44, 78), generator=torch.Generator().manual_seed(7)).to("cuda", torch.bfloat16)
    first = _decoder().decode(z)
    _decoder().decode(torch.randn((1, 48, 3, 44, 78), generator=torch.Generator().manual_seed(8)).to("cuda",
                                                                                                     torch.bfloat16))
    assert torch.equal(_decoder().decode(z), first)


def test_pass_size_does_not_change_the_video():
    """The causal history crosses passes, so any number of latent frames per pass decodes the same video."""
    from wan22_vae_decode import WanVAEDecoder

    z = torch.randn((1, 48, 6, 44, 78), generator=torch.Generator().manual_seed(9)).to("cuda", torch.bfloat16)
    reference_pass = _decoder().decode(z)
    for frames in (1, 16):
        assert torch.equal(WanVAEDecoder(_state_dict(), device="cuda", frames_per_pass=frames).decode(z), reference_pass)
