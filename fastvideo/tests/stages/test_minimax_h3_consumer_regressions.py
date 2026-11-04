# SPDX-License-Identifier: Apache-2.0
"""Host-copy layout changes for frozen offload, without model weights."""
from types import SimpleNamespace

import pytest
import torch

from fastvideo.models import pinned_offload
from fastvideo.pipelines.basic.minimax_h3.stages import minimax_h3_decoding as decoding
from fastvideo.pipelines.basic.minimax_h3.stages.minimax_h3_latent_preparation import MINIMAX_H3_LAYOUT_KEY


def _replace_master_with_packed(module):
    del module.weight
    module.register_buffer("packed_weight", torch.tensor([12, 34], dtype=torch.uint8))


def test_refresh_offload_host_copies_replaces_deleted_master():
    module = torch.nn.Linear(2, 2)
    pinned_offload._host_copies(module, pin=False)
    _replace_master_with_packed(module)
    pinned_offload.refresh_host_copies(module, pin=False)
    host = getattr(module, pinned_offload._HOST_ATTR)
    assert set(host) == {"bias", "buffer:packed_weight"}
    torch.testing.assert_close(host["buffer:packed_weight"], module.packed_weight)


def test_resident_module_does_not_gain_host_mirror():
    module = torch.nn.Linear(2, 2)
    _replace_master_with_packed(module)
    pinned_offload.refresh_host_copies(module, pin=False)
    assert not hasattr(module, pinned_offload._HOST_ATTR)


def test_unload_runs_submodule_device_cache_hooks():
    """Plain-attribute device caches (fused plans, CUDA graphs) must be released with the weights."""
    module = torch.nn.Module()
    child = torch.nn.Module()
    module.add_module("child", child)
    cleared = []
    child.__dict__["clear_offload_caches"] = lambda: cleared.append(True)
    setattr(module, pinned_offload._HOST_ATTR, {})
    pinned_offload.unload(module)
    assert cleared == [True]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="offload reload needs CUDA")
def test_packed_offload_reloads_twice_without_old_weight_names():
    module = torch.nn.Linear(2, 2).requires_grad_(False)
    device = torch.device("cuda", torch.cuda.current_device())
    pinned_offload.load(module, device, pin=False)
    _replace_master_with_packed(module)
    module.packed_weight = module.packed_weight.to(device)
    pinned_offload.refresh_host_copies(module, pin=False)
    for _ in range(2):
        pinned_offload.unload(module)
        assert module.packed_weight.device.type == "cpu"
        pinned_offload.load(module, device, pin=False)
        assert module.packed_weight.device == device
        assert module.packed_weight.tolist() == [12, 34]


def test_decode_refreshes_mirror_before_first_unload(monkeypatch):
    import fastvideo.models.vaes.minimax_h3_nvfp4_decoder as nvfp4

    vae = torch.nn.Linear(2, 2).requires_grad_(False)
    vae.denormalize_latents = lambda z: z
    pinned_offload._host_copies(vae, pin=False)
    layout = SimpleNamespace(num_condition_video_rows=0, num_condition_audio_rows=0)
    batch = SimpleNamespace(extra={MINIMAX_H3_LAYOUT_KEY: layout}, latents=torch.zeros(1, 2),
                            raw_latent_shape=(1, 2, 1, 1, 1), output=None)
    args = SimpleNamespace(vae_parallel_decode=False, pin_cpu_memory=False, vae_cpu_offload=True,
                           output_type="latent")
    monkeypatch.setattr(decoding, "_decode_participation", lambda *a: (None, True, False))
    monkeypatch.setattr(decoding, "_layout", lambda batch: layout)
    monkeypatch.setattr(decoding, "get_local_torch_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(decoding, "h3_dit_patch_size", lambda *a: (1, 1, 1))
    monkeypatch.setattr(decoding, "unpatchify_video_tokens", lambda tokens, *a: tokens)

    def freeze(module):
        _replace_master_with_packed(module)
        return 1

    def unload(module):
        # The real CUDA unload indexes every current tensor name in this mirror.
        host = getattr(module, pinned_offload._HOST_ATTR)
        for name, tensor in pinned_offload._tensors(module):
            torch.testing.assert_close(host[name], tensor)

    monkeypatch.setattr(nvfp4, "freeze_nvfp4_linears_on_device", freeze)
    monkeypatch.setattr(pinned_offload, "unload", unload)
    decoding.MiniMaxH3VideoDecodingStage(vae).forward(batch, args)
