# SPDX-License-Identifier: Apache-2.0
"""Host-copy layout changes and reference-keyed quality dumps, without model weights."""
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from fastvideo import envs
from fastvideo.models import pinned_offload
from fastvideo.pipelines.basic.minimax_h3.reference import MiniMaxH3PreparedReference, prepared_reference_digest
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
    monkeypatch.setattr(decoding, "_maybe_dump_generation", lambda *a: None)
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


def test_reference_contents_and_order_keep_all_quality_dumps(tmp_path, env_overrides):
    red = MiniMaxH3PreparedReference(media_type="image", image=Image.new("RGB", (2, 2), "red"))
    blue = MiniMaxH3PreparedReference(media_type="image", image=Image.new("RGB", (2, 2), "blue"))
    env_overrides.enter_context(envs.FASTVIDEO_H3_DUMP_DIR.override(str(tmp_path)))
    layout = SimpleNamespace(num_condition_video_rows=0, num_condition_audio_rows=0)
    # References have already been freed by latent preparation at this boundary.
    batch = SimpleNamespace(prompt="same prompt", seed=7, raw_latent_shape=(1, 2, 1, 1, 1),
                            latents=torch.ones(1, 2), audio_latents=None, prompt_embeds=[], references=None, extra={})
    digests = []
    for references in ([red], [blue], [red, blue], [blue, red]):
        digest = prepared_reference_digest(references)
        digests.append(digest)
        batch.extra["minimax_h3_reference_digest"] = digest
        decoding._maybe_dump_generation(batch, layout)
    assert len(set(digests)) == 4
    paths = list(tmp_path.glob("*.pt"))
    assert len(paths) == 4
    assert {torch.load(path)["reference_digest"] for path in paths} == set(digests)


def test_input_preparation_retains_digest_when_media_is_released(monkeypatch, tmp_path, env_overrides):
    from fastvideo.pipelines.basic.minimax_h3.stages import minimax_h3_input_preparation as preparation

    reference = MiniMaxH3PreparedReference(media_type="image", image=Image.new("RGB", (2, 2), "red"))
    monkeypatch.setattr(preparation, "validate_references", lambda values: values)
    monkeypatch.setattr(preparation, "prepare_reference", lambda *args: reference)
    monkeypatch.setattr(preparation, "resolve_target_canvas", lambda *args: (480, 832, 16))
    monkeypatch.setattr(preparation, "resolve_target_num_frames", lambda value: value)
    stage = preparation.MiniMaxH3InputPreparationStage(
        SimpleNamespace(latent_channels=2), SimpleNamespace(sampling_rate=24000), ref2va=True)
    batch = SimpleNamespace(image_path=None, pil_image=None, last_image=None, references=["red.png"],
                            num_frames=124, extra={})
    env_overrides.enter_context(envs.FASTVIDEO_H3_DUMP_DIR.override(str(tmp_path)))
    stage._prepare_ref2va(batch)
    batch.references = None
    assert batch.extra["minimax_h3_reference_digest"] == prepared_reference_digest([reference])


def test_audio_content_changes_reference_digest():
    first = MiniMaxH3PreparedReference(media_type="audio", has_audio=True, waveform=torch.zeros(2, 8))
    second = MiniMaxH3PreparedReference(media_type="audio", has_audio=True, waveform=torch.ones(2, 8))
    assert prepared_reference_digest([first]) != prepared_reference_digest([second])
