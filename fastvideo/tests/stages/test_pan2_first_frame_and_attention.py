# SPDX-License-Identifier: Apache-2.0
"""PAN2's first-frame loading (the stored pixels, as the reference reads them) and its FlashAttention-4 default."""

import sys
import types
from types import SimpleNamespace

import numpy as np
import PIL.Image
import pytest
import torch

from fastvideo import envs
from fastvideo.pipelines.basic.pan2 import pan2_pipeline
from fastvideo.pipelines.basic.pan2.stages import PAN2InputValidationStage
from fastvideo.pipelines.stages.input_validation import InputValidationStage


def _args():
    vae = SimpleNamespace(arch_config=SimpleNamespace(spatial_compression_ratio=16, temporal_compression_ratio=4))
    dit = SimpleNamespace(arch_config=SimpleNamespace(patch_size=(1, 2, 2)))
    return SimpleNamespace(pipeline_config=SimpleNamespace(vae_config=vae, dit_config=dit))


def test_an_exif_rotated_first_frame_keeps_its_stored_pixels(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(InputValidationStage, "forward",
                        lambda self, batch, args: seen.append((batch.image_path, batch.pil_image)) or batch)
    stored = np.zeros((8, 16, 3), dtype=np.uint8)
    stored[:, :8] = 255
    path = tmp_path / "rotated.png"
    exif = PIL.Image.Exif()
    exif[0x0112] = 6  # Orientation: rotate 90 degrees clockwise to display upright.
    PIL.Image.fromarray(stored).save(path, exif=exif)

    batch = SimpleNamespace(image_path=str(path), pil_image=None, height=704, width=1248, num_frames=33)
    out = PAN2InputValidationStage().forward(batch, _args())

    # The base stage saw a loaded image and no path to load again; the path is restored for later stages.
    assert seen[0][0] is None and seen[0][1] is out.pil_image
    assert out.image_path == str(path)
    assert out.pil_image.size == (16, 8) and (np.asarray(out.pil_image) == stored).all()


@pytest.fixture
def blackwell(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *args: (10, 3))
    monkeypatch.setitem(sys.modules, "fastvideo.attention.utils.flash_attn_cute",
                        types.ModuleType("fastvideo.attention.utils.flash_attn_cute"))
    for name in pan2_pipeline._FA4_READERS:
        monkeypatch.delitem(sys.modules, name, raising=False)
    with envs.FASTVIDEO_FA4.override(None):
        yield monkeypatch


def test_pan2_turns_flash_attention_4_on_for_blackwell(blackwell):
    assert pan2_pipeline.default_to_flash_attention_4() is True
    assert envs.FASTVIDEO_FA4.get() is True


@pytest.mark.parametrize("value", [False, True])
def test_an_explicit_fastvideo_fa4_is_kept(blackwell, value):
    envs.FASTVIDEO_FA4.set(value)
    assert pan2_pipeline.default_to_flash_attention_4() is False
    assert envs.FASTVIDEO_FA4.get() is value


def test_hopper_keeps_the_default_backend(blackwell):
    blackwell.setattr(torch.cuda, "get_device_capability", lambda *args: (9, 0))
    assert pan2_pipeline.default_to_flash_attention_4() is False
    assert not envs.FASTVIDEO_FA4.is_set()


def test_without_fa4_installed_the_default_backend_stays(blackwell):
    blackwell.setitem(sys.modules, "fastvideo.attention.utils.flash_attn_cute", None)
    assert pan2_pipeline.default_to_flash_attention_4() is False
    assert not envs.FASTVIDEO_FA4.is_set()


def test_attention_set_up_before_the_pipeline_is_not_switched(blackwell):
    blackwell.setitem(sys.modules, pan2_pipeline._FA4_READERS[0], types.ModuleType("already imported"))
    assert pan2_pipeline.default_to_flash_attention_4() is False
    assert not envs.FASTVIDEO_FA4.is_set()
