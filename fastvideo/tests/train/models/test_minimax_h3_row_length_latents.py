# SPDX-License-Identifier: Apache-2.0
"""Without a fixed training shape, H3 rows train at their own length instead of being cropped."""
from types import SimpleNamespace

import pytest
import torch

from fastvideo.pipelines.basic.minimax_h3.packing import audio_latent_num_frames, video_latent_num_frames
from fastvideo.train.models.minimax_h3.minimax_h3 import MiniMaxH3Model


def _model(num_latent_t: int, num_frames: int) -> MiniMaxH3Model:
    model = MiniMaxH3Model.__new__(MiniMaxH3Model)
    model.training_config = SimpleNamespace(data=SimpleNamespace(
        native_shape_bucketing=False, num_latent_t=num_latent_t, num_frames=num_frames))
    return model


def _row(num_frames: int, audio_frames: int | None = None) -> dict[str, torch.Tensor]:
    if audio_frames is None:
        audio_frames = audio_latent_num_frames(num_frames)
    return {
        "vae_latent": torch.zeros(1, 24, video_latent_num_frames(num_frames), 4, 2),
        "audio_latent": torch.zeros(1, 2, 32, audio_frames),
    }


def _resolve(model: MiniMaxH3Model, raw_batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    return model._resolve_clean_latents(raw_batch, "data", torch.float32, torch.device("cpu"))


@pytest.mark.parametrize("num_frames", [124, 158, 243, 362])
def test_unset_shape_keeps_each_row_length(num_frames: int) -> None:
    video, audio = _resolve(_model(0, 0), _row(num_frames))

    assert video.shape[2] == video_latent_num_frames(num_frames)
    assert audio.shape[-1] == audio_latent_num_frames(num_frames)


def test_unset_shape_rejects_mismatched_latent_clocks() -> None:
    with pytest.raises(ValueError, match="158 video frames need 263"):
        _resolve(_model(0, 0), _row(158, audio_frames=264))


def test_fixed_shape_still_crops_and_warns_once(monkeypatch: pytest.MonkeyPatch) -> None:
    import fastvideo.train.models.minimax_h3.minimax_h3 as module

    warnings = []
    monkeypatch.setattr(module.logger, "warning_once", warnings.append)

    video, audio = _resolve(_model(37, 124), _row(243))

    assert video.shape[2] == 37
    assert audio.shape[-1] == audio_latent_num_frames(124)
    assert len(warnings) == 1 and "Cropping vae_latent from 72 to" in warnings[0]


def test_fixed_shape_at_its_own_length_does_not_warn(monkeypatch: pytest.MonkeyPatch) -> None:
    import fastvideo.train.models.minimax_h3.minimax_h3 as module

    warnings = []
    monkeypatch.setattr(module.logger, "warning_once", warnings.append)

    _resolve(_model(37, 124), _row(124))

    assert warnings == []
