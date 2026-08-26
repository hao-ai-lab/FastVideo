# SPDX-License-Identifier: Apache-2.0
"""Non-output SPMD ranks must not decode or materialize audio payloads."""

from types import SimpleNamespace
from unittest.mock import Mock

import torch

from fastvideo.pipelines import ForwardBatch
from fastvideo.pipelines.basic.ltx2.stages.ltx2_audio_decoding import LTX2AudioDecodingStage
from fastvideo.pipelines.basic.magi_human.stages.audio_decoding import MagiHumanAudioDecodingStage
from fastvideo.pipelines.basic.mmaudio.stages import MMAudioDecodingStage
from fastvideo.pipelines.basic.stable_audio.stages.decoding import StableAudioDecodingStage


def _non_output_args(*, output_type="pil"):
    return SimpleNamespace(
        is_output_rank=False,
        output_type=output_type,
        pipeline_config=SimpleNamespace(),
    )


def test_ltx2_non_output_rank_skips_audio_decoder_and_host_payload():
    audio_decoder = Mock()
    vocoder = Mock()
    stage = LTX2AudioDecodingStage(audio_decoder, vocoder)
    batch = ForwardBatch(
        data_type="video",
        extra={
            "ltx2_audio_latents": torch.ones(2),
            "audio": torch.ones(1),
            "audio_sample_rate": 24000,
        },
    )

    result = stage.forward(batch, _non_output_args())

    audio_decoder.to.assert_not_called()
    vocoder.to.assert_not_called()
    assert "audio" not in result.extra
    assert "audio_sample_rate" not in result.extra
    assert "ltx2_audio_latents" not in result.extra


def test_stable_audio_non_output_rank_skips_latent_and_waveform_cpu_copy():
    vae = Mock()
    stage = StableAudioDecodingStage(vae)
    batch = ForwardBatch(
        data_type="audio",
        latents=torch.ones(1, 2, 3),
        extra={
            "audio": torch.ones(1),
            "decoded_audio": torch.ones(1),
        },
    )

    result = stage.forward(batch, _non_output_args(output_type="latent"))

    vae.to.assert_not_called()
    assert result.output is not None and result.output.numel() == 0
    assert result.latents is None
    assert "audio" not in result.extra
    assert "decoded_audio" not in result.extra


def test_mmaudio_non_output_rank_skips_decoder_and_vocoder():
    audio_vae = Mock()
    vocoder = Mock()
    stage = MMAudioDecodingStage(audio_vae, vocoder)
    batch = ForwardBatch(
        data_type="audio",
        latents=torch.ones(1, 2, 3),
        extra={"audio": torch.ones(1)},
    )

    result = stage.forward(batch, _non_output_args(output_type="latent"))

    audio_vae.to.assert_not_called()
    vocoder.to.assert_not_called()
    assert result.output is not None and result.output.numel() == 0
    assert result.latents is None
    assert "audio" not in result.extra


def test_magi_human_non_output_rank_skips_audio_decode():
    audio_vae = Mock()
    stage = MagiHumanAudioDecodingStage(audio_vae)
    batch = ForwardBatch(
        data_type="video",
        audio_latents=torch.ones(1, 2, 3),
        output=torch.ones(1, 3, 1, 2, 2),
        extra={"audio": torch.ones(1)},
    )

    result = stage.forward(batch, _non_output_args())

    audio_vae.decode.assert_not_called()
    assert result.output is not None and result.output.numel() > 0
    assert result.latents is None
    assert result.audio_latents is None
    assert "audio" not in result.extra
