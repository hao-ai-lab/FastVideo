# SPDX-License-Identifier: Apache-2.0

import torch

from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.models.dits.ltx2 import (
    LTX2Transformer3DModel,
    LTX2VideoOnlyTransformer3DModel,
)
from fastvideo.pipelines.basic.ltx2.stages.ltx2_audio_decoding import (
    LTX2AudioDecodingStage,
)
from fastvideo.pipelines.basic.ltx2.stages.ltx2_denoising import (
    LTX2DenoisingStage,
)
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch


class _RecordingVideoOnlyTransformer(LTX2VideoOnlyTransformer3DModel):

    def __init__(self) -> None:
        torch.nn.Module.__init__(self)
        self.calls = []

    def forward(self, hidden_states, **kwargs):
        self.calls.append(kwargs)
        return torch.zeros_like(hidden_states)


class _RecordingAudioVideoTransformer(LTX2Transformer3DModel):

    def __init__(self) -> None:
        torch.nn.Module.__init__(self)
        self.calls = []

    def forward(self, hidden_states, **kwargs):
        self.calls.append(kwargs)
        return torch.zeros_like(hidden_states), torch.zeros_like(kwargs["audio_hidden_states"])


def _batch() -> ForwardBatch:
    text = torch.ones(1, 1, 4)
    return ForwardBatch(
        data_type="video",
        latents=torch.ones(1, 1, 1, 1, 1),
        prompt_embeds=[text],
        negative_prompt_embeds=[-text],
        num_inference_steps=1,
        num_frames=1,
        fps=24,
        extra={
            "ltx2_audio_prompt_embeds": [text],
            "ltx2_audio_negative_embeds": [-text],
            "ltx2_audio_latents": torch.ones(1, 2, 1, 1),
            "video_position_offset_sec": 2.0,
        },
    )


def test_video_only_denoising_ignores_audio_conditioning_and_decoding() -> None:
    transformer = _RecordingVideoOnlyTransformer()
    batch = _batch()
    batch.ltx2_cfg_scale_audio = 7.0
    batch.ltx2_modality_scale_audio = 3.0
    batch.ltx2_stg_scale_audio = 1.0
    batch.do_classifier_free_guidance = True
    args = FastVideoArgs(model_path="", disable_autocast=True)

    result = LTX2DenoisingStage(
        transformer,
        sigmas_override=[1.0, 0.0],
    ).forward(batch, args)

    assert len(transformer.calls) == 1
    call = transformer.calls[0]
    assert call["audio_hidden_states"] is None
    assert call["audio_encoder_hidden_states"] is None
    assert call["audio_timestep"] is None
    assert call["audio_sigma"] is None
    assert call["video_position_offset_sec"] == 0.0
    assert result.extra["ltx2_audio_latents"] is None

    decoded = LTX2AudioDecodingStage(object(), object()).forward(result, args)
    assert decoded is result
    assert "audio" not in decoded.extra


def test_audio_video_denoising_keeps_audio_conditioning() -> None:
    transformer = _RecordingAudioVideoTransformer()
    batch = _batch()

    result = LTX2DenoisingStage(
        transformer,
        sigmas_override=[1.0, 0.0],
    ).forward(batch, FastVideoArgs(model_path="", disable_autocast=True))

    assert len(transformer.calls) == 1
    call = transformer.calls[0]
    assert call["audio_hidden_states"] is not None
    assert call["audio_encoder_hidden_states"] is not None
    assert call["audio_timestep"] is not None
    assert call["audio_sigma"] is not None
    assert call["video_position_offset_sec"] == 2.0
    assert result.extra["ltx2_audio_latents"] is not None
