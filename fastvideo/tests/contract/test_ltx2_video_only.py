# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from fastvideo.models.dits.ltx2 import (
    EntryClass,
    LTXLocalAttention,
    LTX2Transformer3DModel,
    LTX2VideoOnlyTransformer3DModel,
    LTXModel,
    LTXModelType,
)
from fastvideo.models.registry import ModelRegistry
from fastvideo.platforms import AttentionBackendEnum


def _state_keys(model_type: LTXModelType, *, ltx2_3: bool) -> set[str]:
    with torch.device("meta"):
        model = LTXModel(
            model_type=model_type,
            num_attention_heads=2,
            attention_head_dim=8,
            in_channels=8,
            out_channels=8,
            num_layers=2,
            cross_attention_dim=16,
            caption_channels=16,
            audio_num_attention_heads=2,
            audio_attention_head_dim=4,
            audio_in_channels=8,
            audio_out_channels=8,
            audio_cross_attention_dim=16,
            cross_attention_adaln=ltx2_3,
            caption_proj_before_connector=ltx2_3,
            apply_gated_attention=ltx2_3,
        )
    return {f"model.{key}" for key in model.state_dict()}


def test_video_only_class_registration_keeps_av_default() -> None:
    assert LTX2Transformer3DModel._model_type is LTXModelType.AudioVideo
    assert LTX2VideoOnlyTransformer3DModel._model_type is LTXModelType.VideoOnly
    assert EntryClass == [
        LTX2Transformer3DModel,
        LTX2VideoOnlyTransformer3DModel,
    ]
    model_cls, _ = ModelRegistry.resolve_model_cls(
        "LTX2VideoOnlyTransformer3DModel")
    assert model_cls is LTX2VideoOnlyTransformer3DModel


@pytest.mark.parametrize("ltx2_3", [False, True])
def test_video_only_checkpoint_filter_matches_exact_av_state_difference(
        ltx2_3: bool, monkeypatch: pytest.MonkeyPatch) -> None:

    def init_parameterless_attention(self, *args, **kwargs):
        del args, kwargs
        torch.nn.Module.__init__(self)
        self.backend = AttentionBackendEnum.TORCH_SDPA

    monkeypatch.setattr(LTXLocalAttention, "__init__",
                        init_parameterless_attention)
    av_keys = _state_keys(LTXModelType.AudioVideo, ltx2_3=ltx2_3)
    video_keys = _state_keys(LTXModelType.VideoOnly, ltx2_3=ltx2_3)
    removed_keys = av_keys - video_keys

    assert removed_keys
    assert all(LTX2VideoOnlyTransformer3DModel._is_ignored_checkpoint_key(key)
               for key in removed_keys)
    assert not any(
        LTX2VideoOnlyTransformer3DModel._is_ignored_checkpoint_key(key)
        for key in video_keys)


@pytest.mark.parametrize(
    "key",
    [
        "model",
        "model.audio_typo.weight",
        "model.patchify_proj.weight",
        "model.transformer_blocks.0.attn1.to_q.weight",
        "model.transformer_blocks.0.audio_attn3.to_q.weight",
        "model.transformer_blocks.bad.audio_attn1.to_q.weight",
        "unrelated.weight",
    ],
)
def test_video_only_checkpoint_filter_rejects_near_misses(key: str) -> None:
    assert not LTX2VideoOnlyTransformer3DModel._is_ignored_checkpoint_key(key)
