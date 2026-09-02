# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import torch

from fastvideo.train.methods.rl.rewards.mj_video import (
    MJ_VIDEO_ASPECT_INDICES,
    MJ_VIDEO_ASPECT_TO_CRITERIA,
    MJVideoAspectScorer,
    MJVideoRuntime,
    MJVideoRuntimeConfig,
    mj_video_frame_indices,
)


def test_mj_video_source_aspect_mapping() -> None:
    assert MJ_VIDEO_ASPECT_INDICES["fineness"] == 2
    assert MJ_VIDEO_ASPECT_INDICES["cc"] == 3
    assert MJ_VIDEO_ASPECT_TO_CRITERIA == {
        0: [0, 1, 2, 3, 4],
        1: [5, 6, 7, 8, 9, 10],
        2: [11, 12, 13, 14, 15],
        3: [16, 17, 18, 19, 20, 21, 22],
        4: [23, 24, 25, 26, 27],
    }


def test_mj_video_frame_indices_match_official_sampler() -> None:
    assert mj_video_frame_indices(124) == [
        0,
        15,
        30,
        46,
        61,
        76,
        92,
        107,
    ]
    assert mj_video_frame_indices(4) == [
        0,
        0,
        0,
        1,
        1,
        1,
        2,
        2,
    ]


class _FakeTokenizer:
    pad_token_id = 0


class _FakeModel:
    def __init__(self) -> None:
        self.config = SimpleNamespace()
        self.calls = 0

    def forward(
        self,
        pixel_values,
        input_ids,
        attention_mask,
    ):
        del attention_mask
        self.calls += 1
        batch_size = input_ids.shape[0]
        assert pixel_values.shape[0] == batch_size * 8
        base = torch.arange(
            batch_size,
            dtype=torch.float32,
        )[:, None]
        aspects = base + torch.arange(
            5,
            dtype=torch.float32,
        )[None]
        return SimpleNamespace(
            aspect_scores=aspects,
        )


def _fake_prepare_chat_input(
    config,
    tokenizer,
    pixel_values,
    question,
    generation_config,
    *,
    num_patches_list,
    device,
):
    del config, tokenizer, generation_config
    assert pixel_values.shape[0] == 8
    assert num_patches_list == [1] * 8
    assert question.startswith("Frame1: <image>\n")
    length = 4 + (len(question) % 3)
    ids = torch.arange(
        1,
        length + 1,
        device=device,
    )[None]
    return ids, torch.ones_like(ids)


def test_mj_video_runtime_caches_one_forward_for_two_aspects() -> None:
    model = _FakeModel()
    runtime = MJVideoRuntime(
        MJVideoRuntimeConfig(
            runtime_path="unused",
            model_path="unused",
            base_model_path="unused",
            device="cpu",
            verify_revision=False,
            batch_size=2,
        ),
        model=model,
        tokenizer=_FakeTokenizer(),
        prepare_chat_input=_fake_prepare_chat_input,
    )
    cc = MJVideoAspectScorer(
        aspect="cc",
        runtime=runtime,
    )
    fineness = MJVideoAspectScorer(
        aspect="fineness",
        runtime=runtime,
    )
    media = torch.zeros(
        2,
        3,
        16,
        32,
        32,
        dtype=torch.uint8,
    )
    prompts = ["first prompt", "second prompt"]

    cc_scores = cc(media, prompts)
    fineness_scores = fineness(media, prompts)

    assert model.calls == 1
    assert runtime.forward_calls == 1
    assert torch.equal(
        cc_scores,
        torch.tensor([3.0, 4.0]),
    )
    assert torch.equal(
        fineness_scores,
        torch.tensor([2.0, 3.0]),
    )


def test_mj_video_cache_invalidates_for_new_media_object() -> None:
    model = _FakeModel()
    runtime = MJVideoRuntime(
        MJVideoRuntimeConfig(
            runtime_path="unused",
            model_path="unused",
            base_model_path="unused",
            device="cpu",
            verify_revision=False,
        ),
        model=model,
        tokenizer=_FakeTokenizer(),
        prepare_chat_input=_fake_prepare_chat_input,
    )
    scorer = MJVideoAspectScorer(
        aspect="cc",
        runtime=runtime,
    )
    prompts = ["prompt"]

    scorer(
        torch.zeros(1, 3, 8, 16, 16, dtype=torch.uint8),
        prompts,
    )
    scorer(
        torch.zeros(1, 3, 8, 16, 16, dtype=torch.uint8),
        prompts,
    )

    assert model.calls == 2


def test_mj_video_runtime_config_rejects_source_drift() -> None:
    for kwargs in (
        {"num_segments": 16},
        {"input_size": 224},
        {"max_num": 2},
        {"dtype": "float16"},
    ):
        try:
            MJVideoRuntimeConfig(
                runtime_path="runtime",
                model_path="model",
                base_model_path="base",
                device="cpu",
                verify_revision=False,
                **kwargs,
            )
        except ValueError:
            pass
        else:
            raise AssertionError(
                f"Expected source-drift config to fail: {kwargs}"
            )
