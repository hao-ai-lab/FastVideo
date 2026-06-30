# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from fastvideo.api.sampling_param import SamplingParam
from fastvideo.batching.signature import (
    can_dynamic_batch,
    dynamic_batch_signature,
    resolution_key,
)


def _request(prompt: str = "a prompt", **overrides) -> SamplingParam:
    request = SamplingParam(prompt=prompt, height=256, width=384, num_frames=17, num_inference_steps=4)
    for key, value in overrides.items():
        setattr(request, key, value)
    return request


def test_dynamic_batch_signature_excludes_request_local_fields() -> None:
    first = _request(seed=1, output_path="/tmp/a.mp4", save_video=True, return_frames=False)
    second = _request(seed=2, output_path="/tmp/b.mp4", save_video=False, return_frames=True)

    assert dynamic_batch_signature(first) == dynamic_batch_signature(second)


def test_can_dynamic_batch_accepts_matching_text_requests() -> None:
    first = _request("first", seed=1)
    second = _request("second", seed=2)

    result = can_dynamic_batch(first, second)

    assert result.can_batch is True
    assert result.reason is None


def test_can_dynamic_batch_rejects_sampling_mismatch() -> None:
    first = _request(guidance_scale=1.0)
    second = _request(guidance_scale=3.0)

    result = can_dynamic_batch(first, second)

    assert result.can_batch is False
    assert result.reason == "sampling_params.guidance_scale"


def test_can_dynamic_batch_rejects_image_conditioning() -> None:
    first = _request()
    second = _request(image_path="/tmp/image.png")

    result = can_dynamic_batch(first, second)

    assert result.can_batch is False
    assert result.reason == "image_path"


def test_resolution_key_uses_generation_shape() -> None:
    assert resolution_key(_request(height=720, width=1280, num_frames=81)) == "720x1280x81"
