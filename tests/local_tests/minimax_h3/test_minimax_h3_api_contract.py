# SPDX-License-Identifier: Apache-2.0

import torch
from PIL import Image

from fastvideo.api.compat import normalize_generation_request, request_to_sampling_param
from fastvideo.api.sampling_param import SamplingParam
from fastvideo.pipelines.basic.minimax_h3.types import MiniMaxH3Reference
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
from fastvideo.utils import shallow_asdict


def test_typed_h3_inputs_reach_forward_batch_without_extension_dicts(monkeypatch) -> None:
    monkeypatch.setattr(SamplingParam, "from_pretrained", classmethod(lambda cls, _: cls()))
    first_image = Image.new("RGB", (2, 2), color="red")
    last_image = torch.randn(3, 2, 2)
    references = [MiniMaxH3Reference(source="reference.wav", media_type="audio")]
    video_latents = torch.randn(1, 4, 2, 3, 3)
    audio_latents = torch.randn(2, 8, 5)
    request = normalize_generation_request({
        "prompt": "a synthetic contract test",
        "inputs": {
            "pil_image": first_image,
            "last_image": last_image,
            "references": references,
            "latents": video_latents,
            "audio_latents": audio_latents,
        },
    })

    sampling_param = request_to_sampling_param(request, model_path="local/minimax-h3-contract")
    batch = ForwardBatch(**shallow_asdict(sampling_param))

    assert isinstance(batch.pil_image, Image.Image)
    assert batch.pil_image.tobytes() == first_image.tobytes()
    assert torch.equal(batch.last_image, last_image)
    assert batch.references == references
    assert torch.equal(batch.latents, video_latents)
    assert torch.equal(batch.audio_latents, audio_latents)
