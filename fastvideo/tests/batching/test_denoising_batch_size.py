# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest
import torch

from fastvideo.pipelines.stages.denoising import _validate_denoising_batch_size


def test_validate_denoising_batch_size_accepts_dynamic_batch() -> None:
    latents = torch.zeros(2, 4, 1, 1, 1)
    prompt_embeds = [torch.zeros(2, 8, 16)]

    _validate_denoising_batch_size(latents, prompt_embeds, name="prompt embeddings")


def test_validate_denoising_batch_size_rejects_num_videos_per_prompt_mismatch() -> None:
    latents = torch.zeros(2, 4, 1, 1, 1)
    prompt_embeds = [torch.zeros(1, 8, 16)]

    with pytest.raises(ValueError, match="latents have batch size 2.*prompt embeddings.*batch size 1"):
        _validate_denoising_batch_size(latents, prompt_embeds, name="prompt embeddings")
