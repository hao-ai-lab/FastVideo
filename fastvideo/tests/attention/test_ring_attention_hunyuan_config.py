# SPDX-License-Identifier: Apache-2.0
"""Reject unsupported Hunyuan Ring layouts before allocating model layers."""

import pytest

from fastvideo.models.dits import hunyuanvideo


@pytest.mark.parametrize("ring_size", [1, 2, 4])
def test_hunyuan_ring_validation_precedes_weight_allocation(monkeypatch, ring_size):
    monkeypatch.setattr(hunyuanvideo, "get_ring_size", lambda: ring_size)

    class WeightAllocationReached(Exception):
        pass

    def stop_at_first_layer(*args, **kwargs):
        raise WeightAllocationReached

    monkeypatch.setattr(hunyuanvideo, "PatchEmbed", stop_at_first_layer)
    config = hunyuanvideo.HunyuanVideoConfig()
    if ring_size > 1:
        with pytest.raises(NotImplementedError, match="replicated text Q/K/V.*ring_size=1"):
            hunyuanvideo.HunyuanVideoTransformer3DModel(config, {})
    else:
        with pytest.raises(WeightAllocationReached):
            hunyuanvideo.HunyuanVideoTransformer3DModel(config, {})
