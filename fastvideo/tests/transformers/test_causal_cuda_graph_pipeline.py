# SPDX-License-Identifier: Apache-2.0
"""One-GPU SFWan regression: actual graph replay must preserve complete video output."""

import numpy as np
import pytest
import torch

from fastvideo import VideoGenerator, envs
from fastvideo.api import EngineConfig, GeneratorConfig, OffloadConfig, PipelineSelection
from fastvideo.models.wan.pipeline_config import SelfForcingWanT2V480PConfig

MODEL = "wlsaidhi/SFWan2.1-T2V-1.3B-Diffusers"
REVISION = "4b44356635ae5e927ca552a220f768022be76004"


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires one CUDA GPU and SFWan weights")
def test_causal_cuda_graph_pipeline_matches_eager_and_resets_per_request(env_overrides):
    env_overrides.enter_context(envs.FASTVIDEO_ATTENTION_BACKEND.override("FLASH_ATTN"))
    outputs = []
    request = {
        "prompt": "A fox walks through a sunlit forest.",
        "sampling": {"num_frames": 81, "height": 480, "width": 832, "num_inference_steps": 4, "seed": 1024},
        "output": {"save_video": False, "return_frames": True},
    }
    changed_request = {
        "prompt": "A bird flies above a calm lake.",
        "sampling": {"num_frames": 129, "height": 384, "width": 640, "num_inference_steps": 4, "seed": 1025},
        "output": {"save_video": False, "return_frames": True},
    }
    for enabled in (False, True):
        config = GeneratorConfig(
            model_path=MODEL, revision=REVISION,
            engine=EngineConfig(num_gpus=1, enable_causal_cuda_graph=enabled,
                                offload=OffloadConfig(dit=False, dit_layerwise=False)),
            pipeline=PipelineSelection(experimental={"pipeline_config": SelfForcingWanT2V480PConfig(
                causal_local_attn_size=9, causal_rope_cache_policy="relativistic")}),
        )
        generator = VideoGenerator.from_config(config)
        try:
            # Confirm actual backend rather than accepting a silent fallback.
            model = generator.executor.driver_worker.worker.pipeline.get_module("transformer")
            assert model.blocks[0].attn1.attn.backend.name == "FLASH_ATTN"
            first = generator.generate(request)
            assert torch.isfinite(first.samples).all()
            changed = generator.generate(changed_request)
            assert torch.isfinite(changed.samples).all()
            outputs.append((first, changed))
            if enabled:
                second = generator.generate(request)
                for result in (first, changed, second):
                    stats = result.extra["causal_cuda_graph"]
                    assert stats["capture_count"] == 3
                    assert stats["replay_count"] > 0
                    assert stats["reset_count"] == 0
                torch.testing.assert_close(first.samples, second.samples, atol=0, rtol=0)
                for earlier, repeated in zip(first.frames, second.frames, strict=True):
                    np.testing.assert_array_equal(earlier, repeated)
        finally:
            generator.shutdown()
    sizes = ((480, 832, 81), (384, 640, 129))
    for eager_result, graph_result, size in zip(outputs[0], outputs[1], sizes, strict=True):
        torch.testing.assert_close(eager_result.samples, graph_result.samples, atol=0, rtol=0)
        assert eager_result.size == graph_result.size == size
        assert len(eager_result.frames) == len(graph_result.frames) == size[2]
        for eager, graphed in zip(eager_result.frames, graph_result.frames, strict=True):
            assert eager.shape == graphed.shape == (*size[:2], 3)
            np.testing.assert_array_equal(eager, graphed)
