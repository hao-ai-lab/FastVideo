# SPDX-License-Identifier: Apache-2.0
"""Cosmos Predict pipeline parity checks."""
from __future__ import annotations

import gc
import os
from typing import Any

import pytest
import torch
from torch.testing import assert_close

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Requires CUDA",
)

from fastvideo.forward_context import set_forward_context
from fastvideo.distributed import initialize_model_parallel
os.environ.setdefault("FASTVIDEO_ATTENTION_BACKEND", "TORCH_SDPA")


def _run_fastvideo_pipeline(request_kwargs: dict[str, Any]) -> torch.Tensor:
    from fastvideo.api.sampling_param import SamplingParam
    from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
    from fastvideo.utils import shallow_asdict
    from fastvideo.fastvideo_args import FastVideoArgs
    from fastvideo.registry import get_pipeline_config_classes
    from fastvideo.pipelines.basic.cosmos_predict.pipeline_cosmos_predict import CosmosPredictPipeline

    # Ensure model parallel is initialized for distributed components if needed
    try:
        initialize_model_parallel(1, 1, 1, 1)
    except AssertionError:
        pass  # already initialized

    fastvideo_args = FastVideoArgs(
        model_path="nvidia/Cosmos-1.0-Prompt2World-7B-Video",
    )
    
    sampling_param = SamplingParam.from_pretrained(fastvideo_args.model_path)
    sampling_param.update({key: value for key, value in request_kwargs.items() if key not in {"prompt"}})
    sampling_param.prompt = request_kwargs["prompt"]

    batch = ForwardBatch(
        **shallow_asdict(sampling_param),
        eta=0.0,
        n_tokens=sampling_param.num_frames * (sampling_param.height // 8) * (sampling_param.width // 8),
    )
    
    pipeline = CosmosPredictPipeline(fastvideo_args.model_path, fastvideo_args)
    pipeline.create_pipeline_stages(fastvideo_args)

    with set_forward_context(current_timestep=0, attn_metadata=None):
        output_batch = pipeline.forward(batch, fastvideo_args)
    assert output_batch.latents is not None
    return output_batch.latents[0].detach().cpu()


@pytest.mark.skip(reason="Needs Cosmos Predict 7B weights downloaded via HF login.")
def test_cosmos_predict_pipeline_parity():
    """Test FastVideo Cosmos Predict pipeline against diffusers."""
    import diffusers

    prompt = "A cute dog walking."
    request_kwargs = {
        "prompt": prompt,
        "num_inference_steps": 2,
        "guidance_scale": 7.0,
        "height": 128,
        "width": 128,
        "num_frames": 9,
    }

    # FastVideo
    torch.manual_seed(42)
    fastvideo_latents = _run_fastvideo_pipeline(request_kwargs)
    
    # Official
    torch.manual_seed(42)
    official_pipeline = diffusers.pipelines.cosmos.pipeline_cosmos2_5_predict.Cosmos2_5_PredictBasePipeline.from_pretrained(
        "nvidia/Cosmos-1.0-Prompt2World-7B-Video",
        torch_dtype=torch.bfloat16
    ).to("cuda")
    
    official_latents = official_pipeline(
        prompt=prompt,
        num_inference_steps=2,
        guidance_scale=7.0,
        height=128,
        width=128,
        num_frames=9,
        output_type="latent",
        return_dict=False
    )[0].detach().cpu()
    
    assert_close(fastvideo_latents, official_latents, atol=2e-2, rtol=2e-2)

