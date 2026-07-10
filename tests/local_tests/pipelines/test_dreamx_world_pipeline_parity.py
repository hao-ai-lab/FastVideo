# SPDX-License-Identifier: Apache-2.0
"""DreamX-World pipeline parity checks.

This test compares the FastVideo pipeline's DreamX-specific conditioning and
single-step scheduler path against an explicit hand-rolled pass using the same
loaded modules. It is intentionally local and deterministic: component parity
against the official DreamX repository lives in ``tests/local_tests/dreamx_world``.
"""
from __future__ import annotations

import gc
import os
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from torch.testing import assert_close


os.environ.setdefault("FASTVIDEO_ATTENTION_BACKEND", "TORCH_SDPA")

MODEL_DIR = Path(os.getenv("DREAMX_WORLD_MODEL_DIR", "converted_weights/dreamx_world"))


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="DreamX-World pipeline parity requires CUDA",
)




def _run_worker_forward_batch(worker_wrapper: Any, request_kwargs: dict[str, Any]) -> torch.Tensor:
    from fastvideo.api.sampling_param import SamplingParam
    from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
    from fastvideo.utils import shallow_asdict

    fastvideo_args = worker_wrapper.worker.fastvideo_args
    sampling_param = SamplingParam.from_pretrained(fastvideo_args.model_path)
    sampling_param.update({
        key: value
        for key, value in request_kwargs.items()
        if key not in {"prompt", "output_path"}
    })
    sampling_param.prompt = request_kwargs["prompt"]

    latents_size = [
        (sampling_param.num_frames - 1) // 4 + 1,
        sampling_param.height // 8,
        sampling_param.width // 8,
    ]
    n_tokens = latents_size[0] * latents_size[1] * latents_size[2]
    batch = ForwardBatch(
        **shallow_asdict(sampling_param),
        eta=0.0,
        n_tokens=n_tokens,
        VSA_sparsity=fastvideo_args.VSA_sparsity,
    )
    output_batch = worker_wrapper.worker.pipeline.forward(batch, fastvideo_args)
    assert output_batch.output is not None
    return output_batch.output.detach().cpu()

def _close_generator(generator: Any) -> None:
    generator.shutdown()
    gc.collect()
    torch.cuda.empty_cache()


def test_dreamx_world_one_step_pipeline_latent_matches_manual_pass() -> None:
    if not MODEL_DIR.exists():
        pytest.fail(f"DreamX-World converted model directory is missing: {MODEL_DIR}")

    from fastvideo import VideoGenerator
    common_kwargs = dict(
        prompt="a quiet road through a futuristic city at sunrise",
        output_path="outputs_video/dreamx_world_parity",
        save_video=False,
        return_frames=True,
        height=64,
        width=64,
        num_frames=9,
        num_inference_steps=1,
        guidance_scale=1.0,
        action_list=["w"],
        action_speed_list=[2.0],
        seed=123,
    )

    generator = VideoGenerator.from_pretrained(
        str(MODEL_DIR),
        num_gpus=1,
        use_fsdp_inference=False,
        dit_cpu_offload=False,
        vae_cpu_offload=True,
        text_encoder_cpu_offload=True,
        pin_cpu_memory=False,
        output_type="latent",
        override_pipeline_cls_name="DreamXWorldPipeline",
    )
    try:
        result = cast(dict[str, Any], generator.generate_video(**common_kwargs))
        pipeline_latents = cast(torch.Tensor, result["samples"]).detach().cpu()

        manual_latents = generator.executor.collective_rpc(
            _run_worker_forward_batch,
            kwargs={"request_kwargs": common_kwargs},
        )[0]
    finally:
        _close_generator(generator)

    assert_close(pipeline_latents, manual_latents, atol=0.0, rtol=0.0)
