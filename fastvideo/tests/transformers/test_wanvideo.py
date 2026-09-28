# SPDX-License-Identifier: Apache-2.0
"""Full-checkpoint Wan parity against Diffusers; requires one CUDA GPU and weights."""

import os

import pytest
import torch
from diffusers import WanTransformer3DModel
from torch.testing import assert_close

from fastvideo.configs.pipelines import PipelineConfig
from fastvideo.forward_context import set_forward_context
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.models.loader.component_loader import TransformerLoader
from fastvideo.tests.golden_gate._wan_checkpoint import component_path
from fastvideo.models.wan.config import WanVideoConfig
from fastvideo.pipelines.pipeline_batch_info import ForwardBatch

os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", "29503")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Wan transformer parity requires one CUDA GPU")
@pytest.mark.usefixtures("distributed_setup")
def test_wan_transformer():
    transformer_path = str(component_path("transformer"))
    device = torch.device("cuda:0")
    precision = torch.bfloat16
    args = FastVideoArgs(
        model_path=transformer_path,
        dit_cpu_offload=True,
        pipeline_config=PipelineConfig(dit_config=WanVideoConfig(), dit_precision="bf16"),
    )
    args.device = device

    loader = TransformerLoader()
    candidate = loader.load(transformer_path, args).to(dtype=precision).eval()
    reference = WanTransformer3DModel.from_pretrained(transformer_path, torch_dtype=precision).to(device).eval()

    # Keep temporal and non-square spatial coverage with 288 tokens after
    # patching, while retaining the full model and real checkpoint loader.
    hidden_states = torch.randn(1, 16, 3, 16, 24, device=device, dtype=precision)
    encoder_hidden_states = torch.randn(1, 31, 4096, device=device, dtype=precision)
    timestep = torch.tensor([500], device=device, dtype=precision)
    forward_batch = ForwardBatch(data_type="dummy")

    with torch.inference_mode(), torch.amp.autocast("cuda", dtype=precision):
        expected = reference(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep,
            return_dict=False,
        )[0]
        with set_forward_context(
                current_timestep=0,
                attn_metadata=None,
                forward_batch=forward_batch,
        ):
            actual = candidate(hidden_states=hidden_states,
                               encoder_hidden_states=encoder_hidden_states,
                               timestep=timestep)

    assert actual.shape == expected.shape == hidden_states.shape
    assert_close(actual, expected, atol=1e-1, rtol=1e-2)
