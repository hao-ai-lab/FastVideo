# SPDX-License-Identifier: Apache-2.0
"""Opt-in parity of the ported latent upscaler against the author's ComfyUI node.

Needs the release weights (OPEN_H3_REGEN_WEIGHTS) and a checkout of NVlabs/Sana on the ``sol-engine`` branch
(OPEN_H3_REGEN_REF, the models/Open-H3-Regenerate-2K directory). GPU only: the node runs the network in bf16.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import types
from pathlib import Path

import pytest
import torch

from fastvideo.models.upsamplers.minimax_h3_latent_upscaler import MiniMaxH3LatentUpscaler, upscale_latent

WEIGHTS = os.environ.get("OPEN_H3_REGEN_WEIGHTS")
REFERENCE = os.environ.get("OPEN_H3_REGEN_REF")
CHECKPOINT = "latent_upsampler/minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors"

pytestmark = pytest.mark.skipif(not (WEIGHTS and REFERENCE and torch.cuda.is_available()),
                                reason="needs OPEN_H3_REGEN_WEIGHTS, OPEN_H3_REGEN_REF and a GPU")


def _load_author_node(checkpoint: str):
    folder = types.ModuleType("folder_paths")
    folder.folder_names_and_paths = {"latent_upscale_models": ([], set())}
    folder.get_full_path_or_raise = lambda category, name: checkpoint
    sys.modules["folder_paths"] = folder
    sys.modules["comfy.model_management"] = None
    sys.modules["comfy_api.latest"] = None
    spec = importlib.util.spec_from_file_location("author_upscaler",
                                                  Path(REFERENCE) / "open_h3" / "upscaler_node.py")
    node = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(node)
    node.io.NodeOutput = lambda value: (value, )
    return node


@pytest.mark.parametrize("shape", [(24, 37, 48, 84), (24, 37, 36, 64)])
def test_upscaler_matches_the_author_node(shape):
    checkpoint = str(Path(WEIGHTS) / CHECKPOINT)
    node = _load_author_node(checkpoint)
    torch.manual_seed(0)
    latent = torch.randn(shape)
    mode = {"mode": node.UpscaleMode.TARGET_DIMENSIONS, "width": 2560, "height": 1440}
    with torch.inference_mode():
        (expected, ) = node.MinimaxH3LatentUpscaler3D.execute({"samples": latent[None].float()},
                                                              "upscaler",
                                                              mode,
                                                              align=32,
                                                              enable_temporal_chunking=False,
                                                              force_unload=False,
                                                              device="cuda",
                                                              precision="bf16")
    model = MiniMaxH3LatentUpscaler.from_checkpoint(checkpoint).to("cuda", torch.bfloat16)
    actual = upscale_latent(model, latent.cuda(), (1440, 2560)).cpu()
    assert torch.equal(actual, expected["samples"].float()[0])
