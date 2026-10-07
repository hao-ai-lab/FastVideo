# SPDX-License-Identifier: Apache-2.0
"""Real-weights parity of the Kandinsky6 SR latent upscaler against the previous FastVideo implementation.

The previous implementation (a general re-implementation of the training code's upscaler family) was itself checked
against the ``k6_video`` reference; the current module implements only the released architecture.  Both run the
released ``latent_upscaler/`` weights on one fixed latent tile for x2 and x4, in fp32 and in bf16 under bf16 autocast
(the production call), and the outputs must be bitwise equal.

Needs a CUDA device and (skips otherwise):

    KANDINSKY6_SR_BUNDLE          local copy of an official Diffusers SR repo (both repos share ``latent_upscaler/``)
    KANDINSKY6_SR_PREVIOUS_ROOT   a FastVideo checkout that still has ``fastvideo/models/upsamplers/kandinsky6_sr_lu/``
                                  (e.g. commit 12be8fdd); it runs in a subprocess because both packages are
                                  named ``fastvideo``

Run: ``pytest tests/local_tests/kandinsky6_sr/test_kandinsky6_sr_latent_upscaler_parity.py -v -s``.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from . import lu_parity_common as common

BUNDLE = os.environ.get("KANDINSKY6_SR_BUNDLE")
PREVIOUS_ROOT = os.environ.get("KANDINSKY6_SR_PREVIOUS_ROOT")
LU_DIR = str(Path(BUNDLE or "") / "latent_upscaler")

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="latent-upscaler parity needs a CUDA device"),
    pytest.mark.skipif(not BUNDLE or not list(Path(LU_DIR).glob("*.safetensors")),
                       reason="set KANDINSKY6_SR_BUNDLE to a local SR repo with latent_upscaler weights"),
]


@pytest.fixture(scope="module")
def previous(tmp_path_factory):
    if not PREVIOUS_ROOT or not (Path(PREVIOUS_ROOT) / "fastvideo/models/upsamplers/kandinsky6_sr_lu").is_dir():
        pytest.skip("set KANDINSKY6_SR_PREVIOUS_ROOT to a FastVideo checkout with the previous latent upscaler")
    out = tmp_path_factory.mktemp("lu_parity") / "previous.pt"
    env = dict(os.environ, PYTHONPATH=PREVIOUS_ROOT)
    subprocess.run([sys.executable, common.__file__, "--impl", "previous", "--lu-dir", LU_DIR, "--out",
                    str(out)],
                   cwd=PREVIOUS_ROOT,
                   env=env,
                   check=True)
    result = torch.load(out)
    assert Path(result["fastvideo_file"]).resolve().is_relative_to(Path(PREVIOUS_ROOT).resolve())
    return result


def _build_current():
    from fastvideo.configs.models.upsamplers.kandinsky6_sr import Kandinsky6SRLatentUpscalerConfig
    from fastvideo.models.upsamplers.kandinsky6_sr import Kandinsky6SRLatentUpscalerBank

    config = json.loads((Path(LU_DIR) / "config.json").read_text())
    with torch.device("meta"):
        return Kandinsky6SRLatentUpscalerBank(Kandinsky6SRLatentUpscalerConfig(**config))


@pytest.fixture(scope="module")
def current():
    bank = _build_current()
    bank.load_state_dict(common.load_checkpoint(LU_DIR), strict=True, assign=True)
    bank.eval()
    outputs = common.run_outputs(bank, lambda b, z, s: b(z, s))
    return {
        "outputs": outputs,
        "num_parameters": sum(p.numel() for p in bank.parameters()),
        "state_keys": list(bank.state_dict()),
        "backend_flags": common.backend_flags(),
    }


def test_state_dict_matches_the_previous_implementation(previous, current):
    assert current["state_keys"] == previous["state_keys"]
    assert current["num_parameters"] == previous["num_parameters"]


def test_partial_checkpoint_fails_the_strict_load():
    state = common.load_checkpoint(LU_DIR)
    state.pop(next(key for key in state if key.startswith("_models.2x.x2_branch.")))
    with pytest.raises(RuntimeError, match="Missing key"):
        _build_current().load_state_dict(state, strict=True, assign=True)


@pytest.mark.parametrize("dtype", common.DTYPES)
@pytest.mark.parametrize("scale", common.SCALES)
def test_outputs_are_bitwise_equal(previous, current, dtype, scale):
    assert current["backend_flags"] == previous["backend_flags"]
    key = f"{dtype}_x{scale}"
    got, want = current["outputs"][key], previous["outputs"][key]
    n, c, t, h, w = common.LATENT_SHAPE
    assert got.shape == want.shape == (n, c, t, h * scale, w * scale)
    assert torch.isfinite(got).all()
    max_diff = (got - want).abs().max().item()
    print(f"{key}: max |diff| = {max_diff}")
    assert torch.equal(got, want), f"{key}: max |diff| = {max_diff}"


def test_production_loader_matches_the_previous_bf16_outputs(previous):
    """``UpsamplerLoader`` (config from the pipeline config, bf16 build, strict load) reproduces the bf16 outputs."""
    from fastvideo.fastvideo_args import FastVideoArgs
    from fastvideo.models.loader.component_loader import PipelineComponentLoader

    args = FastVideoArgs.from_kwargs(model_path=BUNDLE, num_gpus=1, use_fsdp_inference=False, dit_cpu_offload=False,
                                     vae_cpu_offload=False, pin_cpu_memory=False)
    index = json.loads((Path(BUNDLE) / "model_index.json").read_text())
    args._model_index_class_names = {key: spec[1] for key, spec in index.items() if isinstance(spec, list)}
    library = index["latent_upscaler"][0]
    bank = PipelineComponentLoader.load_module("latent_upscaler", LU_DIR, library, args)
    assert type(bank).__name__ == "Kandinsky6SRLatentUpscalerBank" and bank.scales == (2, 4)
    assert next(bank.parameters()).dtype == torch.bfloat16 and not bank.training
    assert all(not p.requires_grad for p in bank.parameters())
    z = torch.randn(common.LATENT_SHAPE, generator=torch.Generator().manual_seed(common.SEED))
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for scale in common.SCALES:
            got = bank(z.to("cuda", torch.bfloat16), scale).float().cpu()
            assert torch.equal(got, previous["outputs"][f"bf16_x{scale}"]), f"x{scale}"
