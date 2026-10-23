# SPDX-License-Identifier: Apache-2.0
"""Shared fixtures of the Kandinsky6 SR CPU tests (tiny random models, no GPU, no weights, no reference package)."""
from __future__ import annotations

import os
import sys
from contextlib import ExitStack
from pathlib import Path

import pytest
import torch
import fastvideo.envs as envs

sys.path.insert(0, str(Path(__file__).resolve().parent))

import k6_sr_tiny  # noqa: E402

@pytest.fixture(autouse=True)
def _single_process_dist_env():
    """torch.distributed needs a rendezvous even for one process. Fill only what is missing (never overwrite what a CI
    runner leased) and undo it after the test so the settings do not leak into other tests of the same session."""
    with ExitStack() as stack:
        if "MASTER_ADDR" not in os.environ:
            stack.enter_context(envs.override_external("MASTER_ADDR", "localhost"))
        if "MASTER_PORT" not in os.environ:
            stack.enter_context(envs.override_external("MASTER_PORT", "29531"))
        if "RANK" not in os.environ:
            stack.enter_context(envs.override_external("RANK", "0"))
        if "WORLD_SIZE" not in os.environ:
            stack.enter_context(envs.override_external("WORLD_SIZE", "1"))
        if "LOCAL_RANK" not in os.environ:
            stack.enter_context(envs.override_external("LOCAL_RANK", "0"))
        yield


@pytest.fixture(scope="session")
def distilled_bundle(tmp_path_factory) -> Path:
    """Tiny distilled repo (DX head + ``PiflowScheduler``) in the official Diffusers ``Kandinsky6SRPipeline`` layout.
    The directory name carries no SR marker on purpose: routing must work from the model_index ``_class_name`` alone."""
    return k6_sr_tiny.write_official_sr_repo(tmp_path_factory.mktemp("k6sr_official") / "repo")


@pytest.fixture(scope="session")
def flow_matching_bundle(tmp_path_factory) -> Path:
    """Tiny flow-matching repo: a single-width head, the release ``FlowMatchEulerDiscreteScheduler`` config (shift 5.0),
    the root ``sr_config.json`` and the dict-valued ``_kandinsky6_sr`` model_index entry."""
    return k6_sr_tiny.write_official_sr_repo(tmp_path_factory.mktemp("k6sr_flow_matching") / "repo", piflow=None)


@pytest.fixture()
def cpu_device(monkeypatch):
    """Load and run every SR component on CPU (``get_local_torch_device`` falls back to MPS without CUDA)."""
    import fastvideo.models.loader.component_loader as component_loader
    import fastvideo.pipelines.stages.kandinsky6_sr as sr_stages

    cpu = torch.device("cpu")
    monkeypatch.setattr(component_loader, "get_local_torch_device", lambda: cpu)
    monkeypatch.setattr(sr_stages, "get_local_torch_device", lambda: cpu)
    return cpu


@pytest.fixture()
def tiny_resolutions(monkeypatch):
    """Shrink the trained base resolutions so the tiny models get tiny tiles."""
    from fastvideo.pipelines.basic.kandinsky6_sr import tiling

    monkeypatch.setitem(tiling.RESOLUTIONS, 512, list(k6_sr_tiny.TINY_RESOLUTIONS[512]))
    return k6_sr_tiny.TINY_RESOLUTIONS


@pytest.fixture()
def cpu_args(cpu_device):
    """Factory of ``FastVideoArgs`` resolved through the registry for a bundle directory, configured for CPU tests."""
    from fastvideo.fastvideo_args import FastVideoArgs

    def make(bundle: Path) -> FastVideoArgs:
        args = FastVideoArgs.from_kwargs(model_path=str(bundle), num_gpus=1, dit_cpu_offload=False,
                                         dit_layerwise_offload=False, vae_cpu_offload=False, use_fsdp_inference=False,
                                         pin_cpu_memory=False)
        args.pipeline_config.dit_precision = "fp32"
        args.pipeline_config.vae_precision = "fp32"
        args.pipeline_config.upsampler_precision = "fp32"
        return args

    return make


@pytest.fixture()
def fv_args(distilled_bundle, cpu_args):
    """``FastVideoArgs`` for the session's tiny bundle (see ``distilled_bundle``)."""
    return cpu_args(distilled_bundle)
