# SPDX-License-Identifier: Apache-2.0
"""Run real pipeline assembly with weight-free modules, not source-text assertions."""

import importlib
from types import SimpleNamespace

import pytest
import torch

from fastvideo.tests.stages._denoising_fixtures import RecordingDenoiser, TinyVAE, _args, _batch, _patch_denoising_module


@pytest.mark.parametrize("module_name, sampler, first_frame", [
    ("wan.wan_pipeline", "WanDenoisingStage", True),
    ("wan.wan_i2v_pipeline", "WanDenoisingStage", False),
    ("wan.wan_v2v_pipeline", "WanDenoisingStage", False),
    ("wan.lucy_edit_pipeline", "WanDenoisingStage", False),
    ("wan.wan_dmd_pipeline", "DmdDenoisingStage", False),
    ("wan.wan_i2v_dmd_pipeline", "DmdDenoisingStage", False),
    ("wan.wan_causal_pipeline", "CausalDenoisingStage", False),
    ("wan.wan_causal_dmd_pipeline", "CausalDMDDenosingStage", True),
    ("dreamx_world.dreamx_world_pipeline", "WanDenoisingStage", True),
    ("turbodiffusion.turbodiffusion_pipeline", "WanDenoisingStage", False),
    ("turbodiffusion.turbodiffusion_i2v_pipeline", "WanDenoisingStage", False),
])
def test_pipeline_wires_family_sampler_and_owns_scheduler(monkeypatch, module_name, sampler, first_frame):
    _patch_denoising_module(monkeypatch, "1.0")
    from fastvideo.pipelines.stages.base import PipelineStage
    from fastvideo.models.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
    module = importlib.import_module(f"fastvideo.pipelines.basic.{module_name}")

    class PreparedStage:
        def __init__(self, *args, **kwargs):
            self.kwargs = kwargs

    for name, value in list(vars(module).items()):
        if isinstance(value, type) and issubclass(value, PipelineStage) and name not in {sampler, "WanFirstFrameEncodingStage"}:
            monkeypatch.setattr(module, name, PreparedStage)
    model = RecordingDenoiser()
    model.blocks = [None]
    model.config.arch_config = SimpleNamespace(num_frames_per_block=3, sliding_window_num_frames=21)
    pipeline = module.EntryClass.__new__(module.EntryClass)
    pipeline.modules = {"transformer": model, "transformer_2": None, "vae": TinyVAE(),
                        "scheduler": FlowMatchEulerDiscreteScheduler(shift=8.0)}
    pipeline.get_module = lambda name, *default: pipeline.modules.get(name)
    stages = {}
    pipeline.add_stage = lambda *, stage_name, stage: stages.setdefault(stage_name, stage)
    args = _args()
    args.pipeline_config.flow_shift = 3.0
    pipeline.initialize_pipeline(args)
    pipeline.create_pipeline_stages(args)
    stage = stages["denoising_stage"]
    assert type(stage).__name__ == sampler
    assert type(stage).__module__.startswith("fastvideo.pipelines.basic.wan.stages.")
    assert stage.vae is None
    assert ("first_frame_encoding_stage" in stages) is first_frame
    if first_frame:
        assert list(stages).index("first_frame_encoding_stage") < list(stages).index("denoising_stage")
        assert stages["first_frame_encoding_stage"].vae is pipeline.modules["vae"]
    preparation = stages["latent_preparation_stage"].kwargs["scheduler"]
    if sampler == "CausalDMDDenosingStage":
        assert "timestep_preparation_stage" not in stages
    else:
        assert stages["timestep_preparation_stage"].kwargs["scheduler"] is preparation
    if sampler == "DmdDenoisingStage":
        assert stage.scheduler is not preparation
        assert stage.scheduler.config.shift == 8.0
        preparation.set_timesteps(3)
        assert len(stage.scheduler.timesteps) == 1000
    else:
        assert stage.scheduler is preparation


@pytest.mark.parametrize("causal", [False, True])
def test_first_frame_preparation_validates_and_clears_request_state(monkeypatch, causal):
    from fastvideo.pipelines.basic.wan.stages import conditioning
    monkeypatch.setattr(conditioning, "get_local_torch_device", lambda: torch.device("cpu"))
    args, batch = _args(), _batch()
    args.pipeline_config.ti2v_task = not causal
    batch.pil_image = torch.zeros(1, 3, 1, 16, 32)
    stage = conditioning.WanFirstFrameEncodingStage(TinyVAE(), causal=causal)
    assert stage.verify_input(batch, args).is_valid()
    stage.forward(batch, args)
    assert stage.verify_output(batch, args).is_valid()
    expected = (torch.full((1, 2, 1, 2, 4), 0.5) - stage.vae.shift_factor) * stage.vae.scaling_factor
    torch.testing.assert_close(batch.first_frame_latent, expected, atol=0, rtol=0)
    batch.pil_image = None
    stage.forward(batch, args)
    assert batch.first_frame_latent is None
    assert stage.vae.encode_calls == 1
