# SPDX-License-Identifier: Apache-2.0
"""PAN2 guardrails: the prompt and video checks against a stand-in `pan2_guardrail`; no weights or CUDA needed."""

import importlib.machinery
import sys
import types
from types import SimpleNamespace

import pytest
import torch

from fastvideo.configs.pipelines.pan2 import PAN2PipelineConfig
from fastvideo.pipelines.basic.pan2 import guardrails as pan2_guardrails
from fastvideo.pipelines.basic.pan2.guardrails import (
    PAN2Guardrails,
    PAN2GuardrailViolationError,
    PAN2PromptGuardrailStage,
    PAN2VideoGuardrailStage,
)


class FakeSafetyChecker:
    """Blocks prompts containing "unsafe" and videos whose first pixel is white; records what it was given."""

    def __init__(self):
        self.text_guard = torch.nn.Linear(1, 1)
        self.video_guard = torch.nn.Linear(1, 1)
        self.prompts, self.videos, self.devices = [], [], []

    @classmethod
    def from_pretrained(cls):
        return cls()

    def to(self, device):
        self.devices.append(str(device))
        return self

    def check_text_safety(self, prompt):
        self.prompts.append(prompt)
        return "unsafe" not in prompt

    def check_video_safety(self, frames):
        self.videos.append(frames)
        return None if frames[0, 0, 0, 0] == 255 else frames


@pytest.fixture
def fake_package(monkeypatch):
    module = types.ModuleType("pan2_guardrail")
    module.__spec__ = importlib.machinery.ModuleSpec("pan2_guardrail", None)
    module.PAN2SafetyChecker = FakeSafetyChecker
    monkeypatch.setitem(sys.modules, "pan2_guardrail", module)
    monkeypatch.setattr(pan2_guardrails, "get_local_torch_device", lambda: torch.device("cpu"))
    return module


def _args(output_type="pil"):
    return SimpleNamespace(output_type=output_type)


def test_guardrails_are_on_by_default():
    config = PAN2PipelineConfig()
    assert config.enable_guardrails is True
    assert config.offload_guardrail_models is False


def test_a_missing_package_says_how_to_install_or_disable(monkeypatch):
    monkeypatch.setattr(pan2_guardrails.importlib.util, "find_spec", lambda name: None)
    with pytest.raises(ImportError, match="pan2-guardrail.*enable_guardrails=False"):
        PAN2Guardrails()


def test_the_checker_loads_on_the_gpu_unless_offloaded(fake_package):
    assert PAN2Guardrails().checker.devices == ["cpu"]
    assert PAN2Guardrails(offload_to_cpu=True).checker.devices == []


def test_a_safe_prompt_passes_and_an_unsafe_one_is_blocked(fake_package):
    guardrails = PAN2Guardrails()
    stage = PAN2PromptGuardrailStage(guardrails)

    batch = SimpleNamespace(prompt="A red fox in the snow.")
    assert stage.forward(batch, _args()) is batch
    with pytest.raises(PAN2GuardrailViolationError, match="prompt was blocked"):
        stage.forward(SimpleNamespace(prompt="an unsafe prompt"), _args())
    assert guardrails.checker.prompts == ["A red fox in the snow.", "an unsafe prompt"]


def test_latent_output_is_refused_before_any_compute(fake_package):
    guardrails = PAN2Guardrails()
    with pytest.raises(ValueError, match="latent output"):
        PAN2PromptGuardrailStage(guardrails).forward(SimpleNamespace(prompt="A red fox."), _args("latent"))
    assert guardrails.checker.prompts == []


def test_each_video_is_checked_as_the_delivered_uint8_frames(fake_package):
    guardrails = PAN2Guardrails()
    stage = PAN2VideoGuardrailStage(guardrails)
    # Two [3, T=2, H=4, W=5] videos in [0, 1]; values outside it saturate as in the generator.
    output = torch.rand(2, 3, 2, 4, 5)
    output[0, :, 0, 0, 0] = 0.5
    output[1, :, 0, 0, 0] = -0.1

    batch = SimpleNamespace(output=output)
    assert stage.forward(batch, _args()) is batch
    first, second = guardrails.checker.videos
    assert first.shape == (2, 4, 5, 3) and first.dtype == "uint8"
    assert first[0, 0, 0].tolist() == [127, 127, 127] and second[0, 0, 0].tolist() == [0, 0, 0]
    assert (first == (output[0] * 255).clamp(0, 255).to(torch.uint8).permute(1, 2, 3, 0).numpy()).all()


def test_a_blocked_video_raises(fake_package):
    output = torch.zeros(1, 3, 2, 4, 5)
    output[0, :, 0, 0, 0] = 1.0
    with pytest.raises(PAN2GuardrailViolationError, match="video was blocked"):
        PAN2VideoGuardrailStage(PAN2Guardrails()).forward(SimpleNamespace(output=output), _args())


def test_offloaded_guards_visit_the_device_only_for_their_check(fake_package, monkeypatch):
    moves = []
    guardrails = PAN2Guardrails(offload_to_cpu=True)
    for name in ("text_guard", "video_guard"):
        guard = getattr(guardrails.checker, name)
        monkeypatch.setattr(guard, "to", lambda device, name=name: moves.append((name, str(device))))

    PAN2PromptGuardrailStage(guardrails).forward(SimpleNamespace(prompt="A red fox."), _args())
    PAN2VideoGuardrailStage(guardrails).forward(SimpleNamespace(output=torch.zeros(1, 3, 1, 2, 2)), _args())
    assert moves == [("text_guard", "cpu"), ("text_guard", "cpu"), ("video_guard", "cpu"), ("video_guard", "cpu")]


class FakeGroup:
    """A two-rank sequence-parallel group seen from ``rank``; a peer receives ``replies`` in order."""

    def __init__(self, rank, replies=()):
        self.rank_in_group, self.world_size = rank, 2
        self.replies, self.broadcasts = list(replies), []

    def broadcast_object(self, obj, src=0):
        self.broadcasts.append(obj)
        return obj if self.rank_in_group == src else self.replies.pop(0)


def _sp_group(monkeypatch, group):
    monkeypatch.setattr(pan2_guardrails, "model_parallel_is_initialized", lambda: True)
    monkeypatch.setattr(pan2_guardrails, "get_sp_group", lambda: group)
    return group


def test_ranks_other_than_the_first_take_the_broadcast_verdict(fake_package, monkeypatch):
    group = _sp_group(monkeypatch, FakeGroup(rank=1, replies=[(True, None), (True, False)]))
    guardrails = PAN2Guardrails()
    # This rank loads no checker and runs no check: it raises on rank 0's verdict.
    assert guardrails.checker is None
    with pytest.raises(PAN2GuardrailViolationError):
        guardrails.check_prompt("A red fox.")
    assert group.broadcasts == [None, None]


def test_a_failed_load_on_the_first_rank_reaches_the_other_ranks_at_once(fake_package, monkeypatch):
    def fail():
        raise OSError("no guard weights")

    monkeypatch.setattr(FakeSafetyChecker, "from_pretrained", staticmethod(fail))
    first = _sp_group(monkeypatch, FakeGroup(rank=0))
    with pytest.raises(OSError, match="no guard weights"):
        PAN2Guardrails()
    assert first.broadcasts == [(False, "OSError: no guard weights")]

    _sp_group(monkeypatch, FakeGroup(rank=1, replies=first.broadcasts))
    with pytest.raises(RuntimeError, match="failed to load the guardrails: OSError: no guard weights"):
        PAN2Guardrails()


def test_a_failed_check_on_the_first_rank_reaches_the_other_ranks_at_once(fake_package, monkeypatch):
    first = _sp_group(monkeypatch, FakeGroup(rank=0))
    guardrails = PAN2Guardrails()
    monkeypatch.setattr(guardrails.checker, "check_text_safety", lambda prompt: 1 / 0)
    with pytest.raises(ZeroDivisionError):
        guardrails.check_prompt("A red fox.")
    assert first.broadcasts[-1] == (False, "ZeroDivisionError: division by zero")

    _sp_group(monkeypatch, FakeGroup(rank=1, replies=[(True, None), first.broadcasts[-1]]))
    with pytest.raises(RuntimeError, match="failed to run the guardrail check"):
        PAN2Guardrails().check_prompt("A red fox.")


@pytest.mark.parametrize("config", [None, SimpleNamespace(enable_guardrails=True, offload_guardrail_models=True)])
def test_the_pipeline_checks_the_prompt_first_and_the_video_last(monkeypatch, config):
    from fastvideo.pipelines.basic.pan2 import pan2_pipeline

    for name in ("PAN2InputValidationStage", "PAN2TextEncodingStage", "PAN2TimestepPreparationStage",
                 "PAN2ConditioningStage", "PAN2DenoisingStage", "DecodingStage", "LatentPreparationStage"):
        monkeypatch.setattr(pan2_pipeline, name, lambda *args, **kwargs: None)
    built = []
    monkeypatch.setattr(pan2_pipeline, "PAN2Guardrails", lambda **kwargs: built.append(kwargs) or "guardrails")
    pipeline = pan2_pipeline.PAN2Pipeline.__new__(pan2_pipeline.PAN2Pipeline)
    stages = {}
    pipeline.add_stage = lambda stage_name, stage: stages.__setitem__(stage_name, stage)
    pipeline.get_module = lambda name: None

    pipeline.create_pipeline_stages(SimpleNamespace(pipeline_config=config))

    names = list(stages)
    assert names[:2] == ["input_validation_stage", "prompt_guardrail_stage"]
    assert names[-2:] == ["decoding_stage", "video_guardrail_stage"]
    assert stages["prompt_guardrail_stage"].guardrails == stages["video_guardrail_stage"].guardrails == "guardrails"
    assert built == [{"offload_to_cpu": config is not None}]


def test_the_pipeline_has_no_guardrail_stages_when_they_are_off(monkeypatch):
    from fastvideo.pipelines.basic.pan2 import pan2_pipeline

    for name in ("PAN2InputValidationStage", "PAN2TextEncodingStage", "PAN2TimestepPreparationStage",
                 "PAN2ConditioningStage", "PAN2DenoisingStage", "DecodingStage", "LatentPreparationStage"):
        monkeypatch.setattr(pan2_pipeline, name, lambda *args, **kwargs: None)
    monkeypatch.setattr(pan2_pipeline, "PAN2Guardrails", lambda **kwargs: pytest.fail("guardrails were loaded"))
    pipeline = pan2_pipeline.PAN2Pipeline.__new__(pan2_pipeline.PAN2Pipeline)
    stages = {}
    pipeline.add_stage = lambda stage_name, stage: stages.__setitem__(stage_name, stage)
    pipeline.get_module = lambda name: None

    pipeline.create_pipeline_stages(SimpleNamespace(pipeline_config=SimpleNamespace(enable_guardrails=False)))

    assert not any("guardrail" in name for name in stages)
