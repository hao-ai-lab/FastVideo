import types

import torch

from fastvideo.api.sampling_param import SamplingParam
from fastvideo.pipelines import ForwardBatch, TrainingBatch
from fastvideo.training.distillation_pipeline import DistillationPipeline


class _PromptEncodingStage:

    def __init__(self):
        self.calls = 0
        self.prompts = []

    def __call__(self, batch: ForwardBatch, _training_args):
        self.calls += 1
        self.prompts.append(batch.prompt)
        batch.prompt_embeds.append(torch.ones(1, 2, 3))
        assert batch.prompt_attention_mask is not None
        batch.prompt_attention_mask.append(torch.ones(1, 2))
        return batch


class _ValidationPipeline:

    def __init__(self):
        self.prompt_encoding_stage = _PromptEncodingStage()


class _DistillationPipelineForTest(DistillationPipeline):

    def initialize_validation_pipeline(self, training_args):
        self.validation_pipeline_init_calls += 1
        self.validation_pipeline = _ValidationPipeline()


def _make_pipeline():
    pipeline = object.__new__(_DistillationPipelineForTest)
    pipeline.validation_pipeline_init_calls = 0
    return pipeline


def test_negative_prompt_conditioning_initializes_without_validation_enabled(monkeypatch):
    pipeline = _make_pipeline()
    training_args = types.SimpleNamespace(model_path="Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
    monkeypatch.setattr(
        SamplingParam,
        "from_pretrained",
        staticmethod(lambda _model_path: types.SimpleNamespace(negative_prompt="low quality")),
    )

    pipeline._ensure_negative_prompt_conditioning(training_args)

    assert pipeline.validation_pipeline_init_calls == 1
    assert pipeline.validation_pipeline.prompt_encoding_stage.calls == 1
    assert pipeline.validation_pipeline.prompt_encoding_stage.prompts == ["low quality"]
    assert torch.equal(pipeline.negative_prompt_embeds, torch.ones(1, 2, 3))
    assert torch.equal(pipeline.negative_prompt_attention_mask, torch.ones(1, 2))

    pipeline._ensure_negative_prompt_conditioning(training_args)

    assert pipeline.validation_pipeline_init_calls == 1
    assert pipeline.validation_pipeline.prompt_encoding_stage.calls == 1


def test_initialized_negative_prompt_conditioning_builds_unconditional_kwargs():
    pipeline = _make_pipeline()
    training_args = types.SimpleNamespace(model_path="Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
    pipeline._ensure_negative_prompt_conditioning(training_args, negative_prompt="low quality")
    training_batch = TrainingBatch()
    text_dict = {
        "encoder_hidden_states": pipeline.negative_prompt_embeds,
        "encoder_attention_mask": pipeline.negative_prompt_attention_mask,
    }
    noise_input = torch.zeros(1, 2, 3, 4, 5)
    timestep = torch.tensor([1])

    pipeline._build_distill_input_kwargs(noise_input, timestep, text_dict, training_batch)

    assert training_batch.input_kwargs["encoder_hidden_states"] is pipeline.negative_prompt_embeds
    assert training_batch.input_kwargs["encoder_attention_mask"] is pipeline.negative_prompt_attention_mask
    assert training_batch.input_kwargs["hidden_states"].shape == (1, 3, 2, 4, 5)
