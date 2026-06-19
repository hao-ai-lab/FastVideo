from types import SimpleNamespace

import torch

from fastvideo.train.models.interleave_thinker import (
    INTERLEAVE_GUIDANCE_PLANNER_PROMPT,
    INTERLEAVE_PLANNER_PROMPT,
    InterleavePlannerStep,
    InterleaveThinkerPlannerModel,
    extract_interleave_plan,
)
from fastvideo.train.models.interleave_thinker.qwen_actor import (
    _PlaceholderActorModule,
)
from fastvideo.train.utils.config import load_run_config


class _FakeBackendPlanner(InterleaveThinkerPlannerModel):

    @property
    def device(self):
        return torch.device("cpu")


class _FakeProcessor:

    def __init__(self):
        self.tokenizer = SimpleNamespace(pad_token_id=0, eos_token_id=2)

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize,
        add_generation_prompt,
        return_dict,
        return_tensors,
    ):
        del tokenize, add_generation_prompt, return_dict, return_tensors
        assert messages[0]["role"] == "user"
        return {
            "input_ids": torch.arange(1, 4).unsqueeze(0),
            "attention_mask": torch.ones(1, 3, dtype=torch.long),
        }

    def batch_decode(self, sequences, **kwargs):
        del sequences, kwargs
        return [
            """
            <think>plan</think>
            <answer>
            {"execution_plan": [
              {"step_number": 1, "step_name": "Sketch", "instruction": "Draw a cat", "prompt": "a clean cat sketch", "auxiliary_text": null}
            ]}
            </answer>
            """
        ]


class _FakeQwen(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.generate_kwargs = None

    def generate(self, **kwargs):
        self.generate_kwargs = kwargs
        input_ids = kwargs["input_ids"]
        suffix = torch.tensor([[9, 10]], dtype=input_ids.dtype)
        return torch.cat([input_ids, suffix], dim=1)


def test_extract_interleave_plan_accepts_json_answer_block():
    parsed = extract_interleave_plan("""
    <think>ok</think>
    <answer>
    {"execution_plan": [
      {"step_number": 1, "step_name": "Base", "instruction": "Draw a cube", "prompt": "a blue cube", "auxiliary_text": null}
    ]}
    </answer>
    """)

    assert parsed is not None
    assert parsed.steps == (InterleavePlannerStep(
        step_number=1,
        step_name="Base",
        instruction="Draw a cube",
        prompt="a blue cube",
        auxiliary_text=None,
    ), )


def test_extract_interleave_plan_accepts_upstream_python_literal_answer_block():
    parsed = extract_interleave_plan("""
    <answer>
    {'execution_plan': [
      {'step_number': '2', 'step_name': 'Color', 'instruction': 'Color the cube', 'prompt': 'make the cube red', 'auxiliary_text': None}
    ]}
    </answer>
    """)

    assert parsed is not None
    assert parsed.steps[0].step_number == 2
    assert parsed.steps[0].step_name == "Color"
    assert parsed.steps[0].auxiliary_text is None


def test_interleave_thinker_planner_builds_text_only_messages_without_backend():
    model = InterleaveThinkerPlannerModel(load_backend=False)

    messages = model.build_messages({"instruction": "Show how to draw a cat step by step."})

    content = messages[0]["content"]
    images = [part for part in content if part["type"] == "image"]
    text = "\n".join(part["text"] for part in content if part["type"] == "text")
    assert images == []
    assert "Show how to draw a cat step by step." in text
    assert "execution_plan" in text
    assert "{text_input}" in INTERLEAVE_PLANNER_PROMPT


def test_interleave_thinker_planner_builds_image_conditioned_messages_without_backend():
    model = InterleaveThinkerPlannerModel(load_backend=False)

    messages = model.build_messages({
        "instruction": "Continue this process in two steps.",
        "input_image_paths": ["step1.png", "step2.png"],
    })

    content = messages[0]["content"]
    images = [part for part in content if part["type"] == "image"]
    text = "\n".join(part["text"] for part in content if part["type"] == "text")
    assert images == [{
        "type": "image",
        "image": "step1.png",
    }, {
        "type": "image",
        "image": "step2.png",
    }]
    assert "Continue this process in two steps." in text
    assert "Multimodal Sequence Planner" in text
    assert "{text_input}" in INTERLEAVE_GUIDANCE_PLANNER_PROMPT


def test_interleave_thinker_planner_fake_backend_generates_parseable_plan():
    model = _FakeBackendPlanner(load_backend=False)
    assert isinstance(model.transformer, _PlaceholderActorModule)
    model.processor = _FakeProcessor()
    model.transformer = _FakeQwen()

    plans = model.generate_interleave_plans(
        {
            "items": [{
                "instruction": "Show how to draw a cat step by step."
            }]
        },
        num_generations=1,
        temperature=0.7,
        top_p=0.9,
        max_new_tokens=12,
    )

    assert len(plans) == 1
    assert plans[0]["plan"] is not None
    assert plans[0]["steps"][0].prompt == "a clean cat sketch"
    assert model.transformer.generate_kwargs["max_new_tokens"] == 12
    assert model.transformer.generate_kwargs["num_return_sequences"] == 1


def test_interleave_thinker_planner_config_parses_public_yaml():
    cfg = load_run_config("examples/train/configs/interleave_thinker/planner_smoke.yaml")

    assert cfg.models["student"]["_target_"] == (
        "fastvideo.train.models.interleave_thinker.InterleaveThinkerPlannerModel")
    assert cfg.models["student"]["init_from"] == "InterleaveThinker/InterleaveThinker-Planner-8B"
    assert cfg.models["student"]["processor_from"] == "Qwen/Qwen3-VL-8B-Instruct"
    assert cfg.models["student"]["trainable"] is False
