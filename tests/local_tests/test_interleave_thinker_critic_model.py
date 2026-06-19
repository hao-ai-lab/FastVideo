import json
from types import SimpleNamespace

import torch

from fastvideo.train.models.interleave_thinker.critic import (
    _PlaceholderActorModule,
)
from fastvideo.train.models.interleave_thinker import (
    INTERLEAVE_CRITIC_PROMPT,
    InterleaveThinkerCriticModel,
)
from fastvideo.train.utils.training_config import (
    DataConfig,
    TrainingConfig,
)


class _FakeBackendCritic(InterleaveThinkerCriticModel):

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
        has_assistant = any(message["role"] == "assistant" for message in messages)
        length = 5 if has_assistant else 3
        return {
            "input_ids": torch.arange(1, length + 1).unsqueeze(0),
            "attention_mask": torch.ones(1, length, dtype=torch.long),
        }

    def batch_decode(self, sequences, **kwargs):
        del kwargs
        return [
            '<think>ok</think><answer>{"previous_step_success": true, "refine_prompt": "better"}</answer>'
            for _ in sequences
        ]


class _FakeQwen(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.generate_kwargs = None
        self.last_labels = None

    def generate(self, **kwargs):
        self.generate_kwargs = kwargs
        input_ids = kwargs["input_ids"]
        num_return_sequences = int(kwargs["num_return_sequences"])
        suffix = torch.tensor([[9, 10]], dtype=input_ids.dtype)
        return torch.cat([input_ids.repeat(num_return_sequences, 1), suffix.repeat(num_return_sequences, 1)], dim=1)

    def forward(self, **kwargs):
        input_ids = kwargs["input_ids"]
        vocab_size = max(16, int(input_ids.max().detach().cpu()) + 1)
        logits = torch.zeros(
            *input_ids.shape,
            vocab_size,
            dtype=self.weight.dtype,
            device=input_ids.device,
        )
        for idx in range(input_ids.shape[1] - 1):
            next_token = input_ids[:, idx + 1]
            logits[:, idx].scatter_(1, next_token[:, None], self.weight.expand(input_ids.shape[0], 1))
        labels = kwargs.get("labels")
        if labels is not None:
            self.last_labels = labels.detach().clone()
            trainable_fraction = (labels != -100).float().mean()
            return SimpleNamespace(
                loss=self.weight.pow(2).sum() * trainable_fraction,
                logits=logits,
            )
        return SimpleNamespace(logits=logits)


def test_interleave_thinker_critic_builds_qwen_vl_messages_without_loading_backend():
    model = InterleaveThinkerCriticModel(load_backend=False)

    messages = model.build_messages({
        "origin_prompt": "draw a vase",
        "previous_prompt": "a vase on a table",
        "origin_image_path": "before.png",
        "edited_image_path": "after.png",
    })

    content = messages[0]["content"]
    images = [part for part in content if part["type"] == "image"]
    text = "\n".join(part["text"] for part in content if part["type"] == "text")
    assert messages[0]["role"] == "user"
    assert images == [{
        "type": "image",
        "image": "before.png",
    }, {
        "type": "image",
        "image": "after.png",
    }]
    assert "draw a vase" in text
    assert "a vase on a table" in text


def test_interleave_thinker_critic_prefers_previous_image_as_before_image():
    model = InterleaveThinkerCriticModel(load_backend=False)

    messages = model.build_messages({
        "origin_prompt": "draw a vase",
        "previous_prompt": "a vase on a table",
        "origin_image_path": "original.png",
        "previous_image_path": "before.png",
        "edited_image_path": "after.png",
    })

    images = [part for part in messages[0]["content"] if part["type"] == "image"]
    assert images == [{
        "type": "image",
        "image": "before.png",
    }, {
        "type": "image",
        "image": "after.png",
    }]


def test_interleave_thinker_critic_initializes_jsonl_dataloader(tmp_path):
    data_path = tmp_path / "critic_rl.jsonl"
    data_path.write_text(
        json.dumps({
            "origin_prompt": "draw a chair",
            "rewritten_prompt": "a wooden chair",
            "origin_image_path": "interleave/before.png",
            "edited_image_path": "interleave/after.png",
            "evaluation": {
                "success": True,
                "semantics": 7.0,
                "quality": 8.0,
            },
        }) + "\n")
    image_dir = tmp_path / "images"
    model = InterleaveThinkerCriticModel(load_backend=False, image_dir=str(image_dir))

    model.init_preprocessors(TrainingConfig(data=DataConfig(data_path=str(data_path), train_batch_size=1)))
    batch = next(iter(model.dataloader))

    assert batch["items"][0]["origin_prompt"] == "draw a chair"
    assert batch["items"][0]["origin_image_path"] == str(image_dir / "interleave/before.png")
    assert batch["items"][0]["edited_image_path"] == str(image_dir / "interleave/after.png")
    assert batch["items"][0]["evaluation"]["success"] is True
    assert "{original_instruction}" in INTERLEAVE_CRITIC_PROMPT


def test_interleave_thinker_critic_fake_backend_generates_rollouts():
    model = _FakeBackendCritic(load_backend=False)
    model.processor = _FakeProcessor()
    model.transformer = _FakeQwen()

    rollouts = model.generate_interleave_responses(
        {
            "items": [{
                "origin_prompt": "draw a chair",
                "previous_prompt": "a wooden chair",
                "origin_image_path": "before.png",
                "edited_image_path": "after.png",
            }]
        },
        num_generations=2,
        temperature=0.7,
        top_p=0.9,
        max_new_tokens=8,
    )

    assert len(rollouts) == 2
    assert all("previous_step_success" in rollout["response"] for rollout in rollouts)
    assert rollouts[0]["group_key"] == rollouts[1]["group_key"]
    assert rollouts[0]["sample_index"] == 0
    assert len(rollouts[0]["old_logprobs"]) == 2
    assert rollouts[0]["response_mask"] == [1.0, 1.0]
    assert model.transformer.generate_kwargs["num_return_sequences"] == 2
    assert model.transformer.generate_kwargs["max_new_tokens"] == 8


def test_interleave_thinker_critic_fake_backend_trains_response_tokens_only():
    model = _FakeBackendCritic(load_backend=False)
    assert isinstance(model.transformer, _PlaceholderActorModule)
    model.processor = _FakeProcessor()
    model.transformer = _FakeQwen()
    optimizer = torch.optim.SGD(model.transformer.parameters(), lr=0.1)
    before = float(model.transformer.weight.detach())

    loss_map, metrics = model.train_interleave_rollouts(
        rollouts=[{
            "origin_prompt": "draw a chair",
            "previous_prompt": "a wooden chair",
            "origin_image_path": "before.png",
            "edited_image_path": "after.png",
            "response": '<think>ok</think><answer>{"previous_step_success": true, "refine_prompt": "better"}</answer>',
        }],
        advantages=torch.tensor([1.0]),
        rewards={},
        optimizer=optimizer,
        gradient_accumulation_steps=1,
        max_grad_norm=0.0,
    )

    assert "total_loss" in loss_map
    assert metrics["actor/policy_loss"] == -1.0
    assert metrics["actor/clipped_fraction"] == 0.0
    assert metrics["actor/mean_ratio"] == 1.0
    assert metrics["actor/response_tokens"] == 2.0
    assert float(model.transformer.weight.detach()) > before
