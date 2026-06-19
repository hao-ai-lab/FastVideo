# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from types import SimpleNamespace

import torch

from fastvideo.train.methods.fine_tuning import InterleaveThinkerSFTMethod
from fastvideo.train.models.interleave_thinker import (
    InterleaveThinkerCriticModel,
    InterleaveThinkerPlannerModel,
)
from fastvideo.train.utils.config import load_run_config
from fastvideo.train.utils.training_config import (
    DataConfig,
    OptimizerConfig,
    TrainingConfig,
    TrainingLoopConfig,
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


class _FakeQwen(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.last_labels = None

    def forward(self, **kwargs):
        labels = kwargs["labels"]
        self.last_labels = labels.detach().clone()
        trainable_fraction = (labels != -100).float().mean()
        return SimpleNamespace(loss=self.weight.pow(2).sum() * trainable_fraction)


def test_interleave_thinker_sft_method_trains_response_tokens_only():
    model = _FakeBackendCritic(load_backend=False, trainable=True)
    model.processor = _FakeProcessor()
    model.transformer = _FakeQwen()
    method = InterleaveThinkerSFTMethod(
        cfg=SimpleNamespace(
            method={},
            validation={},
            training=TrainingConfig(
                data=DataConfig(data_path="", train_batch_size=1),
                optimizer=OptimizerConfig(learning_rate=0.1),
                loop=TrainingLoopConfig(max_train_steps=1),
            ),
        ),
        role_models={"student": model},
    )
    before = float(model.transformer.weight.detach())

    loss_map, outputs, metrics = method.single_train_step(
        {
            "items": [{
                "origin_prompt": "draw a chair",
                "previous_prompt": "a wooden chair",
                "origin_image_path": "before.png",
                "edited_image_path": "after.png",
                "response": '<answer>{"previous_step_success": true, "refine_prompt": "ok"}</answer>',
            }]
        },
        iteration=0,
    )
    method.backward(loss_map, outputs, grad_accum_rounds=1)
    method.optimizers_schedulers_step(0)

    assert set(loss_map) == {"total_loss", "sft_loss"}
    assert metrics["sft/response_tokens"] == 2.0
    assert metrics["sft/num_items"] == 1.0
    assert model.transformer.last_labels.tolist()[0][:3] == [-100, -100, -100]
    assert all(label != -100 for label in model.transformer.last_labels.tolist()[0][3:])
    assert float(model.transformer.weight.detach()) < before


def test_qwen_actor_dataset_kind_uses_planner_sft_normalizer(tmp_path):
    image_dir = tmp_path / "images"
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "planner_sft.json").write_text(
        json.dumps([{
            "messages": [
                {
                    "role": "user",
                    "content": "draw a cat"
                },
                {
                    "role": "assistant",
                    "content": "<answer>{\"execution_plan\": []}</answer>"
                },
            ],
            "images": ["planner/cat.png"],
        }]),
        encoding="utf-8",
    )
    model = InterleaveThinkerPlannerModel(
        load_backend=False,
        dataset_kind="planner_sft",
        image_dir=str(image_dir),
    )

    model.init_preprocessors(TrainingConfig(data=DataConfig(data_path=str(data_dir), train_batch_size=1)))
    batch = next(iter(model.dataloader))

    assert batch["items"][0]["instruction"] == "draw a cat"
    assert batch["items"][0]["response"] == '<answer>{"execution_plan": []}</answer>'
    assert batch["items"][0]["images"] == [str(image_dir / "planner/cat.png")]


def test_interleave_thinker_sft_configs_parse_public_yaml():
    planner_cfg = load_run_config("examples/train/configs/interleave_thinker/planner_sft_lora.yaml")
    critic_cfg = load_run_config("examples/train/configs/interleave_thinker/critic_sft_lora.yaml")

    assert planner_cfg.models["student"]["dataset_kind"] == "planner_sft"
    assert planner_cfg.method["_target_"] == "fastvideo.train.methods.fine_tuning.InterleaveThinkerSFTMethod"
    assert planner_cfg.models["student"]["lora"]["enable"] is True
    assert critic_cfg.models["student"]["dataset_kind"] == "critic_sft"
    assert critic_cfg.method["_target_"] == "fastvideo.train.methods.fine_tuning.InterleaveThinkerSFTMethod"
