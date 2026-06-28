# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch

from fastvideo.train.models.interleave_thinker import qwen_actor as qwen_actor_module
from fastvideo.train.models.interleave_thinker.qwen_actor import (
    Qwen3VLActorBase,
    _distributed_token_gradient_scale,
    _qwen_sharding_root,
    _qwen_transformer_block_condition,
    _rank_independent_rng,
    _resolve_hsdp_dimensions,
)
from fastvideo.train.utils.config import load_run_config
from fastvideo.train.utils.training_config import (
    DataConfig,
    DistributedConfig,
    TrainingConfig,
)


class _FakeActor(Qwen3VLActorBase):

    @property
    def device(self):
        return torch.device("cpu")

    def build_messages(self, item):
        return [{
            "role": "user",
            "content": [{
                "type": "text",
                "text": str(item.get("prompt", "prompt")),
            }],
        }]


class _FakeProcessor:

    def __init__(self, *, prompt_length=2, empty_full_length=3, full_length=4):
        self.prompt_length = prompt_length
        self.empty_full_length = empty_full_length
        self.full_length = full_length
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
        assistant_messages = [message for message in messages if message["role"] == "assistant"]
        if not assistant_messages:
            length = self.prompt_length
        else:
            assistant_text = "".join(
                str(part.get("text", "")) for part in assistant_messages[-1].get("content", [])
                if isinstance(part, dict))
            length = self.empty_full_length if not assistant_text else self.full_length
        return {
            "input_ids": torch.arange(1, length + 1).unsqueeze(0),
            "attention_mask": torch.ones(1, length, dtype=torch.long),
        }

    def batch_decode(self, sequences, **kwargs):
        del kwargs
        return ["response" for _ in sequences]


class _FakeQwenPolicy(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.0))
        self.generate_kwargs = None

    def forward(self, **kwargs):
        input_ids = kwargs["input_ids"]
        batch, seq_len = input_ids.shape
        logits = torch.zeros(batch, seq_len, 16, dtype=self.weight.dtype)
        for idx in range(seq_len - 1):
            next_token = input_ids[:, idx + 1]
            logits[:, idx].scatter_(1, next_token[:, None], self.weight.expand(batch, 1))
        return SimpleNamespace(logits=logits)

    def generate(self, **kwargs):
        self.generate_kwargs = kwargs
        input_ids = kwargs["input_ids"]
        suffix = torch.tensor([[9]], dtype=input_ids.dtype)
        return torch.cat([input_ids, suffix], dim=1)


def _make_fake_actor(*, max_prompt_length=8, max_response_length=4):
    actor = _FakeActor(
        init_from="unused",
        load_backend=False,
        max_prompt_length=max_prompt_length,
        max_response_length=max_response_length,
    )
    actor.processor = _FakeProcessor()
    actor.transformer = _FakeQwenPolicy()
    return actor


def test_grpo_update_is_invariant_to_micro_batch_partitioning():
    rollouts = [{
        "prompt": f"prompt-{index}",
        "response": "response",
    } for index in range(5)]
    advantages = torch.ones(5)

    weights = []
    for micro_batch_size in (2, 5):
        actor = _make_fake_actor()
        optimizer = torch.optim.SGD(actor.transformer.parameters(), lr=0.5)
        actor.train_interleave_rollouts(
            rollouts=rollouts,
            advantages=advantages,
            optimizer=optimizer,
            update_micro_batch_size=micro_batch_size,
            gradient_accumulation_steps=1,
        )
        weights.append(actor.transformer.weight.detach().clone())

    assert torch.allclose(weights[0], weights[1], atol=1.0e-6)


def test_actor_rejects_tokenized_prompts_over_configured_limit():
    actor = _make_fake_actor(max_prompt_length=1)

    with pytest.raises(ValueError, match="max_prompt_length=1"):
        actor.generate_qwen_responses(actor.build_messages({"prompt": "too long"}))


def test_actor_caps_generation_at_configured_response_limit():
    actor = _make_fake_actor(max_response_length=3)

    actor.generate_qwen_responses(
        actor.build_messages({"prompt": "prompt"}),
        max_new_tokens=99,
        temperature=0.0,
    )

    assert actor.transformer.generate_kwargs["max_new_tokens"] == 3


def test_actor_rejects_tokenized_responses_over_configured_limit():
    actor = _make_fake_actor(max_prompt_length=4, max_response_length=1)
    actor.processor = _FakeProcessor(full_length=5)

    with pytest.raises(ValueError, match="max_response_length=1"):
        actor.response_logprobs_from_messages(actor.build_messages({"prompt": "prompt"}), "response")


def test_actor_accepts_response_content_at_exact_configured_limit():
    actor = _make_fake_actor(max_prompt_length=4, max_response_length=1)

    logprobs, mask = actor.response_logprobs_from_messages(
        actor.build_messages({"prompt": "prompt"}),
        "response",
    )

    assert logprobs.numel() == 2
    assert mask.tolist() == [1.0, 1.0]


def test_actor_enables_synced_generation_for_distributed_fsdp(monkeypatch):
    actor = _make_fake_actor()
    monkeypatch.setattr(qwen_actor_module, "_distributed_actor_world_size", lambda: 2)

    actor.generate_qwen_responses(actor.build_messages({"prompt": "prompt"}), temperature=0.0)

    assert actor.transformer.generate_kwargs["synced_gpus"] is True


def test_interleave_dataloader_restores_shuffle_position(tmp_path):
    data_path = tmp_path / "records.jsonl"
    data_path.write_text("".join(json.dumps({"prompt": f"prompt-{index}"}) + "\n" for index in range(8)))
    training_config = TrainingConfig(
        data=DataConfig(
            data_path=str(data_path),
            train_batch_size=2,
            dataloader_num_workers=0,
            seed=17,
        ))

    actor = _FakeActor(init_from="unused", load_backend=False)
    actor.init_preprocessors(training_config)
    iterator = iter(actor.dataloader)
    next(iterator)
    state = actor.dataloader.state_dict()
    expected = next(iterator)

    resumed_actor = _FakeActor(init_from="unused", load_backend=False)
    resumed_actor.init_preprocessors(training_config)
    resumed_actor.dataloader.load_state_dict(state)

    assert next(iter(resumed_actor.dataloader)) == expected


def test_interleave_dataloader_shards_records_across_distributed_ranks(monkeypatch, tmp_path):
    data_path = tmp_path / "records.jsonl"
    data_path.write_text("".join(json.dumps({"prompt": f"prompt-{index}"}) + "\n" for index in range(8)))
    training_config = TrainingConfig(
        data=DataConfig(
            data_path=str(data_path),
            train_batch_size=2,
            dataloader_num_workers=0,
            seed=23,
        ))
    monkeypatch.setattr(qwen_actor_module.dist, "is_available", lambda: True)
    monkeypatch.setattr(qwen_actor_module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(qwen_actor_module.dist, "get_world_size", lambda: 2)

    rank_records = []
    epoch_orders = []
    for rank in (0, 1):
        monkeypatch.setattr(qwen_actor_module.dist, "get_rank", lambda rank=rank: rank)
        actor = _FakeActor(init_from="unused", load_backend=False)
        actor.init_preprocessors(training_config)
        epoch_orders.append((list(actor.dataloader.sampler), list(actor.dataloader.sampler)))
        rank_records.append({
            item["prompt"]
            for batch in actor.dataloader
            for item in batch["items"]
        })

    assert rank_records[0].isdisjoint(rank_records[1])
    assert rank_records[0] | rank_records[1] == {f"prompt-{index}" for index in range(8)}
    assert all(first_epoch != second_epoch for first_epoch, second_epoch in epoch_orders)


def test_distributed_sampler_restores_rank_local_epoch_position(monkeypatch, tmp_path):
    data_path = tmp_path / "records.jsonl"
    data_path.write_text("".join(json.dumps({"prompt": f"prompt-{index}"}) + "\n" for index in range(8)))
    training_config = TrainingConfig(data=DataConfig(data_path=str(data_path), train_batch_size=2, seed=31))
    monkeypatch.setattr(qwen_actor_module.dist, "is_available", lambda: True)
    monkeypatch.setattr(qwen_actor_module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(qwen_actor_module.dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(qwen_actor_module.dist, "get_rank", lambda: 1)

    actor = _FakeActor(init_from="unused", load_backend=False)
    actor.init_preprocessors(training_config)
    sampler_iterator = iter(actor.dataloader.sampler)
    next(sampler_iterator)
    state = actor.dataloader.sampler.state_dict()
    expected = next(sampler_iterator)

    resumed_actor = _FakeActor(init_from="unused", load_backend=False)
    resumed_actor.init_preprocessors(training_config)
    resumed_actor.dataloader.sampler.load_state_dict(state)

    assert next(iter(resumed_actor.dataloader.sampler)) == expected


def test_distributed_actor_rejects_unsupported_tensor_parallelism():
    actor = _FakeActor(
        init_from="unused",
        load_backend=False,
        training_config=TrainingConfig(
            distributed=DistributedConfig(
                num_gpus=8,
                tp_size=4,
                sp_size=1,
                hsdp_replicate_dim=1,
                hsdp_shard_dim=8,
            )),
    )

    with pytest.raises(ValueError, match="do not implement tensor parallelism"):
        actor._validate_distributed_config(device_map=None)


def test_hsdp_auto_shard_dimension_uses_configured_gpu_count():
    distributed = DistributedConfig(
        num_gpus=8,
        tp_size=1,
        sp_size=1,
        hsdp_replicate_dim=1,
        hsdp_shard_dim=-1,
    )

    assert _resolve_hsdp_dimensions(distributed, num_gpus=8) == (1, 8)


def test_distributed_token_gradient_scale_uses_global_denominator(monkeypatch):
    monkeypatch.setattr(qwen_actor_module.dist, "is_available", lambda: True)
    monkeypatch.setattr(qwen_actor_module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(qwen_actor_module.dist, "get_world_size", lambda: 2)

    def fake_all_reduce(value, op):
        del op
        value.fill_(12.0)

    monkeypatch.setattr(qwen_actor_module.dist, "all_reduce", fake_all_reduce)

    rank_zero_scale = _distributed_token_gradient_scale(2.0, device=torch.device("cpu"))
    rank_one_scale = _distributed_token_gradient_scale(10.0, device=torch.device("cpu"))

    assert rank_zero_scale == rank_one_scale == 1.0 / 6.0


def test_rank_independent_rng_repeats_adapter_initialization_and_restores_state():
    torch.manual_seed(101)
    before = torch.rand(1)
    with _rank_independent_rng(7, device=torch.device("cpu")):
        first_adapter = torch.rand(4)
    after = torch.rand(1)

    torch.manual_seed(999)
    with _rank_independent_rng(7, device=torch.device("cpu")):
        second_adapter = torch.rand(4)

    torch.manual_seed(101)
    assert torch.equal(before, torch.rand(1))
    assert torch.equal(after, torch.rand(1))
    assert torch.equal(first_adapter, second_adapter)


@pytest.mark.parametrize(
    "config_path",
    [
        "examples/train/configs/interleave_thinker/planner_sft_lora.yaml",
        "examples/train/configs/interleave_thinker/critic_sft_lora.yaml",
        "examples/train/configs/rl/interleave_thinker/planner_grpo.yaml",
        "examples/train/configs/rl/interleave_thinker/critic_grpo.yaml",
    ],
)
def test_public_distributed_actor_configs_use_supported_hsdp(config_path):
    distributed = load_run_config(config_path).training.distributed

    assert distributed.num_gpus == 8
    assert distributed.tp_size == 1
    assert distributed.sp_size == 1
    assert distributed.hsdp_replicate_dim == 1
    assert distributed.hsdp_shard_dim == 8


def test_qwen_shard_condition_matches_module_list_blocks():
    class VisionBlock(torch.nn.Linear):
        pass

    model = torch.nn.ModuleDict({
        "language": torch.nn.ModuleList([torch.nn.Linear(2, 2)]),
        "visual": torch.nn.ModuleList([VisionBlock(2, 2)]),
    })
    condition = _qwen_transformer_block_condition(model)

    assert condition("language.0", model["language"][0]) is True
    assert condition("language", model["language"]) is False
    assert condition("visual.0", model["visual"][0]) is False


def test_qwen_sharding_root_uses_peft_forward_and_generate_base():
    base_model = torch.nn.Sequential(torch.nn.ModuleList([torch.nn.Linear(2, 2)]))

    class FakePeftModel(torch.nn.Module):

        def __init__(self):
            super().__init__()
            self.base = base_model

        def get_base_model(self):
            return self.base

    wrapper = FakePeftModel()

    assert _qwen_sharding_root(wrapper) is base_model
    assert _qwen_sharding_root(base_model) is base_model
