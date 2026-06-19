# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import torch

from fastvideo.train.methods.rl.common.grpo import compute_grpo_loss
from fastvideo.train.models.interleave_thinker import InterleaveThinkerCriticModel


class _FakeBackendCritic(InterleaveThinkerCriticModel):

    @property
    def device(self):
        return torch.device("cpu")


class _FakeProcessor:

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
        input_ids = [3, 4, 1, 2] if has_assistant else [3, 4]
        return {
            "input_ids": torch.tensor([input_ids], dtype=torch.long),
            "attention_mask": torch.ones(1, len(input_ids), dtype=torch.long),
        }


class _FakeQwenPolicy(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, **kwargs):
        input_ids = kwargs["input_ids"]
        batch, seq_len = input_ids.shape
        logits = torch.zeros(batch, seq_len, 8, dtype=self.weight.dtype, device=input_ids.device)
        for idx in range(seq_len - 1):
            next_token = input_ids[:, idx + 1]
            logits[:, idx].scatter_(1, next_token[:, None], self.weight.expand(batch, 1))
        return SimpleNamespace(logits=logits)


def test_compute_grpo_loss_clips_ratios_and_masks_tokens():
    result = compute_grpo_loss(
        current_logprobs=torch.log(torch.tensor([[1.5, 1.0], [0.5, 2.0]])),
        old_logprobs=torch.zeros(2, 2),
        advantages=torch.tensor([1.0, -1.0]),
        response_mask=torch.tensor([[1.0, 1.0], [1.0, 0.0]]),
        clip_range=0.2,
    )

    assert torch.isclose(result.policy_loss, torch.tensor(-1.4 / 3.0), atol=1.0e-6)
    assert torch.isclose(result.clipped_fraction, torch.tensor(2.0 / 3.0), atol=1.0e-6)
    assert torch.isclose(result.mean_ratio, torch.tensor(1.0), atol=1.0e-6)
    assert result.token_count.item() == 3.0


def test_compute_grpo_loss_adds_optional_reference_kl():
    reference_logprobs = torch.log(torch.tensor([[0.5, 2.0]]))
    result = compute_grpo_loss(
        current_logprobs=torch.zeros(1, 2),
        old_logprobs=torch.zeros(1, 2),
        advantages=torch.zeros(1),
        response_mask=torch.ones(1, 2),
        reference_logprobs=reference_logprobs,
        kl_coef=0.25,
    )
    expected_kl = ((0.5 - torch.log(torch.tensor(0.5)) - 1.0) +
                   (2.0 - torch.log(torch.tensor(2.0)) - 1.0)) / 2.0

    assert torch.isclose(result.kl_loss, expected_kl, atol=1.0e-6)
    assert torch.isclose(result.total_loss, 0.25 * expected_kl, atol=1.0e-6)


def test_critic_grpo_update_uses_response_logprobs_for_positive_advantage():
    model = _FakeBackendCritic(load_backend=False, trainable=True)
    model.processor = _FakeProcessor()
    model.transformer = _FakeQwenPolicy()
    optimizer = torch.optim.SGD(model.transformer.parameters(), lr=0.5)
    before = float(model.transformer.weight.detach())

    loss_map, metrics = model.train_interleave_rollouts(
        rollouts=[{
            "origin_prompt": "draw a chair",
            "previous_prompt": "a wooden chair",
            "response": '<answer>{"previous_step_success": true, "refine_prompt": "ok"}</answer>',
        }],
        advantages=torch.tensor([1.0]),
        optimizer=optimizer,
        lr_scheduler=None,
        gradient_accumulation_steps=1,
        clip_range=0.2,
        kl_coef=0.0,
        update_micro_batch_size=1,
    )

    assert torch.isclose(loss_map["total_loss"], torch.tensor(-1.0), atol=1.0e-6)
    assert metrics["actor/policy_loss"] == -1.0
    assert metrics["actor/clipped_fraction"] == 0.0
    assert metrics["actor/mean_ratio"] == 1.0
    assert metrics["actor/response_tokens"] == 2.0
    assert float(model.transformer.weight.detach()) > before
