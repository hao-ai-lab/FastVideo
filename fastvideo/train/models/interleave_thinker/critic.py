# SPDX-License-Identifier: Apache-2.0
"""InterleaveThinker Qwen3-VL critic actor adapter."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TYPE_CHECKING

import torch

from fastvideo.train.models.interleave_thinker.qwen_actor import (
    Qwen3VLActorBase,
    _PlaceholderActorModule as _PlaceholderActorModule,
    batch_to_items,
    first_string,
    rollout_group_key,
)

if TYPE_CHECKING:
    from fastvideo.train.utils.lora import LoraConfig
    from fastvideo.train.utils.training_config import TrainingConfig

INTERLEAVE_CRITIC_PROMPT = """
<image><image>
# Generation/Edit Evaluation and Prompt Refinement System

You are an expert image editing evaluator and prompt engineer. Your task is to:
1. Evaluate the edited image and output the result in boolean format (True/False).
2. If you think the edited image is not good enough (False), generate an optimized rewritten prompt that addresses the original shortcomings; if you think it is good enough (True), output the original rewritten prompt.

## Input Information
You have been presented with two images in sequence:
- Original Image: The input image before editing. For the initial generation step, this may be a pure white or blank canvas.
- Generated/Edited Image: The resulting image after applying the instruction or prompt.

Original User Instruction: "{original_instruction}"
Rewritten Prompt: "{rewritten_prompt}"

## Evaluation Instructions
Check both intent matching and anomaly / logic errors. The step is successful only if the generated or edited image satisfies the instruction and has no fatal artifacts, collateral damage, impossible anatomy, identity loss, or unrelated changes.

## Prompt Refinement Strategy
If the previous step is not good enough, explain what failed and produce a clearer rewritten prompt that fixes the issue while preserving what worked. If the previous step is good enough, return the original rewritten prompt.

## Output
Return exactly:
<think>
Detailed evaluation and refinement reasoning.
</think>
<answer>
{{"previous_step_success": true_or_false, "refine_prompt": "prompt text"}}
</answer>
"""


class InterleaveThinkerCriticModel(Qwen3VLActorBase):
    """Qwen3-VL actor wrapper for InterleaveThinker critic RL."""

    def __init__(
        self,
        *,
        init_from: str = "InterleaveThinker/Critic-SFT-8B",
        processor_from: str | None = None,
        training_config: TrainingConfig | None = None,
        trainable: bool = True,
        load_backend: bool = True,
        image_dir: str = "",
        torch_dtype: str = "auto",
        device_map: str | dict[str, Any] | None = None,
        attn_implementation: str | None = None,
        trust_remote_code: bool = False,
        use_cache: bool = False,
        freeze_vision_tower: bool = True,
        freeze_multi_modal_projector: bool = True,
        enable_gradient_checkpointing: bool = True,
        max_prompt_length: int = 16384,
        max_response_length: int = 4096,
        prompt_template: str = INTERLEAVE_CRITIC_PROMPT,
        lora: LoraConfig | dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self.prompt_template = prompt_template
        super().__init__(
            init_from=init_from,
            processor_from=processor_from,
            training_config=training_config,
            trainable=trainable,
            load_backend=load_backend,
            image_dir=image_dir,
            torch_dtype=torch_dtype,
            device_map=device_map,
            attn_implementation=attn_implementation,
            trust_remote_code=trust_remote_code,
            use_cache=use_cache,
            freeze_vision_tower=freeze_vision_tower,
            freeze_multi_modal_projector=freeze_multi_modal_projector,
            enable_gradient_checkpointing=enable_gradient_checkpointing,
            max_prompt_length=max_prompt_length,
            max_response_length=max_response_length,
            lora=lora,
            **kwargs,
        )

    @torch.no_grad()
    def generate_interleave_responses(
        self,
        batch: dict[str, Any],
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        num_generations = max(1, int(kwargs.get("num_generations", 1) or 1))
        temperature = float(kwargs.get("temperature", 1.0) or 1.0)
        top_p = float(kwargs.get("top_p", 1.0) or 1.0)
        max_new_tokens = int(kwargs.get("max_new_tokens") or self.max_response_length)

        rollouts: list[dict[str, Any]] = []
        for item_idx, item in enumerate(batch_to_items(batch)):
            decoded = self.generate_qwen_responses(
                self.build_messages(item),
                num_generations=num_generations,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
            )
            for generation_idx, response in enumerate(decoded):
                rollout = dict(item)
                rollout["response"] = response
                rollout.setdefault("sample_index", item_idx)
                rollout.setdefault("generation_index", generation_idx)
                rollout.setdefault("group_key", rollout_group_key(rollout, item_idx))
                rollouts.append(rollout)
        return rollouts

    def train_interleave_rollouts(
        self,
        **kwargs: Any,
    ) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
        self._require_backend()
        rollouts = kwargs["rollouts"]
        advantages = kwargs["advantages"].detach().to(self.device).float()
        optimizer = kwargs.get("optimizer")
        lr_scheduler = kwargs.get("lr_scheduler")
        max_grad_norm = float(kwargs.get("max_grad_norm", 0.0) or 0.0)
        grad_accum = max(1, int(kwargs.get("gradient_accumulation_steps", 1) or 1))

        if optimizer is None:
            raise RuntimeError("InterleaveThinkerCriticModel.train_interleave_rollouts() requires an optimizer")
        optimizer.zero_grad(set_to_none=True)
        self.transformer.train()

        weighted_losses: list[torch.Tensor] = []
        nll_values: list[torch.Tensor] = []
        token_counts: list[int] = []
        loss_scale = max(1, len(rollouts) * grad_accum)
        for idx, rollout in enumerate(rollouts):
            nll, token_count = self._response_nll(rollout)
            weighted = nll * advantages[idx]
            (weighted / loss_scale).backward()
            weighted_losses.append(weighted.detach())
            nll_values.append(nll.detach())
            token_counts.append(token_count)

        if max_grad_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.transformer.parameters() if p.requires_grad],
                max_grad_norm,
            )
        optimizer.step()
        if lr_scheduler is not None:
            lr_scheduler.step()

        total_loss = torch.stack(weighted_losses).mean() if weighted_losses else torch.zeros((), device=self.device)
        mean_nll = torch.stack(nll_values).mean() if nll_values else torch.zeros((), device=self.device)
        return (
            {
                "total_loss": total_loss.detach()
            },
            {
                "actor/policy_loss": float(total_loss.detach().cpu()),
                "actor/response_nll": float(mean_nll.detach().cpu()),
                "actor/mean_advantage": float(advantages.mean().detach().cpu()),
                "actor/response_tokens": float(sum(token_counts) / max(1, len(token_counts))),
            },
        )

    def build_messages(
        self,
        item: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        prompt = self.prompt_template.format(
            original_instruction=str(item.get("origin_prompt", item.get("prompt", "")) or ""),
            rewritten_prompt=str(item.get("previous_prompt", item.get("rewritten_prompt", "")) or ""),
        )
        return self.build_text_image_messages(prompt, _item_image_paths(item))

    def _response_nll(
        self,
        rollout: Mapping[str, Any],
    ) -> tuple[torch.Tensor, int]:
        return self.response_nll_from_messages(
            self.build_messages(rollout),
            str(rollout.get("response", "") or ""),
        )

    def backward(
        self,
        loss: torch.Tensor,
        ctx: Any,
        *,
        grad_accum_rounds: int,
    ) -> None:
        del loss, ctx, grad_accum_rounds
        raise NotImplementedError("InterleaveThinkerCriticModel uses train_interleave_rollouts().")


def _item_image_paths(item: Mapping[str, Any], ) -> list[str]:
    before = first_string(item, "previous_image_path", "origin_image_path", "input_image_path")
    after = first_string(item, "edited_image_path", "generated_image_path", "output_image_path")
    image_paths = [value for value in (before, after) if value]
    if len(image_paths) == 1:
        image_paths.append(image_paths[0])
    return image_paths[:2]


__all__ = ["INTERLEAVE_CRITIC_PROMPT", "InterleaveThinkerCriticModel", "_PlaceholderActorModule"]
