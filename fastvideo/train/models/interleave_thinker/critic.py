# SPDX-License-Identifier: Apache-2.0
"""InterleaveThinker Qwen3-VL critic actor adapter."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, TYPE_CHECKING

import torch

from fastvideo.train.models.interleave_thinker.qwen_actor import (
    Qwen3VLActorBase,
    _PlaceholderActorModule as _PlaceholderActorModule,
    batch_to_items,
    coerce_logprob_vector,
    coerce_response_mask,
    first_string,
    pad_1d_tensors,
    rollout_group_key,
)
from fastvideo.train.models.interleave_thinker.data import InterleaveDatasetKind
from fastvideo.train.methods.rl.common.grpo import compute_grpo_loss

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
        dataset_kind: InterleaveDatasetKind | None = None,
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
            dataset_kind=dataset_kind,
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
                old_logprobs, response_mask = self.response_logprobs_from_messages(
                    self.build_messages(item),
                    response,
                )
                rollout["old_logprobs"] = old_logprobs.detach().cpu().tolist()
                rollout["response_mask"] = response_mask.detach().cpu().tolist()
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
        clip_range = float(kwargs.get("clip_range", 0.2) or 0.0)
        kl_coef = float(kwargs.get("kl_coef", 0.0) or 0.0)
        update_micro_batch_size = kwargs.get("update_micro_batch_size")
        if update_micro_batch_size is None:
            update_micro_batch_size = len(rollouts)
        update_micro_batch_size = max(1, int(update_micro_batch_size or 1))
        max_grad_norm = float(kwargs.get("max_grad_norm", 0.0) or 0.0)
        grad_accum = max(1, int(kwargs.get("gradient_accumulation_steps", 1) or 1))

        if optimizer is None:
            raise RuntimeError("InterleaveThinkerCriticModel.train_interleave_rollouts() requires an optimizer")
        optimizer.zero_grad(set_to_none=True)
        self.transformer.train()

        if int(advantages.shape[0]) != len(rollouts):
            raise ValueError("advantages count must match rollout count")

        loss_results = []
        token_counts: list[float] = []
        loss_scale = max(1, grad_accum)
        for start in range(0, len(rollouts), update_micro_batch_size):
            end = min(start + update_micro_batch_size, len(rollouts))
            current_logprobs, old_logprobs, response_masks, reference_logprobs = self._grpo_logprob_batch(
                rollouts[start:end])
            result = compute_grpo_loss(
                current_logprobs=current_logprobs,
                old_logprobs=old_logprobs,
                advantages=advantages[start:end],
                response_mask=response_masks,
                clip_range=clip_range,
                reference_logprobs=reference_logprobs,
                kl_coef=kl_coef,
            )
            (result.total_loss / loss_scale).backward()
            loss_results.append(result)
            token_counts.append(float(result.token_count.detach().cpu()))

        if max_grad_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.transformer.parameters() if p.requires_grad],
                max_grad_norm,
            )
        optimizer.step()
        if lr_scheduler is not None:
            lr_scheduler.step()

        total_tokens = max(1.0, sum(token_counts))
        total_loss = _weighted_result_mean(loss_results, "total_loss", token_counts, self.device, total_tokens)
        policy_loss = _weighted_result_mean(loss_results, "policy_loss", token_counts, self.device, total_tokens)
        kl_loss = _weighted_result_mean(loss_results, "kl_loss", token_counts, self.device, total_tokens)
        approx_kl = _weighted_result_mean(loss_results, "approx_kl", token_counts, self.device, total_tokens)
        clipped_fraction = _weighted_result_mean(
            loss_results,
            "clipped_fraction",
            token_counts,
            self.device,
            total_tokens,
        )
        mean_ratio = _weighted_result_mean(loss_results, "mean_ratio", token_counts, self.device, total_tokens)
        return (
            {
                "total_loss": total_loss.detach()
            },
            {
                "actor/policy_loss": float(policy_loss.detach().cpu()),
                "actor/kl_loss": float(kl_loss.detach().cpu()),
                "actor/approx_kl": float(approx_kl.detach().cpu()),
                "actor/clipped_fraction": float(clipped_fraction.detach().cpu()),
                "actor/mean_ratio": float(mean_ratio.detach().cpu()),
                "actor/mean_advantage": float(advantages.mean().detach().cpu()),
                "actor/response_tokens": float(total_tokens / max(1, len(rollouts))),
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

    def _grpo_logprob_batch(
        self,
        rollouts: Sequence[Mapping[str, Any]],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        current_logprob_rows: list[torch.Tensor] = []
        old_logprob_rows: list[torch.Tensor] = []
        mask_rows: list[torch.Tensor] = []
        reference_rows: list[torch.Tensor] = []
        has_reference = False
        for rollout in rollouts:
            response = str(rollout.get("response", "") or "")
            current_logprobs, default_mask = self.response_logprobs_from_messages(
                self.build_messages(rollout),
                response,
            )
            expected_len = int(current_logprobs.numel())
            old_logprobs = coerce_logprob_vector(
                rollout.get("old_logprobs", rollout.get("old_logprob")),
                expected_len=expected_len,
                device=self.device,
                name="old_logprobs",
            )
            if old_logprobs is None:
                old_logprobs = current_logprobs.detach()
            response_mask = coerce_response_mask(
                rollout.get("response_mask"),
                expected_len=expected_len,
                device=self.device,
            )
            if response_mask is None:
                response_mask = default_mask
            reference_logprobs = coerce_logprob_vector(
                rollout.get("reference_logprobs", rollout.get("ref_logprobs")),
                expected_len=expected_len,
                device=self.device,
                name="reference_logprobs",
            )
            current_logprob_rows.append(current_logprobs)
            old_logprob_rows.append(old_logprobs)
            mask_rows.append(response_mask)
            if reference_logprobs is None:
                reference_rows.append(current_logprobs.detach())
            else:
                has_reference = True
                reference_rows.append(reference_logprobs)
        return (
            pad_1d_tensors(current_logprob_rows),
            pad_1d_tensors(old_logprob_rows),
            pad_1d_tensors(mask_rows),
            pad_1d_tensors(reference_rows) if has_reference else None,
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


def _weighted_result_mean(
    results: Sequence[Any],
    field: str,
    weights: Sequence[float],
    device: torch.device,
    total_weight: float,
) -> torch.Tensor:
    if not results:
        return torch.zeros((), device=device)
    values = []
    for result, weight in zip(results, weights, strict=True):
        values.append(getattr(result, field).detach() * float(weight))
    return torch.stack(values).sum() / float(total_weight)


__all__ = ["INTERLEAVE_CRITIC_PROMPT", "InterleaveThinkerCriticModel", "_PlaceholderActorModule"]
