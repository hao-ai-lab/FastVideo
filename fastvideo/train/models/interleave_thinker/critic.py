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
from fastvideo.train.models.interleave_thinker.data import InterleaveDatasetKind

if TYPE_CHECKING:
    from fastvideo.train.utils.lora import LoraConfig
    from fastvideo.train.utils.training_config import TrainingConfig

INTERLEAVE_CRITIC_PROMPT = """<image><image>

# Generation/Edit Evaluation and Prompt Refinement System

You are an expert image editing evaluator and prompt engineer. Your task is to:
1. Evaluate the edited image and output the result in boolean format (True/False).
2. If you think the edited image is not good enough (False), generate an optimized rewritten prompt that addresses the original shortcomings; if you think it is good enough (True), output the [Original Rewritten Prompt].

## Input Information
You have been presented with two images in sequence:
- Original Image: The input image before editing. (NOTE: For the initial generation step, this will be a pure white/blank canvas).
- Generated/Edited Image: The resulting image after applying the instruction/prompt.

Now, here are the instructions that were involved in this process:
Original User Instruction (user's initial request): "{original_instruction}"
Rewritten Prompt (last refined instruction that was used. **NOTE: If this is empty, you must base your evaluation and refinement entirely on the Original User Instruction**): "{rewritten_prompt}"

## Evaluation Instructions
**Evaluate Previous Step (Strict 2-Part Check)**: Carefully compare the **Before Image** and the **After Image**. You must evaluate based on two strict criteria. If the image fails *either* criteria, the step is a FAILURE.
1. **Criterion A (Intent Matching)**: If the Before Image is pure white, evaluate if the After Image successfully generated the Previous Step from scratch. Otherwise, observe the delta (differences). Did the changes match the key meaning and necessary details of the Previous Step?
2. **Criterion B (Anomaly & Logic Detection - CRITICAL)**: You must actively play the role of a "Fault Finder". Do NOT just check if the requested object exists; you MUST check HOW it exists. Scan the After Image for any of the following fatal errors:
   - **Anatomical/Biological Errors**: Extra/missing limbs or fingers, body parts emerging from impossible or anatomically incorrect places (e.g., a hand growing out of a chest, stomach, or a wall), distorted faces.
   - **Collateral Damage**: Unintended alterations to unrelated areas, background bleeding, or the original subject losing its identity.

## Prompt Refinement Strategy (if NOT GOOD ENOUGH, False)

When generating a new rewritten prompt, analyze:

1. **What went wrong?**
   - Compare original instruction → rewritten prompt → generated/edited result. *(If Rewritten Prompt is empty, directly compare Original Instruction → Result).*
   - Identify gaps between intent and execution
   - Determine if the issue is clarity, specificity, or contradiction

2. **Refinement Approaches:**

   **If this is an Initial Generation task (Before image was blank):**
   - **Establish Foundation:** Translate the raw user instruction into a comprehensive Text-to-Image prompt.
   - **Enrich Details:** Clearly define the main subject, background/environment, lighting, camera angle, composition, and art style.
   - **Prevent Ambiguity:** Fill in missing visual details that the user might have implied but didn't explicitly state to prevent the model from hallucinating incorrectly.
   - **Remove Redundent:** Remove the description which is not contained in raw user instruction but appeared in image, especially the text.

   **If the rewritten prompt was too vague:**
   - Add more specific descriptors (exact colors, positions, sizes)
   - Include spatial relationships and context
   - Specify interaction with existing elements

   **If the rewritten prompt was contradictory:**
   - Resolve conflicts between requirements
   - Prioritize core intent over secondary details
   - Simplify complex multi-part instructions

   **If important details were lost:**
   - Explicitly state preservation requirements
   - Add "maintain [aspect]" or "preserve [feature]" clauses
   - Reference specific elements from the original image

   **If positioning/scale was wrong:**
   - Use more precise spatial descriptors
   - Add relative size/scale indicators
   - Specify foreground/midground/background placement

   **If style/appearance was incorrect:**
   - Use more specific visual vocabulary
   - Add reference to original image's style elements
   - Include material/texture/lighting specifications

   **If the edit was over/under-processed:**
   - Add modifiers like "subtle", "gentle", "dramatic", "significant"
   - Specify degree of change more clearly
   - Balance enhancement with naturalness

3. **Leverage All Information:**
   - Reference what's visible in the original image
   - Learn from what the previous rewritten prompt missed
   - Use the edited image as feedback on what went wrong
   - Maintain what worked, fix what didn't

## Output
The output consists of three parts:
1. A Statement - Analysis process and reasoning;
2. A Boolean - Judge whether the edited images is good enough;
3. A prompt — either the optimized rewritten prompt or the original rewritten prompt.

Here is a output example:

<think>
Detailed explanation of evaluation and new rewritten prompt. If edited image is good enough, explain why it meets requirements. If not good enough, explain specific shortcomings.
</think>

<answer>
{{
   'previous_step_success': 'boolean (True ONLY IF the Intent Check is successful AND the Anomaly Check finds ZERO errors. If ANY anomaly is detected, this MUST be False.)',
   'refine_prompt': '[Improved rewritten prompt that addresses identified issues and enhances clarity, specificity, and preservation requirements] if NOT GOOD ENOUGH (False), [original rewritten prompt] if GOOD ENOUGH (True)'
}}
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
        temperature_value = kwargs.get("temperature", 1.0)
        top_p_value = kwargs.get("top_p", 1.0)
        temperature = 1.0 if temperature_value is None else float(temperature_value)
        top_p = 1.0 if top_p_value is None else float(top_p_value)
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

    def build_messages(
        self,
        item: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        prompt = self.prompt_template.format(
            original_instruction=str(item.get("origin_prompt", item.get("prompt", "")) or ""),
            rewritten_prompt=str(item.get("previous_prompt", item.get("rewritten_prompt", "")) or ""),
        )
        return self.build_text_image_messages(prompt, _item_image_paths(item))


def _item_image_paths(item: Mapping[str, Any], ) -> list[str]:
    before = first_string(item, "previous_image_path", "origin_image_path", "input_image_path")
    after = first_string(item, "edited_image_path", "generated_image_path", "output_image_path")
    image_paths = [value for value in (before, after) if value]
    if len(image_paths) == 1:
        image_paths.append(image_paths[0])
    return image_paths[:2]


__all__ = ["INTERLEAVE_CRITIC_PROMPT", "InterleaveThinkerCriticModel", "_PlaceholderActorModule"]
