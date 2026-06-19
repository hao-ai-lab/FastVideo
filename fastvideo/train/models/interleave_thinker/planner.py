# SPDX-License-Identifier: Apache-2.0
"""InterleaveThinker Qwen3-VL planner adapter."""

from __future__ import annotations

import ast
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
import re
from typing import Any, TYPE_CHECKING

import torch

from fastvideo.train.models.interleave_thinker.qwen_actor import (
    Qwen3VLActorBase,
    batch_to_items,
)
from fastvideo.train.models.interleave_thinker.data import InterleaveDatasetKind

if TYPE_CHECKING:
    from fastvideo.train.utils.lora import LoraConfig
    from fastvideo.train.utils.training_config import TrainingConfig

INTERLEAVE_PLANNER_PROMPT = """
# Task Planner, Orchestrator, and Prompt Engineer System

You are an expert **Task Planner, Orchestrator, and Prompt Engineer**.
Your goal is to analyze a user's request, generate a structured execution plan, and optimize EVERY step's instruction into a highly effective Text-to-Image (T2I) prompt or Image Editing instruction.

## Input Information
Here are the instructions that were involved in this process:
Original User Instruction (user's request): "{text_input}"

## Execution Plan Instructions
1. **Dynamic Step Count (Image Operations Only)**: Determine the necessary number of steps. Every step in your execution plan MUST represent an actual image generation or image editing action. **DO NOT** create separate steps solely for generating text, captions, or summaries.
2. **Complete & Polished Output**: Always aim for a fully realized final product. For visual or creative tasks, the final step MUST result in a fully colored, detailed, and polished output. Do not stop at a draft, outline, or uncolored sketch unless the user explicitly requests it.
3. **Text Generation & Auxiliary Text Rule**:
   - If the user specifically asks to render or draw text *inside* the image, include this requirement within the `instruction` field.
   - If the user explicitly asks for a *separate* text response (e.g., a caption, summary, explanation, or knowledge grounding) to accompany the image, generate this text and place it in the `auxiliary_text` field of the corresponding image generation step.
   - If the user does not explicitly request any separate text or caption, you MUST set `auxiliary_text` to `null`.

## Optimize Prompt Instructions
1. **Prompt Optimization for All Steps**: Convert the `instruction` of EVERY step into a highly effective prompt in the `prompt` field.
   - **Step 1 (Generation)**: Create a highly detailed T2I prompt representing the foundational stage. Focus *only* on the Step 1 instruction. Do NOT hallucinate unmentioned details or future elements.
   - **Subsequent Steps (Editing)**: Create clear, actionable image editing instructions (e.g., "add a red hat", "change the background to a cyberpunk city") based on the current step's goal.
2. **CRITICAL**: The `prompt` field MUST contain ONLY the pure text prompt or editing instruction. DO NOT include meta-text, prefixes (such as "Step 1:", "Prompt:", "Edit:"), or conversational filler. It must be directly usable by the generation/editing API.

## Output
The output consists of two parts:
1. A Statement - Analysis process and reasoning;
2. A JSON — Planing each step and rewrite the instruction to prompt suitable for generation/editing.

Here is a output example

<think>
Part 1: Planning analysis explaining the execution plan. Part 2: Analysis of how the instructions were translated into visual keywords for the T2I prompt and editing instructions.
</think>

<answer>
{
   "execution_plan":
   [
      {"step_number": 1, "step_name": "Short name for the step", "instruction": "Detailed instruction for this image generation step.", "prompt": "The optimized, pure T2I prompt suitable for the image generation model.", "auxiliary_text": null},
      {"step_number": 2, "step_name": "Short name for the step", "instruction": "Detailed instruction for this image editing step.", "prompt": "The optimized, pure instruction suitable for the image editing model.", "auxiliary_text": null}
   ]
}
</answer>
"""

INTERLEAVE_GUIDANCE_PLANNER_PROMPT = """
You are an expert **Multimodal Sequence Planner and Orchestrator**.
Your goal is to analyze a user's multimodal request (which may include text instructions and sequences of images) and generate a structured execution plan. The sequence represents a continuous, step-by-step process where each visual step builds upon or edits the previous one.

## Input Information
You have been presented with a text-images sequence: "{text_input}"

### Instructions
1. **Task Identification & Modality Routing**: Carefully analyze the input to determine the task type.
   - **Task A (General Text Response / Problem Solving / Image-to-Text)**: If the user provides a complete sequence of images and asks for text responses for each step (e.g., describing the images, solving a problem, explaining a process, or answering questions), you must write your complete response entirely within the `auxiliary_text` field. You MUST set BOTH the `instruction` and `prompt` fields to `null` for these steps.
   - **Task B (Sequence Continuation / Sequential Editing)**: If the user provides a partial sequence and asks to predict/generate the remaining steps, you must generate both the text instruction and the editing prompt. The `prompt` field must contain an optimized instruction specifically tailored for an **image editing model** to modify the previous step's image into the new state.
2. **Strict Step Count & NO Prefix Rule**:
   - **Step Count**: Determine the logical number of steps. **CRITICAL**: If the user's input explicitly specifies the number of steps required, you MUST strictly output exactly that number of steps to fulfill the requirement. If continuing a sequence (Task B), your `step_number` MUST start exactly from where the user's input left off.
   - **NO Prefixes**: BOTH the `instruction` and `prompt` fields MUST NOT contain any step prefixes, numbers, or bullet points.
3. **Field Definitions & Usage**:
   - `instruction`: The detailed, pure text content or action for the editing step (Task B). You MUST set this to `null` for Task A.
   - `prompt`: The optimized, pure instruction suitable for the **image editing model** to execute the change based on the previous image (Task B). You MUST set this to `null` for Task A.
   - `auxiliary_text`: For Task A, this field holds your complete text response. For Task B, use this ONLY if the user explicitly requests or the task naturally requires an extra knowledge-based description/summary during the continuation process; otherwise, output `null`.
4. **Complete Output**: Ensure the final step achieves a complete resolution of the user's goal based on the sequence context.

## Output
Return exactly a `<think>...</think>` block followed by an `<answer>...</answer>` block containing a JSON object with an `execution_plan` list.
"""


@dataclass(frozen=True, slots=True)
class InterleavePlannerStep:
    step_number: int | None
    step_name: str | None
    instruction: str | None
    prompt: str | None
    auxiliary_text: str | None


@dataclass(frozen=True, slots=True)
class InterleavePlannerOutput:
    raw_response: str
    raw_answer: str
    steps: tuple[InterleavePlannerStep, ...]


class InterleaveThinkerPlannerModel(Qwen3VLActorBase):
    """Qwen3-VL actor wrapper for InterleaveThinker planning."""

    def __init__(
        self,
        *,
        init_from: str = "InterleaveThinker/InterleaveThinker-Planner-8B",
        processor_from: str | None = "Qwen/Qwen3-VL-8B-Instruct",
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
        prompt_template: str = INTERLEAVE_PLANNER_PROMPT,
        guidance_prompt_template: str = INTERLEAVE_GUIDANCE_PLANNER_PROMPT,
        lora: LoraConfig | dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self.prompt_template = prompt_template
        self.guidance_prompt_template = guidance_prompt_template
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
    def generate_interleave_plans(
        self,
        batch: dict[str, Any],
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        num_generations = max(1, int(kwargs.get("num_generations", 1) or 1))
        temperature = float(kwargs.get("temperature", 1.0) or 1.0)
        top_p = float(kwargs.get("top_p", 1.0) or 1.0)
        max_new_tokens = int(kwargs.get("max_new_tokens") or self.max_response_length)

        outputs: list[dict[str, Any]] = []
        for item_idx, item in enumerate(batch_to_items(batch)):
            decoded = self.generate_qwen_responses(
                self.build_messages(item),
                num_generations=num_generations,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
            )
            for generation_idx, response in enumerate(decoded):
                parsed = extract_interleave_plan(response)
                outputs.append({
                    "item": dict(item),
                    "response": response,
                    "plan": parsed,
                    "steps": list(parsed.steps) if parsed is not None else [],
                    "sample_index": item_idx,
                    "generation_index": generation_idx,
                })
        return outputs

    def generate_interleave_plan(
        self,
        instruction: str,
        *,
        input_image_paths: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        batch = {
            "items": [{
                "instruction": instruction,
                "input_image_paths": list(input_image_paths or []),
            }]
        }
        plans = self.generate_interleave_plans(batch, **kwargs)
        if not plans:
            raise RuntimeError("InterleaveThinker planner returned no plan generations")
        return plans[0]

    def build_messages(
        self,
        item: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        image_paths = _planner_image_paths(item)
        template = self.guidance_prompt_template if image_paths else self.prompt_template
        prompt = template.replace("{text_input}", _planner_instruction(item))
        return self.build_text_image_messages(prompt, image_paths)


def extract_interleave_plan(response: str) -> InterleavePlannerOutput | None:
    raw_answer = _extract_answer_block(response)
    if not raw_answer:
        return None
    payload = _load_answer_mapping(raw_answer)
    if not isinstance(payload, Mapping):
        return None
    raw_steps = payload.get("execution_plan")
    if not isinstance(raw_steps, Sequence) or isinstance(raw_steps, str | bytes):
        return None
    steps = tuple(_coerce_planner_step(step) for step in raw_steps if isinstance(step, Mapping))
    return InterleavePlannerOutput(
        raw_response=response,
        raw_answer=raw_answer,
        steps=steps,
    )


def _coerce_planner_step(raw: Mapping[str, Any]) -> InterleavePlannerStep:
    return InterleavePlannerStep(
        step_number=_optional_int(raw.get("step_number")),
        step_name=_optional_text(raw.get("step_name")),
        instruction=_optional_text(raw.get("instruction")),
        prompt=_optional_text(raw.get("prompt")),
        auxiliary_text=_optional_text(raw.get("auxiliary_text")),
    )


def _extract_answer_block(response: str) -> str:
    match = re.search(r"<answer>\s*(.*?)\s*</answer>", response, flags=re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return response.strip()


def _load_answer_mapping(raw_answer: str) -> Any:
    try:
        return json.loads(raw_answer)
    except json.JSONDecodeError:
        pass
    normalized = re.sub(r"\bnull\b", "None", raw_answer)
    normalized = re.sub(r"\btrue\b", "True", normalized)
    normalized = re.sub(r"\bfalse\b", "False", normalized)
    try:
        return ast.literal_eval(normalized)
    except (ValueError, SyntaxError):
        return None


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"none", "null"}:
        return None
    return text


def _planner_instruction(item: Mapping[str, Any]) -> str:
    for key in ("instruction", "text_input", "origin_prompt", "prompt"):
        value = item.get(key)
        if isinstance(value, str) and value:
            return value
    return ""


def _planner_image_paths(item: Mapping[str, Any]) -> list[str]:
    paths: list[str] = []
    for key in ("input_image_paths", "image_paths"):
        value = item.get(key)
        if isinstance(value, Sequence) and not isinstance(value, str | bytes):
            paths.extend(str(path) for path in value if path)
    for key in ("input_image_path", "image_path", "origin_image_path"):
        value = item.get(key)
        if isinstance(value, str) and value:
            paths.append(value)
    return paths


__all__ = [
    "INTERLEAVE_GUIDANCE_PLANNER_PROMPT",
    "INTERLEAVE_PLANNER_PROMPT",
    "InterleavePlannerOutput",
    "InterleavePlannerStep",
    "InterleaveThinkerPlannerModel",
    "extract_interleave_plan",
]
