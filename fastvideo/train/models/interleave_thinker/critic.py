# SPDX-License-Identifier: Apache-2.0
"""InterleaveThinker Qwen3-VL critic actor adapter."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
import os
from pathlib import Path
import re
from types import SimpleNamespace
from typing import Any, Literal, TYPE_CHECKING

import torch
from torch.utils.data import DataLoader, Dataset

from fastvideo.pipelines import TrainingBatch
from fastvideo.train.models.base import ModelBase

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


class _InterleaveJSONLDataset(Dataset):
    """Tiny JSON/JSONL dataset for InterleaveThinker critic RL records."""

    def __init__(
        self,
        data_path: str,
    ) -> None:
        self.records = list(_load_interleave_records(data_path))
        if not self.records:
            raise ValueError(f"No InterleaveThinker records found at {data_path!r}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(
        self,
        index: int,
    ) -> dict[str, Any]:
        return dict(self.records[index])


class _PlaceholderActorModule(torch.nn.Module):
    """Checkpoint-visible placeholder used when backend loading is disabled."""


class InterleaveThinkerCriticModel(ModelBase):
    """Qwen3-VL actor wrapper for InterleaveThinker critic RL.

    The wrapper loads an InterleaveThinker critic checkpoint through
    Transformers and exposes the two hooks consumed by
    :class:`InterleaveThinkerRLMethod`:

    - ``generate_interleave_responses`` for grouped critic rollouts.
    - ``train_interleave_rollouts`` for an advantage-weighted policy update.

    Set ``load_backend=false`` in tests or dry-run configs to skip heavyweight
    checkpoint loading while keeping the YAML target importable.
    """

    def __init__(
        self,
        *,
        init_from: str = "InterleaveThinker/Critic-SFT-8B",
        processor_from: str | None = None,
        training_config: TrainingConfig | None = None,
        trainable: bool = True,
        load_backend: bool = True,
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
        del kwargs
        super().__init__(trainable=trainable, lora=lora)
        self.init_from = str(init_from)
        self.processor_from = str(processor_from or init_from)
        self.training_config = training_config
        self.max_prompt_length = int(max_prompt_length)
        self.max_response_length = int(max_response_length)
        self.prompt_template = prompt_template
        self.processor: Any | None = None
        self.dataloader: Any = None
        self.start_step = 0
        self.noise_scheduler = SimpleNamespace(num_train_timesteps=0)

        if not load_backend:
            self.transformer = _PlaceholderActorModule()
            return

        self.processor, self.transformer = self._load_backend(
            init_from=self.init_from,
            processor_from=self.processor_from,
            torch_dtype=torch_dtype,
            device_map=device_map,
            attn_implementation=attn_implementation,
            trust_remote_code=trust_remote_code,
            use_cache=use_cache,
        )
        self._apply_trainable_policy(
            trainable=trainable,
            freeze_vision_tower=freeze_vision_tower,
            freeze_multi_modal_projector=freeze_multi_modal_projector,
        )
        if trainable and enable_gradient_checkpointing and hasattr(self.transformer, "gradient_checkpointing_enable"):
            self.transformer.gradient_checkpointing_enable()
        if self._enable_lora_if_configured(self.transformer):
            return

    def _load_backend(
        self,
        *,
        init_from: str,
        processor_from: str,
        torch_dtype: str,
        device_map: str | dict[str, Any] | None,
        attn_implementation: str | None,
        trust_remote_code: bool,
        use_cache: bool,
    ) -> tuple[Any, torch.nn.Module]:
        from transformers import AutoProcessor
        import transformers

        processor = AutoProcessor.from_pretrained(
            processor_from,
            trust_remote_code=trust_remote_code,
        )
        model_cls = getattr(transformers, "AutoModelForImageTextToText", None)
        if model_cls is None:
            model_cls = transformers.AutoModelForCausalLM
        load_kwargs: dict[str, Any] = {
            "trust_remote_code": trust_remote_code,
            "use_cache": use_cache,
        }
        dtype_arg = _resolve_transformers_dtype(torch_dtype)
        if dtype_arg is not None:
            load_kwargs["dtype"] = dtype_arg
        if device_map is not None:
            load_kwargs["device_map"] = device_map
        if attn_implementation:
            load_kwargs["attn_implementation"] = attn_implementation
        model = model_cls.from_pretrained(init_from, **load_kwargs)
        if device_map is None:
            model = model.to(self.device)
        return processor, model

    def _apply_trainable_policy(
        self,
        *,
        trainable: bool,
        freeze_vision_tower: bool,
        freeze_multi_modal_projector: bool,
    ) -> None:
        for name, param in self.transformer.named_parameters():
            requires_grad = bool(trainable)
            lower_name = name.lower()
            if freeze_vision_tower and _is_vision_parameter(lower_name):
                requires_grad = False
            if freeze_multi_modal_projector and _is_mm_projector_parameter(lower_name):
                requires_grad = False
            param.requires_grad_(requires_grad)

    def init_preprocessors(
        self,
        training_config: TrainingConfig,
    ) -> None:
        self.training_config = training_config
        data_path = str(getattr(training_config.data, "data_path", "") or "")
        if not data_path:
            return
        if not os.path.exists(os.path.expanduser(data_path)):
            raise FileNotFoundError(f"InterleaveThinker data_path not found: {data_path}")
        dataset = _InterleaveJSONLDataset(data_path)
        self.dataloader = DataLoader(
            dataset,
            batch_size=max(1, int(training_config.data.train_batch_size or 1)),
            shuffle=True,
            num_workers=max(0, int(training_config.data.dataloader_num_workers or 0)),
            collate_fn=_collate_interleave_records,
        )

    @torch.no_grad()
    def generate_interleave_responses(
        self,
        batch: dict[str, Any],
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        self._require_backend()
        self.transformer.eval()
        num_generations = max(1, int(kwargs.get("num_generations", 1) or 1))
        temperature = float(kwargs.get("temperature", 1.0) or 1.0)
        top_p = float(kwargs.get("top_p", 1.0) or 1.0)
        max_new_tokens = int(kwargs.get("max_new_tokens") or self.max_response_length)

        rollouts: list[dict[str, Any]] = []
        for item_idx, item in enumerate(_batch_to_items(batch)):
            messages = self.build_messages(item)
            inputs = self._apply_chat_template(
                messages,
                add_generation_prompt=True,
            )
            input_ids = inputs["input_ids"]
            generate_kwargs: dict[str, Any] = {
                "max_new_tokens": max_new_tokens,
                "num_return_sequences": num_generations,
                "do_sample": num_generations > 1 or temperature > 0.0,
                "temperature": temperature,
                "top_p": top_p,
            }
            pad_token_id = getattr(getattr(self.processor, "tokenizer", None), "pad_token_id", None)
            eos_token_id = getattr(getattr(self.processor, "tokenizer", None), "eos_token_id", None)
            if pad_token_id is not None:
                generate_kwargs["pad_token_id"] = pad_token_id
            if eos_token_id is not None:
                generate_kwargs["eos_token_id"] = eos_token_id
            generated_ids = self.transformer.generate(**inputs, **generate_kwargs)
            prompt_len = int(input_ids.shape[-1])
            decoded = self.processor.batch_decode(
                [output_ids[prompt_len:] for output_ids in generated_ids],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            for generation_idx, response in enumerate(decoded):
                rollout = dict(item)
                rollout["response"] = str(response).strip()
                rollout.setdefault("sample_index", item_idx)
                rollout.setdefault("generation_index", generation_idx)
                rollout.setdefault("group_key", _rollout_group_key(rollout, item_idx))
                rollouts.append(rollout)
        if self._trainable:
            self.transformer.train()
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
        for idx, rollout in enumerate(rollouts):
            nll, token_count = self._response_nll(rollout)
            weighted = nll * advantages[idx]
            (weighted / grad_accum).backward()
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
        images = _item_image_paths(item)
        content: list[dict[str, Any]] = []
        parts = re.split(r"(?i)<image>", prompt)
        for idx, text in enumerate(parts):
            if text:
                content.append({
                    "type": "text",
                    "text": text,
                })
            if idx < len(images):
                content.append({
                    "type": "image",
                    "image": images[idx],
                })
        return [{
            "role": "user",
            "content": content,
        }]

    def _response_nll(
        self,
        rollout: Mapping[str, Any],
    ) -> tuple[torch.Tensor, int]:
        response = str(rollout.get("response", "") or "")
        if not response:
            raise ValueError("InterleaveThinker rollout response is empty")
        prompt_messages = self.build_messages(rollout)
        full_messages = [*prompt_messages, {
            "role": "assistant",
            "content": response,
        }]
        prompt_inputs = self._apply_chat_template(
            prompt_messages,
            add_generation_prompt=True,
        )
        full_inputs = self._apply_chat_template(
            full_messages,
            add_generation_prompt=False,
        )
        labels = full_inputs["input_ids"].clone()
        prompt_len = int(prompt_inputs["input_ids"].shape[-1])
        labels[:, :prompt_len] = -100
        token_count = int((labels != -100).sum().detach().cpu())
        if token_count == 0:
            raise ValueError("InterleaveThinker rollout has no trainable response tokens")
        outputs = self.transformer(**full_inputs, labels=labels)
        return outputs.loss, token_count

    def _apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        *,
        add_generation_prompt: bool,
    ) -> Mapping[str, torch.Tensor]:
        assert self.processor is not None
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
            return_dict=True,
            return_tensors="pt",
        )
        return _move_to_device(inputs, self.device)

    def _require_backend(self) -> None:
        if self.processor is None or isinstance(self.transformer, _PlaceholderActorModule):
            raise RuntimeError("InterleaveThinkerCriticModel was created with load_backend=false; "
                               "enable backend loading for generation or RL updates.")

    def prepare_batch(
        self,
        raw_batch: dict[str, Any],
        *,
        generator: torch.Generator,
        latents_source: Literal["data", "zeros"] = "data",
    ) -> TrainingBatch:
        del raw_batch, generator, latents_source
        raise NotImplementedError("InterleaveThinkerCriticModel is not a diffusion ModelBase.")

    def add_noise(
        self,
        clean_latents: torch.Tensor,
        noise: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        del clean_latents, noise, timestep
        raise NotImplementedError("InterleaveThinkerCriticModel is not a diffusion ModelBase.")

    def predict_noise(
        self,
        noisy_latents: torch.Tensor,
        timestep: torch.Tensor,
        batch: TrainingBatch,
        *,
        conditional: bool,
        cfg_uncond: dict[str, Any] | None = None,
        attn_kind: Literal["dense", "vsa"] = "dense",
    ) -> torch.Tensor:
        del noisy_latents, timestep, batch, conditional, cfg_uncond, attn_kind
        raise NotImplementedError("InterleaveThinkerCriticModel is not a diffusion ModelBase.")

    def backward(
        self,
        loss: torch.Tensor,
        ctx: Any,
        *,
        grad_accum_rounds: int,
    ) -> None:
        del loss, ctx, grad_accum_rounds
        raise NotImplementedError("InterleaveThinkerCriticModel uses train_interleave_rollouts().")


def _load_interleave_records(data_path: str, ) -> list[dict[str, Any]]:
    path = Path(os.path.expanduser(data_path))
    files = sorted(path.glob("*.json*")) if path.is_dir() else [path]
    records: list[dict[str, Any]] = []
    for file_path in files:
        if file_path.suffix == ".jsonl":
            for line in file_path.read_text().splitlines():
                if line.strip():
                    raw = json.loads(line)
                    if isinstance(raw, Mapping):
                        records.append(dict(raw))
        elif file_path.suffix == ".json":
            raw_json = json.loads(file_path.read_text())
            if isinstance(raw_json, list):
                records.extend(dict(item) for item in raw_json if isinstance(item, Mapping))
            elif isinstance(raw_json, Mapping):
                records.append(dict(raw_json))
        else:
            raise ValueError(f"Unsupported InterleaveThinker data file: {file_path}")
    return records


def _collate_interleave_records(records: Sequence[Mapping[str, Any]], ) -> dict[str, Any]:
    return {"items": [dict(record) for record in records]}


def _batch_to_items(batch: Mapping[str, Any], ) -> list[dict[str, Any]]:
    if "items" in batch and isinstance(batch["items"], Sequence):
        return [dict(item) for item in batch["items"]]
    length = 1
    for value in batch.values():
        if isinstance(value, Sequence) and not isinstance(value, str | bytes):
            length = len(value)
            break
    items: list[dict[str, Any]] = []
    for idx in range(length):
        item: dict[str, Any] = {}
        for key, value in batch.items():
            if isinstance(value, Sequence) and not isinstance(value, str | bytes) and len(value) == length:
                item[key] = value[idx]
            else:
                item[key] = value
        items.append(item)
    return items


def _item_image_paths(item: Mapping[str, Any], ) -> list[str]:
    image_paths: list[str] = []
    for key in ("origin_image_path", "previous_image_path", "edited_image_path"):
        value = item.get(key)
        if isinstance(value, str) and value:
            image_paths.append(value)
    if len(image_paths) == 1:
        image_paths.append(image_paths[0])
    return image_paths[:2]


def _move_to_device(
    value: Any,
    device: torch.device,
) -> Any:
    if torch.is_tensor(value):
        return value.to(device)
    if hasattr(value, "to"):
        return value.to(device)
    if isinstance(value, Mapping):
        return {key: _move_to_device(item, device) for key, item in value.items()}
    return value


def _resolve_transformers_dtype(raw: str, ) -> Any:
    value = str(raw or "auto").lower()
    if value == "auto":
        return "auto"
    if value in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if value in {"fp16", "float16"}:
        return torch.float16
    if value in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported torch_dtype for InterleaveThinker critic: {raw!r}")


def _is_vision_parameter(lower_name: str, ) -> bool:
    return any(token in lower_name for token in ("visual", "vision_tower", "vision_model"))


def _is_mm_projector_parameter(lower_name: str, ) -> bool:
    return any(token in lower_name for token in ("mm_projector", "multi_modal_projector", "visual_merger"))


def _rollout_group_key(
    rollout: Mapping[str, Any],
    index: int,
) -> str:
    for key in ("group_key", "problem_id", "sample_index", "origin_prompt", "prompt"):
        value = rollout.get(key)
        if value is not None:
            return str(value)
    return str(index)


__all__ = ["INTERLEAVE_CRITIC_PROMPT", "InterleaveThinkerCriticModel"]
