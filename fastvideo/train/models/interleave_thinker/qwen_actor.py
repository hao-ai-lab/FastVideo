# SPDX-License-Identifier: Apache-2.0
"""Shared Qwen3-VL actor utilities for InterleaveThinker adapters."""

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
from fastvideo.train.models.interleave_thinker.data import (
    InterleaveDatasetKind,
    load_interleave_dataset,
)

if TYPE_CHECKING:
    from fastvideo.train.utils.lora import LoraConfig
    from fastvideo.train.utils.training_config import TrainingConfig


class InterleaveJSONLDataset(Dataset):
    """Tiny JSON/JSONL dataset for InterleaveThinker records."""

    def __init__(
        self,
        data_path: str,
        *,
        image_dir: str = "",
        dataset_kind: InterleaveDatasetKind | None = None,
    ) -> None:
        if dataset_kind is None:
            self.records = list(load_interleave_records(
                data_path,
                image_dir=image_dir,
            ))
        else:
            self.records = list(load_interleave_dataset(
                data_path,
                kind=dataset_kind,
                image_dir=image_dir,
            ))
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


class Qwen3VLActorBase(ModelBase):
    """Shared Transformers Qwen3-VL wrapper for planner and critic actors."""

    def __init__(
        self,
        *,
        init_from: str,
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
        lora: LoraConfig | dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        del kwargs
        super().__init__(trainable=trainable, lora=lora)
        self.init_from = str(init_from)
        self.processor_from = str(processor_from or init_from)
        self.image_dir = str(image_dir or "")
        self.training_config = training_config
        self.dataset_kind = dataset_kind
        self.max_prompt_length = int(max_prompt_length)
        self.max_response_length = int(max_response_length)
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
        self._enable_lora_if_configured(self.transformer)

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
        model_cls = _preferred_qwen_model_class(transformers)
        load_kwargs: dict[str, Any] = {
            "trust_remote_code": trust_remote_code,
            "use_cache": use_cache,
        }
        dtype_arg = resolve_transformers_dtype(torch_dtype)
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
            if freeze_vision_tower and is_vision_parameter(lower_name):
                requires_grad = False
            if freeze_multi_modal_projector and is_mm_projector_parameter(lower_name):
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
        dataset = InterleaveJSONLDataset(
            data_path,
            image_dir=self.image_dir,
            dataset_kind=self.dataset_kind,
        )
        self.dataloader = DataLoader(
            dataset,
            batch_size=max(1, int(training_config.data.train_batch_size or 1)),
            shuffle=True,
            num_workers=max(0, int(training_config.data.dataloader_num_workers or 0)),
            collate_fn=collate_interleave_records,
        )

    def generate_qwen_responses(
        self,
        messages: list[dict[str, Any]],
        *,
        num_generations: int = 1,
        temperature: float = 1.0,
        top_p: float = 1.0,
        max_new_tokens: int | None = None,
    ) -> list[str]:
        self._require_backend()
        self.transformer.eval()
        num_return_sequences = max(1, int(num_generations or 1))
        max_tokens = int(max_new_tokens or self.max_response_length)
        inputs = self._apply_chat_template(
            messages,
            add_generation_prompt=True,
        )
        input_ids = inputs["input_ids"]
        generate_kwargs: dict[str, Any] = {
            "max_new_tokens": max_tokens,
            "num_return_sequences": num_return_sequences,
            "do_sample": num_return_sequences > 1 or float(temperature) > 0.0,
            "temperature": float(temperature),
            "top_p": float(top_p),
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
        if self._trainable:
            self.transformer.train()
        return [str(response).strip() for response in decoded]

    def response_nll_from_messages(
        self,
        prompt_messages: list[dict[str, Any]],
        response: str,
    ) -> tuple[torch.Tensor, int]:
        self._require_backend()
        if not response:
            raise ValueError("InterleaveThinker rollout response is empty")
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

    def response_logprobs_from_messages(
        self,
        prompt_messages: list[dict[str, Any]],
        response: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return per-token logprobs and mask for the assistant response."""
        self._require_backend()
        if not response:
            raise ValueError("InterleaveThinker rollout response is empty")
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
        input_ids = full_inputs["input_ids"]
        prompt_len = int(prompt_inputs["input_ids"].shape[-1])
        response_ids = input_ids[:, prompt_len:]
        if int(response_ids.shape[-1]) == 0:
            raise ValueError("InterleaveThinker rollout has no response tokens")

        outputs = self.transformer(**full_inputs)
        logits = _output_logits(outputs)
        token_logits = logits[:, prompt_len - 1:-1, :]
        if int(token_logits.shape[1]) != int(response_ids.shape[1]):
            token_logits = token_logits[:, -int(response_ids.shape[1]):, :]
        if int(token_logits.shape[1]) != int(response_ids.shape[1]):
            raise ValueError("Could not align response logits with response token ids")

        logprobs = torch.log_softmax(token_logits.float(), dim=-1)
        response_logprobs = logprobs.gather(-1, response_ids.unsqueeze(-1)).squeeze(-1)
        response_mask = torch.ones_like(response_logprobs, dtype=torch.float32)
        attention_mask = full_inputs.get("attention_mask")
        if isinstance(attention_mask, torch.Tensor):
            response_mask = attention_mask[:, prompt_len:].to(device=response_logprobs.device, dtype=torch.float32)
        if float(response_mask.sum().detach().cpu()) <= 0.0:
            raise ValueError("InterleaveThinker rollout has no unmasked response tokens")
        return response_logprobs.squeeze(0), response_mask.squeeze(0)

    def compute_interleave_sft_loss(
        self,
        batch: Mapping[str, Any],
    ) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
        self._require_backend()
        self.transformer.train()
        losses: list[torch.Tensor] = []
        token_counts: list[int] = []
        for item in batch_to_items(batch):
            response = _sft_response_from_item(item)
            nll, token_count = self.response_nll_from_messages(
                self.build_messages(item),
                response,
            )
            losses.append(nll)
            token_counts.append(token_count)
        if not losses:
            raise ValueError("InterleaveThinker SFT batch is empty")
        loss = torch.stack(losses).mean()
        return (
            {
                "total_loss": loss,
                "sft_loss": loss,
            },
            {
                "sft/response_tokens": float(sum(token_counts) / max(1, len(token_counts))),
                "sft/num_items": float(len(token_counts)),
            },
        )

    def build_messages(
        self,
        item: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        raise NotImplementedError(f"{type(self).__name__} must implement build_messages().")

    def build_text_image_messages(
        self,
        prompt: str,
        image_paths: Sequence[str] | None = None,
    ) -> list[dict[str, Any]]:
        images = list(image_paths or [])
        content: list[dict[str, Any]] = []
        parts = re.split(r"(?i)<image>", prompt)
        consumed_images = 0
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
                consumed_images += 1
        for image_path in images[consumed_images:]:
            content.append({
                "type": "image",
                "image": image_path,
            })
        return [{
            "role": "user",
            "content": content,
        }]

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
        return move_to_device(inputs, self.device)

    def _require_backend(self) -> None:
        if self.processor is None or isinstance(self.transformer, _PlaceholderActorModule):
            raise RuntimeError(f"{type(self).__name__} was created with load_backend=false; "
                               "enable backend loading for generation or training updates.")

    def prepare_batch(
        self,
        raw_batch: dict[str, Any],
        *,
        generator: torch.Generator,
        latents_source: Literal["data", "zeros"] = "data",
    ) -> TrainingBatch:
        del raw_batch, generator, latents_source
        raise NotImplementedError(f"{type(self).__name__} is not a diffusion ModelBase.")

    def add_noise(
        self,
        clean_latents: torch.Tensor,
        noise: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        del clean_latents, noise, timestep
        raise NotImplementedError(f"{type(self).__name__} is not a diffusion ModelBase.")

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
        raise NotImplementedError(f"{type(self).__name__} is not a diffusion ModelBase.")

    def backward(
        self,
        loss: torch.Tensor,
        ctx: Any,
        *,
        grad_accum_rounds: int,
    ) -> None:
        del loss, ctx, grad_accum_rounds
        raise NotImplementedError(f"{type(self).__name__} uses role-specific training hooks.")


def load_interleave_records(
    data_path: str,
    *,
    image_dir: str = "",
) -> list[dict[str, Any]]:
    path = Path(os.path.expanduser(data_path))
    files = sorted(path.glob("*.json*")) if path.is_dir() else [path]
    records: list[dict[str, Any]] = []
    for file_path in files:
        if file_path.suffix == ".jsonl":
            for line in file_path.read_text().splitlines():
                if line.strip():
                    raw = json.loads(line)
                    if isinstance(raw, Mapping):
                        records.append(normalize_interleave_record(raw, image_dir=image_dir))
        elif file_path.suffix == ".json":
            raw_json = json.loads(file_path.read_text())
            if isinstance(raw_json, list):
                records.extend(
                    normalize_interleave_record(item, image_dir=image_dir) for item in raw_json
                    if isinstance(item, Mapping))
            elif isinstance(raw_json, Mapping):
                records.append(normalize_interleave_record(raw_json, image_dir=image_dir))
        else:
            raise ValueError(f"Unsupported InterleaveThinker data file: {file_path}")
    return records


def _sft_response_from_item(item: Mapping[str, Any]) -> str:
    for key in ("response", "completion", "target"):
        value = item.get(key)
        if isinstance(value, str) and value:
            return value
    messages = item.get("messages")
    if isinstance(messages, Sequence) and not isinstance(messages, str | bytes):
        for message in messages:
            if not isinstance(message, Mapping):
                continue
            if message.get("role") != "assistant":
                continue
            content = message.get("content")
            if isinstance(content, str) and content:
                return content
    raise ValueError("InterleaveThinker SFT item requires response, completion, target, or assistant message")


def normalize_interleave_record(
    record: Mapping[str, Any],
    *,
    image_dir: str = "",
) -> dict[str, Any]:
    normalized = dict(record)
    for key in (
            "origin_image_path",
            "previous_image_path",
            "edited_image_path",
            "generated_image_path",
            "input_image_path",
            "output_image_path",
            "image_path",
    ):
        value = normalized.get(key)
        if isinstance(value, str) and value:
            normalized[key] = resolve_image_path(value, image_dir=image_dir)
    for key in ("input_image_paths", "image_paths"):
        value = normalized.get(key)
        if isinstance(value, Sequence) and not isinstance(value, str | bytes):
            normalized[key] = [
                resolve_image_path(item, image_dir=image_dir) if isinstance(item, str) and item else item
                for item in value
            ]
    return normalized


def resolve_image_path(
    value: str,
    *,
    image_dir: str = "",
) -> str:
    expanded = os.path.expanduser(value)
    if not image_dir or os.path.isabs(expanded) or looks_like_uri(expanded):
        return expanded
    return str(Path(os.path.expanduser(image_dir)) / expanded)


def looks_like_uri(value: str) -> bool:
    return "://" in value or value.startswith("data:")


def collate_interleave_records(records: Sequence[Mapping[str, Any]], ) -> dict[str, Any]:
    return {"items": [dict(record) for record in records]}


def batch_to_items(batch: Mapping[str, Any], ) -> list[dict[str, Any]]:
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


def coerce_logprob_vector(
    value: Any,
    *,
    expected_len: int,
    device: torch.device,
    name: str,
) -> torch.Tensor | None:
    if value is None:
        return None
    if torch.is_tensor(value):
        tensor = value.detach().to(device=device, dtype=torch.float32).flatten()
    elif isinstance(value, Sequence) and not isinstance(value, str | bytes):
        tensor = torch.tensor(list(value), device=device, dtype=torch.float32).flatten()
    else:
        raise TypeError(f"{name} must be a tensor or sequence of floats")
    if int(tensor.numel()) != int(expected_len):
        raise ValueError(f"{name} length {int(tensor.numel())} does not match response token count {expected_len}")
    return tensor


def coerce_response_mask(
    value: Any,
    *,
    expected_len: int,
    device: torch.device,
) -> torch.Tensor | None:
    if value is None:
        return None
    if torch.is_tensor(value):
        tensor = value.detach().to(device=device, dtype=torch.float32).flatten()
    elif isinstance(value, Sequence) and not isinstance(value, str | bytes):
        tensor = torch.tensor(list(value), device=device, dtype=torch.float32).flatten()
    else:
        raise TypeError("response_mask must be a tensor or sequence of numbers")
    if int(tensor.numel()) != int(expected_len):
        raise ValueError(
            f"response_mask length {int(tensor.numel())} does not match response token count {expected_len}")
    return tensor


def pad_1d_tensors(
    values: Sequence[torch.Tensor],
    *,
    pad_value: float = 0.0,
) -> torch.Tensor:
    if not values:
        raise ValueError("Cannot pad an empty tensor sequence")
    max_len = max(int(value.numel()) for value in values)
    if max_len <= 0:
        raise ValueError("Cannot pad empty tensors")
    device = values[0].device
    dtype = values[0].dtype
    padded = torch.full((len(values), max_len), float(pad_value), device=device, dtype=dtype)
    for idx, value in enumerate(values):
        flat = value.to(device=device, dtype=dtype).flatten()
        padded[idx, :int(flat.numel())] = flat
    return padded


def first_string(
    item: Mapping[str, Any],
    *keys: str,
) -> str | None:
    for key in keys:
        value = item.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def move_to_device(
    value: Any,
    device: torch.device,
) -> Any:
    if torch.is_tensor(value):
        return value.to(device)
    if hasattr(value, "to"):
        return value.to(device)
    if isinstance(value, Mapping):
        return {key: move_to_device(item, device) for key, item in value.items()}
    return value


def resolve_transformers_dtype(raw: str, ) -> Any:
    value = str(raw or "auto").lower()
    if value == "auto":
        return "auto"
    if value in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if value in {"fp16", "float16"}:
        return torch.float16
    if value in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported torch_dtype for InterleaveThinker actor: {raw!r}")


def is_vision_parameter(lower_name: str, ) -> bool:
    return any(token in lower_name for token in ("visual", "vision_tower", "vision_model"))


def is_mm_projector_parameter(lower_name: str, ) -> bool:
    return any(token in lower_name for token in ("mm_projector", "multi_modal_projector", "visual_merger"))


def rollout_group_key(
    rollout: Mapping[str, Any],
    index: int,
) -> str:
    for key in ("group_key", "problem_id", "sample_index", "origin_prompt", "prompt"):
        value = rollout.get(key)
        if value is not None:
            return str(value)
    return str(index)


def _output_logits(outputs: Any) -> torch.Tensor:
    if hasattr(outputs, "logits"):
        logits = outputs.logits
    elif isinstance(outputs, Mapping):
        logits = outputs.get("logits")
    else:
        logits = None
    if not torch.is_tensor(logits):
        raise TypeError("Qwen actor forward output must include tensor logits")
    return logits


def _preferred_qwen_model_class(transformers_module: Any) -> Any:
    for class_name in (
            "Qwen3VLForConditionalGeneration",
            "AutoModelForImageTextToText",
            "AutoModelForVision2Seq",
            "AutoModelForCausalLM",
    ):
        model_cls = getattr(transformers_module, class_name, None)
        if model_cls is not None:
            return model_cls
    raise RuntimeError("Transformers does not provide a compatible Qwen3-VL model class")


__all__ = [
    "InterleaveJSONLDataset",
    "Qwen3VLActorBase",
    "_PlaceholderActorModule",
    "batch_to_items",
    "collate_interleave_records",
    "coerce_logprob_vector",
    "coerce_response_mask",
    "first_string",
    "load_interleave_records",
    "normalize_interleave_record",
    "pad_1d_tensors",
    "resolve_image_path",
    "rollout_group_key",
]
