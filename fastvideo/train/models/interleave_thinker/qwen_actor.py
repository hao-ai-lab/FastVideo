# SPDX-License-Identifier: Apache-2.0
"""Shared Qwen3-VL actor utilities for InterleaveThinker adapters."""

from __future__ import annotations

from collections.abc import Generator, Mapping, Sequence
from contextlib import contextmanager
import json
import os
from pathlib import Path
import re
from typing import Any, TYPE_CHECKING

import torch
import torch.distributed as dist
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler
from torchdata.stateful_dataloader import StatefulDataLoader

from fastvideo.logger import init_logger
from fastvideo.train.methods.rl.common.grpo import compute_grpo_loss
from fastvideo.train.models.base import RoleModelBase
from fastvideo.train.models.interleave_thinker.data import (
    InterleaveDatasetKind,
    load_interleave_dataset,
)

if TYPE_CHECKING:
    from fastvideo.train.utils.lora import LoraConfig
    from fastvideo.train.utils.training_config import TrainingConfig

logger = init_logger(__name__)

DEFAULT_QWEN_LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]


class _EpochAwareDistributedSampler(DistributedSampler):
    """Advance the deterministic distributed shuffle after each full epoch."""

    _indices: list[int] | None = None
    _cursor: int = 0

    def __iter__(self):
        if self._indices is None:
            self._indices = list(super().__iter__())
        return self

    def __next__(self) -> int:
        assert self._indices is not None
        if self._cursor >= len(self._indices):
            self.epoch += 1
            self._cursor = 0
            self._indices = None
            raise StopIteration
        index = self._indices[self._cursor]
        self._cursor += 1
        return index

    def state_dict(self) -> dict[str, int]:
        return {
            "epoch": int(self.epoch),
            "cursor": int(self._cursor),
        }

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        self.epoch = int(state_dict.get("epoch", 0))
        self._cursor = int(state_dict.get("cursor", 0))
        self._indices = list(super().__iter__())
        if self._cursor < 0 or self._cursor > len(self._indices):
            raise ValueError(f"Invalid distributed sampler cursor {self._cursor} for {len(self._indices)} indices")


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


class Qwen3VLActorBase(RoleModelBase):
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
        if self.max_prompt_length <= 0:
            raise ValueError("max_prompt_length must be > 0")
        if self.max_response_length <= 0:
            raise ValueError("max_response_length must be > 0")
        self.processor: Any | None = None
        self._assistant_template_overhead: int | None = None
        self.dataloader: Any = None
        self.start_step = 0

        if not load_backend:
            self.transformer = _PlaceholderActorModule()
            return

        self._validate_distributed_config(
            device_map=device_map,
            freeze_vision_tower=freeze_vision_tower,
            freeze_multi_modal_projector=freeze_multi_modal_projector,
        )

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
        self._enable_peft_lora_if_configured()
        self.transformer = self._apply_distributed_sharding(self.transformer)

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
            "low_cpu_mem_usage": True,
        }
        dtype_arg = resolve_transformers_dtype(torch_dtype)
        if dtype_arg is not None:
            load_kwargs["dtype"] = dtype_arg
        if device_map is not None:
            load_kwargs["device_map"] = device_map
        if attn_implementation:
            load_kwargs["attn_implementation"] = attn_implementation
        model = model_cls.from_pretrained(init_from, **load_kwargs)
        distributed_config = getattr(self.training_config, "distributed", None)
        configured_world_size = max(1, int(getattr(distributed_config, "num_gpus", 1) or 1))
        if device_map is None and configured_world_size == 1:
            model = model.to(self.device)
        return processor, model

    def _validate_distributed_config(
        self,
        *,
        device_map: str | dict[str, Any] | None,
        freeze_vision_tower: bool = True,
        freeze_multi_modal_projector: bool = True,
    ) -> None:
        distributed_config = getattr(self.training_config, "distributed", None)
        if distributed_config is None:
            return
        num_gpus = max(1, int(getattr(distributed_config, "num_gpus", 1) or 1))
        if num_gpus == 1:
            return

        tp_size = max(1, int(getattr(distributed_config, "tp_size", 1) or 1))
        if tp_size != 1:
            raise ValueError("InterleaveThinker Qwen actors do not implement tensor parallelism; "
                             "set training.distributed.tp_size=1 and use HSDP sharding instead")
        sp_size = max(1, int(getattr(distributed_config, "sp_size", 1) or 1))
        if sp_size != 1:
            raise ValueError("InterleaveThinker Qwen actors do not implement sequence parallelism; "
                             "set training.distributed.sp_size=1 and use HSDP sharding instead")
        if device_map is not None:
            raise ValueError("InterleaveThinker distributed actor training is incompatible with models.*.device_map")
        if not freeze_vision_tower or not freeze_multi_modal_projector:
            raise ValueError("InterleaveThinker distributed actor training requires the conditionally executed vision "
                             "tower and multimodal projector to remain frozen")

        replicate_dim, shard_dim = _resolve_hsdp_dimensions(distributed_config, num_gpus=num_gpus)
        if replicate_dim * shard_dim != num_gpus:
            raise ValueError("InterleaveThinker HSDP dimensions must multiply to training.distributed.num_gpus; "
                             f"got {replicate_dim} * {shard_dim} != {num_gpus}")
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError(
                "InterleaveThinker multi-GPU actor loading requires an initialized distributed process group; "
                "launch through torchrun and fastvideo.train.entrypoint.train")
        if dist.get_world_size() != num_gpus:
            raise ValueError(
                "training.distributed.num_gpus must match the torchrun world size for InterleaveThinker actors; "
                f"got num_gpus={num_gpus}, world_size={dist.get_world_size()}")

    def _apply_distributed_sharding(
        self,
        model: torch.nn.Module,
    ) -> torch.nn.Module:
        distributed_config = getattr(self.training_config, "distributed", None)
        num_gpus = max(1, int(getattr(distributed_config, "num_gpus", 1) or 1))
        if num_gpus == 1:
            return model

        from torch.distributed import init_device_mesh
        from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy

        replicate_dim, shard_dim = _resolve_hsdp_dimensions(distributed_config, num_gpus=num_gpus)
        device_mesh = init_device_mesh(
            self.device.type,
            mesh_shape=(replicate_dim, shard_dim),
            mesh_dim_names=("replicate", "shard"),
        )
        sharding_root = _qwen_sharding_root(model)
        param_dtype = _first_floating_parameter_dtype(sharding_root)
        mp_policy = MixedPrecisionPolicy(
            param_dtype=param_dtype,
            reduce_dtype=torch.float32,
            output_dtype=None,
            cast_forward_inputs=False,
        )
        shard_condition = _qwen_transformer_block_condition(sharding_root)
        sharded_blocks = 0
        # Hugging Face loads the checkpoint on CPU. FSDP moves and shards each
        # group bottom-up, avoiding a full unsharded 8B accelerator copy.
        for name, module in reversed(list(sharding_root.named_modules())):
            if shard_condition(name, module):
                fully_shard(
                    module,
                    reshard_after_forward=True,
                    mesh=device_mesh,
                    mp_policy=mp_policy,
                )
                sharded_blocks += 1
        if sharded_blocks == 0:
            raise ValueError("No Qwen transformer blocks matched the HSDP sharding policy")
        fully_shard(
            sharding_root,
            reshard_after_forward=True,
            mesh=device_mesh,
            mp_policy=mp_policy,
        )
        logger.info(
            "Applied progressive FSDP2/HSDP to %s with %d blocks, replicate_dim=%d shard_dim=%d",
            type(self).__name__,
            sharded_blocks,
            replicate_dim,
            shard_dim,
        )
        return model

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

    def _enable_peft_lora_if_configured(self) -> bool:
        cfg = self._lora_config
        if cfg is None or not cfg.enable:
            return False
        if not self._trainable:
            raise ValueError("PEFT LoRA training requires trainable=true for the role model")
        if cfg.rank is None:
            raise ValueError("PEFT LoRA requires lora.rank when lora.enable=true")

        from peft import LoraConfig as PeftLoraConfig
        from peft import TaskType, get_peft_model

        target_modules = list(cfg.target_modules or DEFAULT_QWEN_LORA_TARGET_MODULES)
        peft_config = PeftLoraConfig(
            r=int(cfg.rank),
            lora_alpha=int(cfg.alpha or cfg.rank),
            target_modules=target_modules,
            lora_dropout=0.0,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        seed = int(getattr(getattr(self.training_config, "data", None), "seed", 0) or 0)
        with _rank_independent_rng(seed, device=self.device):
            self.transformer = get_peft_model(self.transformer, peft_config)
        self._num_lora_layers = sum(1 for module in self.transformer.modules() if hasattr(module, "lora_A"))
        trainable_params = sum(param.numel() for param in self.transformer.parameters() if param.requires_grad)
        total_params = sum(param.numel() for param in self.transformer.parameters())
        logger.info(
            "Enabled PEFT LoRA for %s with rank=%d alpha=%d targets=%s trainable=%d/%d",
            type(self).__name__,
            int(cfg.rank),
            int(cfg.alpha or cfg.rank),
            target_modules,
            trainable_params,
            total_params,
        )
        return True

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
        seed = int(training_config.data.seed or 0)
        generator = torch.Generator().manual_seed(seed)
        sampler = _distributed_actor_sampler(dataset, seed=seed)
        self.dataloader = StatefulDataLoader(
            dataset,
            batch_size=max(1, int(training_config.data.train_batch_size or 1)),
            shuffle=sampler is None,
            sampler=sampler,
            num_workers=max(0, int(training_config.data.dataloader_num_workers or 0)),
            collate_fn=collate_interleave_records,
            generator=generator,
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
        requested_max_tokens = int(max_new_tokens or self.max_response_length)
        max_tokens = min(requested_max_tokens, self.max_response_length)
        inputs = self._apply_chat_template(
            messages,
            add_generation_prompt=True,
            max_length=self.max_prompt_length,
            limit_name="max_prompt_length",
        )
        input_ids = inputs["input_ids"]
        temperature = float(temperature)
        top_p = float(top_p)
        generate_kwargs: dict[str, Any] = {"max_new_tokens": max_tokens}
        if _distributed_actor_world_size() > 1:
            generate_kwargs["synced_gpus"] = True
        uses_custom_sampling = num_return_sequences > 1 or temperature > 0.0 or top_p < 1.0
        if uses_custom_sampling:
            generate_kwargs["do_sample"] = True
            if num_return_sequences > 1:
                generate_kwargs["num_return_sequences"] = num_return_sequences
            if temperature > 0.0:
                generate_kwargs["temperature"] = temperature
            if top_p < 1.0:
                generate_kwargs["top_p"] = top_p

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
        full_messages = [*prompt_messages, make_assistant_text_message(response)]
        prompt_inputs = self._apply_chat_template(
            prompt_messages,
            add_generation_prompt=True,
            max_length=self.max_prompt_length,
            limit_name="max_prompt_length",
        )
        template_overhead = self._assistant_response_template_overhead()
        full_inputs = self._apply_chat_template(
            full_messages,
            add_generation_prompt=False,
            max_length=self.max_prompt_length + self.max_response_length + template_overhead,
            limit_name="max_prompt_length + max_response_length + assistant template overhead",
        )
        labels = full_inputs["input_ids"].clone()
        prompt_len = int(prompt_inputs["input_ids"].shape[-1])
        self._validate_response_length(
            full_inputs["input_ids"],
            prompt_len=prompt_len,
            template_overhead=template_overhead,
        )
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
        full_messages = [*prompt_messages, make_assistant_text_message(response)]
        prompt_inputs = self._apply_chat_template(
            prompt_messages,
            add_generation_prompt=True,
            max_length=self.max_prompt_length,
            limit_name="max_prompt_length",
        )
        template_overhead = self._assistant_response_template_overhead()
        full_inputs = self._apply_chat_template(
            full_messages,
            add_generation_prompt=False,
            max_length=self.max_prompt_length + self.max_response_length + template_overhead,
            limit_name="max_prompt_length + max_response_length + assistant template overhead",
        )
        input_ids = full_inputs["input_ids"]
        prompt_len = int(prompt_inputs["input_ids"].shape[-1])
        self._validate_response_length(
            input_ids,
            prompt_len=prompt_len,
            template_overhead=template_overhead,
        )
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

    @torch.no_grad()
    def reference_logprobs_for_interleave_rollouts(
        self,
        rollouts: Sequence[Mapping[str, Any]],
    ) -> list[list[float]]:
        """Compute frozen-policy response logprobs for GRPO KL terms."""
        self._require_backend()
        was_training = bool(getattr(self.transformer, "training", False))
        self.transformer.eval()
        try:
            rows: list[list[float]] = []
            for rollout in rollouts:
                response = str(rollout.get("response", "") or "")
                logprobs, _ = self.response_logprobs_from_messages(
                    self.build_messages(rollout),
                    response,
                )
                rows.append(logprobs.detach().cpu().float().flatten().tolist())
            return rows
        finally:
            if was_training:
                self.transformer.train()

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

        if optimizer is None:
            raise RuntimeError(f"{type(self).__name__}.train_interleave_rollouts() requires an optimizer")
        optimizer.zero_grad(set_to_none=True)
        self.transformer.train()

        if int(advantages.shape[0]) != len(rollouts):
            raise ValueError("advantages count must match rollout count")

        loss_results = []
        token_counts: list[float] = []
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
            # ``compute_grpo_loss`` returns a token mean. Accumulate token sums
            # so differently sized microbatches produce the same global mean.
            (result.total_loss * result.token_count.detach()).backward()
            loss_results.append(result)
            token_counts.append(float(result.token_count.detach().cpu()))

        total_tokens = max(1.0, sum(token_counts))
        gradient_scale = _distributed_token_gradient_scale(total_tokens, device=self.device)
        for parameter in self.transformer.parameters():
            if parameter.grad is not None:
                parameter.grad.mul_(gradient_scale)

        if max_grad_norm > 0.0:
            from fastvideo.train.utils.optimizer import clip_grad_norm_if_needed

            clip_grad_norm_if_needed(self.transformer, max_grad_norm)
        optimizer.step()
        if lr_scheduler is not None:
            lr_scheduler.step()

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
                image_content: dict[str, Any] = {
                    "type": "image",
                    "image": images[idx],
                }
                if len(images) >= 5:
                    image_content["max_pixels"] = 384 * 384
                content.append(image_content)
                consumed_images += 1
        for image_path in images[consumed_images:]:
            image_content = {
                "type": "image",
                "image": image_path,
            }
            if len(images) >= 5:
                image_content["max_pixels"] = 384 * 384
            content.append(image_content)
        return [{
            "role": "user",
            "content": content,
        }]

    def _apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        *,
        add_generation_prompt: bool,
        max_length: int,
        limit_name: str,
    ) -> Mapping[str, torch.Tensor]:
        assert self.processor is not None
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
            return_dict=True,
            return_tensors="pt",
        )
        input_ids = inputs.get("input_ids")
        if not torch.is_tensor(input_ids):
            raise TypeError("Qwen processor chat template must return tensor input_ids")
        token_length = int(input_ids.shape[-1])
        if token_length > max_length:
            raise ValueError(f"InterleaveThinker tokenized input length {token_length} exceeds configured "
                             f"{limit_name}={max_length}")
        return move_to_device(inputs, self.device)

    def _validate_response_length(
        self,
        input_ids: torch.Tensor,
        *,
        prompt_len: int,
        template_overhead: int,
    ) -> None:
        response_len = max(0, int(input_ids.shape[-1]) - int(prompt_len) - int(template_overhead))
        if response_len > self.max_response_length:
            raise ValueError(f"InterleaveThinker response length {response_len} exceeds configured "
                             f"max_response_length={self.max_response_length}")

    def _assistant_response_template_overhead(self) -> int:
        if self._assistant_template_overhead is not None:
            return self._assistant_template_overhead
        assert self.processor is not None
        prompt_messages = [{
            "role": "user",
            "content": [{
                "type": "text",
                "text": "x",
            }],
        }]
        prompt_inputs = self.processor.apply_chat_template(
            prompt_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        empty_response_inputs = self.processor.apply_chat_template(
            [*prompt_messages, make_assistant_text_message("")],
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            return_tensors="pt",
        )
        prompt_ids = prompt_inputs.get("input_ids")
        empty_response_ids = empty_response_inputs.get("input_ids")
        if not torch.is_tensor(prompt_ids) or not torch.is_tensor(empty_response_ids):
            raise TypeError("Qwen processor chat template must return tensor input_ids")
        self._assistant_template_overhead = max(0, int(empty_response_ids.shape[-1]) - int(prompt_ids.shape[-1]))
        return self._assistant_template_overhead

    def _require_backend(self) -> None:
        if self.processor is None or isinstance(self.transformer, _PlaceholderActorModule):
            raise RuntimeError(f"{type(self).__name__} was created with load_backend=false; "
                               "enable backend loading for generation or training updates.")


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


def make_assistant_text_message(response: str) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": [{
            "type": "text",
            "text": response,
        }],
    }


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


def _distributed_actor_world_size() -> int:
    if not dist.is_available() or not dist.is_initialized():
        return 1
    return int(dist.get_world_size())


def _distributed_actor_sampler(
    dataset: Dataset,
    *,
    seed: int,
) -> _EpochAwareDistributedSampler | None:
    world_size = _distributed_actor_world_size()
    if world_size <= 1:
        return None
    return _EpochAwareDistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=int(dist.get_rank()),
        shuffle=True,
        seed=int(seed),
        drop_last=False,
    )


def _distributed_token_gradient_scale(
    local_token_count: float,
    *,
    device: torch.device,
) -> float:
    """Scale FSDP-averaged token-sum gradients into one global token mean."""
    world_size = _distributed_actor_world_size()
    if world_size <= 1:
        return 1.0 / max(1.0, float(local_token_count))

    global_token_count = torch.tensor(float(local_token_count), device=device, dtype=torch.float64)
    dist.all_reduce(global_token_count, op=dist.ReduceOp.SUM)
    # FSDP averages gradients across its data-parallel ranks. Restore the
    # summed numerator before dividing by the global token denominator.
    return float(world_size) / max(1.0, float(global_token_count.detach().cpu()))


def _resolve_hsdp_dimensions(
    distributed_config: Any,
    *,
    num_gpus: int,
) -> tuple[int, int]:
    replicate_dim = int(getattr(distributed_config, "hsdp_replicate_dim", 1) or 1)
    raw_shard_dim = getattr(distributed_config, "hsdp_shard_dim", -1)
    shard_dim = num_gpus if raw_shard_dim is None or int(raw_shard_dim) == -1 else int(raw_shard_dim)
    if replicate_dim <= 0 or shard_dim <= 0:
        raise ValueError("InterleaveThinker HSDP dimensions must be positive (hsdp_shard_dim may be -1 for auto)")
    return replicate_dim, shard_dim


@contextmanager
def _rank_independent_rng(
    seed: int,
    *,
    device: torch.device,
) -> Generator[None, None, None]:
    cuda_devices: list[int] = []
    if device.type == "cuda" and torch.cuda.is_available():
        cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()]
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(int(seed))
        yield


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


def _first_floating_parameter_dtype(model: torch.nn.Module, ) -> torch.dtype:
    for parameter in model.parameters():
        if parameter.is_floating_point():
            return parameter.dtype
    return torch.float32


def _qwen_sharding_root(model: torch.nn.Module, ) -> torch.nn.Module:
    """Return the module invoked by both PEFT ``forward`` and ``generate``."""
    get_base_model = getattr(model, "get_base_model", None)
    if callable(get_base_model):
        base_model = get_base_model()
        if isinstance(base_model, torch.nn.Module):
            return base_model
    return model


def _qwen_transformer_block_condition(model: torch.nn.Module, ) -> Any:
    """Return an FSDP condition covering Hugging Face Qwen block modules."""
    no_split_class_names: set[str] = set()
    module_list_children: set[int] = set()
    for module in model.modules():
        raw_no_split = getattr(module, "_no_split_modules", None)
        if isinstance(raw_no_split, Sequence) and not isinstance(raw_no_split, str | bytes):
            no_split_class_names.update(str(name) for name in raw_no_split)
        if isinstance(module, torch.nn.ModuleList):
            module_list_children.update(id(child) for child in module)

    if not no_split_class_names and not module_list_children:
        raise ValueError("Could not identify transformer blocks for InterleaveThinker HSDP sharding")

    def is_transformer_block(_name: str, module: torch.nn.Module) -> bool:
        lower_name = _name.lower()
        lower_type = type(module).__name__.lower()
        if "vision" in lower_name or "visual" in lower_name or "vision" in lower_type or "visual" in lower_type:
            # Image-conditioned ranks may execute the vision tower while
            # text-only ranks skip it. Keep those frozen parameters in the root
            # FSDP group so every rank runs collectives in the same order.
            return False
        return id(module) in module_list_children or type(module).__name__ in no_split_class_names

    return is_transformer_block


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
