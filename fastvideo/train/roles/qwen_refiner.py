# SPDX-License-Identifier: Apache-2.0
"""Qwen prompt-refiner training role for PromptRL.

Wraps ``Qwen/Qwen2.5-VL-3B-Instruct`` (or any compatible causal LM via
``model_kind="causal_lm"`` for tests) as a non-diffusion
:class:`~fastvideo.train.roles.base.TrainRoleBase`:

* the base model and vision tower stay frozen,
* a PEFT LoRA adapter on ``q_proj``/``k_proj``/``v_proj``/``o_proj`` is
  the only trainable surface,
* the adapter-disabled model doubles as the frozen reference policy,
  avoiding a duplicate base-model copy,
* the role is replicated across ranks; LoRA gradients are synchronized
  with an optional DDP wrapper (``maybe_wrap_ddp``).
"""

from __future__ import annotations

import contextlib
from typing import Any, Literal
from collections.abc import Iterator

import torch

from fastvideo.logger import init_logger
from fastvideo.train.roles.base import TrainRoleBase
from fastvideo.train.utils.lora import LoraConfig

logger = init_logger(__name__)

REFINER_LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]

ModelKind = Literal["vlm", "causal_lm"]


class QwenPromptRefinerRole(TrainRoleBase):
    """Language-model role that rewrites prompts for Wan generation."""

    def __init__(
        self,
        *,
        init_from: str,
        training_config: Any = None,
        trainable: bool = True,
        lora: LoraConfig | dict[str, Any] | None = None,
        model_kind: ModelKind = "vlm",
        torch_dtype: str = "bfloat16",
        attn_implementation: str = "sdpa",
        max_prompt_tokens: int = 1024,
        init_adapter_from: str | None = None,
        model: torch.nn.Module | None = None,
        tokenizer: Any = None,
        device: str | torch.device | None = None,
    ) -> None:
        self._trainable = bool(trainable)
        self._init_from = str(init_from)
        self._lora_config = LoraConfig.coerce(lora)
        self._model_kind = model_kind
        self._torch_dtype = torch_dtype
        self._attn_implementation = attn_implementation
        self._max_prompt_tokens = int(max_prompt_tokens)
        self._device_override = device
        self._ddp_model: torch.nn.Module | None = None

        if model is None:
            model = self._load_model()
        if tokenizer is None:
            tokenizer = self._load_tokenizer()
        self.model = model
        self.tokenizer = tokenizer
        self._freeze_base()
        if self._trainable:
            self._enable_lora()
        if init_adapter_from is not None:
            # Milestone handoff: joint training starts from the
            # prompt-only refiner checkpoint (Wan's LoRA stays at its
            # zero initialization).
            self.load_adapter(init_adapter_from)
        self.model.to(self.device)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load_model(self) -> torch.nn.Module:
        import transformers

        dtype = getattr(torch, self._torch_dtype)
        if self._model_kind == "vlm":
            model_cls = getattr(transformers, "Qwen2_5_VLForConditionalGeneration", None)
            if model_cls is None:
                raise ImportError("transformers does not provide "
                                  "Qwen2_5_VLForConditionalGeneration")
        else:
            model_cls = transformers.AutoModelForCausalLM
        logger.info("Loading prompt refiner %s (kind=%s)", self._init_from, self._model_kind)
        return model_cls.from_pretrained(
            self._init_from,
            torch_dtype=dtype,
            attn_implementation=self._attn_implementation,
        )

    def _load_tokenizer(self) -> Any:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(self._init_from)


    # ------------------------------------------------------------------
    # TrainRoleBase plumbing
    # ------------------------------------------------------------------

    @property
    def device(self) -> torch.device:
        if self._device_override is not None:
            return torch.device(self._device_override)
        return super().device

    def checkpoint_modules(self) -> dict[str, torch.nn.Module]:
        return {"model": self.model}

    def trainable_parameters(self) -> list[torch.nn.Parameter]:
        return [p for p in self.model.parameters() if p.requires_grad]

    def on_train_start(self) -> None:
        self.maybe_wrap_ddp()

    # ------------------------------------------------------------------
    # Freezing + LoRA
    # ------------------------------------------------------------------

    def _freeze_base(self) -> None:
        self.model.requires_grad_(False)
        # Vision tower stays frozen explicitly (it never receives LoRA).
        for attr in ("visual", "vision_tower", "vision_model"):
            tower = getattr(self.model, attr, None)
            if tower is not None:
                tower.requires_grad_(False)
        self.model.eval()

    def _enable_lora(self) -> None:
        cfg = self._lora_config
        if cfg is None or not cfg.enable:
            raise ValueError("QwenPromptRefinerRole requires lora.enable=true with "
                             "an explicit rank (refiner training is LoRA-only)")
        from peft import LoraConfig as PeftLoraConfig
        from peft import get_peft_model

        target_modules = list(cfg.target_modules or REFINER_LORA_TARGET_MODULES)
        peft_config = PeftLoraConfig(
            r=int(cfg.rank),  # type: ignore[arg-type]
            lora_alpha=int(cfg.alpha) if cfg.alpha is not None else int(cfg.rank),  # type: ignore[arg-type]
            target_modules=target_modules,
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.model = get_peft_model(self.model, peft_config)
        # Frozen base + frozen vision tower: verify only LoRA trains.
        trainable = [(n, p.numel()) for n, p in self.model.named_parameters() if p.requires_grad]
        if not trainable:
            raise ValueError("Refiner LoRA produced no trainable parameters")
        non_lora = [n for n, _ in trainable if "lora_" not in n]
        if non_lora:
            raise RuntimeError(f"Non-LoRA refiner parameters are trainable: {non_lora[:5]}")
        logger.info("Refiner LoRA enabled on %d modules (%d trainable params)",
                    len(trainable), sum(n for _, n in trainable))

    # ------------------------------------------------------------------
    # Adapter (reference policy) contexts
    # ------------------------------------------------------------------

    @contextlib.contextmanager
    def adapter_disabled(self) -> Iterator[None]:
        """Temporarily disable the LoRA adapter (frozen reference policy)."""
        disabled = False
        try:
            self.model.disable_adapter_layers()
            disabled = True
        except AttributeError:
            pass
        try:
            yield
        finally:
            if disabled:
                self.model.enable_adapter_layers()

    def maybe_wrap_ddp(self) -> None:
        """Wrap the model in DDP for synchronized LoRA gradients.

        No-op when distributed training is inactive, when already
        wrapped, or on non-CUDA devices (single-process tests).
        """
        import torch.distributed as dist
        from torch.nn.parallel import DistributedDataParallel

        if self._ddp_model is not None:
            return
        if not (dist.is_available() and dist.is_initialized()):
            return
        if dist.get_world_size() <= 1:
            return
        if self.device.type != "cuda":
            logger.warning("Refiner DDP wrap skipped on non-CUDA device %s", self.device)
            return
        # DDP broadcasts parameters from rank 0 at construction, which
        # also aligns PEFT's RNG-dependent LoRA init across ranks.
        self._ddp_model = DistributedDataParallel(
            self.model,
            device_ids=[self.device.index],
            output_device=self.device.index,
            broadcast_buffers=False,
            find_unused_parameters=False,
        )
        logger.info("Wrapped prompt refiner in DDP (rank %d)", dist.get_rank())

    def _forward_model(self) -> torch.nn.Module:
        return self._ddp_model if self._ddp_model is not None else self.model

    # ------------------------------------------------------------------
    # Prompt rendering + generation
    # ------------------------------------------------------------------

    def render_chat(self, instruction: str) -> str:
        """Render the refiner chat template for one instruction."""
        messages = [{"role": "user", "content": instruction}]
        apply_template = getattr(self.tokenizer, "apply_chat_template", None)
        if callable(apply_template):
            return str(
                apply_template(messages, tokenize=False, add_generation_prompt=True))
        return instruction + "\n"

    @torch.no_grad()
    def generate_refinements(
        self,
        prompts: list[str],
        *,
        instructions: list[str] | None = None,
        max_new_tokens: int = 256,
        temperature: float = 1.0,
        top_p: float = 1.0,
        seed: int | None = None,
    ) -> list[str]:
        """Sample one completion per prompt from the current adapter."""
        if instructions is None:
            instructions = prompts
        was_training = self.model.training
        ddp_model = self._ddp_model
        was_ddp_training = ddp_model.training if ddp_model is not None else None
        self.model.eval()
        if ddp_model is not None:
            ddp_model.eval()
        try:
            rendered = [self.render_chat(text) for text in instructions]
            encoded = self.tokenizer(
                rendered,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self._max_prompt_tokens,
            ).to(self.device)
            # transformers>=5 dropped the ``generator`` kwarg from
            # generate(); fork the global RNG for a seeded sample
            # instead (device-independent across torch versions).
            if seed is None:
                rng_context = contextlib.nullcontext()
            elif self.device.type == "cuda":
                rng_context = torch.random.fork_rng(devices=[self.device])
            else:
                rng_context = torch.random.fork_rng(devices=[])
            with rng_context:
                if seed is not None:
                    torch.manual_seed(int(seed))
                    if self.device.type == "cuda":
                        torch.cuda.manual_seed_all(int(seed))
                output = self.model.generate(
                    **encoded,
                    max_new_tokens=int(max_new_tokens),
                    do_sample=True,
                    temperature=float(temperature),
                    top_p=float(top_p),
                    pad_token_id=self._pad_token_id(),
                )
            prompt_length = int(encoded["input_ids"].shape[1])
            completions = output[:, prompt_length:]
            return self.tokenizer.batch_decode(completions, skip_special_tokens=True)
        finally:
            if ddp_model is not None and was_ddp_training is not None:
                ddp_model.train(was_ddp_training)
            self.model.train(was_training)

    # ------------------------------------------------------------------
    # Sequence log probabilities
    # ------------------------------------------------------------------

    def sequence_logprobs(
        self,
        prompts: list[str],
        completions: list[str],
        *,
        use_adapter: bool = True,
        requires_grad: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Summed completion log probs per sample.

        Returns ``(sum_logprobs, token_counts)``, each shape ``[B]``.
        ``use_adapter=False`` evaluates the frozen reference policy via
        the adapter-disabled context.  With ``requires_grad=True`` the
        log probs keep gradients for the GRPO loss.
        """
        token_logprobs, mask = self.token_logprobs(
            prompts,
            completions,
            use_adapter=use_adapter,
            requires_grad=requires_grad,
        )
        summed = (token_logprobs * mask).sum(dim=-1)
        counts = mask.sum(dim=-1)
        return summed, counts

    def token_logprobs(
        self,
        prompts: list[str],
        completions: list[str],
        *,
        use_adapter: bool = True,
        requires_grad: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-token completion log probs.

        Returns ``(token_logprobs, mask)`` of shapes ``[B, L]`` where
        padded positions have mask 0 and log prob 0.
        """
        if len(prompts) != len(completions):
            raise ValueError(f"prompts ({len(prompts)}) and completions "
                             f"({len(completions)}) length mismatch")
        rendered = [self.render_chat(p) for p in prompts]
        prompt_ids = self.tokenizer(
            rendered,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self._max_prompt_tokens,
        )["input_ids"]
        completion_ids = self.tokenizer(
            completions,
            add_special_tokens=False,
            return_tensors="pt",
            padding=True,
        )["input_ids"]

        batch_size = len(prompts)
        input_rows: list[torch.Tensor] = []
        label_rows: list[torch.Tensor] = []
        for row in range(batch_size):
            p_ids = prompt_ids[row][prompt_ids[row] != self._pad_token_id()]
            c_ids = completion_ids[row][completion_ids[row] != self._pad_token_id()]
            input_rows.append(torch.cat([p_ids, c_ids], dim=0))
            labels = torch.full_like(input_rows[-1], -100)
            labels[p_ids.shape[0]:] = c_ids
            label_rows.append(labels)

        max_len = max(row.shape[0] for row in input_rows)
        pad_id = self._pad_token_id()
        input_ids = torch.full((batch_size, max_len), pad_id, dtype=torch.long)
        labels = torch.full((batch_size, max_len), -100, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
        for row in range(batch_size):
            length = input_rows[row].shape[0]
            input_ids[row, :length] = input_rows[row]
            labels[row, :length] = label_rows[row]
            attention_mask[row, :length] = 1
        device = self.device
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        attention_mask = attention_mask.to(device)

        context = (contextlib.nullcontext() if use_adapter else self.adapter_disabled())
        grad_context = (torch.enable_grad() if requires_grad else torch.no_grad())
        # Reference passes run on the raw model (no DDP reducer needed
        # under no_grad); current-policy passes go through DDP when
        # wrapped so LoRA gradients stay synchronized.
        forward_model = self._forward_model() if use_adapter else self.model
        with context, grad_context:
            logits = forward_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            ).logits
            shift_logits = logits[:, :-1, :].float()
            shift_labels = labels[:, 1:]
            logprobs = torch.log_softmax(shift_logits, dim=-1)
            token_logprobs = logprobs.gather(
                dim=-1, index=shift_labels.clamp_min(0).unsqueeze(-1)).squeeze(-1)
            mask = (shift_labels != -100).to(token_logprobs.dtype)
            token_logprobs = token_logprobs * mask
        return token_logprobs, mask


    def _pad_token_id(self) -> int:
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id
        if pad_id is None:
            raise ValueError("Refiner tokenizer needs a pad or eos token")
        return int(pad_id)

    # ------------------------------------------------------------------
    # Adapter export
    # ------------------------------------------------------------------

    def save_adapter(self, output_dir: str) -> None:
        """Save the PEFT adapter (adapter_model.safetensors + config)."""
        import os

        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(output_dir)
        logger.info("Saved refiner adapter to %s", output_dir)

    def load_adapter(self, adapter_dir: str) -> None:
        """Load adapter weights from a PEFT directory into this role."""
        import os

        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file

        weights_path = os.path.join(adapter_dir, "adapter_model.safetensors")
        if not os.path.isfile(weights_path):
            raise FileNotFoundError(f"No adapter_model.safetensors in {adapter_dir}")
        set_peft_model_state_dict(self.model, load_file(weights_path))
        self.model.to(self.device)
        logger.info("Loaded refiner adapter from %s", adapter_dir)
