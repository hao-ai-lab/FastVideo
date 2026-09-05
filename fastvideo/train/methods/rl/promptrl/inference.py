# SPDX-License-Identifier: Apache-2.0
"""PromptRL inference-time prompt refiner.

Loads a PromptRL bundle (``manifest.json`` + ``refiner/`` PEFT adapter)
and exposes ``refine(prompt)`` returning the original prompt, the
refined prompt (with fallback to the original on malformed output), the
raw completion, and the format-valid flag.  This helper intentionally
stays separate from ``VideoGenerator`` — generation behavior is
unchanged in v1.
"""

from __future__ import annotations

import contextlib
import os
from dataclasses import dataclass
from typing import Any

import torch

from fastvideo.logger import init_logger
from fastvideo.train.methods.rl.promptrl.bundle import (
    BundleManifest,
    load_bundle_manifest,
)
from fastvideo.train.methods.rl.promptrl.prompts import (
    parse_answer_tag,
    render_refinement_prompt,
)

logger = init_logger(__name__)


@dataclass(frozen=True, slots=True)
class RefinementResult:
    """Output of one prompt refinement."""

    original_prompt: str
    refined_prompt: str
    raw_completion: str
    format_valid: bool


class PromptRefiner:
    """Inference-time prompt refiner loaded from a PromptRL bundle."""

    def __init__(
        self,
        *,
        manifest: BundleManifest,
        model: torch.nn.Module,
        tokenizer: Any,
        device: str | torch.device = "cuda",
    ) -> None:
        self.manifest = manifest
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()

    @classmethod
    def from_bundle(
        cls,
        bundle_dir: str,
        *,
        device: str | torch.device = "cuda",
        torch_dtype: str = "bfloat16",
        model: torch.nn.Module | None = None,
        tokenizer: Any = None,
    ) -> PromptRefiner:
        """Load the refiner adapter of a PromptRL bundle.

        ``model``/``tokenizer`` may be injected directly (tests); by
        default the base model named in the manifest is loaded and the
        ``refiner/`` PEFT adapter applied on top.
        """
        manifest = load_bundle_manifest(bundle_dir)
        refiner_dir = os.path.join(bundle_dir, "refiner")
        if not os.path.isdir(refiner_dir):
            raise FileNotFoundError(f"Bundle {bundle_dir} has no refiner/ directory")

        if tokenizer is None:
            from transformers import AutoTokenizer

            tokenizer_source = (refiner_dir if os.path.exists(
                os.path.join(refiner_dir, "tokenizer_config.json")) else manifest.base_refiner_model)
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)

        if model is None:
            import transformers
            from peft import PeftModel

            dtype = getattr(torch, torch_dtype)
            model_cls = getattr(transformers, "Qwen2_5_VLForConditionalGeneration", None)
            if model_cls is None:
                model_cls = transformers.AutoModelForCausalLM
            base = model_cls.from_pretrained(manifest.base_refiner_model, torch_dtype=dtype)
            model = PeftModel.from_pretrained(base, refiner_dir)

        return cls(manifest=manifest, model=model, tokenizer=tokenizer, device=device)


    # ------------------------------------------------------------------

    @torch.no_grad()
    def refine(
        self,
        prompt: str,
        *,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        seed: int | None = None,
    ) -> RefinementResult:
        """Refine *prompt* with the bundled adapter.

        Malformed completions (missing ``<answer>...</answer>``) fall
        back to the original prompt, mirroring training-time behavior.
        """
        sampling = self.manifest.refiner_sampling
        max_new_tokens = int(max_new_tokens or sampling.get("max_new_tokens", 256))
        temperature = float(temperature if temperature is not None else sampling.get("temperature", 1.0))
        top_p = float(top_p if top_p is not None else sampling.get("top_p", 1.0))

        instruction = render_refinement_prompt(
            prompt, template_version=self.manifest.prompt_template_version)
        messages = [{"role": "user", "content": instruction}]
        apply_template = getattr(self.tokenizer, "apply_chat_template", None)
        if callable(apply_template):
            rendered = apply_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            rendered = instruction + "\n"
        encoded = self.tokenizer(rendered, return_tensors="pt").to(self.device)

        sample = temperature > 0.0
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
                max_new_tokens=max_new_tokens,
                do_sample=sample,
                temperature=temperature if sample else None,
                top_p=top_p if sample else None,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )
        completion_ids = output[:, encoded["input_ids"].shape[1]:]
        raw_completion = self.tokenizer.batch_decode(
            completion_ids, skip_special_tokens=True)[0]

        parsed = parse_answer_tag(raw_completion)
        refined = parsed.refined_prompt if parsed.format_valid else prompt
        return RefinementResult(
            original_prompt=prompt,
            refined_prompt=refined,
            raw_completion=raw_completion,
            format_valid=parsed.format_valid,
        )
