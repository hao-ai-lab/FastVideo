# SPDX-License-Identifier: Apache-2.0
"""Reason1 (Qwen2.5-VL) text encoder."""

import os
from dataclasses import dataclass
from collections.abc import Iterable

import torch
from transformers import AutoProcessor

from fastvideo.configs.models.encoders import BaseEncoderOutput, Reason1Config
from fastvideo.logger import init_logger
from fastvideo.models.encoders.base import TextEncoder
from fastvideo.models.loader.weight_utils import default_weight_loader
from fastvideo.platforms import AttentionBackendEnum

from fastvideo.models.encoders.qwen2_5_vl_custom import (
    Qwen2_5_VLForConditionalGenerationSimple,
    Qwen2_5_VLConfig,
    get_rope_index,
)

logger = init_logger(__name__)


@dataclass(frozen=True)
class _WeightsSource:
    """Mimic `TextEncoderLoader.Source` (avoid import cycles)."""

    model_or_path: str
    prefix: str = ""
    fall_back_to_pt: bool = True
    allow_patterns_overrides: list[str] | None = None




class Reason1TextEncoder(TextEncoder):
    """Reason1 (Qwen2.5-VL) text encoder."""

    _supported_attention_backends: tuple[AttentionBackendEnum, ...] = (
        AttentionBackendEnum.FLASH_ATTN,
        AttentionBackendEnum.TORCH_SDPA,
    )
    
    def __init__(self, config: Reason1Config, prefix: str = "", checkpoint_path: str | None = None):
        super().__init__(config)
        
        self.prefix = prefix
        self.quant_config = None  # For future quantization support
        
        self.embedding_concat_strategy = config.arch_config.embedding_concat_strategy
        self.n_layers_per_group = config.arch_config.n_layers_per_group
        self.num_embedding_padding_tokens = config.arch_config.num_embedding_padding_tokens
        
        config_path = checkpoint_path if checkpoint_path else config.tokenizer_type
        
        logger.info("Initializing Reason1TextEncoder (Qwen2.5-VL) from %s", config_path)
        try:
            from transformers import AutoConfig as HFAutoConfig
            hf_config = HFAutoConfig.from_pretrained(
                config_path,
                trust_remote_code=True,
            )
        except Exception as e:
            logger.warning("Failed to load HF config from %s (%s). Using default Qwen2.5-VL-7B config.",
                           config_path, e)
            hf_config = Qwen2_5_VLConfig(
                    hidden_size=3584,
                    intermediate_size=18944,
                    max_window_layers=28,
                    num_attention_heads=28,
                    num_hidden_layers=28,
                    num_key_value_heads=4,
                    tie_word_embeddings=False,
                    vocab_size=152064,
            )
        
        hf_config.output_hidden_states = True
        
        if hasattr(config.arch_config, '_attn_implementation') and config.arch_config._attn_implementation:
            hf_config._attn_implementation = config.arch_config._attn_implementation
        else:
            hf_config._attn_implementation = "flash_attention_2"
        logger.info("Reason1 attention implementation: %s", getattr(hf_config, "_attn_implementation", None))
        
        with torch.device("meta"):
            self.model = Qwen2_5_VLForConditionalGenerationSimple(hf_config)
        
        self.processor = AutoProcessor.from_pretrained(
            config_path,
            trust_remote_code=True,
        )

        weights_override = os.getenv("FASTVIDEO_REASON1_WEIGHTS_PATH")
        if weights_override:
            self.secondary_weights = (
                _WeightsSource(
                    model_or_path=weights_override,
                    prefix="",
                    fall_back_to_pt=True,
                    allow_patterns_overrides=None,
                ),
            )
            logger.info("Reason1TextEncoder: overlaying weights from %s", weights_override)
        
        self._weights_loaded = False
    
    def forward(
        self,
        input_ids: torch.Tensor | None,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        output_hidden_states: bool | None = None,
        **kwargs,
    ) -> BaseEncoderOutput:
        # Cosmos2.5 alignment: keep attention_mask=None.
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=None,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_hidden_states=True,
            return_dict=True,
            pixel_values=kwargs.get('pixel_values'),
            pixel_values_videos=kwargs.get('pixel_values_videos'),
            image_grid_thw=kwargs.get('image_grid_thw'),
            video_grid_thw=kwargs.get('video_grid_thw'),
        )
        
        hidden_states = outputs.hidden_states
        last_hidden_state = hidden_states[-1]
        
        return BaseEncoderOutput(
            last_hidden_state=last_hidden_state,
            hidden_states=hidden_states if output_hidden_states else None,
            attention_mask=None,
        )
    
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        first_weight = None
        weights_list = []
        for name, weight in weights:
            if first_weight is None:
                first_weight = weight
                self.model = self.model.to_empty(device=weight.device)
                self.model.init_weights(buffer_device=weight.device)
            weights_list.append((name, weight))
        
        params_dict = dict(self.model.named_parameters())
        loaded_params: set[str] = set()
        skipped_weights = {"lm_head": 0, "visual": 0, "decoder": 0}
        
        for name, loaded_weight in weights_list:
            if "lm_head" in name:
                skipped_weights["lm_head"] += 1
                continue
            if "visual" in name:
                skipped_weights["visual"] += 1
                continue
            if "decoder" in name:
                skipped_weights["decoder"] += 1
                continue
            
            # Handle stacked params mapping (for quantized models)
            for param_name, weight_name, shard_id in self.config.arch_config.stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                if name.endswith(".bias") and name not in params_dict:
                    continue
                
                if name not in params_dict:
                    continue
                
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                param_name_with_prefix = f"model.{name}" if self.prefix == "" else f"{self.prefix}.{name}"
                loaded_params.add(param_name_with_prefix)
                break
            else:
                if name.endswith(".bias") and name not in params_dict:
                    continue
                
                if name not in params_dict:
                    continue
                
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                param_name_with_prefix = f"model.{name}" if self.prefix == "" else f"{self.prefix}.{name}"
                loaded_params.add(param_name_with_prefix)
        if first_weight is not None:
            self.model = self.model.to(first_weight.device)
        
        all_params = set(f"model.{name}" if self.prefix == "" else f"{self.prefix}.{name}" 
                        for name in params_dict)
        loaded_params.update(all_params)
        
        # Mark weights as loaded
        self._weights_loaded = True
        return loaded_params
    
    def compute_text_embeddings_online(
        self,
        data_batch: dict[str, list[str]],
        input_caption_key: str,
    ) -> torch.Tensor:
        prompts = data_batch[input_caption_key]
        return self.compute_text_embeddings(prompts)
    
    def compute_text_embeddings(
        self,
        prompts: list[str],
        device: str | torch.device = "cuda",
    ) -> torch.Tensor:
        """Compute embeddings for a list of prompts."""
        input_ids_batch = []
        
        tok = getattr(self.processor, "tokenizer", None)
        if tok is None:
            raise RuntimeError("Reason1TextEncoder requires processor.tokenizer")
        pad_id = getattr(tok, "pad_id", None)
        if pad_id is None:
            pad_id = getattr(tok, "pad_token_id", None)
        if pad_id is None:
            pad_id = getattr(self.model.config, "pad_token_id", None)
        if pad_id is None:
            pad_id = 0
        
        for prompt in prompts:
            conversations = [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": "You are a helpful assistant who will provide prompts to an image generator.",
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt,
                        }
                    ],
                },
            ]
            
            try:
                tokenizer_output = tok.apply_chat_template(
                    conversations,
                    tokenize=True,
                    add_generation_prompt=False,
                    add_vision_id=False,
                )
            except TypeError:
                tokenizer_output = tok.apply_chat_template(
                conversations,
                tokenize=True,
                add_generation_prompt=False,
            )

            if isinstance(tokenizer_output, dict) and "input_ids" in tokenizer_output:
                input_ids = tokenizer_output["input_ids"]
                if hasattr(input_ids, "tolist"):
                    input_ids = input_ids.tolist()
            else:
                input_ids = tokenizer_output
                if hasattr(input_ids, "tolist"):
                    input_ids = input_ids.tolist()
                if isinstance(input_ids, list) and len(input_ids) == 1 and isinstance(
                        input_ids[0], list):
                    input_ids = input_ids[0]
                if not isinstance(input_ids, list):
                    raise RuntimeError(
                        f"Unexpected chat_template output type: {type(tokenizer_output)}"
                    )
            
            if self.num_embedding_padding_tokens > len(input_ids):
                pad_len = self.num_embedding_padding_tokens - len(input_ids)
                input_ids = input_ids + [pad_id] * pad_len
            else:
                input_ids = input_ids[:self.num_embedding_padding_tokens]
            
            input_ids = torch.LongTensor(input_ids).to(device=device)
            input_ids_batch.append(input_ids)
        
        input_ids_batch = torch.stack(input_ids_batch, dim=0)

        # Cosmos2.5 alignment: keep attention_mask=None.
        target_device = input_ids_batch.device
        try:
            embed_device = self.model.model.embed_tokens.weight.device  # type: ignore[attr-defined]
        except Exception:
            embed_device = None
        if embed_device is not None and embed_device != target_device:
            self.model = self.model.to(target_device)
        
        with torch.no_grad():
            position_ids, _ = get_rope_index(
                self.model.config,
                input_ids_batch,
                image_grid_thw=None,
                video_grid_thw=None,
                second_per_grid_ts=None,
                attention_mask=None,
            )
            position_ids = position_ids.to(target_device)
                
            outputs = self.model.model(
                input_ids=input_ids_batch,
                position_ids=position_ids,
                attention_mask=None,
                output_hidden_states=True,
                return_dict=True,
                use_cache=False,
            )
            hidden_states = outputs.hidden_states
        
        normalized_hidden_states = []
        for layer_idx in range(1, len(hidden_states)):
            normalized_state = self._mean_normalize(hidden_states[layer_idx])
            normalized_hidden_states.append(normalized_state)
        
        if self.embedding_concat_strategy == "full_concat":
            text_embeddings = torch.cat(normalized_hidden_states, dim=-1)
        elif self.embedding_concat_strategy == "mean_pooling":
            text_embeddings = torch.stack(normalized_hidden_states).mean(dim=0)
        elif self.embedding_concat_strategy == "pool_every_n_layers_and_concat":
            pooled_embeddings = []
            for i in range(0, len(normalized_hidden_states), self.n_layers_per_group):
                group = normalized_hidden_states[i : i + self.n_layers_per_group]
                pooled = torch.stack(group).mean(dim=0)
                pooled_embeddings.append(pooled)
            text_embeddings = torch.cat(pooled_embeddings, dim=-1)
        else:
            raise ValueError(
                f"Unknown embedding_concat_strategy: {self.embedding_concat_strategy}"
            )
        
        return text_embeddings
    @staticmethod
    def _mean_normalize(tensor: torch.Tensor) -> torch.Tensor:
        return (tensor - tensor.mean(dim=-1, keepdim=True)) / (
            tensor.std(dim=-1, keepdim=True) + 1e-8
        )

# Entry point for model registry
EntryClass = Reason1TextEncoder
