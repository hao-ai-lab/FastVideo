import glob
from importlib import util
import os
from dataclasses import dataclass, field
from typing import Literal

import safetensors
import torch
from peft import LoraConfig, get_peft_model
from reward_model import Qwen2VLRewardModelBT
from transformers import AutoProcessor, TrainingArguments


@dataclass
class TrainingConfig(TrainingArguments):
    max_length: int | None = None
    dataset_num_proc: int | None = None
    center_rewards_coefficient: float | None = None
    disable_flash_attn2: bool = field(default=False)

    vision_lr: float | None = None
    merger_lr: float | None = None
    special_token_lr: float | None = None

    conduct_eval: bool | None = True
    load_from_pretrained: str = None
    load_from_pretrained_step: int = None
    logging_epochs: float | None = None
    eval_epochs: float | None = None
    save_epochs: float | None = None
    remove_unused_columns: bool | None = False

    save_full_model: bool | None = False


@dataclass
class PEFTLoraConfig:
    lora_enable: bool = False
    vision_lora: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list[str] | None = None
    lora_namespan_exclude: list[str] | None = None
    lora_modules_to_save: list[str] | None = None
    lora_task_type: str = "CAUSAL_LM"
    use_rslora: bool = False
    num_lora_modules: int = -1

    def __post_init__(self):
        if isinstance(self.lora_target_modules, list) and len(self.lora_target_modules) == 1:
            self.lora_target_modules = self.lora_target_modules[0]

        if isinstance(self.lora_namespan_exclude, list) and len(self.lora_namespan_exclude) == 1:
            self.lora_namespan_exclude = self.lora_namespan_exclude[0]


@dataclass
class ModelConfig:
    model_name_or_path: str | None = None
    model_revision: str = "main"

    output_dim: int = 1

    use_special_tokens: bool = False

    freeze_vision_tower: bool = field(default=False)
    freeze_llm: bool = field(default=False)
    tune_merger: bool = field(default=False)

    torch_dtype: Literal["auto", "bfloat16", "float16", "float32"] | None = None
    trust_remote_code: bool = False
    attn_implementation: str | None = None
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    bnb_4bit_quant_type: Literal["fp4", "nf4"] = "nf4"
    use_bnb_nested_quant: bool = False
    reward_token: Literal["last", "mean", "special"] = "last"
    loss_type: Literal["bt", "reg", "btt", "margin", "constant_margin", "scaled"] = "regular"

    def __post_init__(self):
        if self.load_in_8bit and self.load_in_4bit:
            raise ValueError("You can't use 8 bit and 4 bit precision at the same time")


def find_target_linear_names(model, num_lora_modules=-1, lora_namespan_exclude=None):
    linear_cls = torch.nn.Linear
    embedding_cls = torch.nn.Embedding
    excluded = lora_namespan_exclude or []
    lora_module_names = []

    for name, module in model.named_modules():
        if any(ex_keyword in name for ex_keyword in excluded):
            continue

        if isinstance(module, (linear_cls, embedding_cls)):
            lora_module_names.append(name)

    if num_lora_modules > 0:
        lora_module_names = lora_module_names[-num_lora_modules:]
    return lora_module_names


def _get_quantization_config(model_config):
    if not model_config.load_in_8bit and not model_config.load_in_4bit:
        return None
    from transformers import BitsAndBytesConfig

    if model_config.load_in_8bit:
        return BitsAndBytesConfig(load_in_8bit=True)
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=model_config.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=model_config.use_bnb_nested_quant,
    )


def create_model_and_processor(
    model_config,
    peft_lora_config,
    training_args,
    cache_dir=None,
):
    torch_dtype = (
        model_config.torch_dtype
        if model_config.torch_dtype in ["auto", None]
        else getattr(torch, model_config.torch_dtype)
    )
    quantization_config = _get_quantization_config(model_config)
    model_kwargs = dict(
        revision=model_config.model_revision,
        device_map="auto" if quantization_config is not None else None,
        quantization_config=quantization_config,
        use_cache=True if training_args.gradient_checkpointing else False,
    )

    processor = AutoProcessor.from_pretrained(
        model_config.model_name_or_path, padding_side="right", cache_dir=cache_dir
    )

    special_token_ids = None
    if model_config.use_special_tokens:
        special_tokens = ["<|VQ_reward|>", "<|MQ_reward|>", "<|TA_reward|>"]
        processor.tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
        special_token_ids = processor.tokenizer.convert_tokens_to_ids(special_tokens)

    has_flash_attn = util.find_spec("flash_attn") is not None
    model = Qwen2VLRewardModelBT.from_pretrained(
        model_config.model_name_or_path,
        output_dim=model_config.output_dim,
        reward_token=model_config.reward_token,
        special_token_ids=special_token_ids,
        torch_dtype=torch_dtype,
        attn_implementation=(
            "flash_attention_2" if not training_args.disable_flash_attn2 and has_flash_attn else "sdpa"
        ),
        cache_dir=cache_dir,
        **model_kwargs,
    )
    if model_config.use_special_tokens:
        model.resize_token_embeddings(len(processor.tokenizer))

    if training_args.bf16:
        model.to(torch.bfloat16)
    if training_args.fp16:
        model.to(torch.float16)

    if peft_lora_config.lora_enable:
        target_modules = find_target_linear_names(
            model,
            num_lora_modules=peft_lora_config.num_lora_modules,
            lora_namespan_exclude=peft_lora_config.lora_namespan_exclude,
        )
        peft_config = LoraConfig(
            target_modules=target_modules,
            r=peft_lora_config.lora_r,
            lora_alpha=peft_lora_config.lora_alpha,
            lora_dropout=peft_lora_config.lora_dropout,
            task_type=peft_lora_config.lora_task_type,
            use_rslora=peft_lora_config.use_rslora,
            bias="none",
            modules_to_save=peft_lora_config.lora_modules_to_save,
        )
        model = get_peft_model(model, peft_config)
    else:
        peft_config = None

    model.config.tokenizer_padding_side = processor.tokenizer.padding_side
    model.config.pad_token_id = processor.tokenizer.pad_token_id

    return model, processor, peft_config


def _insert_adapter_name_into_state_dict(
    state_dict: dict[str, torch.Tensor], adapter_name: str, parameter_prefix: str
) -> dict[str, torch.Tensor]:
    peft_model_state_dict = {}
    for key, val in state_dict.items():
        if parameter_prefix in key:
            suffix = key.split(parameter_prefix)[1]
            if "." in suffix:
                suffix_to_replace = ".".join(suffix.split(".")[1:])
                key = key.replace(suffix_to_replace, f"{adapter_name}.{suffix_to_replace}")
            else:
                key = f"{key}.{adapter_name}"
            peft_model_state_dict[key] = val
        else:
            peft_model_state_dict[key] = val
    return peft_model_state_dict


def load_model_from_checkpoint(model, checkpoint_dir, checkpoint_step):
    checkpoint_paths = glob.glob(os.path.join(checkpoint_dir, "checkpoint-*"))
    checkpoint_paths.sort(key=lambda x: int(x.split("-")[-1]), reverse=True)

    if checkpoint_step is None or checkpoint_step == -1:
        if checkpoint_paths:
            checkpoint_path = checkpoint_paths[0]
        else:
            checkpoint_path = checkpoint_dir
    else:
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint-{checkpoint_step}")
        if checkpoint_path not in checkpoint_paths:
            checkpoint_path = checkpoint_paths[0] if checkpoint_paths else checkpoint_dir

    checkpoint_step = checkpoint_path.split("checkpoint-")[-1].split("/")[0]

    full_ckpt = os.path.join(checkpoint_path, "model.pth")
    lora_ckpt = os.path.join(checkpoint_path, "adapter_model.safetensors")
    non_lora_ckpt = os.path.join(checkpoint_path, "non_lora_state_dict.pth")
    if os.path.exists(full_ckpt):
        model_state_dict = torch.load(full_ckpt, map_location="cpu")
        model.load_state_dict(model_state_dict)
    else:
        lora_state_dict = safetensors.torch.load_file(lora_ckpt)
        non_lora_state_dict = torch.load(non_lora_ckpt, map_location="cpu")

        lora_state_dict = _insert_adapter_name_into_state_dict(
            lora_state_dict, adapter_name="default", parameter_prefix="lora_"
        )

        model_state_dict = model.state_dict()
        model_state_dict.update(non_lora_state_dict)
        model_state_dict.update(lora_state_dict)
        model.load_state_dict(model_state_dict)

    return model, checkpoint_step
