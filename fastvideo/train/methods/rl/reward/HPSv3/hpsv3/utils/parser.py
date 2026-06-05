import sys
import yaml
from pathlib import Path
from typing import Any, Optional, Union, Tuple, List, Literal
from omegaconf import OmegaConf
from transformers import HfArgumentParser
from dataclasses import dataclass, field
from transformers import TrainingArguments

@dataclass
class DataConfig:
    train_json_list: List[str] = field(default_factory=lambda: ["/path/to/dataset/meta_data.json"])
    val_json_list: List[str] = field(default_factory=lambda: ["/path/to/dataset/meta_data.json"])
    test_json_list: List[str] = field(default_factory=lambda: ["/path/to/dataset/meta_data.json"])
    soft_label: bool = False
    confidence_threshold: Optional[float] = None
    max_pixels: Optional[int] = 256 * 28 * 28  # Default max pixels
    min_pixels: Optional[int] = 256 * 28 * 28
    with_instruction: bool = True
    tied_threshold: Optional[float] = None

@dataclass
class TrainingConfig(TrainingArguments):
    max_grad_norm: Optional[float] = 1.0
    dataset_num_proc: Optional[int] = None
    center_rewards_coefficient: Optional[float] = None
    disable_flash_attn2: bool = field(default=False)
    disable_dropout: bool = field(default=False)

    vision_lr: Optional[float] = None
    merger_lr: Optional[float] = None
    rm_head_lr: Optional[float] = None
    special_token_lr: Optional[float] = None

    conduct_eval: Optional[bool] = True
    load_from_pretrained: str = None
    load_from_pretrained_step: int = None
    logging_epochs: Optional[float] = None
    eval_epochs: Optional[float] = None
    save_epochs: Optional[float] = None
    remove_unused_columns: Optional[bool] = False

    save_full_model: Optional[bool] = False
    
    # Visualization parameters
    visualization_steps: Optional[int] = 100
    max_viz_samples: Optional[int] = 4

@dataclass
class PEFTLoraConfig:
    lora_enable: bool = False
    vision_lora: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: Optional[List[str]] = None
    lora_namespan_exclude: Optional[List[str]] = None
    lora_modules_to_save: Optional[List[str]] = None
    lora_task_type: str = "CAUSAL_LM"
    use_rslora: bool = False
    num_lora_modules: int = -1

    def __post_init__(self):
        if (
            isinstance(self.lora_target_modules, list)
            and len(self.lora_target_modules) == 1
        ):
            self.lora_target_modules = self.lora_target_modules[0]

        if (
            isinstance(self.lora_namespan_exclude, list)
            and len(self.lora_namespan_exclude) == 1
        ):
            self.lora_namespan_exclude = self.lora_namespan_exclude[0]


@dataclass
class ModelConfig:
    model_name_or_path: Optional[str] = None
    model_revision: str = "main"
    rm_head_type: str = "default"
    rm_head_kwargs: Optional[dict] = None
    output_dim: int = 1

    use_special_tokens: bool = False

    freeze_vision_tower: bool = field(default=False)
    freeze_llm: bool = field(default=False)
    tune_merger: bool = field(default=False)
    trainable_visual_layers: Optional[int] = -1

    torch_dtype: Optional[Literal["auto", "bfloat16", "float16", "float32"]] = None
    trust_remote_code: bool = False
    attn_implementation: Optional[str] = None
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    bnb_4bit_quant_type: Literal["fp4", "nf4"] = "nf4"
    use_bnb_nested_quant: bool = False
    reward_token: Literal["last", "mean", "special"] = "last"
    loss_type: Literal["bt", "reg", "btt", "margin", "constant_margin", "scaled"] = (
        "regular"
    )
    loss_hyperparameters: dict = field(default_factory=lambda: {})
    checkpoint_path: Optional[str] = None
    
    def __post_init__(self):
        if self.load_in_8bit and self.load_in_4bit:
            raise ValueError("You can't use 8 bit and 4 bit precision at the same time")

        # if isinstance(self.lora_target_modules, list) and len(self.lora_target_modules) == 1:
        #     self.lora_target_modules = self.lora_target_modules[0]

        # if isinstance(self.lora_namespan_exclude, list) and len(self.lora_namespan_exclude) == 1:
        #     self.lora_namespan_exclude = self.lora_namespan_exclude[0]


########## Functions for get trainable modules' parameters ##########

def parse_args_with_yaml(
    dataclass_types: Tuple[type, ...], 
    config_path: str = None,
    allow_extra_keys: bool = True,
    is_train: bool = True,
) -> Tuple[Any, ...]:
    """
    Parse arguments using HfArgumentParser with OmegaConf for YAML support.
    
    Args:
        dataclass_types: Tuple of dataclass types for HfArgumentParser
        args: Optional arguments (if None, will read from sys.argv)
        allow_extra_keys: Whether to allow extra keys in config
    
    Returns:
        Tuple of parsed dataclass instances
    """
    # Read arguments from command line or provided args
    # Load YAML config and merge with command line overrides
    args = OmegaConf.to_container(OmegaConf.load(config_path))
    if not is_train:
        args.pop('deepspeed', None)

    # Parse with HfArgumentParser
    parser = HfArgumentParser(dataclass_types)
    return parser.parse_dict(args, allow_extra_keys=allow_extra_keys), config_path


if __name__ == "__main__":
    data_config, training_args, model_config, peft_lora_config = parse_args_with_yaml(
        (DataConfig, TrainingConfig, ModelConfig, PEFTLoraConfig)
    )