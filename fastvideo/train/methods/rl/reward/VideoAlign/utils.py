import os
import glob
from dataclasses import dataclass, field
from typing import List, Literal, Optional

import safetensors
import torch
from transformers import TrainingArguments

########## DataClass For Configure ##########

@dataclass
class TrainingConfig(TrainingArguments):
    max_length: Optional[int] = None
    dataset_num_proc: Optional[int] = None
    center_rewards_coefficient: Optional[float] = None
    disable_flash_attn2: bool = field(default=False)

    vision_lr: Optional[float] = None
    merger_lr: Optional[float] = None
    special_token_lr: Optional[float] = None

    conduct_eval: Optional[bool] = True
    load_from_pretrained: str = None
    load_from_pretrained_step: int = None
    logging_epochs: Optional[float] = None
    eval_epochs: Optional[float] = None
    save_epochs: Optional[float] = None
    remove_unused_columns: Optional[bool] = False

    save_full_model: Optional[bool] = False

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
        if isinstance(self.lora_target_modules, list) and len(self.lora_target_modules) == 1:
            self.lora_target_modules = self.lora_target_modules[0]

        if isinstance(self.lora_namespan_exclude, list) and len(self.lora_namespan_exclude) == 1:
            self.lora_namespan_exclude = self.lora_namespan_exclude[0]

@dataclass
class ModelConfig:
    model_name_or_path: Optional[str] = None
    model_revision: str = "main"

    output_dim: int = 1

    use_special_tokens: bool = False

    freeze_vision_tower: bool = field(default=False)
    freeze_llm: bool = field(default=False)
    tune_merger: bool = field(default=False)

    torch_dtype: Optional[Literal["auto", "bfloat16", "float16", "float32"]] = None
    trust_remote_code: bool = False
    attn_implementation: Optional[str] = None
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    bnb_4bit_quant_type: Literal["fp4", "nf4"] = "nf4"
    use_bnb_nested_quant: bool = False
    reward_token: Literal["last", "mean", "special"] = "last"
    loss_type: Literal["bt", "reg", "btt", "margin", "constant_margin", "scaled"] = "regular"

    def __post_init__(self):
        if self.load_in_8bit and self.load_in_4bit:
            raise ValueError("You can't use 8 bit and 4 bit precision at the same time")

        # if isinstance(self.lora_target_modules, list) and len(self.lora_target_modules) == 1:
        #     self.lora_target_modules = self.lora_target_modules[0]

        # if isinstance(self.lora_namespan_exclude, list) and len(self.lora_namespan_exclude) == 1:
        #     self.lora_namespan_exclude = self.lora_namespan_exclude[0]

########## Functions for get trainable modules' parameters ##########

def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param

# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return

def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return

########## Load Models From Folder ##########

def _insert_adapter_name_into_state_dict(
    state_dict: dict[str, torch.Tensor], adapter_name: str, parameter_prefix: str
) -> dict[str, torch.Tensor]:
    """Utility function to remap the state_dict keys to fit the PEFT model by inserting the adapter name."""
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


def save_video(tensor, path):
    from torchvision.io import write_video
    tensor = tensor * 255.0
    tensor = tensor.permute(0, 2, 3, 1)
    tensor = tensor.clamp(0, 255).byte()
    write_video(path, tensor, 4, video_codec='h264')


def load_model_from_checkpoint(
    model, checkpoint_dir, checkpoint_step
):
    checkpoint_paths = glob.glob(os.path.join(checkpoint_dir, "checkpoint-*"))
    checkpoint_paths.sort(key=lambda x: int(x.split("-")[-1]), reverse=True)

    if checkpoint_step is None or checkpoint_step == -1:
        # get the latest checkpoint
        checkpoint_path = checkpoint_paths[0]
        print(f"===> Checkpoint step is not provided, using the latest checkpoint: {checkpoint_path}")
    else:
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint-{checkpoint_step}")
        if checkpoint_path not in checkpoint_paths:
            checkpoint_path = checkpoint_paths[0]
            print(f"===> Checkpoint step {checkpoint_step} not found, using the latest checkpoint: {checkpoint_path}")
        else:
            print(f"===> Checkpoint step {checkpoint_step} found, using the specified checkpoint: {checkpoint_path}")
    
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

        lora_state_dict = _insert_adapter_name_into_state_dict(lora_state_dict, adapter_name="default", parameter_prefix="lora_")
        
        model_state_dict = model.state_dict()
        model_state_dict.update(non_lora_state_dict)
        model_state_dict.update(lora_state_dict)
        model.load_state_dict(model_state_dict)

    return model, checkpoint_step