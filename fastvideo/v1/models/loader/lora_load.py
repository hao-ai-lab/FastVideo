from typing import (Any, Callable, DefaultDict, Dict, Generator, Hashable, List,
                    Optional, Tuple, Type, Union)

import torch
from torch import nn
from fastvideo.v1.utils import maybe_download_lora
import safetensors 

def load_lora_adapter(
        model: nn.Module,
        adapter_name_or_path: str) -> Dict[str, torch.Tensor]:
    """
    Load the lora weights into the model in-plance and return the adapter 
    weight state dict.
    Args:
        model: The (transformer) model to load the lora weights into
        adapter_name_or_path: The name or hf repo id of the lora weights
    """
    weight_path = maybe_download_lora(adapter_name_or_path)
    state_dict = safetensors.torch.load_file(weight_path)