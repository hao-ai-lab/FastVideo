from typing import (Any, Callable, DefaultDict, Dict, Generator, Hashable, List,
                    Optional, Tuple, Type, Union)

import torch
from torch import nn

def load_lora_adapter(
        model: nn.Module,
        adapter_path: str) -> Dict[str, torch.Tensor]:
    """
    Load the lora weights into the model in-plance and return the adapter 
    weight state dict.
    """
    adapter = torch.load(adapter_path)
    raise NotImplementedError("TODO")