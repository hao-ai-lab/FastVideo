import inspect
import math
from typing import Callable, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn
from einops import rearrange, repeat

from fastvideo.utils.parallel_states import get_sequence_parallel_state, nccl_info

class NewMochiRoPE(nn.Module):
    def _get_positions(
        self,
        num_frames: int,
        height: int,
        width: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        scale = (self.target_area / (height * width)) ** 0.5
        t = torch.arange(num_frames * nccl_info.world_size, device=device, dtype=dtype)
        h = self._centers(-height * scale / 2, height * scale / 2, height, device, dtype)
        w = self._centers(-width * scale / 2, width * scale / 2, width, device, dtype)

        grid_t, grid_h, grid_w = torch.meshgrid(t, h, w, indexing="ij")

        positions = torch.stack([grid_t, grid_h, grid_w], dim=-1).view(-1, 3)
        return positions