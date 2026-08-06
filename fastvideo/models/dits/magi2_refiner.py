# Copyright (c) 2026 SandAI. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""MAGI-2 super-resolution model."""

import importlib
import logging
import math
import os
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import (
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
)

import torch
import torch.nn as nn
from einops import rearrange, repeat
from flash_attn.layers.rotary import apply_rotary_emb
from magi_compiler import magi_compile
from magi_compiler.api import magi_register_custom_op
from magi_compiler.config import CompileConfig
from torch import Tensor
from torch.nn import Parameter

from fastvideo.configs.models.dits.magi2 import Magi2RefinerArchConfig, Magi2RefinerVideoConfig
from fastvideo.models.dits.base import BaseDiT
from fastvideo.models.dits.magi2_runtime import psm
from fastvideo.models.dits.magi2_runtime.all_to_all_primitive import (
    batch_scatter_head_gather_seqlen,
    scatter_seqlen_gather_head,
)
from fastvideo.models.dits.magi2_runtime.context_parallel import ulysses_scheduler


logger = logging.getLogger(__name__)

ModelConfig = Magi2RefinerArchConfig


def _resolve_params_dtype(value: torch.dtype | str) -> torch.dtype:
    """Convert a serialized MAGI-2 refiner parameter dtype into a PyTorch dtype."""
    if isinstance(value, torch.dtype):
        return value
    dtype_by_name = {
        "bfloat16": torch.bfloat16,
        "torch.bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "torch.float16": torch.float16,
        "float32": torch.float32,
        "torch.float32": torch.float32,
    }
    try:
        return dtype_by_name[value]
    except KeyError as error:
        raise ValueError(f"Unsupported MAGI-2 refiner parameter dtype: {value!r}") from error

# ============================================================
# _merge_enum
# ============================================================


# Define the MLP activation type
class MLPActivationType(Enum):
    """Enumeration of supported activation functions for MLP"""

    SWIGLU7 = "swiglu7"


class Modality(IntEnum):
    VIDEO = 0
    AUDIO = 1
    TEXT = 2


# ============================================================
# _merge_activation
# ============================================================


def swiglu7(
    x, alpha: float = 1.702, limit: float = 7.0, out_dtype: Optional[torch.dtype] = None
):
    """Apply the clamped SwiGLU7 activation and cast to the requested dtype."""
    out_dtype = x.dtype if out_dtype is None else out_dtype
    x = x.to(torch.float32)
    x_glu, x_linear = x[..., ::2], x[..., 1::2]
    # Clamp the input values
    x_glu = x_glu.clamp(min=None, max=limit)
    x_linear = x_linear.clamp(min=-limit, max=limit)
    out_glu = x_glu * torch.sigmoid(alpha * x_glu)
    # Note we add an extra bias of 1 to the linear layer (from GPT-OSS)
    return (out_glu * (x_linear + 1)).to(out_dtype)


def create_activation_func(activation_type: MLPActivationType) -> Callable:
    if activation_type is MLPActivationType.SWIGLU7:
        return swiglu7
    raise ValueError(f"Unknown activation type: {activation_type}")


# ============================================================
# _merge_schemas
# ============================================================


@dataclass
class FFAHandler:
    q_ranges: torch.Tensor
    k_ranges: torch.Tensor
    max_seqlen_q: int
    max_seqlen_k: int
    attn_type_map: torch.Tensor
    softmax_scale: Optional[float] = None
    auto_range_merge: bool = False
    sparse_load: bool = False


@dataclass
class MoEConfig:
    num_experts: int = 1
    num_shared_experts: int = 0
    type: Literal["token_choice", "expert_choice", "video_only_expert_choice"] = (
        "token_choice"
    )
    score_func: Literal["softmax", "sigmoid"] = "softmax"
    route_norm: bool = False
    route_scale: float = 1.0
    top_k: Optional[int] = None  # for token_choice
    capacity_factor: Optional[int] = (
        None  # for expert_choice and video_only_expert_choice
    )
    moe_layers: list[int] = field(default_factory=list)
    with_zero_experts: bool = True

    def __post_init__(self):
        if self.top_k is not None and self.capacity_factor is not None:
            raise ValueError("top_k and capacity_factor cannot be set at the same time")


# ============================================================
# _merge_modality_dispatcher
# ============================================================


class ModalityDispatcher:
    permuted_modality_mapping: torch.Tensor
    group_size: torch.Tensor
    num_modalities: int

    @torch._dynamo.disable
    def __init__(self, modality_mapping: torch.Tensor, num_modalities: int):
        """
        Initialize the dispatcher.
        Runs once at creation and precomputes all required mappings.
        """
        self.modality_mapping = modality_mapping
        self.num_modalities = num_modalities

        self.permuted_modality_mapping = self._precompute_permute_mapping(
            modality_mapping
        )

        self.group_size = torch.bincount(
            self.permuted_modality_mapping, minlength=num_modalities
        ).to(torch.int32)
        group_size_cpu = [int(x) for x in self.group_size.to("cpu").tolist()]

        # Carrier tensor: shape encodes per-modality token counts. Only the
        # shape is ever read, so it lives on meta and owns no storage --
        # a real tensor would allocate num_video * num_text elements.
        # tolist() above triggers a graph break when called inside
        # @torch.compile, so this code always runs in eager.
        # mark_unbacked prevents Dynamo from unifying symbols when some
        # modalities have 0 tokens (e.g. total == video when text=0).
        self._size_carrier = torch.empty(group_size_cpu, device="meta")
        if not torch.compiler.is_compiling():
            for i in range(num_modalities):
                torch._dynamo.decorators.mark_unbacked(self._size_carrier, i)

    @property
    def group_size_cpu(self) -> list[int]:
        return [self._size_carrier.shape[i] for i in range(self.num_modalities)]

    def _precompute_permute_mapping(self, modality_mapping):
        """Cache permutations that group tokens by modality."""
        # 1. Compute forward and inverse permutation mappings
        # argsort is efficient O(N log N)
        self.permute_mapping = torch.argsort(modality_mapping)
        self.inv_permute_mapping = torch.argsort(self.permute_mapping)

        # 2. Compute group sizes
        # bincount is very efficient counting
        permuted_modality_mapping = modality_mapping[self.permute_mapping]

        return permuted_modality_mapping

    def dispatch(self, x: torch.Tensor) -> list[torch.Tensor]:
        grouped_tensors = torch.split(x, self.group_size_cpu, dim=0)
        return list(grouped_tensors)

    def undispatch(self, *processed_groups: list[torch.Tensor]) -> torch.Tensor:
        return torch.cat(processed_groups, dim=0)

    @staticmethod
    def permute(x: torch.Tensor, permute_mapping: torch.Tensor) -> torch.Tensor:
        """Apply forward permutation to tensor."""
        return x[permute_mapping]

    @staticmethod
    def inv_permute(x: torch.Tensor, inv_permute_mapping: torch.Tensor) -> torch.Tensor:
        """Apply inverse permutation to tensor."""
        return x[inv_permute_mapping]


# ============================================================
# _merge__common
# ============================================================

DYNAMIC_LORA_RANK = 0


def register_lora_module(cls_or_module):
    return cls_or_module


class MultiModalityRMSNorm(nn.Module):
    __constants__ = ["dim", "eps", "num_modality"]
    dim: int
    eps: float
    num_modality: int

    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
        device: torch.device | None = None,
        num_modality: int = 1,
    ):
        """Initialize modality-specific root-mean-square scaling weights."""
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.num_modality = num_modality

        self.weight = torch.nn.Parameter(
            torch.zeros(dim * num_modality, device=device, dtype=torch.float32)
        )
        if num_modality > 1:
            self.forward = self.forward_multi_experts
        else:
            self.forward = self.forward_single_expert

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.weight)

    def rms(self, x: torch.Tensor) -> torch.Tensor:
        t = x.float()
        t = t * torch.rsqrt(torch.mean(t**2, dim=-1, keepdim=True) + self.eps)
        return t

    def forward_multi_experts(
        self, x: torch.Tensor, modality_dispatcher: ModalityDispatcher
    ) -> torch.Tensor:
        """Normalize tokens and apply the scale for each modality group."""
        original_dtype = x.dtype
        t = self.rms(x)

        weight_chunked = self.weight.chunk(self.num_modality, dim=0)
        t_list = modality_dispatcher.dispatch(t)
        for i in range(self.num_modality):
            t_list[i] = t_list[i] * (weight_chunked[i] + 1)
        t = modality_dispatcher.undispatch(*t_list)

        return t.to(original_dtype)

    def forward_single_expert(
        self, x: torch.Tensor, modality_dispatcher: Optional[ModalityDispatcher] = None
    ) -> torch.Tensor:
        t, original_dtype = x.float(), x.dtype
        t = t * torch.rsqrt(torch.mean(t**2, dim=-1, keepdim=True) + self.eps)
        return (t * (self.weight + 1)).to(original_dtype)


def _bf16_compute_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    output_dtype: Optional[torch.dtype],
    compute_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Linear forward in bf16 compute precision, cast output to output_dtype."""
    input_cast = input.to(compute_dtype)
    weight_cast = weight.to(compute_dtype)
    output = torch.matmul(input_cast, weight_cast.t())
    if bias is not None:
        output = output + bias.to(compute_dtype)
    return output.to(output_dtype)


@register_lora_module
class BaseLinear(nn.Module):
    __constants__ = ["in_features", "out_features", "num_layers", "num_experts"]
    in_features: int
    out_features: int
    num_layers_for_initialization: int
    num_experts: int
    weight: Tensor

    def __init__(
        self,
        in_features,
        out_features,
        num_layers_for_initialization,
        num_experts,
        bias=True,
        device=None,
        dtype=None,
    ):
        """Initialize stacked projection weights for one or more experts."""
        super().__init__()
        factory_kwargs = {"device": device, "dtype": torch.bfloat16}
        self.in_features = in_features
        self.out_features = out_features
        self.num_layers_for_initialization = num_layers_for_initialization
        self.num_experts = num_experts
        self.use_bias = bias
        self.weight = Parameter(
            torch.empty((out_features * num_experts, in_features), **factory_kwargs)
        )
        if bias:
            self.bias = Parameter(
                torch.empty(out_features * num_experts, **factory_kwargs)
            )
        else:
            self.register_parameter("bias", None)

    def forward(
        self,
        input: torch.Tensor,
        output_dtype: Optional[torch.dtype] = None,
        modality_dispatcher: Optional[ModalityDispatcher] = None,
    ) -> torch.Tensor:
        output_dtype = input.dtype if output_dtype is None else output_dtype
        return _bf16_compute_linear(
            input, self.weight, self.bias, output_dtype, torch.bfloat16
        )

    # ── LoRA Bypass ──

    def lora_layout(self) -> Dict[str, Tuple]:
        return {
            "lora_A": (DYNAMIC_LORA_RANK, self.in_features),
            "lora_B": (self.out_features, DYNAMIC_LORA_RANK),
        }

    def lora_forward(self, input, *args, **kwargs):
        out = self.orig_forward(input, *args, **kwargs)
        x = input.to(self.lora_A.dtype)
        delta = torch.matmul(
            torch.matmul(x, self.lora_A.T) * self.lora_scale, self.lora_B.T
        )
        return out + delta.to(out.dtype)


@register_lora_module
class NativeMoELinear(BaseLinear):
    def forward(
        self,
        input: torch.Tensor,
        output_dtype: Optional[torch.dtype] = None,
        modality_dispatcher: Optional[ModalityDispatcher] = None,
    ) -> torch.Tensor:
        """Apply the matching linear expert to each dispatched modality group."""
        output_dtype = input.dtype if output_dtype is None else output_dtype

        input_list = modality_dispatcher.dispatch(input)  # type: ignore
        weight_chunked = self.weight.chunk(self.num_experts, dim=0)

        if self.bias is not None:
            bias_chunked = self.bias.chunk(self.num_experts, dim=0)

        for i in range(self.num_experts):
            input_list[i] = _bf16_compute_linear(
                input_list[i],
                weight_chunked[i],
                bias_chunked[i] if self.bias is not None else None,
                output_dtype,
                torch.bfloat16,
            )
        return modality_dispatcher.undispatch(*input_list)  # type: ignore

    # ── LoRA Bypass (overrides BaseLinear: 3-D per-expert) ──

    def lora_layout(self) -> Dict[str, Tuple]:
        return {
            "lora_A": (self.num_experts, DYNAMIC_LORA_RANK, self.in_features),
            "lora_B": (self.num_experts, self.out_features, DYNAMIC_LORA_RANK),
        }

    def lora_forward(self, input, output_dtype=None, modality_dispatcher=None):
        """Add per-expert low-rank updates to the mixture-of-experts output."""
        out = self.orig_forward(
            input, output_dtype=output_dtype, modality_dispatcher=modality_dispatcher
        )
        num_experts = self.num_experts
        input_list = modality_dispatcher.dispatch(input)
        deltas = []
        for i in range(num_experts):
            x_i = input_list[i].to(self.lora_A.dtype)
            delta_i = torch.matmul(
                torch.matmul(x_i, self.lora_A[i].T) * self.lora_scale, self.lora_B[i].T
            )
            deltas.append(delta_i.to(out.dtype))
        return out + modality_dispatcher.undispatch(*deltas)





def create_linear(
    in_features,
    out_features,
    num_layers=1,
    num_experts=1,
    bias=True,
    device=None,
    dtype=None,
) -> BaseLinear | NativeMoELinear:
    """Construct a dense or modality-expert linear projection."""
    if num_experts == 1:
        return BaseLinear(
            in_features, out_features, num_layers, num_experts, bias, device, dtype
        )
    return NativeMoELinear(
        in_features, out_features, num_layers, num_experts, bias, device, dtype
    )


# ============================================================
# _merge_element_wise_rope
# ============================================================


class ElementWiseFourierEmbed(nn.Module):
    def __init__(
        self,
        dim: int,
        max_res: int = 224,
        temperature: float = 10000.0,
        in_pixels: bool = True,
        linear_bands: bool = False,
        learnable: bool = False,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ):
        """Configure frequency bands for coordinate-wise Fourier embeddings.

        Args:
            dim: Output feature dimension, total channels, must be divisible by 6
            max_res: Max pixel frequency resolution for generating pixel-domain frequency bands
            temperature: Temperature parameter for inverse frequency mode
            in_pixels: True -> pixel frequency bands, False -> inverse frequency bands
            linear_bands: Whether pixel frequency bands are linearly spaced
            learnable: Whether to make frequency bands learnable parameters
        """
        super().__init__()
        self.dim = dim
        self.in_pixels = in_pixels
        self.learnable = learnable
        self.temperature = temperature
        self.max_res = max_res
        self.linear_bands = linear_bands
        self.device = device
        self.dtype = dtype
        # Make frequency bands learnable parameters or register as buffer
        bands = self.get_default_bands()
        if self.learnable:
            self.bands = nn.Parameter(bands)
        else:
            self.register_buffer("bands", bands)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """Encode token coordinates as element-wise Fourier features.

        Args:
            coords: [L,9], column order is (time, row, col, T, H, W, ref_T, ref_H, ref_W)
        Returns:
            emb: [L, dim] element-wise Fourier embedding
        """
        # Use slicing instead of unbind + stack to reduce intermediate tensors
        coords_xyz = coords[:, :3]  # [L,3] -> (t, h, w)
        sizes = coords[:, 3:6]  # [L,3] -> (T, H, W)
        refs = coords[:, 6:9]  # [L,3] -> (ref_T, ref_H, ref_W)

        # Compute scaling factors
        scales = (refs - 1) / (sizes - 1)  # [L,3]

        # NOTE: For points where both ref and size are 1, scale factor is 1; otherwise error
        scales[(refs == 1) & (sizes == 1)] = 1
        assert not scales.isnan().any(), "scales has nan"
        assert not scales.isinf().any(), "scales has inf"

        # Center-align, only for h,w dimensions, not for t dimension
        centers = (sizes - 1) / 2  # [L,3]
        centers[:, 0] = 0  # No centering for temporal dimension
        coords_xyz = coords_xyz - centers  # [L,3]

        # Project to frequency bands in one pass: [L,3,B]
        proj = coords_xyz.unsqueeze(-1) * scales.unsqueeze(-1) * self.bands

        # Compute sin & cos and concatenate
        sin_proj = proj.sin()  # [L,3,B]
        cos_proj = proj.cos()

        return torch.cat((sin_proj, cos_proj), dim=1).flatten(1)

    def reset_parameters(self):
        bands = self.get_default_bands()
        self.bands.copy_(bands)

    def get_default_bands(self):
        if self.in_pixels:
            raise NotImplementedError("in_pixels are not implemented yet")
        else:
            bands = freq_bands(
                self.dim // 8, temperature=self.temperature, step=1, device=self.device
            ).to(self.dtype)
        return bands


def get_coords(
    shape: list[int],
    ref_feat_shape: list[int],
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
):
    """
    Generate feature grid coordinates with original and reference size info
    Args:
        feat_shape: [T, H, W] original feature map shape
        ref_feat_shape: [T_ref, H_ref, W_ref] reference feature map shape
        device: Device for coordinate tensors
    Returns:
        coords: Tensor of shape (T*H*W, 9), containing (t, h, w, T, H, W, ref_T, ref_H, ref_W)
    """
    ori_t, ori_h, ori_w = shape
    ref_t, ref_h, ref_w = ref_feat_shape

    # Generate index ranges
    time_rng = torch.arange(ori_t, device=device, dtype=dtype)
    height_rng = torch.arange(ori_h, device=device, dtype=dtype)
    width_rng = torch.arange(ori_w, device=device, dtype=dtype)

    # Create 3D meshgrid (T, H, W)
    time_grid, height_grid, width_grid = torch.meshgrid(
        time_rng, height_rng, width_rng, indexing="ij"
    )

    # Stack and flatten
    coords_grid = torch.stack([time_grid, height_grid, width_grid], dim=-1)
    coords_flat = coords_grid.reshape(-1, 3)

    # Construct and expand size metadata
    meta = torch.tensor(
        [ori_t, ori_h, ori_w, ref_t, ref_h, ref_w], device=device, dtype=dtype
    )
    meta_expanded = meta.expand(coords_flat.size(0), -1)

    # Concatenate and return
    return torch.cat([coords_flat, meta_expanded], dim=-1)


# ============================================================
# _merge_nd_rotary_pos_embedding
# ============================================================


def ndgrid(*tensors) -> Tuple[torch.Tensor, ...]:
    """generate N-D grid in dimension order.

    The ndgrid function is like meshgrid except that the order of the first two input arguments are switched.

    That is, the statement
    [X1,X2,X3] = ndgrid(x1,x2,x3)

    produces the same result as

    [X2,X1,X3] = meshgrid(x2,x1,x3)

    This naming is based on MATLAB, the purpose is to avoid confusion due to torch's change to make
    torch.meshgrid behaviour move from matching ndgrid ('ij') indexing to numpy meshgrid defaults of ('xy').

    """
    try:
        return torch.meshgrid(*tensors, indexing="ij")
    except TypeError:
        # old PyTorch < 1.10 will follow this path as it does not have indexing arg,
        # the old behaviour of meshgrid was 'ij'
        return torch.meshgrid(*tensors)


def meshgrid(*tensors) -> Tuple[torch.Tensor, ...]:
    """generate N-D grid in spatial dim order.

    The meshgrid function is similar to ndgrid except that the order of the
    first two input and output arguments is switched.

    That is, the statement

    [X,Y,Z] = meshgrid(x,y,z)
    produces the same result as

    [Y,X,Z] = ndgrid(y,x,z)
    Because of this, meshgrid is better suited to problems in two- or three-dimensional Cartesian space,
    while ndgrid is better suited to multidimensional problems that aren't spatially based.
    """

    # NOTE: this will throw in PyTorch < 1.10 as meshgrid did not support indexing arg or have
    # capability of generating grid in xy order before then.
    return torch.meshgrid(*tensors, indexing="xy")


def pixel_freq_bands(
    num_bands: int,
    max_freq: float = 224.0,
    linear_bands: bool = True,
    device: Optional[torch.device] = None,
):
    """Generate pixel-domain Fourier frequency bands scaled by pi."""
    if linear_bands:
        bands = torch.linspace(
            1.0, max_freq / 2, num_bands, dtype=torch.float32, device=device
        )
    else:
        bands = 2 ** torch.linspace(
            0, math.log(max_freq, 2) - 1, num_bands, dtype=torch.float32, device=device
        )
    return bands * torch.pi


def freq_bands(
    num_bands: int,
    temperature: float = 10000.0,
    step: int = 2,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Generate inverse-frequency bands with exponential spacing."""
    exp = (
        torch.arange(0, num_bands, step, dtype=torch.int64, device=device).to(
            torch.float32
        )
        / num_bands
    )
    bands = 1.0 / (temperature**exp)
    return bands


def build_sincos2d_pos_embed(
    feat_shape: List[int],
    dim: int = 64,
    temperature: float = 10000.0,
    reverse_coord: bool = False,
    interleave_sin_cos: bool = False,
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Build a two-dimensional sinusoidal embedding for a feature grid.

    Args:
        feat_shape:
        dim:
        temperature:
        reverse_coord: stack grid order W, H instead of H, W
        interleave_sin_cos: sin, cos, sin, cos stack instead of sin, sin, cos, cos
        dtype:
        device:

    Returns:

    """
    assert dim % 4 == 0, (
        "Embed dimension must be divisible by 4 for sin-cos 2D position embedding"
    )
    pos_dim = dim // 4
    bands = freq_bands(pos_dim, temperature=temperature, step=1, device=device)

    if reverse_coord:
        feat_shape = feat_shape[::-1]  # stack W, H instead of H, W
    grid = (
        torch.stack(
            ndgrid(
                [
                    torch.arange(s, device=device, dtype=torch.int64).to(torch.float32)
                    for s in feat_shape
                ]
            )
        )
        .flatten(1)
        .transpose(0, 1)
    )
    pos2 = grid.unsqueeze(-1) * bands.unsqueeze(0)
    # FIXME add support for unflattened spatial dim?

    stack_dim = (
        2 if interleave_sin_cos else 1
    )  # stack sin, cos, sin, cos  instead of sin sin cos cos
    pos_emb = torch.stack([torch.sin(pos2), torch.cos(pos2)], dim=stack_dim).flatten(1)
    return pos_emb.to(dtype=dtype)


def build_fourier_pos_embed(
    feat_shape: List[int],
    bands: Optional[torch.Tensor] = None,
    num_bands: int = 64,
    max_res: int = 224,
    temperature: float = 10000.0,
    linear_bands: bool = False,
    include_grid: bool = False,
    in_pixels: bool = True,
    ref_feat_shape: Optional[List[float]] = None,
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
) -> List[torch.Tensor]:
    """Build sine and cosine Fourier features for a feature grid.

    Args:
        feat_shape: Feature shape for embedding.
        bands: Pre-calculated frequency bands.
        num_bands: Number of frequency bands (determines output dim).
        max_res: Maximum resolution for pixel based freq.
        temperature: Temperature for non-pixel freq.
        linear_bands: Linear band spacing for pixel based freq.
        include_grid: Include the spatial grid in output.
        in_pixels: Output in pixel freq.
        ref_feat_shape: Reference feature shape for resize / fine-tune.
        dtype: Output dtype.
        device: Output device.

    Returns:

    """
    if bands is None:
        if in_pixels:
            bands = pixel_freq_bands(
                num_bands, float(max_res), linear_bands=linear_bands, device=device
            )
        else:
            bands = freq_bands(
                num_bands, temperature=temperature, step=1, device=device
            )
    else:
        if device is None:
            device = bands.device
        if dtype is None:
            dtype = bands.dtype

    if in_pixels:
        t = [
            torch.linspace(-1.0, 1.0, steps=s, device=device, dtype=torch.float32)
            for s in feat_shape
        ]
    else:
        t = [
            torch.arange(s, device=device, dtype=torch.int64).to(torch.float32)
            for s in feat_shape
        ]
        # align spatial center (H/2,W/2) to (0,0)
        t[1] = t[1] - (feat_shape[1] - 1) / 2
        t[2] = t[2] - (feat_shape[2] - 1) / 2
    if ref_feat_shape is not None:
        # eva's scheme for resizing rope embeddings (ref shape = pretrain)
        # aligning to the endpoint e.g [0,1,2] -> [0, 0.4, 0.8, 1.2, 1.6, 2]
        t_rescaled = []
        for x, f, r in zip(t, feat_shape, ref_feat_shape):
            if f == 1:
                assert r == 1, "ref_feat_shape must be 1 when feat_shape is 1"
                t_rescaled.append(x)
            else:
                t_rescaled.append(x / (f - 1) * (r - 1))
    else:
        t_rescaled = t

    grid = torch.stack(ndgrid(t_rescaled), dim=-1)
    grid = grid.unsqueeze(-1)
    pos = grid * bands

    pos_sin, pos_cos = pos.sin().to(dtype=dtype), pos.cos().to(dtype)
    out = [grid, pos_sin, pos_cos] if include_grid else [pos_sin, pos_cos]
    return out


class FourierEmbed(nn.Module):
    def __init__(
        self,
        max_res: int = 224,
        num_bands: int = 64,
        concat_grid=True,
        keep_spatial=False,
    ):
        """Initialize fixed Fourier bands for input feature grids."""
        super().__init__()
        self.max_res = max_res
        self.num_bands = num_bands
        self.concat_grid = concat_grid
        self.keep_spatial = keep_spatial
        self.register_buffer(
            "bands", pixel_freq_bands(max_res, num_bands), persistent=False
        )

    def forward(self, x):
        """Append Fourier positional features to an image feature tensor."""
        B, C = x.shape[:2]
        feat_shape = x.shape[2:]
        emb = build_fourier_pos_embed(
            feat_shape,
            self.bands,
            include_grid=self.concat_grid,
            dtype=x.dtype,
            device=x.device,
        )
        emb = torch.cat(emb, dim=-1)
        emb = emb.transpose(-1, -2).flatten(len(feat_shape))
        batch_expand = (B,) + (-1,) * (x.ndim - 1)

        # FIXME support nD
        if self.keep_spatial:
            x = torch.cat(
                [x, emb.unsqueeze(0).expand(batch_expand).permute(0, 3, 1, 2)], dim=1
            )
        else:
            x = torch.cat(
                [x.permute(0, 2, 3, 1), emb.unsqueeze(0).expand(batch_expand)], dim=-1
            )
            x = x.reshape(B, feat_shape.numel(), -1)

        return x


def rot(x):
    return torch.stack([-x[..., 1::2], x[..., ::2]], -1).reshape(x.shape)


def apply_rot_embed(x: torch.Tensor, sin_emb, cos_emb):
    if sin_emb.ndim == 3:
        return x * cos_emb.unsqueeze(1).expand_as(x) + rot(x) * sin_emb.unsqueeze(
            1
        ).expand_as(x)
    return x * cos_emb + rot(x) * sin_emb


def apply_rot_embed_list(x: List[torch.Tensor], sin_emb, cos_emb):
    if isinstance(x, torch.Tensor):
        x = [x]
    return [t * cos_emb + rot(t) * sin_emb for t in x]


def apply_rot_embed_cat(x: torch.Tensor, emb):
    sin_emb, cos_emb = emb.tensor_split(2, -1)
    if sin_emb.ndim == 3:
        return x * cos_emb.unsqueeze(1).expand_as(x) + rot(x) * sin_emb.unsqueeze(
            1
        ).expand_as(x)
    return x * cos_emb.unsqueeze(1).unsqueeze(2) + rot(x) * sin_emb.unsqueeze(
        1
    ).unsqueeze(2)


def rotate_half(x, interleaved=False):
    if not interleaved:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
    else:
        x1, x2 = x[..., ::2], x[..., 1::2]
        return rearrange(
            torch.stack((-x2, x1), dim=-1), "... d two -> ... (d two)", two=2
        )


def apply_rotary_emb_torch(x, cos, sin, interleaved=False):
    """Apply rotary embeddings to the rotary channels of attention heads.

    x: (batch_size, seqlen, nheads, headdim)
    cos, sin: (seqlen, rotary_dim / 2) or (batch_size, seqlen, rotary_dim / 2)
    """
    ro_dim = cos.shape[-1] * 2
    assert ro_dim <= x.shape[-1]
    cos = repeat(
        cos, "... d -> ... 1 (2 d)" if not interleaved else "... d -> ... 1 (d 2)"
    )
    sin = repeat(
        sin, "... d -> ... 1 (2 d)" if not interleaved else "... d -> ... 1 (d 2)"
    )
    return torch.cat(
        [
            x[..., :ro_dim] * cos + rotate_half(x[..., :ro_dim], interleaved) * sin,
            x[..., ro_dim:],
        ],
        dim=-1,
    )


def apply_keep_indices_nlc(x, pos_embed, keep_indices):
    pos_embed = pos_embed.unsqueeze(0).expand(x.shape[0], -1, -1)
    pos_embed = pos_embed.gather(
        1, keep_indices.unsqueeze(-1).expand(-1, -1, pos_embed.shape[-1])
    )
    return pos_embed


def build_rotary_pos_embed(
    feat_shape: List[int],
    bands: Optional[torch.Tensor] = None,
    dim: int = 64,
    max_res: int = 224,
    temperature: float = 10000.0,
    linear_bands: bool = False,
    in_pixels: bool = True,
    ref_feat_shape: Optional[List[float]] = None,
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
):
    """Build split sine and cosine rotary embeddings for a feature grid.

    Args:
        feat_shape: Spatial shape of the target tensor for embedding.
        bands: Optional pre-generated frequency bands
        dim: Output dimension of embedding tensor.
        max_res: Maximum resolution for pixel mode.
        temperature: Temperature (inv freq) for non-pixel mode
        linear_bands: Linearly (instead of log) spaced bands for pixel mode
        in_pixels: Pixel vs language (inv freq) mode.
        dtype: Output dtype.
        device: Output device.

    Returns:

    """
    sin_emb, cos_emb = build_fourier_pos_embed(
        feat_shape,
        bands=bands,
        num_bands=dim // 8,
        max_res=max_res,
        temperature=temperature,
        linear_bands=linear_bands,
        in_pixels=in_pixels,
        ref_feat_shape=ref_feat_shape,
        device=device,
        dtype=dtype,
    )
    num_spatial_dim = 1
    # this would be much nicer as a .numel() call to torch.Size(), but torchscript sucks
    for x in feat_shape:
        num_spatial_dim *= x

    # In the original timm implementation, a repeat_interleave is performed
    # But in flash attn, the repeat_interleave is done on the fly
    # Here we align with flash attn implementation and skip repeat_interleave
    # sin_emb = sin_emb.reshape(num_spatial_dim, -1).repeat_interleave(2, -1)
    # cos_emb = cos_emb.reshape(num_spatial_dim, -1).repeat_interleave(2, -1)
    sin_emb = sin_emb.reshape(num_spatial_dim, -1)
    cos_emb = cos_emb.reshape(num_spatial_dim, -1)
    return sin_emb, cos_emb


class RotaryEmbedding(nn.Module):
    """Rotary position embedding

    NOTE: This is my initial attempt at impl rotary embedding for spatial use, it has not
    been well tested, and will likely change. It will be moved to its own file.

    The following impl/resources were referenced for this impl:
    * https://github.com/lucidrains/vit-pytorch/blob/6f3a5fcf0bca1c5ec33a35ef48d97213709df4ba/vit_pytorch/rvt.py
    * https://blog.eleuther.ai/rotary-embeddings/
    """

    def __init__(
        self,
        dim,
        max_res=224,
        temperature=10000,
        in_pixels=True,
        linear_bands: bool = False,
        feat_shape: Optional[List[int]] = None,
        ref_feat_shape: Optional[List[float]] = None,
    ):
        """Initialize cached bands or shape-specific rotary embeddings."""
        super().__init__()
        self.dim = dim
        self.max_res = max_res
        self.temperature = temperature
        self.in_pixels = in_pixels
        self.feat_shape = feat_shape
        self.ref_feat_shape = ref_feat_shape

        if feat_shape is None:
            # only cache bands
            if in_pixels:
                bands = pixel_freq_bands(
                    dim // 8, float(max_res), linear_bands=linear_bands
                )
            else:
                bands = freq_bands(dim // 8, temperature=temperature, step=1)
                logger.debug("bands: %s", bands)
            self.register_buffer("bands", bands, persistent=False)
            self.pos_embed_sin = None
            self.pos_embed_cos = None
        else:
            # cache full sin/cos embeddings if shape provided up front
            emb_sin, emb_cos = build_rotary_pos_embed(
                feat_shape=feat_shape,
                dim=dim,
                max_res=max_res,
                linear_bands=linear_bands,
                in_pixels=in_pixels,
                ref_feat_shape=self.ref_feat_shape,
                temperature=self.temperature,
            )
            self.bands = None
            self.register_buffer("pos_embed_sin", emb_sin, persistent=False)
            self.register_buffer("pos_embed_cos", emb_cos, persistent=False)

    def get_embed(self, shape: Optional[List[int]] = None):
        """Return cached rotary embeddings or build them for a feature shape."""
        if self.bands is not None:
            # rebuild embeddings every call, use if target shape changes
            assert shape is not None  # type: ignore[unreachable]
            return build_rotary_pos_embed(
                shape,
                self.bands,
                in_pixels=self.in_pixels,
                temperature=self.temperature,
            )
        else:
            return self.pos_embed_sin, self.pos_embed_cos

    def forward(self, x):
        # assuming channel-first tensor where spatial dim are >= 2
        sin_emb, cos_emb = self.get_embed(x.shape[2:])
        return apply_rot_embed(x, sin_emb, cos_emb)


class RotaryEmbeddingCat(nn.Module):
    """Rotary position embedding w/ concatenatd sin & cos

    The following impl/resources were referenced for this impl:
    * https://github.com/lucidrains/vit-pytorch/blob/6f3a5fcf0bca1c5ec33a35ef48d97213709df4ba/vit_pytorch/rvt.py
    * https://blog.eleuther.ai/rotary-embeddings/
    """

    def __init__(
        self,
        dim,
        max_res=224,
        temperature=10000,
        in_pixels=True,
        linear_bands: bool = False,
        feat_shape: Optional[List[int]] = None,
        ref_feat_shape: Optional[List[float]] = None,
    ):
        """Initialize optional shape-specific concatenated rotary embeddings."""
        super().__init__()
        self.dim = dim
        self.max_res = max_res
        self.temperature = temperature
        self.in_pixels = in_pixels
        self.linear_bands = linear_bands
        self.feat_shape = feat_shape
        self.ref_feat_shape = ref_feat_shape

        if feat_shape is None:
            self.bands = None
            self.pos_embed = None
        else:
            # cache full sin/cos embeddings if shape provided up front
            embeds = build_rotary_pos_embed(
                feat_shape=feat_shape,
                dim=dim,
                max_res=max_res,
                linear_bands=linear_bands,
                in_pixels=in_pixels,
                ref_feat_shape=self.ref_feat_shape,
                temperature=self.temperature,
            )
            self.bands = None
            self.register_buffer("pos_embed", torch.cat(embeds, -1), persistent=False)

    def get_embed(
        self,
        shape: Optional[List[int]] = None,
        ref_feat_shape: Optional[List[float]] = None,
    ):
        """Build concatenated rotary embeddings or return cached values."""
        if shape is not None:
            # rebuild bands and embeddings every call, use if target shape changes
            embeds = build_rotary_pos_embed(
                feat_shape=shape,
                dim=self.dim,
                max_res=self.max_res,
                linear_bands=self.linear_bands,
                in_pixels=self.in_pixels,
                ref_feat_shape=ref_feat_shape
                if ref_feat_shape
                else self.ref_feat_shape,
                temperature=self.temperature,
                device=torch.cuda.current_device(),
            )
            return torch.cat(embeds, -1)
        elif self.pos_embed is not None:
            return self.pos_embed  # type: ignore[unreachable]
        else:
            assert False, (
                "get_embed() requires pre-computed pos_embed or valid shape w/ pre-computed bands"
            )

    def forward(self, x):
        # assuming channel-first tensor where spatial dim are >= 2
        pos_embed = self.get_embed(x.shape[2:])
        return apply_rot_embed_cat(x, pos_embed)


class LearnableRotaryEmbeddingCat(nn.Module):
    """Rotary position embedding w/ concatenatd sin & cos

    The following impl/resources were referenced for this impl:
    * https://github.com/lucidrains/vit-pytorch/blob/6f3a5fcf0bca1c5ec33a35ef48d97213709df4ba/vit_pytorch/rvt.py
    * https://blog.eleuther.ai/rotary-embeddings/
    """

    def __init__(
        self,
        dim,
        max_res=224,
        temperature=10000,
        in_pixels=True,
        linear_bands: bool = False,
        feat_shape: Optional[List[int]] = None,
        ref_feat_shape: Optional[List[int]] = None,
    ):
        """Initialize learnable rotary frequency bands for spatial features."""
        super().__init__()
        self.dim = dim
        self.max_res = max_res
        self.temperature = temperature
        self.in_pixels = in_pixels
        self.linear_bands = linear_bands
        self.feat_shape = feat_shape
        self.ref_feat_shape = ref_feat_shape
        self.bands = nn.Parameter(self.get_default_bands())
        setattr(self.bands, "sequence_parallel", True)

    def get_default_bands(self):
        """Create initial bands for the configured coordinate domain."""
        if self.in_pixels:
            bands = pixel_freq_bands(
                self.dim // 8,
                float(self.max_res),
                linear_bands=self.linear_bands,
                device=torch.cuda.current_device(),
            )
        else:
            bands = freq_bands(
                self.dim // 8,
                temperature=self.temperature,
                step=1,
                device=torch.cuda.current_device(),
            )
        return bands

    def get_embed(
        self, shape: Optional[List[int]], ref_feat_shape: Optional[List[float]] = None
    ):
        """Build concatenated rotary embeddings from learned frequency bands."""
        # rebuild bands and embeddings every call, use if target shape changes
        embeds = build_rotary_pos_embed(
            feat_shape=shape,  # type: ignore[arg-type]
            bands=self.bands,  # use learned bands
            dim=self.dim,
            max_res=self.max_res,
            linear_bands=self.linear_bands,
            in_pixels=self.in_pixels,
            ref_feat_shape=ref_feat_shape if ref_feat_shape else self.ref_feat_shape,  # type: ignore[arg-type]
            temperature=self.temperature,
            device=torch.cuda.current_device(),
        )
        return torch.cat(embeds, -1)


# ============================================================
# _merge_custom_ops
# ============================================================

HAS_MAGI_ATTENTION = importlib.util.find_spec("magi_attention") is not None
HAS_FA3 = importlib.util.find_spec("flash_attn_interface") is not None


def is_hopper_arch() -> bool:
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    return major >= 9


@magi_register_custom_op(
    "magi2::flash_attn_func", infer_output_meta_fn=["query"], is_subgraph_boundary=True
)
def flash_attn_func(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
) -> torch.Tensor:
    """Select and run the FlashAttention kernel that matches GPU support."""
    if HAS_FA3 and is_hopper_arch():
        from flash_attn_interface import flash_attn_func as fa3_flash_attn_func

        return fa3_flash_attn_func(query, key, value)
    else:
        from flash_attn.flash_attn_interface import (
            flash_attn_func as fa2_flash_attn_func,
        )

        return fa2_flash_attn_func(query, key, value)


def _split_q_range_with_no_overlap(
    q_ranges: torch.Tensor, k_ranges: torch.Tensor
) -> Tuple[List[List[int]], List[List[List[int]]]]:
    """Split overlapping query spans and retain their associated key ranges."""
    range_boundary = torch.unique(q_ranges, sorted=True).tolist()
    candidates = [
        [start, end, []] for start, end in zip(range_boundary[:-1], range_boundary[1:])
    ]
    q_ranges = q_ranges.tolist()
    k_ranges = k_ranges.tolist()
    for q_range, k_range in zip(q_ranges, k_ranges):
        q_start, q_end = q_range
        for q_range_cand in candidates:
            if q_start <= q_range_cand[0] and q_range_cand[1] <= q_end:
                q_range_cand[2].append(k_range)
    q_ranges_out = []
    k_ranges_out = []
    for q_range_cand in candidates:
        if len(q_range_cand[2]) > 0:
            q_ranges_out.append(q_range_cand[0:2])
            k_ranges_out.append(q_range_cand[2])
    return q_ranges_out, k_ranges_out


def _flash_attn_with_correction(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    q_ranges: List[List[int]],
    k_range_list: List[List[List[int]]],
):
    """Combine disjoint key-range attention with log-sum-exp correction."""
    output = torch.zeros_like(query)
    output_lse = torch.zeros(
        (query.shape[0], query.shape[1]), dtype=torch.float32, device=query.device
    )

    from flash_attn.flash_attn_interface import flash_attn_func

    for q_range, k_ranges in zip(q_ranges, k_range_list):
        q_start, q_end = q_range
        qo_out, qo_lse = None, None
        for k_range in k_ranges:
            k_start, k_end = k_range
            cur_qo_out, cur_qo_lse, _ = flash_attn_func(
                query[q_start:q_end].unsqueeze(0),
                key[k_start:k_end].unsqueeze(0),
                value[k_start:k_end].unsqueeze(0),
                return_attn_probs=True,
            )
            cur_qo_out, cur_qo_lse = cur_qo_out.squeeze(0), cur_qo_lse.squeeze(0)

            if qo_out is None:
                qo_out = cur_qo_out
                qo_lse = cur_qo_lse
            else:
                qo_lse[qo_lse == torch.inf] = -torch.inf
                cur_qo_lse[cur_qo_lse == torch.inf] = -torch.inf
                max_lse = torch.max(qo_lse, cur_qo_lse)
                qo_se, cur_qo_se = (
                    torch.exp(qo_lse - max_lse),
                    torch.exp(cur_qo_lse - max_lse),
                )
                sum_se = qo_se + cur_qo_se
                qo_scale, cur_qo_scale = qo_se / sum_se, cur_qo_se / sum_se

                qo_out = qo_out * qo_scale.permute(1, 0).unsqueeze(
                    -1
                ) + cur_qo_out * cur_qo_scale.permute(1, 0).unsqueeze(-1)
                qo_lse = torch.log(sum_se) + max_lse

        output[q_start:q_end] = qo_out
        output_lse[q_start:q_end, :] = qo_lse.permute(1, 0)
    return output, output_lse


def _flex_flash_attn_func_meta(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, *args, **kwargs
) -> Tuple[torch.Tensor, torch.Tensor]:
    output = torch.empty_like(query)
    output_lse = torch.empty(
        (query.shape[0], query.shape[1]), dtype=torch.float32, device=query.device
    )
    return output, output_lse


_PARAM_RENAMES = {"auto_range_merge": "range_merge", "sparse_load": "block_sparse"}


def _adapt_magi_attn_kwargs(fn, kwargs: dict) -> dict:
    """Translate legacy kwarg names to current magi_attention API and vice versa."""
    import inspect

    params = set(inspect.signature(fn).parameters)
    out = {}
    for key, value in kwargs.items():
        if key in params:
            out[key] = value
        elif key in _PARAM_RENAMES and _PARAM_RENAMES[key] in params:
            out[_PARAM_RENAMES[key]] = value
    return out

@magi_register_custom_op(
    "magi2::flex_flash_attn_func",
    infer_output_meta_fn=_flex_flash_attn_func_meta,
    is_subgraph_boundary=True,
)
def flex_flash_attn_func(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    q_ranges: torch.Tensor,
    k_ranges: torch.Tensor,
    attn_type_map: torch.Tensor | None = None,
    max_seqlen_q: int | None = None,
    auto_range_merge: bool = False,
    sparse_load: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run range-restricted attention through the available kernel."""
    if HAS_MAGI_ATTENTION and is_hopper_arch():
        from magi_attention.api import flex_flash_attn_func as magi_flex_flash_attn_func

        kwargs = _adapt_magi_attn_kwargs(
            magi_flex_flash_attn_func,
            {"max_seqlen_q": max_seqlen_q, "auto_range_merge": auto_range_merge, "sparse_load": sparse_load},
        )
        output, meta = magi_flex_flash_attn_func(
            query, key, value, q_ranges, k_ranges, attn_type_map, **kwargs
        )
        return output, meta.lse
    return _custom_flex_flash_attn_func(query, key, value, q_ranges, k_ranges)


def _custom_flex_flash_attn_func(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    q_ranges: torch.Tensor,
    k_ranges: torch.Tensor,
    **kwargs,
):
    q_ranges, k_range_list = _split_q_range_with_no_overlap(q_ranges, k_ranges)
    return _flash_attn_with_correction(query, key, value, q_ranges, k_range_list)


# ============================================================
# _merge_local_attn
# ============================================================


# num_video_tokens = 26000
# num_audio_and_txt_tokens = 101+500
# num_frames = 26
# frame_base_recept_field = 2
def calc_local_qk_range(
    num_video_tokens, num_audio_and_txt_tokens, num_frames, frame_receptive_field
):
    """Build local-video and global cross-modality query and key ranges."""
    token_per_frame = num_video_tokens // num_frames
    total_tokens = num_video_tokens + num_audio_and_txt_tokens

    q_range_list = []
    k_range_list = []

    for i in range(num_frames):
        local_q_range = torch.tensor([i * token_per_frame, (i + 1) * token_per_frame])
        local_k_range = torch.tensor(
            [
                (i - frame_receptive_field) * token_per_frame,
                (i + frame_receptive_field + 1) * token_per_frame,
            ]
        )

        q_range_list.append(local_q_range)
        k_range_list.append(local_k_range)
    local_q_range = torch.stack(q_range_list, dim=0)
    local_k_range = torch.stack(k_range_list, dim=0)

    local_k_range[local_k_range < 0] = 0
    local_k_range[local_k_range > num_video_tokens] = num_video_tokens

    video_q_range = torch.tensor([[0, num_video_tokens]])
    video_k_range = torch.tensor(
        [[num_video_tokens, num_video_tokens + num_audio_and_txt_tokens]]
    )

    at_q_ranges = torch.tensor([[num_video_tokens, total_tokens]])
    at_k_ranges = torch.tensor([[0, total_tokens]])

    q_ranges = (
        torch.cat([local_q_range, video_q_range, at_q_ranges], dim=0)
        .to(torch.int32)
        .to("cuda", non_blocking=True)
    )
    k_ranges = (
        torch.cat([local_k_range, video_k_range, at_k_ranges], dim=0)
        .to(torch.int32)
        .to("cuda", non_blocking=True)
    )

    return (q_ranges, k_ranges)


def calc_local_attn_ffa_handler(
    num_video_tokens, num_audio_and_txt_tokens, num_frames, frame_receptive_field
):
    """Create the attention descriptor for local video and cross-modality ranges."""
    q_ranges, k_ranges = calc_local_qk_range(
        num_video_tokens, num_audio_and_txt_tokens, num_frames, frame_receptive_field
    )
    max_seqlen_q = num_video_tokens + num_audio_and_txt_tokens
    max_seqlen_k = num_video_tokens + num_audio_and_txt_tokens
    attn_type_map = torch.zeros([q_ranges.shape[0]], device="cuda", dtype=torch.int32)
    softmax_scale = None

    ffa_handler = FFAHandler(
        q_ranges=q_ranges,
        k_ranges=k_ranges,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        attn_type_map=attn_type_map,
        softmax_scale=softmax_scale,
    )
    return ffa_handler


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _magi_compile_enabled(_=None) -> bool:
    return not _env_flag("MAGI2_DISABLE_MAGI_COMPILE", False) and not _env_flag(
        "TORCH_COMPILE_DISABLE", False
    )


# ============================================================
# _merge_modeling
# ============================================================


def build_dataclass_from_config(dataclass_cls, config, overrides=None):
    """Build a dataclass instance from a pydantic config, applying optional overrides."""
    fields = (
        {f.name for f in dataclass_cls.__dataclass_fields__.values()}
        if hasattr(dataclass_cls, "__dataclass_fields__")
        else set()
    )
    kwargs = {}
    for field_name in fields:
        if overrides and field_name in overrides:
            kwargs[field_name] = overrides[field_name]
        elif hasattr(config, field_name):
            kwargs[field_name] = getattr(config, field_name)
    return dataclass_cls(**kwargs)


def register_model(name, aliases=None):
    """No-op decorator for inference (model registry not needed)."""

    def decorator(cls):
        return cls

    return decorator


@dataclass
class VarlenHandler(object):
    cu_seqlens_q: torch.Tensor
    cu_seqlens_k: torch.Tensor
    max_seqlen_q: int
    max_seqlen_k: int


def _flash_attn_with_cp_meta(q: torch.Tensor, *args, **kwargs) -> torch.Tensor:
    return torch.empty_like(q, dtype=torch.bfloat16).squeeze(0)


@magi_register_custom_op(
    "magi2::flash_attn_with_cp",
    infer_output_meta_fn=_flash_attn_with_cp_meta,
    is_subgraph_boundary=True,
)
def flash_attn_with_cp(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, cp_split_sizes: List[int]
) -> torch.Tensor:
    """Run dense attention with context-parallel head and sequence exchange.

    Args:
        q: bsz, seq_len, n_heads, head_dim
        k: bsz, seq_len, n_heads, head_dim
        v: bsz, seq_len, n_heads, head_dim
        rope: seq_len, rotary_dim

    Returns:
        self_attn_out: bsz, seq_len, n_heads, head_dim
    """
    q, k, v = q.to(torch.bfloat16), k.to(torch.bfloat16), v.to(torch.bfloat16)

    if psm.get_world_size("cp") > 1:
        q, k, v = batch_scatter_head_gather_seqlen(
            [q.squeeze(0), k.squeeze(0), v.squeeze(0)],
            cp_split_sizes,
            psm.get_parallel_group("cp"),
        )
        q = q.unsqueeze(0)
        k = k.unsqueeze(0)
        v = v.unsqueeze(0)

    self_attn_out = torch.ops.magi2.flash_attn_func(q, k, v).squeeze(0)

    if psm.get_world_size("cp") > 1:
        self_attn_out = scatter_seqlen_gather_head(
            self_attn_out, cp_split_sizes, psm.get_parallel_group("cp"), async_op=False
        )
        self_attn_out = rearrange(
            self_attn_out, "(cp sq) hn hd -> sq (cp hn) hd", cp=psm.get_world_size("cp")
        )

    return self_attn_out


@magi_register_custom_op(
    "magi2::flex_flash_attn_with_cp",
    infer_output_meta_fn=_flash_attn_with_cp_meta,
    is_subgraph_boundary=True,
)
def flex_flash_attn_with_cp(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_ranges: torch.Tensor,
    k_ranges: torch.Tensor,
    attn_type_map: torch.Tensor,
    max_seqlen_q: int,
    auto_range_merge: bool,
    sparse_load: bool,
    cp_split_sizes: List[int],
) -> torch.Tensor:
    """Run range-restricted attention with context-parallel token exchange.

    Args:
        q: bsz, seq_len, n_heads, head_dim
        k: bsz, seq_len, n_heads, head_dim
        v: bsz, seq_len, n_heads, head_dim
        q_ranges: query ranges for flex attention
        k_ranges: key ranges for flex attention
        attn_type_map: attention type for each q/k range
        max_seqlen_q: maximum query range length for tile scheduling
        auto_range_merge: merge repeated query ranges before launching magi attention
        sparse_load: enable sparse-load optimization in magi attention

    Returns:
        self_attn_out: bsz, seq_len, n_heads, head_dim
    """
    q, k, v = (
        q.to(torch.bfloat16).squeeze(0),
        k.to(torch.bfloat16).squeeze(0),
        v.to(torch.bfloat16).squeeze(0),
    )

    if psm.get_world_size("cp") > 1:
        q, k, v = batch_scatter_head_gather_seqlen(
            [q, k, v], cp_split_sizes, psm.get_parallel_group("cp")
        )

    out, _ = torch.ops.magi2.flex_flash_attn_func(
        q,
        k,
        v,
        q_ranges,
        k_ranges,
        attn_type_map,
        max_seqlen_q,
        auto_range_merge,
    )

    if psm.get_world_size("cp") > 1:
        out = scatter_seqlen_gather_head(
            out, cp_split_sizes, psm.get_parallel_group("cp"), async_op=False
        )
        out = rearrange(
            out, "(cp sq) hn hd -> sq (cp hn) hd", cp=psm.get_world_size("cp")
        )

    return out


# Define the Attention module
@dataclass
class AttentionConfig:
    """
    Args:
        hidden_size (int): Hidden dimension size of the attention module.
        num_heads_q (int): Number of query heads.
        num_heads_kv (int): Number of key-value heads.
        head_dim (int): Dimension size of each head.
        params_dtype (torch.dtype): Dtype of the parameters.

    """

    hidden_size: int
    num_heads_q: int
    num_heads_kv: int
    head_dim: int
    params_dtype: torch.dtype
    checkpoint_qk_layernorm_rope: bool
    num_modality: int
    num_layers: int
    use_local_attn: bool = False
    enable_attn_gating: bool = False


class Attention(torch.nn.Module):
    """
    Multi-head attention module with support for both self-attention and cross-attention.
    """

    config: AttentionConfig

    def __init__(self, config: AttentionConfig):
        """Initialize multimodal attention normalization and projections."""
        super().__init__()
        self.config = config

        self.pre_norm = MultiModalityRMSNorm(
            config.hidden_size, eps=1e-6, num_modality=config.num_modality
        )

        # Linear projection for query, key, value for self-attention
        self.linear_qkv = create_linear(
            config.hidden_size,
            config.num_heads_q * config.head_dim
            + config.num_heads_kv * config.head_dim * 2,
            num_experts=config.num_modality,
            bias=False,
            dtype=config.params_dtype,
            num_layers=config.num_layers,
        )
        if config.enable_attn_gating:
            self.linear_g = create_linear(
                config.hidden_size,
                config.num_heads_q,
                num_experts=config.num_modality,
                bias=False,
                dtype=config.params_dtype,
                num_layers=config.num_layers,
            )

        # Output projection for self-attention outputs
        self.linear_proj = create_linear(
            config.num_heads_q * config.head_dim,
            config.hidden_size,
            bias=False,
            num_experts=config.num_modality,
            dtype=config.params_dtype,
            num_layers=config.num_layers,
        )

        # Layer normalization for query and key in self-attention
        self.q_norm = MultiModalityRMSNorm(
            config.head_dim, num_modality=config.num_modality
        )
        self.k_norm = MultiModalityRMSNorm(
            config.head_dim, num_modality=config.num_modality
        )

        self.q_size = config.num_heads_q * config.head_dim
        self.kv_size = config.num_heads_kv * config.head_dim

    def reset_parameters(self):
        if hasattr(self.linear_proj, "reset_parameters_output_layer"):
            self.linear_proj.reset_parameters_output_layer()

    def forward(
        self,
        hidden_states: torch.Tensor,
        rope: torch.Tensor,
        permute_mapping: torch.Tensor,
        inv_permute_mapping: torch.Tensor,
        varlen_handler: VarlenHandler,
        local_attn_handler: FFAHandler,  # Key for distributed attention
        modality_dispatcher: ModalityDispatcher,
        cp_split_sizes: List[int],
    ) -> torch.Tensor:
        """
        Forward pass for the attention module.

        Args:
            hidden_states(torch.Tensor): Input tensor for self-attention
            rope(torch.Tensor): Rotary position embeddings

        Returns:
            out(torch.Tensor): Combined output from self-attention and cross-attention


        Shape:
            - hidden_states: (num_tokens, hidden_size)
            - rope: (num_tokens, embed_dim)
            - out: (num_tokens, hidden_size)

        """
        # Project input to query, query_xattn, key, value for self-attention

        hidden_states = self.pre_norm(
            hidden_states, modality_dispatcher=modality_dispatcher
        ).to(torch.bfloat16)
        qkv: torch.Tensor = self.linear_qkv(
            hidden_states, modality_dispatcher=modality_dispatcher
        ).to(torch.float32)

        q, k, v = torch.split(qkv, [self.q_size, self.kv_size, self.kv_size], dim=1)
        q = q.view(-1, self.config.num_heads_q, self.config.head_dim)
        k = k.view(-1, self.config.num_heads_kv, self.config.head_dim)
        v = v.view(-1, self.config.num_heads_kv, self.config.head_dim)
        if self.config.enable_attn_gating:
            g = self.linear_g(
                hidden_states, modality_dispatcher=modality_dispatcher
            ).to(torch.float32)
            # IMPORTANT: Do NOT use `g.view(k.shape[0], num_heads, -1)` here.
            # ModalityDispatcher marks per-modality sizes as unbacked symbols (u0,u1,u2),
            # so k.shape[0] = u0+u1+u2. Using -1 in view forces Inductor to guard on
            # `u0+u1+u2 > 0`, which is forbidden for unbacked symbols and raises
            # GuardOnDataDependentSymNode at compile time.
            g = g.unsqueeze(-1)

        q = self.q_norm(q, modality_dispatcher=modality_dispatcher)
        k = self.k_norm(k, modality_dispatcher=modality_dispatcher)

        q = ModalityDispatcher.inv_permute(q, inv_permute_mapping).unsqueeze(0)
        k = ModalityDispatcher.inv_permute(k, inv_permute_mapping).unsqueeze(0)
        v = ModalityDispatcher.inv_permute(v, inv_permute_mapping).unsqueeze(0)

        sin_emb, cos_emb = rope.tensor_split(2, -1)
        q = apply_rotary_emb(q, cos_emb, sin_emb)
        k = apply_rotary_emb(k, cos_emb, sin_emb)

        if self.config.use_local_attn:
            auto_range_merge = bool(
                getattr(local_attn_handler, "auto_range_merge", False)
            )
            sparse_load = bool(getattr(local_attn_handler, "sparse_load", False))
            self_attn_out = flex_flash_attn_with_cp(
                q,
                k,
                v,
                local_attn_handler.q_ranges,
                local_attn_handler.k_ranges,
                local_attn_handler.attn_type_map,
                local_attn_handler.max_seqlen_q,
                auto_range_merge,
                sparse_load,
                cp_split_sizes,
            )
        else:
            self_attn_out = flash_attn_with_cp(q, k, v, cp_split_sizes)
        self_attn_out = ModalityDispatcher.permute(self_attn_out, permute_mapping)

        if self.config.enable_attn_gating:
            self_attn_out = self_attn_out * torch.sigmoid(g)

        self_attn_out = self_attn_out.view(
            -1, self.config.num_heads_q * self.config.head_dim
        ).to(torch.bfloat16)
        out = self.linear_proj(self_attn_out, modality_dispatcher=modality_dispatcher)

        return out


@dataclass
class MLPConfig:
    """Configuration for the MLP module

    Args:
        hidden_size (int): Hidden dimension size.
        intermediate_size (int): Intermediate Hidden dimension of the mlp layer.
        activation_type (MLPActivationType): Activation function type.
        params_dtype (torch.dtype): Dtype of the parameters.
    """

    hidden_size: int
    intermediate_size: int
    activation_type: MLPActivationType
    params_dtype: torch.dtype
    num_modality: int = 1
    num_layers: int = 1
    gated_act: bool = False


class MLP(torch.nn.Module):
    """MLP module with traditional architecture (up-projection, activation, and down-projection)"""

    config: MLPConfig

    def __init__(self, config: MLPConfig):
        """Initialize modality-aware gated feed-forward projections."""
        super().__init__()

        num_experts = config.num_modality

        self.pre_norm = MultiModalityRMSNorm(
            config.hidden_size, num_modality=config.num_modality
        )
        if config.gated_act:
            intermediate_size_up = config.intermediate_size * 2
        else:
            intermediate_size_up = config.intermediate_size

        # Combined projection for up-projection and gate
        self.up_gate_proj = create_linear(
            config.hidden_size,
            intermediate_size_up,
            bias=False,
            dtype=config.params_dtype,
            num_layers=config.num_layers,
            num_experts=num_experts,
        )

        # Down-projection back to hidden size
        self.down_proj = create_linear(
            config.intermediate_size,
            config.hidden_size,
            bias=False,
            dtype=config.params_dtype,
            num_layers=config.num_layers,
            num_experts=num_experts,
        )

        # Initialize activation function based on configuration
        self.activation_func = create_activation_func(config.activation_type)

    def forward(
        self, x: torch.Tensor, modality_dispatcher: ModalityDispatcher
    ) -> torch.Tensor:
        """Forward pass of the MLP module.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            output (torch.Tensor): Output tensor

        Shape:
            - x: (num_tokens, hidden_size)
            - output: (num_tokens, hidden_size)
        """

        x = self.pre_norm(x, modality_dispatcher=modality_dispatcher).to(torch.bfloat16)

        # Project input to up-projection and gate
        x = self.up_gate_proj(x, modality_dispatcher=modality_dispatcher).to(
            torch.float32
        )

        x = self.activation_func(x).to(torch.bfloat16)

        # Element-wise multiplication with gate and down-projection
        x = self.down_proj(x, modality_dispatcher=modality_dispatcher).to(torch.float32)
        return x

    def extra_repr(self) -> str:
        return f"{self.up_gate_proj.weight.shape=}, {self.down_proj.weight.shape=}"


# Define the Adapter module
@dataclass
class AdapterConfig:
    """
    Configuration for the Adapter module that handles various input embeddings and conditioning
    """

    hidden_size: int
    num_attention_heads: int

    # caption
    text_in_channels: int
    video_in_channels: int
    audio_in_channels: int

    params_dtype: torch.dtype


class Adapter(torch.nn.Module):
    """
    Adapter module that handles input embeddings, timestep embeddings, and caption conditioning
    """

    config: AdapterConfig

    def __init__(self, config: AdapterConfig):
        """Initialize modality projections and rotary coordinate embedding."""
        super().__init__()
        self.config = config

        # Linear projection for input features
        self.video_embedder = nn.Linear(
            config.video_in_channels, config.hidden_size, bias=True, dtype=torch.float32
        )

        # Projection for text embeddings for cross-attention
        self.text_embedder = nn.Linear(
            config.text_in_channels, config.hidden_size, bias=True, dtype=torch.float32
        )

        self.audio_embedder = nn.Linear(
            config.audio_in_channels, config.hidden_size, bias=True, dtype=torch.float32
        )

        # Initialize rotary position embeddings
        self.rope = ElementWiseFourierEmbed(
            config.hidden_size // config.num_attention_heads,
            in_pixels=False,
            learnable=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        coords_mapping: torch.Tensor,
        video_mask: torch.Tensor,
        audio_mask: torch.Tensor,
        text_mask: torch.Tensor,
    ):
        """
        Forward pass of the Adapter module

        Args:
            x: (num_tokens, max(image_token_channels, txt_token_channels))
            coords_mapping: (num_tokens, )
            video_mask: (num_tokens, )
            audio_mask: (num_tokens, )
            text_mask: (num_tokens, )

        Returns:
            output_x: (num_tokens, hidden_size)
            rope: (num_tokens, hidden_size // num_attention_heads)

        """

        # Calculate rotary position embeddings
        rope = self.rope(coords_mapping)

        output_x = torch.zeros(
            x.shape[0], self.config.hidden_size, device=x.device, dtype=x.dtype
        )

        output_x[text_mask] = self.text_embedder(
            x[text_mask, : self.config.text_in_channels],
            # output_dtype=torch.float32
        )

        # Project input features to hidden dimension
        output_x[audio_mask] = self.audio_embedder(
            x[audio_mask, : self.config.audio_in_channels],
            #  output_dtype=torch.float32
        )

        output_x[video_mask] = self.video_embedder(
            x[video_mask, : self.config.video_in_channels],
            #  output_dtype=torch.float32
        )

        return output_x, rope


class TransformerLayer(torch.nn.Module):
    """
    A single transformer layer with attention, MLP, and adaptive layer normalization
    """

    def __init__(self, config: ModelConfig, layer_idx: int):
        """Initialize one refiner layer from its indexed architecture settings."""
        super().__init__()
        num_modality = 3 if layer_idx in config.mm_layers else 1
        use_local_attn = layer_idx in config.local_attn_layers
        self.post_norm = layer_idx in config.post_norm_layers
        # Build attention and MLP modules
        attention_config = build_dataclass_from_config(
            AttentionConfig,
            config,
            {"num_modality": num_modality, "use_local_attn": use_local_attn},
        )
        self.attention: Attention = Attention(attention_config)

        if layer_idx in config.layer_activation_types:
            activation_type = MLPActivationType(
                config.layer_activation_types[layer_idx]
            )
        else:
            activation_type = MLPActivationType(config.activation_type)
        if activation_type in [MLPActivationType.SWIGLU7]:
            gated_act = True
            intermediate_size = int(config.hidden_size * 4 * 2 / 3) // 4 * 4
        else:
            gated_act = False
            intermediate_size = config.hidden_size * 4
        mlp_config = build_dataclass_from_config(
            MLPConfig,
            config,
            {
                "num_modality": num_modality,
                "activation_type": activation_type,
                "intermediate_size": intermediate_size,
                "gated_act": gated_act,
            },
        )
        self.mlp: MLP = MLP(mlp_config)
        if self.post_norm:
            self.attn_post_norm = MultiModalityRMSNorm(
                config.hidden_size, num_modality=num_modality
            )
            self.mlp_post_norm = MultiModalityRMSNorm(
                config.hidden_size, num_modality=num_modality
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        rope: torch.Tensor,
        permute_mapping: torch.Tensor,
        inv_permute_mapping: torch.Tensor,
        varlen_handler: VarlenHandler,
        local_attn_handler: FFAHandler,  # Key for distributed attention
        modality_dispatcher: ModalityDispatcher,
        cp_split_sizes: List[int],
    ) -> torch.Tensor:
        """
        Forward pass of the transformer layer

        Args:
            hidden_states: Input tensor
            rope: Rotary position embeddings
            magi_attn_key: Key for distributed self-attention computation

        Returns:
            hidden_states(torch.Tensor): Processed hidden states

        Shape:
            - hidden_states: (num_tokens, hidden_size)
            - rope: (num_tokens, embed_dim)
        """
        # Apply Attention
        attn_out = self.attention(
            hidden_states,
            rope,
            permute_mapping,
            inv_permute_mapping,
            varlen_handler,
            local_attn_handler,
            modality_dispatcher,
            cp_split_sizes,
        )
        if self.post_norm:
            attn_out = self.attn_post_norm(
                attn_out, modality_dispatcher=modality_dispatcher
            )
        hidden_states = hidden_states + attn_out

        # Apply MLP
        mlp_out = self.mlp(hidden_states, modality_dispatcher)
        if self.post_norm:
            mlp_out = self.mlp_post_norm(
                mlp_out, modality_dispatcher=modality_dispatcher
            )
        hidden_states = hidden_states + mlp_out

        return hidden_states


keep_preview_model_once = True


def config_patch(compile_config: CompileConfig) -> CompileConfig:
    global keep_preview_model_once
    if keep_preview_model_once:
        keep_preview_model_once = False
    else:
        compile_config.offload_config.gpu_resident_weight_ratio = 0.0
    return compile_config


@magi_compile(config_patch=config_patch, enable_if=_magi_compile_enabled)
class TransformerBlock(torch.nn.Module):
    """
    A stack of transformer layers
    """

    def __init__(self, model_config: ModelConfig):
        super().__init__()
        # Build all transformer layers
        self.layers: list[TransformerLayer] = nn.ModuleList()
        for layer_idx in range(model_config.num_layers):
            self.layers.append(TransformerLayer(model_config, layer_idx))

    def forward(
        self,
        x: torch.Tensor,
        rope: torch.Tensor,
        permute_mapping: torch.Tensor,
        inv_permute_mapping: torch.Tensor,
        varlen_handler: VarlenHandler,
        local_attn_handler: FFAHandler,  # Key for distributed attention
        modality_dispatcher: ModalityDispatcher,
        cp_split_sizes: List[int],
    ) -> torch.Tensor:
        """
        Forward pass of the transformer block

        Args:
            x(torch.Tensor): Input tensor
            rope(torch.Tensor): Rotary position embeddings
            local_attn_handler(FFAHandler): Key for distributed attention
            video_mask(torch.Tensor): Mask for video modality
            audio_mask(torch.Tensor): Mask for audio modality
            text_mask(torch.Tensor): Mask for text modality
            modality_dispatcher(ModalityDispatcher): Modality dispatcher
        Returns:
            x(torch.Tensor): Processed input features

        Shape:
            - x: (num_tokens, hidden_size)
            - rope: (num_tokens, embed_dim)
            - modality_mapping: (num_tokens, )
            - video_mask: (num_tokens, )
            - audio_mask: (num_tokens, )
            - text_mask: (num_tokens, )
        """

        for layer in self.layers:
            x = layer(
                x,
                rope,
                permute_mapping,
                inv_permute_mapping,
                varlen_handler,
                local_attn_handler,
                modality_dispatcher,
                cp_split_sizes,
            )

        return x


# Define the Transformer module
@dataclass
class TransformerConfig:
    hidden_size: int  # Size of hidden representations
    video_in_channels: int  # Number of input channels
    audio_in_channels: int  # Number of input channels
    text_in_channels: int  # Number of input channels
    params_dtype: torch.dtype  # Data type for parameters
    post_process_dtype: torch.dtype  # Data type for post-processing


@register_model("magi2_refiner", aliases=["magi2_refiner"])
class PostAdapter(torch.nn.Module):
    """Final per-modality normalization and projection back to latent channels."""

    def __init__(self, config: TransformerConfig) -> None:
        """Initialize video and audio normalization and output projections."""
        super().__init__()
        self.config = config
        self.final_norm_video = MultiModalityRMSNorm(config.hidden_size)
        self.final_norm_audio = MultiModalityRMSNorm(config.hidden_size)
        self.final_linear_video = nn.Linear(
            config.hidden_size,
            config.video_in_channels,
            bias=False,
            dtype=torch.float32,
        )
        self.final_linear_audio = nn.Linear(
            config.hidden_size,
            config.audio_in_channels,
            bias=False,
            dtype=torch.float32,
        )

    def forward(
        self, x: torch.Tensor, video_mask: torch.Tensor, audio_mask: torch.Tensor
    ) -> torch.Tensor:
        """Project normalized video and audio tokens into a latent tensor."""
        x_video = x[video_mask].to(self.final_norm_video.weight.dtype)
        x_video = self.final_norm_video(x_video)
        x_video = self.final_linear_video(x_video)

        x_audio = x[audio_mask].to(self.final_norm_audio.weight.dtype)
        x_audio = self.final_norm_audio(x_audio)
        x_audio = self.final_linear_audio(x_audio)

        x_out = torch.zeros(
            x.shape[0],
            max(self.config.video_in_channels, self.config.audio_in_channels),
            device=x.device,
            dtype=x.dtype,
        )
        x_out[video_mask, : self.config.video_in_channels] = x_video
        x_out[audio_mask, : self.config.audio_in_channels] = x_audio
        return x_out


class Transformer(BaseDiT):
    _config = Magi2RefinerVideoConfig()
    _fsdp_shard_conditions = _config._fsdp_shard_conditions
    _compile_conditions = _config._compile_conditions
    _supported_attention_backends = _config._supported_attention_backends
    param_names_mapping = _config.param_names_mapping
    reverse_param_names_mapping = _config.reverse_param_names_mapping
    lora_param_names_mapping = _config.lora_param_names_mapping

    config: TransformerConfig

    def __init__(
        self,
        config: Magi2RefinerVideoConfig,
        hf_config: dict | None = None,
        **kwargs,
    ) -> None:
        """Construct the production MAGI-2 refiner with official module names."""
        del kwargs
        super().__init__(config=config, hf_config=hf_config or {})
        self.fastvideo_config = config
        model_config = config.arch_config
        model_config.params_dtype = _resolve_params_dtype(model_config.params_dtype)
        self.hidden_size = model_config.hidden_size
        self.num_attention_heads = model_config.num_heads_q
        self.num_channels_latents = model_config.num_channels_latents

        self.config = build_dataclass_from_config(
            TransformerConfig, model_config, {"post_process_dtype": torch.float32}
        )
        adapter_config = build_dataclass_from_config(
            AdapterConfig,
            model_config,
            {
                "num_attention_heads": model_config.num_heads_q,
                "params_dtype": torch.float32,
            },
        )
        self.pre_adapter: Adapter = Adapter(adapter_config)

        self.block: TransformerBlock = TransformerBlock(model_config=model_config)

        self.post_adapter: PostAdapter = PostAdapter(self.config)
        self.__post_init__()

    @torch.compile(dynamic=True, fullgraph=False)
    def forward(
        self,
        x: torch.Tensor,
        coords_mapping: torch.Tensor,
        modality_mapping: torch.Tensor,
        varlen_handler: VarlenHandler,
        local_attn_handler: FFAHandler,  # Key for distributed attention
    ):
        """Refine packed video, audio, and text token features.

        Args:
            x(torch.Tensor): Input features
            coords_mapping(torch.Tensor): Mapping from tokens to coords for rope
            modality_mapping(torch.Tensor): Mapping from tokens to modality

        Returns:
            x(torch.Tensor): Processed input features

        """
        x = ulysses_scheduler().dispatch(x)
        coords_mapping = ulysses_scheduler().dispatch(coords_mapping)
        modality_mapping = ulysses_scheduler().dispatch(modality_mapping)
        cp_split_sizes = ulysses_scheduler().cp_split_sizes

        modality_dispatcher = ModalityDispatcher(modality_mapping, 3)
        permute_mapping, inv_permute_mapping = (
            modality_dispatcher.permute_mapping,
            modality_dispatcher.inv_permute_mapping,
        )
        # assert modality_dispatcher.is_sorted
        video_mask = modality_mapping == Modality.VIDEO
        audio_mask = modality_mapping == Modality.AUDIO
        text_mask = modality_mapping == Modality.TEXT

        # Process inputs through adapter
        x, rope = self.pre_adapter(
            x, coords_mapping, video_mask, audio_mask, text_mask
        )

        # NOTE(LLZ):only convert x to params_dtype, rope stays in float32
        x = x.to(self.config.params_dtype)
        x = ModalityDispatcher.permute(x, permute_mapping)
        x = self.block(
            x,
            rope,
            permute_mapping=permute_mapping,
            inv_permute_mapping=inv_permute_mapping,
            varlen_handler=varlen_handler,
            local_attn_handler=local_attn_handler,
            modality_dispatcher=modality_dispatcher,
            cp_split_sizes=cp_split_sizes,
        )
        x = ModalityDispatcher.inv_permute(x, inv_permute_mapping)

        x_out = self.post_adapter(x, video_mask, audio_mask)

        x_out = ulysses_scheduler().undispatch(x_out)

        return x_out


Magi2RefinerDiT = Transformer
EntryClass = Magi2RefinerDiT
