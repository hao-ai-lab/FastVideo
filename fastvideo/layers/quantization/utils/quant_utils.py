# Adapted from vllm: https://github.com/vllm-project/vllm/blob/v0.7.3/vllm/model_executor/layers/quantization/utils/quant_utils.py
from dataclasses import dataclass
from typing import ClassVar, NamedTuple

import torch

FP8_DTYPE = torch.float8_e4m3fn


def get_fp8_min_max() -> tuple[float, float]:
    finfo = torch.finfo(FP8_DTYPE)
    return finfo.min, finfo.max


class _GroupShape(NamedTuple):
    row: int
    col: int


class GroupShape(_GroupShape):
    """
    Describes quantization group shape.

    Uses -1 to indicate full extent along a dimension:
      - PER_TENSOR: (-1, -1) — one scale for the entire tensor
      - PER_TOKEN:  (1, -1)  — one scale per row (token)
      - PER_CHANNEL: (-1, 1) — one scale per column (output channel)
      - Arbitrary block: (r, c) — e.g. (1, 128) for group-size-128
    """

    PER_TENSOR: ClassVar["GroupShape"]
    PER_TOKEN: ClassVar["GroupShape"]
    PER_CHANNEL: ClassVar["GroupShape"]

    def is_per_tensor(self) -> bool:
        return self.row == -1 and self.col == -1

    def is_per_token(self) -> bool:
        return self.row == 1 and self.col == -1

    def is_per_channel(self) -> bool:
        return self.row == -1 and self.col == 1

    def is_per_group(self) -> bool:
        return self.row == 1 and self.col >= 1


GroupShape.PER_TENSOR = GroupShape(-1, -1)
GroupShape.PER_TOKEN = GroupShape(1, -1)
GroupShape.PER_CHANNEL = GroupShape(-1, 1)


@dataclass(frozen=True)
class ScaleDesc:
    """
    Describes a single quantization scaling factor.

    dtype: data type of the scale
    static: static scale if True, dynamic if False
    group_shape: group shape of the scale
    """

    dtype: torch.dtype
    static: bool
    group_shape: GroupShape

    def __str__(self) -> str:
        shape_names = {
            GroupShape.PER_TENSOR: "per_tensor",
            GroupShape.PER_TOKEN: "per_token",
            GroupShape.PER_CHANNEL: "per_channel",
        }
        group_shape = shape_names.get(self.group_shape, str(self.group_shape))
        mode = "static" if self.static else "dynamic"
        return f"{self.dtype},{mode},{group_shape}"


@dataclass(frozen=True)
class QuantKey:
    """
    Identifies the type of quantization.

    dtype: quantized data type
    scale: scale descriptor
    scale2: second-level scale descriptor
    symmetric: symmetric if True, asymmetric if False
    """

    dtype: torch.dtype
    scale: ScaleDesc
    scale2: ScaleDesc | None = None
    symmetric: bool = True

    def __str__(self) -> str:
        scale2_str = f"scale2({self.scale2})," if self.scale2 else ""
        sym = "symmetric" if self.symmetric else "asymmetric"
        return f"QuantKey({self.dtype},scale({self.scale}),{scale2_str}{sym})"


kStaticTensorScale = ScaleDesc(torch.float32, True, GroupShape.PER_TENSOR)
kFp8StaticTensorSym = QuantKey(FP8_DTYPE, kStaticTensorScale, symmetric=True)

kDynamicTensorScale = ScaleDesc(torch.float32, False, GroupShape.PER_TENSOR)
kFp8DynamicTensorSym = QuantKey(FP8_DTYPE, kDynamicTensorScale, symmetric=True)

kStaticTokenScale = ScaleDesc(torch.float32, True, GroupShape.PER_TOKEN)
kFp8StaticTokenSym = QuantKey(FP8_DTYPE, kStaticTokenScale, symmetric=True)

kDynamicTokenScale = ScaleDesc(torch.float32, False, GroupShape.PER_TOKEN)
kFp8DynamicTokenSym = QuantKey(FP8_DTYPE, kDynamicTokenScale, symmetric=True)

kStaticChannelScale = ScaleDesc(torch.float32, True, GroupShape.PER_CHANNEL)
kFp8StaticChannelSym = QuantKey(FP8_DTYPE, kStaticChannelScale, symmetric=True)

kStatic128BlockScale = ScaleDesc(torch.float32, True, GroupShape(128, 128))
kFp8Static128BlockSym = QuantKey(FP8_DTYPE,
                                 kStatic128BlockScale,
                                 symmetric=True)

kDynamic128Scale = ScaleDesc(torch.float32, False, GroupShape(1, 128))
kFp8Dynamic128Sym = QuantKey(FP8_DTYPE, kDynamic128Scale, symmetric=True)


def is_layer_skipped(prefix: str, ignored_layers: list[str]) -> bool:
    """
    Check whether a layer should be ignored for quantization.
    """
    if prefix in ignored_layers:
        return True
    proj_name = prefix.split(".")[-1]
    return any(layer in (prefix, proj_name) for layer in ignored_layers)