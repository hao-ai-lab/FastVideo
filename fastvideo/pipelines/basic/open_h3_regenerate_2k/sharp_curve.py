# SPDX-License-Identifier: Apache-2.0
"""LMS Sharp curve AdaLN branches.

Fifty AdaLN projections get an extra low-rank term whose input is an 8-D curve interpolated from the model timestep
(``adaln_t_table``, 1025 x 8) instead of the layer input, so it cannot be folded into the LoRA. The term is added to the
projection output through a forward hook; ``set_timesteps`` must be called with the per-forward unique timesteps first.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file

CURVE_BRANCHES_FILE = "lora/sharp_curve_branches.safetensors"
CURVE_TABLE_FILE = "lora/adaln_t_table.bin"
CURVE_TABLE_SHAPE = (1025, 8)
NUM_CURVE_BRANCHES = 50
_KEY_PREFIX = "transformer."
_A_SUFFIX = ".curve_A.weight"
_B_SUFFIX = ".curve_B.weight"


def interpolate_curve(table: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
    """Linear interpolation of ``table`` rows at model times in [0, 1]; ``timesteps`` [U] -> [U, 8]."""
    t = timesteps.float()
    if t.numel() and (t.min() < 0 or t.max() > 1):
        raise ValueError("Sharp curve timesteps must lie in [0, 1].")
    table = table.to(t.device)
    position = t * (len(table) - 1)
    low = position.floor().long().clamp(max=len(table) - 2)
    return torch.lerp(table[low], table[low + 1], (position - low).unsqueeze(-1))


class SharpCurveBranches:

    def __init__(self, weights_dir: str | Path) -> None:
        root = Path(weights_dir)
        raw = torch.frombuffer(bytearray((root / CURVE_TABLE_FILE).read_bytes()), dtype=torch.float32)
        if raw.numel() != CURVE_TABLE_SHAPE[0] * CURVE_TABLE_SHAPE[1]:
            raise ValueError(f"{root / CURVE_TABLE_FILE} must hold a {CURVE_TABLE_SHAPE} float32 table.")
        self.table = raw.reshape(CURVE_TABLE_SHAPE).clone()
        tensors = load_file(str(root / CURVE_BRANCHES_FILE), device="cpu")
        self.factors: dict[str, tuple[torch.Tensor, torch.Tensor]] = {
            key[len(_KEY_PREFIX):-len(_A_SUFFIX)]: (value.float(), tensors[key[:-len(_A_SUFFIX)] + _B_SUFFIX].float())
            for key, value in tensors.items() if key.endswith(_A_SUFFIX)
        }
        if len(self.factors) != NUM_CURVE_BRANCHES:
            raise ValueError(f"Expected {NUM_CURVE_BRANCHES} Sharp curve branches, found {len(self.factors)}.")
        self.coords: torch.Tensor | None = None
        self._attached: set[int] = set()
        self._device_factors: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

    def set_timesteps(self, unique_timesteps: torch.Tensor) -> None:
        self.coords = interpolate_curve(self.table, unique_timesteps)

    def attach(self, transformer: torch.nn.Module) -> None:
        """Register the branch hooks on ``transformer`` once."""
        if id(transformer) in self._attached:
            return
        modules = dict(transformer.named_modules())
        missing = sorted(set(self.factors) - set(modules))
        if missing:
            raise ValueError(f"Sharp curve branches match no transformer module: {missing[:5]}")
        for name in self.factors:
            modules[name].register_forward_hook(self._hook(name))
        self._attached.add(id(transformer))

    def _hook(self, name: str):

        def add_branch(module, inputs, output):
            projected = output[0] if isinstance(output, tuple) else output
            coords = self.coords
            if coords is None or coords.shape[0] != projected.shape[0]:
                raise RuntimeError("Sharp curve timesteps were not set for this forward.")
            factors = self._device_factors.get(name)
            if factors is None or factors[0].device != projected.device:
                factors = tuple(f.to(projected.device) for f in self.factors[name])
                self._device_factors[name] = factors
            delta = F.linear(F.linear(coords, factors[0]), factors[1]).to(projected.dtype)
            projected = projected + delta
            return (projected, *output[1:]) if isinstance(output, tuple) else projected

        return add_branch
