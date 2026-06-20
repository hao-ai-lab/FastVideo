"""FakeDeviceMesh — a torchtitan-style ParallelDims builder, CPU-testable.

On a GPU box this compiles to ``torch.distributed.device_mesh.DeviceMesh``. Here it is a
pure-Python mesh of rank tuples so topology logic is unit-tested without GPUs.
Degree-one axes exist as trivial groups so component code needs no special cases.
"""
from __future__ import annotations

from itertools import product

from v2.core.parallel.plan import ParallelPlan
from v2.core.parallel.validation import validate_plan


class FakeDeviceMesh:

    def __init__(self, plan: ParallelPlan):
        self.plan = plan
        self.order = plan.mesh_order or list(plan.axes.keys())
        self.shape = tuple(plan.degree(a) for a in self.order)
        self.world_size = plan.world_size()

    def group_ranks(self, axis: str) -> list[list[int]]:
        """All collective groups along one axis (each is a list of global ranks)."""
        if axis not in self.order:
            return [[0]]  # degree-one trivial group
        axis_idx = self.order.index(axis)
        groups: list[list[int]] = []
        ranges = [range(d) for d in self.shape]
        seen: set[tuple] = set()
        for coord in product(*ranges):
            key = coord[:axis_idx] + coord[axis_idx + 1:]
            if key in seen:
                continue
            seen.add(key)
            group = []
            for i in range(self.shape[axis_idx]):
                c = list(coord)
                c[axis_idx] = i
                group.append(self._coord_to_rank(tuple(c)))
            groups.append(group)
        return groups

    def _coord_to_rank(self, coord: tuple[int, ...]) -> int:
        rank = 0
        for c, d in zip(coord, self.shape, strict=False):
            rank = rank * d + c
        return rank

    def __repr__(self) -> str:
        return f"FakeDeviceMesh(order={self.order}, shape={self.shape}, world={self.world_size})"


def build_mesh(plan: ParallelPlan, card=None, **kw) -> FakeDeviceMesh:
    validate_plan(plan, card=card, **kw)
    return FakeDeviceMesh(plan)
