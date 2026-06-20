"""Tagged memory pools with reservation-before-admission.

Reservation must be pre-flight: a scheduler that admits a diffusion step and then discovers it
cannot allocate the VAE tile is wrong.

Sleep/wake is component-granular (tags are component names) for RL: drop DiT + caches,
keep VAE/text-encoder resident (CuMem-style).
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass


class OutOfMemory(Exception):
    pass


_res_ctr = itertools.count(1)


@dataclass
class Reservation:
    res_id: int
    tag: str
    nbytes: int
    active: bool = True


class MemoryManager:

    def __init__(self, total_bytes: int = 1 << 40, per_tag_budget: dict[str, int] | None = None):
        self.total_bytes = total_bytes
        self.per_tag_budget = dict(per_tag_budget or {})
        self.reserved = 0
        self._by_tag: dict[str, int] = {}
        self._reservations: dict[int, Reservation] = {}
        self._asleep: set[str] = set()

    @property
    def available(self) -> int:
        return self.total_bytes - self.reserved

    def can_reserve(self, tag: str, nbytes: int) -> bool:
        if nbytes > self.available:
            return False
        budget = self.per_tag_budget.get(tag)
        return budget is None or self._by_tag.get(tag, 0) + nbytes <= budget

    def reserve(self, tag: str, nbytes: int) -> Reservation:
        if not self.can_reserve(tag, nbytes):
            raise OutOfMemory(f"cannot reserve {nbytes} bytes for tag {tag!r} "
                              f"(available={self.available}, used={self._by_tag.get(tag, 0)})")
        res = Reservation(next(_res_ctr), tag, nbytes)
        self._reservations[res.res_id] = res
        self.reserved += nbytes
        self._by_tag[tag] = self._by_tag.get(tag, 0) + nbytes
        return res

    def release(self, res: Reservation) -> None:
        if not res.active:
            return
        res.active = False
        self._reservations.pop(res.res_id, None)
        self.reserved -= res.nbytes
        self._by_tag[res.tag] = max(0, self._by_tag.get(res.tag, 0) - res.nbytes)

    # component-granular sleep/wake (tags = component names) ------------------- #
    def sleep(self, tags: list[str]) -> int:
        freed = 0
        for tag in tags:
            self._asleep.add(tag)
            for res in [r for r in self._reservations.values() if r.tag == tag]:
                freed += res.nbytes
                self.release(res)
        return freed

    def wake(self, tags: list[str]) -> None:
        for tag in tags:
            self._asleep.discard(tag)
