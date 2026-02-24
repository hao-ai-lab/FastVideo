# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Literal

SamplerKind = Literal["ode", "sde"]


def normalize_sampler_kind(
    raw: str | None,
    *,
    where: str,
    default: SamplerKind = "ode",
) -> SamplerKind:
    if raw is None:
        return default

    kind = str(raw).strip().lower()
    if kind == "ode":
        return "ode"
    if kind == "sde":
        return "sde"

    raise ValueError(f"Unknown sampler kind at {where}: {raw!r} (expected ode|sde)")

