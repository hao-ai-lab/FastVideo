#!/usr/bin/env python3
"""Scratch A/B for returning raw LTX-2 velocity during training.

This wraps ``benchmark_fastvideo_train_ltx2_singleton_timestep.py`` so the
two candidates remain independently selectable:

* no flags: exact source control
* ``--raw-velocity``: bypass x0 conversion and velocity reconstruction
* ``--singleton-timestep``: singleton uniform T2V timestep only
* both flags: stacked candidates

The checkout is never edited.  For the raw-velocity candidate the shared
transformer's module-level ``_to_denoised`` helper is temporarily replaced by
an identity and the modular LTX-2 adapter returns that raw output directly.
Validation is disabled by the wrapped benchmark, so inference semantics are
not part of this process-local monkeypatch.
"""

from __future__ import annotations

import json
from pathlib import Path
import runpy
import sys
from typing import Any


BASE_DRIVER = Path("/mnt/benchmark_fastvideo_train_ltx2_singleton_timestep.py")
EXPECTED_STEPS = 30


def _remove_flag(flag: str) -> bool:
    found = False
    kept = [sys.argv[0]]
    for argument in sys.argv[1:]:
        if argument == flag:
            found = True
        else:
            kept.append(argument)
    sys.argv[:] = kept
    return found


def main() -> None:
    raw_velocity = _remove_flag("--raw-velocity")
    singleton_timestep = "--singleton-timestep" in sys.argv
    self_test = "--self-test" in sys.argv
    if not BASE_DRIVER.is_file():
        raise RuntimeError(f"missing wrapped benchmark driver: {BASE_DRIVER}")

    bypass_calls = 0
    if raw_velocity and not self_test:
        import torch

        from fastvideo.forward_context import set_forward_context
        import fastvideo.models.dits.ltx2 as ltx2_dit_module
        from fastvideo.train.models.ltx2 import LTX2Model

        def _return_velocity(
            sample: torch.Tensor,
            velocity: torch.Tensor,
            sigma: torch.Tensor,
            calc_dtype: torch.dtype = torch.float32,
        ) -> torch.Tensor:
            del sample, sigma, calc_dtype
            nonlocal bypass_calls
            bypass_calls += 1
            return velocity

        def _predict_raw_velocity(
            self: LTX2Model,
            noisy_latents: torch.Tensor,
            timestep: torch.Tensor,
            batch: Any,
            *,
            conditional: bool,
            cfg_uncond: dict[str, Any] | None = None,
            attn_kind: str = "dense",
            clean_x: torch.Tensor | None = None,
            aug_t: torch.Tensor | None = None,
        ) -> torch.Tensor:
            if clean_x is not None or aug_t is not None:
                raise NotImplementedError("LTX2Model does not support teacher forcing inputs")
            device_type = self.device.type
            dtype = self._get_training_dtype()
            if conditional:
                text_dict = batch.conditional_dict
                if text_dict is None:
                    raise RuntimeError("Missing conditional_dict in TrainingBatch")
            else:
                text_dict = self._get_uncond_text_dict(batch, cfg_uncond=cfg_uncond)
            if attn_kind not in ("dense", "vsa"):
                raise ValueError(f"Unknown attn_kind: {attn_kind!r}")

            noisy_bcthw = noisy_latents.permute(0, 2, 1, 3, 4).to(dtype)
            with torch.autocast(device_type, dtype=dtype), set_forward_context(
                    current_timestep=batch.timesteps,
                    attn_metadata=batch.attn_metadata,
                    forward_batch=self._make_rope_forward_batch(),
            ):
                input_kwargs = self._build_distill_input_kwargs(
                    noisy_bcthw,
                    timestep,
                    text_dict,
                )
                transformer = self._get_transformer(timestep)
                velocity = transformer(**input_kwargs)

            if isinstance(velocity, tuple):
                velocity = velocity[0]
            return velocity.permute(0, 2, 1, 3, 4)

        ltx2_dit_module._to_denoised = _return_velocity
        LTX2Model.predict_noise = _predict_raw_velocity

    runpy.run_path(str(BASE_DRIVER), run_name="__main__")

    if self_test:
        print(
            "RAW_VELOCITY_SELF_TEST " + json.dumps({
                "raw_velocity": raw_velocity,
                "singleton_timestep": singleton_timestep,
            }, sort_keys=True),
            flush=True,
        )
        return

    import torch.distributed as dist

    from fastvideo.distributed import get_world_group

    world = get_world_group() if dist.is_initialized() else None
    rank = world.rank if world is not None else 0
    world_size = world.world_size if world is not None else 1
    counts: list[int | None] = [None] * world_size
    if world is not None:
        dist.all_gather_object(
            counts,
            bypass_calls,
            group=world.cpu_group,
        )
    else:
        counts[0] = bypass_calls
    expected = EXPECTED_STEPS if raw_velocity else 0
    if any(count != expected for count in counts):
        raise RuntimeError(
            f"raw-velocity bypass count mismatch: expected {expected} per rank, got {counts}"
        )
    if rank == 0:
        print(
            "BF16_VARIANT " + json.dumps({
                "raw_velocity": raw_velocity,
                "singleton_timestep": singleton_timestep,
                "to_denoised_bypass_calls_by_rank": counts,
            }, sort_keys=True),
            flush=True,
        )


if __name__ == "__main__":
    main()
