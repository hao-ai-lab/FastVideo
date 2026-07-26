#!/usr/bin/env python3
"""Audit the PR #1630 MFU numerator by counting executed per-sample training FLOPs.

Wraps each counted microstep (forward ``single_train_step`` plus ``backward``)
in ``torch.utils.flop_counter.FlopCounterMode`` and reports per-rank forward,
backward, and total FLOPs. Run with ``--models.student.attention_backend
TORCH_SDPA`` and ``--training.model.enable_torch_compile false``: compiled
regions and the FA4 custom op bypass the dispatch-mode counter, so FLASH_ATTN
or compiled runs undercount attention. The counter uses PyTorch's SDPA flop
formulas, which charge the flash backward recompute (about 2.5x the attention
forward) rather than the 2x pure-gradient convention.
"""

from __future__ import annotations

import argparse
import json
import os

import torch
import torch.distributed as dist
from torch.utils.flop_counter import FlopCounterMode

COUNT_STEPS = (2, 3)

_records: list[dict[str, int]] = []
_input_shapes: dict[str, list] = {}
_run_config: dict[str, int] = {}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args, overrides = parser.parse_known_args()

    from fastvideo.training.trackers import DummyTracker

    import fastvideo.train.trainer as trainer_module
    from fastvideo.distributed import get_world_group
    from fastvideo.train.entrypoint.train import main as train_main
    from fastvideo.train.methods.fine_tuning.finetune import FineTuneMethod
    from fastvideo.train.models.ltx2 import LTX2Model

    trainer_module.build_tracker = lambda *_a, **_k: DummyTracker()

    original_trainer_init = trainer_module.Trainer.__init__

    def _trainer_init(self, *init_args, **init_kwargs):
        original_trainer_init(self, *init_args, **init_kwargs)
        self.callbacks._callbacks.pop("validation", None)
        _run_config["local_batch_size"] = int(self.training_config.data.train_batch_size)
        _run_config["gradient_accumulation_steps"] = int(
            self.training_config.loop.gradient_accumulation_steps or 1)
        if bool(self.training_config.model.enable_torch_compile):
            raise RuntimeError("FLOP audit requires enable_torch_compile=false")

    trainer_module.Trainer.__init__ = _trainer_init

    original_run = trainer_module.Trainer.run

    def _trainer_run(self, method, **kwargs):
        vae = getattr(getattr(method, "student", None), "vae", None)
        if vae is not None:
            method.student.vae = None
        return original_run(self, method, **kwargs)

    trainer_module.Trainer.run = _trainer_run

    original_build = LTX2Model._build_distill_input_kwargs

    def _build(self, *build_args, **build_kwargs):
        result = original_build(self, *build_args, **build_kwargs)
        if not _input_shapes:
            for key, value in result.items():
                if isinstance(value, torch.Tensor):
                    _input_shapes[key] = [list(value.shape), str(value.dtype)]
        return result

    LTX2Model._build_distill_input_kwargs = _build

    # FineTuneMethod overrides both single_train_step and backward, so the
    # subclass attributes must be patched; base-class patches never fire.
    original_step = FineTuneMethod.single_train_step
    original_backward = FineTuneMethod.backward
    state: dict[str, object] = {"mode": None, "fwd": 0, "step": None}

    def _counted_step(self, batch, step):
        if int(step) in COUNT_STEPS:
            mode = FlopCounterMode(display=False)
            mode.__enter__()
            state["mode"] = mode
            state["step"] = int(step)
        outputs = original_step(self, batch, step)
        if state["mode"] is not None:
            state["fwd"] = state["mode"].get_total_flops()
        return outputs

    def _counted_backward(self, loss_map, outputs, **kwargs):
        result = original_backward(self, loss_map, outputs, **kwargs)
        mode = state["mode"]
        if mode is not None:
            mode.__exit__(None, None, None)
            total = int(mode.get_total_flops())
            fwd = int(state["fwd"])
            _records.append({
                "step": state["step"],
                "forward_flops": fwd,
                "backward_flops": total - fwd,
                "total_flops": total,
            })
            state["mode"] = None
        return result

    FineTuneMethod.single_train_step = _counted_step
    FineTuneMethod.backward = _counted_backward

    train_main(argparse.Namespace(config=args.config, dry_run=False), overrides=overrides or None)

    if len(_records) != len(COUNT_STEPS):
        raise RuntimeError(f"expected {len(COUNT_STEPS)} counted microsteps, got {len(_records)}")

    properties = torch.cuda.get_device_properties(0)
    payload = {
        "records": _records,
        "input_shapes": _input_shapes,
        "run_config": _run_config,
        "device": {
            "name": properties.name,
            "multi_processor_count": properties.multi_processor_count,
            "capability": list(torch.cuda.get_device_capability(0)),
        },
        "torch": torch.__version__,
        "attention_backend_env": os.environ.get("FASTVIDEO_ATTENTION_BACKEND"),
    }
    world = get_world_group()
    payloads: list[dict | None] = [None] * world.world_size
    dist.all_gather_object(payloads, payload, group=world.cpu_group)
    if world.rank != 0:
        return
    totals = {tuple(sorted(r["total_flops"] for r in p["records"])) for p in payloads}
    per_sample = None
    batch = payloads[0]["run_config"]["local_batch_size"]
    steady = payloads[0]["records"][-1]["total_flops"]
    per_sample = steady / batch
    print(
        "FLOP_AUDIT " + json.dumps({
            "by_rank": payloads,
            "rank_total_sets_identical": len(totals) == 1,
            "local_batch_size": batch,
            "per_sample_total_flops_last_counted_step": per_sample,
            "per_sample_tflops": per_sample / 1e12,
        }, sort_keys=True),
        flush=True,
    )


if __name__ == "__main__":
    main()
