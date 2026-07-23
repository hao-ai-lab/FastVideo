#!/usr/bin/env python3
"""Run FastVideo training with scratch-only public FSDP2 performance knobs."""

from __future__ import annotations

import argparse
import gc
import json
import statistics

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FSDPModule

from fastvideo.training.trackers import DummyTracker


_step_times: list[float] = []
_gradient_sync_calls: list[bool] = []
_forced_reshard_calls = 0
_original_log = DummyTracker.log


def _log(self, metrics, step):
    _original_log(self, metrics, step)
    if "step_time_sec" in metrics:
        value = float(metrics["step_time_sec"])
        if not dist.is_initialized() or dist.get_rank() == 0:
            print("BF16_STEP " + json.dumps({"step": step, "step_time_sec": value}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--pg-alloc", action="store_true")
    parser.add_argument("--prefetch-depth", type=int, default=0)
    parser.add_argument("--force-sync", action="store_true")
    parser.add_argument("--force-reshard-after-backward", action="store_true")
    args, overrides = parser.parse_known_args()
    if args.prefetch_depth < 0:
        raise ValueError("--prefetch-depth must be nonnegative")
    DummyTracker.log = _log

    import fastvideo.train.trainer as trainer_module
    import fastvideo.train.utils.moduleloader as moduleloader
    from fastvideo.train.methods.base import TrainingMethod
    from fastvideo.distributed import get_world_group
    from fastvideo.train.entrypoint.train import main as train_main

    original_load_module = moduleloader.load_module_from_path
    original_set_requires_gradient_sync = TrainingMethod.set_requires_gradient_sync

    def _set_requires_gradient_sync(self, enabled: bool) -> None:
        global _forced_reshard_calls
        enabled = True if args.force_sync else enabled
        if not dist.is_initialized() or dist.get_rank() == 0:
            _gradient_sync_calls.append(enabled)
        original_set_requires_gradient_sync(self, enabled)
        if args.force_reshard_after_backward and not enabled:
            for model in self._role_models.values():
                transformer = getattr(model, "transformer", None)
                if getattr(model, "_trainable", False) and isinstance(transformer, FSDPModule):
                    transformer.set_reshard_after_backward(True, recurse=True)
            if not dist.is_initialized() or dist.get_rank() == 0:
                _forced_reshard_calls += 1

    TrainingMethod.set_requires_gradient_sync = _set_requires_gradient_sync

    def _load_module_from_path(*load_args, **load_kwargs):
        module = original_load_module(*load_args, **load_kwargs)
        if load_kwargs.get("module_type") != "transformer":
            return module
        fsdp_modules = [submodule for submodule in module.modules() if isinstance(submodule, FSDPModule)]
        if not fsdp_modules:
            raise RuntimeError("scratch FSDP knob benchmark requires an FSDP-wrapped transformer")
        if args.pg_alloc:
            for fsdp_module in fsdp_modules:
                fsdp_module.set_allocate_memory_from_process_group_for_comm(True)
        if args.prefetch_depth:
            depth = args.prefetch_depth
            for index, fsdp_module in enumerate(fsdp_modules):
                forward = fsdp_modules[index + 1:index + 1 + depth]
                backward = list(reversed(fsdp_modules[max(0, index - depth):index]))
                if forward:
                    fsdp_module.set_modules_to_forward_prefetch(forward)
                if backward:
                    fsdp_module.set_modules_to_backward_prefetch(backward)
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(
                "BF16_FSDP_KNOBS " + json.dumps({
                    "module_count": len(fsdp_modules),
                    "pg_alloc": args.pg_alloc,
                    "prefetch_depth": args.prefetch_depth,
                }, sort_keys=True),
                flush=True,
            )
        return module

    moduleloader.load_module_from_path = _load_module_from_path
    trainer_module.build_tracker = lambda *_args, **_kwargs: DummyTracker()
    original_trainer_init = trainer_module.Trainer.__init__
    original_trainer_run = trainer_module.Trainer.run

    class _Recorder:

        def on_training_step_end(self, _method, metrics, iteration=0):
            del iteration
            _step_times.append(float(metrics["step_time_sec"]))

    def _trainer_init(self, *init_args, **init_kwargs):
        original_trainer_init(self, *init_args, **init_kwargs)
        self.callbacks._callbacks.pop("validation", None)
        self.callbacks._callbacks["_benchmark_recorder"] = _Recorder()

    def _trainer_run(self, method, **kwargs):
        vae = getattr(getattr(method, "student", None), "vae", None)
        if vae is not None:
            method.student.vae = None
            del vae
            gc.collect()
            torch.cuda.empty_cache()
            if not dist.is_initialized() or dist.get_rank() == 0:
                print("BF16_SETUP " + json.dumps({"unused_vae": "unloaded"}), flush=True)
        torch.cuda.reset_peak_memory_stats()
        return original_trainer_run(self, method, **kwargs)

    trainer_module.Trainer.__init__ = _trainer_init
    trainer_module.Trainer.run = _trainer_run

    train_main(argparse.Namespace(config=args.config, dry_run=False), overrides=overrides or None)
    world = get_world_group()
    times_by_rank = [None] * world.world_size
    dist.all_gather_object(times_by_rank, _step_times, group=world.cpu_group)
    peak_memory = torch.tensor(
        [torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()],
        device="cuda",
        dtype=torch.int64,
    )
    dist.all_reduce(peak_memory, op=dist.ReduceOp.MAX)
    if world.rank == 0:
        per_step_max = [max(values) for values in zip(*times_by_rank, strict=True)]
        measured = per_step_max[10:30]
        if len(measured) != 20:
            raise RuntimeError(f"expected 30 steps, got {len(per_step_max)}")
        median = statistics.median(measured)
        print(
            "BF16_RESULT " + json.dumps({
                "median_step_sec": median,
                "model_mfu_percent": 14.444115 / median,
                "samples_per_second": world.world_size / median,
                "world_size": world.world_size,
                "pg_alloc": args.pg_alloc,
                "prefetch_depth": args.prefetch_depth,
                "force_sync": args.force_sync,
                "force_reshard_after_backward": args.force_reshard_after_backward,
                "forced_reshard_calls": _forced_reshard_calls,
                "gradient_sync_false_calls": _gradient_sync_calls.count(False),
                "gradient_sync_true_calls": _gradient_sync_calls.count(True),
                "peak_allocated_max_rank_bytes": int(peak_memory[0]),
                "peak_reserved_max_rank_bytes": int(peak_memory[1]),
            }, sort_keys=True),
            flush=True,
        )


if __name__ == "__main__":
    main()
