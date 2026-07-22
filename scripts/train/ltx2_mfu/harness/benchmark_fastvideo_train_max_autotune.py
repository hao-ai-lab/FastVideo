#!/usr/bin/env python3
"""Benchmark FastVideo training with regional max-autotune and no CUDA graphs."""

import argparse
import json
import statistics

import torch
import torch.distributed as dist

from fastvideo.training.trackers import DummyTracker


_step_times: list[float] = []
_compile_calls = 0
_original_log = DummyTracker.log
_original_compile = torch.compile


def _log(self, metrics, step):
    _original_log(self, metrics, step)
    if "step_time_sec" in metrics and (not dist.is_initialized() or dist.get_rank() == 0):
        print("BF16_STEP " + json.dumps({"step": step, "step_time_sec": float(metrics["step_time_sec"])}),
              flush=True)


def _compile(*args, **kwargs):
    global _compile_calls
    if "mode" in kwargs or "options" in kwargs:
        raise RuntimeError(f"unexpected compile configuration: {kwargs}")
    kwargs["mode"] = "max-autotune-no-cudagraphs"
    _compile_calls += 1
    return _original_compile(*args, **kwargs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args, overrides = parser.parse_known_args()

    torch.compile = _compile
    DummyTracker.log = _log

    import fastvideo.train.trainer as trainer_module
    from fastvideo.distributed import get_world_group
    from fastvideo.train.entrypoint.train import main as train_main

    trainer_module.build_tracker = lambda *_args, **_kwargs: DummyTracker()
    original_trainer_init = trainer_module.Trainer.__init__

    class _Recorder:
        def on_training_step_end(self, _method, metrics, iteration=0):
            _step_times.append(float(metrics["step_time_sec"]))

    def _trainer_init(self, *init_args, **init_kwargs):
        original_trainer_init(self, *init_args, **init_kwargs)
        self.callbacks._callbacks.pop("validation", None)
        self.callbacks._callbacks["_benchmark_recorder"] = _Recorder()

    trainer_module.Trainer.__init__ = _trainer_init
    train_main(argparse.Namespace(config=args.config, dry_run=False), overrides=overrides or None)

    if _compile_calls != 48:
        raise RuntimeError(f"expected 48 regional compile calls, got {_compile_calls}")
    world = get_world_group()
    times_by_rank = [None] * world.world_size
    dist.all_gather_object(times_by_rank, _step_times, group=world.cpu_group)
    if world.rank == 0:
        per_step_max = [max(values) for values in zip(*times_by_rank, strict=True)]
        measured = per_step_max[10:30]
        if len(measured) != 20:
            raise RuntimeError(f"expected 30 steps, got {len(per_step_max)}")
        median = statistics.median(measured)
        print("BF16_COMPILE_MODE " + json.dumps({
            "calls": _compile_calls,
            "mode": "max-autotune-no-cudagraphs",
        }, sort_keys=True), flush=True)
        print("BF16_RESULT " + json.dumps({
            "kind": "regional_max_autotune_no_cudagraphs",
            "median_step_sec": median,
            "samples_per_second": 4.0 / median,
            "model_mfu_percent": 14.444115 / median,
        }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
