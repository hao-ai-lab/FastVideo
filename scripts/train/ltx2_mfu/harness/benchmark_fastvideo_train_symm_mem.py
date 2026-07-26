#!/usr/bin/env python3
"""Benchmark FastVideo training with FSDP2 symmetric-memory all-gather."""

import argparse
import json
import os
import statistics

import torch.distributed as dist

from fastvideo.training.trackers import DummyTracker


_step_times: list[float] = []
_original_log = DummyTracker.log


def _log(self, metrics, step):
    _original_log(self, metrics, step)
    if "step_time_sec" in metrics and (not dist.is_initialized() or dist.get_rank() == 0):
        print("BF16_STEP " + json.dumps({"step": step, "step_time_sec": float(metrics["step_time_sec"])}),
              flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args, overrides = parser.parse_known_args()
    if os.environ.get("NCCL_CTA_POLICY") != "2":
        raise RuntimeError("symmetric-memory all-gather requires NCCL_CTA_POLICY=2")

    import fastvideo.train.models.ltx2.ltx2 as ltx2_module
    import fastvideo.train.trainer as trainer_module
    from fastvideo.distributed import get_world_group
    from fastvideo.train.entrypoint.train import main as train_main
    from torch.distributed.fsdp import FSDPModule

    original_load = ltx2_module.load_module_from_path

    def load_with_symm_mem(**kwargs):
        module = original_load(**kwargs)
        if kwargs.get("module_type") == "transformer":
            if not isinstance(module, FSDPModule):
                raise RuntimeError("expected an FSDP2 transformer")
            configured = []
            for name, submodule in module.named_modules():
                if isinstance(submodule, FSDPModule):
                    submodule.set_force_sum_reduction_for_comms(True)
                    submodule.set_symm_mem_for_comm("NCCL")
                    configured.append(name or "<root>")
            if len(configured) != 49 or configured[0] != "<root>":
                raise RuntimeError(f"expected root plus 48 FSDP2 blocks, got {configured}")
            if dist.get_rank() == 0:
                print(f"BF16_SYMM_MEM enabled on {len(configured)} modules", flush=True)
        return module

    ltx2_module.load_module_from_path = load_with_symm_mem
    DummyTracker.log = _log
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

    world = get_world_group()
    times_by_rank = [None] * world.world_size
    dist.all_gather_object(times_by_rank, _step_times, group=world.cpu_group)
    if world.rank == 0:
        per_step_max = [max(values) for values in zip(*times_by_rank, strict=True)]
        measured = per_step_max[10:30]
        if len(measured) != 20:
            raise RuntimeError(f"expected 30 steps, got {len(per_step_max)}")
        median = statistics.median(measured)
        print("BF16_RESULT " + json.dumps({
            "kind": "fsdp2_symm_mem",
            "median_step_sec": median,
            "samples_per_second": 4.0 / median,
            "model_mfu_percent": 14.444115 / median,
        }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
