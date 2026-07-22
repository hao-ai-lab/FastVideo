import argparse
import sys

import torch

from fastvideo.train.entrypoint.train import main as train_main
from fastvideo.train.trainer import Trainer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--trace-dir", required=True)
    args, overrides = parser.parse_known_args()

    profiler = torch.profiler.profile(
        activities=(torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA),
        schedule=torch.profiler.schedule(wait=10, warmup=1, active=1, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(
            args.trace_dir, use_gzip=True),
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
        with_flops=False,
    )
    original_iter = Trainer._iter_dataloader
    original_run = Trainer.run

    def profiled_iter(self, dataloader):
        iterator = original_iter(self, dataloader)
        while True:
            batch = next(iterator)
            yield batch
            profiler.step()

    def profiled_run(self, *run_args, **run_kwargs):
        with profiler:
            return original_run(self, *run_args, **run_kwargs)

    Trainer._iter_dataloader = profiled_iter
    Trainer.run = profiled_run
    train_args = argparse.Namespace(config=args.config, dry_run=False)
    train_main(train_args, overrides=overrides or None)


if __name__ == "__main__":
    sys.exit(main())
