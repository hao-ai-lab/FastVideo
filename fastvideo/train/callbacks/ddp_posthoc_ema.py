# SPDX-License-Identifier: Apache-2.0
"""Official nitrous PostHocEMA lifecycle for native DDP training."""

from __future__ import annotations

from pathlib import Path
from typing import Any, TYPE_CHECKING

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from fastvideo.logger import init_logger
from fastvideo.train.callbacks.callback import Callback
from fastvideo.train.utils.distributed_strategy import unwrap_ddp_module

if TYPE_CHECKING:
    from fastvideo.train.methods.base import TrainingMethod

logger = init_logger(__name__)


class DDPPostHocEMACallback(Callback):
    """Run the exact ``nitrous_ema.PostHocEMA`` used by MMAudio.

    Official MMAudio owns two complete CUDA EMA models on local rank zero and
    checkpoints them every 5,000 optimizer steps. Other DDP ranks do not hold
    EMA copies. This callback preserves that lifecycle while the ordinary
    FastVideo FSDP callback continues to maintain rank-local CPU shards.
    """

    def __init__(
        self,
        *,
        sigma_rels: list[float] | tuple[float, ...] = (0.05, 0.1),
        update_every: int = 1,
        checkpoint_every: int = 5000,
        checkpoint_folder: str | None = None,
        start_iter: int = 0,
        default_output_sigma: float = 0.05,
        step_size_correction: bool = True,
    ) -> None:
        self.sigma_rels = tuple(float(value) for value in sigma_rels)
        self.update_every = max(1, int(update_every))
        self.checkpoint_every = max(1, int(checkpoint_every))
        self.checkpoint_folder = checkpoint_folder
        self.start_iter = int(start_iter)
        self.default_output_sigma = float(default_output_sigma)
        self.step_size_correction = bool(step_size_correction)

        self._ema: Any | None = None
        self._calls = 0
        self._rank = 0
        self._checkpoint_root: Path | None = None

    def on_train_start(
        self,
        method: TrainingMethod,
        iteration: int = 0,
    ) -> None:
        del iteration
        transformer = method.student.transformer
        if not isinstance(transformer, DistributedDataParallel):
            raise TypeError("DDPPostHocEMACallback requires a native DDP transformer")

        self._rank = dist.get_rank() if dist.is_initialized() else 0
        root = self.checkpoint_folder
        if not root:
            root = str(Path(self.training_config.checkpoint.output_dir) / "posthoc_ema" / "official_ddp")
        self._checkpoint_root = Path(root).expanduser().resolve()

        if self._rank == 0:
            try:
                from nitrous_ema import PostHocEMA
            except ImportError as exc:
                raise ImportError("DDP MMAudio training requires nitrous-ema. Install "
                                  "FastVideo with the mmaudio-train extra.") from exc

            module = unwrap_ddp_module(transformer)
            self._ema = PostHocEMA(
                module,
                sigma_rels=self.sigma_rels,
                update_every=self.update_every,
                checkpoint_every_num_steps=self.checkpoint_every,
                checkpoint_folder=str(self._checkpoint_root),
                step_size_correction=self.step_size_correction,
            ).to(method.student.device)
            logger.info(
                "Official DDP PostHocEMA enabled on rank 0: "
                "sigma_rels=%s checkpoint_every=%d folder=%s",
                self.sigma_rels,
                self.checkpoint_every,
                self._checkpoint_root,
            )

    def on_training_step_end(
        self,
        method: TrainingMethod,
        loss_dict: dict[str, Any],
        iteration: int = 0,
    ) -> None:
        del method, loss_dict
        if iteration < self.start_iter:
            return
        self._calls += 1
        if self._rank == 0:
            if self._ema is None:
                raise RuntimeError("DDP PostHocEMA was not initialized")
            self._ema.update()
            if int(self._ema.step) != self._calls:
                raise RuntimeError("DDP PostHocEMA step diverged from the trainer step")

    def _restore_checkpoint(self) -> None:
        if self._rank != 0 or self._ema is None or self._calls <= 0:
            return
        if self._checkpoint_root is None:
            raise RuntimeError("DDP PostHocEMA checkpoint folder is unset")
        for index, ema_model in enumerate(self._ema.ema_models):
            path = self._checkpoint_root / f"{index}.{self._calls}.pt"
            if not path.is_file():
                raise FileNotFoundError("Cannot resume DDP PostHocEMA: expected matching snapshot "
                                        f"{path}. Keep training checkpoints aligned with "
                                        "checkpoint_every.")
            state = torch.load(
                path,
                map_location=method_device(ema_model),
                weights_only=True,
            )
            ema_model.load_state_dict(state)
        logger.info("Restored official DDP PostHocEMA at step %d", self._calls)

    def state_dict(self) -> dict[str, Any]:
        # Full nitrous states already live in rank-0 snapshots. Keeping only
        # the common step in DCP gives every rank the same callback schema.
        return {"calls": self._calls}

    def load_state_dict(
        self,
        state_dict: dict[str, Any],
    ) -> None:
        self._calls = int(state_dict.get("calls", 0))
        self._restore_checkpoint()

    def on_train_end(
        self,
        method: TrainingMethod,
        iteration: int = 0,
    ) -> None:
        del method, iteration
        if self._rank != 0 or self._ema is None or self._calls <= 0:
            return
        if self._calls % self.checkpoint_every != 0:
            self._ema.checkpoint()
        synthesized = self._ema.synthesize_ema_model(
            sigma_rel=self.default_output_sigma,
            step=self._calls,
            device="cpu",
        )
        if self._checkpoint_root is None:
            raise RuntimeError("DDP PostHocEMA checkpoint folder is unset")
        sigma_name = str(self.default_output_sigma).replace(".", "p")
        path = self._checkpoint_root / (f"mmaudio_ema_final_sigma_{sigma_name}_step_"
                                        f"{self._calls:09d}.pth")
        torch.save(synthesized.ema_model.state_dict(), path)
        logger.info("Saved official synthesized DDP PostHocEMA to %s", path)


def method_device(module: torch.nn.Module) -> torch.device:
    """Return the device of a nitrous EMA module."""
    return next(module.parameters()).device


__all__ = ["DDPPostHocEMACallback"]
