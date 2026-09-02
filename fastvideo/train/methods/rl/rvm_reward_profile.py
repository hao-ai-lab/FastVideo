# SPDX-License-Identifier: Apache-2.0
"""RVM method variant with a scorer-declared distributed reward contract."""

from __future__ import annotations

from typing import Any

import torch.distributed as dist

from fastvideo.train.methods.rl.rvm_local_metrics import (
    RVMWithLocalMetricsMethod,
)


class RVMRewardProfileMethod(RVMWithLocalMetricsMethod):
    """Paper-faithful RVM supporting calibrated multi-model reward profiles.

    The optimization and validation logic is inherited unchanged. This class
    only synchronizes the complete scorer output contract—including calibrated
    raw-value diagnostics—across each sequence-parallel group.
    """

    def on_train_start(self) -> None:
        super().on_train_start()
        if self._sp_group is None:
            raise RuntimeError(
                "RVM reward profile requires an initialized SP group"
            )
        keys: tuple[str, ...] | None = None
        if self._is_sp_leader:
            scorer = self._reward_scorer
            if scorer is None:
                raise RuntimeError(
                    "SP leader did not construct a reward scorer"
                )
            raw_keys = getattr(scorer, "output_keys", None)
            if not isinstance(raw_keys, tuple) or not raw_keys:
                raise RuntimeError(
                    "Reward scorer must expose a nonempty output_keys tuple"
                )
            keys = tuple(str(value) for value in raw_keys)

        payload: list[Any] = [keys]
        if dist.is_available() and dist.is_initialized():
            dist.broadcast_object_list(
                payload,
                src=int(self._sp_group.first_rank),
                group=self._sp_group.cpu_group,
            )
        if not isinstance(payload[0], tuple) or not payload[0]:
            raise RuntimeError(
                "Could not synchronize reward output keys"
            )
        self._reward_keys = list(payload[0])
        self._log_progress(
            "RVM reward profile outputs: "
            + ", ".join(self._reward_keys)
        )


__all__ = ["RVMRewardProfileMethod"]
