# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import torch

from fastvideo.training import training_utils


def test_clip_grad_norm_uses_local_dtensor_shards_for_foreach(monkeypatch) -> None:

    class FakeDTensor:

        def __init__(self, local: torch.Tensor) -> None:
            self.local = local

        def to_local(self) -> torch.Tensor:
            return self.local

    monkeypatch.setattr(torch.distributed.tensor, "DTensor", FakeDTensor)

    local_grad = torch.tensor([3.0, 4.0])
    parameter = SimpleNamespace(grad=FakeDTensor(local_grad))

    training_utils._clip_grads_with_norm_(
        [parameter],
        max_norm=1.0,
        total_norm=torch.tensor(5.0),
    )

    torch.testing.assert_close(local_grad, torch.tensor([0.6, 0.8]))
