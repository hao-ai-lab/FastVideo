# SPDX-License-Identifier: Apache-2.0
"""Config validation for the initial pure Ring Attention integration.

The first version only supports ``ring_size == sp_size`` (pure Ring, no
Ulysses x Ring hybrid) and inference-only. These checks live in
``FastVideoArgs._check_ring_attention_args`` and run from ``__post_init__``,
so they are exercised here purely through the dataclass constructor — no
GPU or distributed process group required.
"""

from __future__ import annotations

import pytest

from fastvideo.fastvideo_args import ExecutionMode, FastVideoArgs

MODEL_PATH = "FastVideo/LTX2-Distilled-Diffusers"


def test_ring_disabled_by_default() -> None:
    args = FastVideoArgs(model_path=MODEL_PATH)
    assert args.ring_size == 1


def test_ring_size_matching_sp_size_is_allowed() -> None:
    args = FastVideoArgs(model_path=MODEL_PATH, num_gpus=4, sp_size=4, ring_size=4)
    assert args.ring_size == 4
    assert args.sp_size == 4


def test_ring_size_zero_rejected() -> None:
    with pytest.raises(ValueError, match="ring_size must be >= 1"):
        FastVideoArgs(model_path=MODEL_PATH, ring_size=0)


def test_ring_without_sequence_parallelism_rejected() -> None:
    with pytest.raises(ValueError, match="requires sequence parallelism"):
        FastVideoArgs(model_path=MODEL_PATH, num_gpus=1, sp_size=1, ring_size=2)


def test_ring_size_mismatch_with_sp_size_rejected() -> None:
    with pytest.raises(NotImplementedError, match="ring_size must equal"):
        FastVideoArgs(model_path=MODEL_PATH, num_gpus=4, sp_size=4, ring_size=2)


def test_ring_training_rejected() -> None:
    with pytest.raises(NotImplementedError, match="training/backward is not supported"):
        FastVideoArgs(
            model_path=MODEL_PATH,
            mode=ExecutionMode.FINETUNING,
            inference_mode=False,
            num_gpus=4,
            sp_size=4,
            ring_size=4,
            hsdp_shard_dim=4,
        )
