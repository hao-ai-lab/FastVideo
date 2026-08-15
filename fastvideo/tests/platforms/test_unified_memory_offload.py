# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the offload gate on unified-memory devices.

Every offload flag moves weights to host memory so the device can drop them.
That trade only pays when the two are separate pools. On a unified-memory device
they are one, so the move frees nothing, the copy is a pure loss, and the peak
becomes the sum instead of the max.

The reason this is decided once in ``check_fastvideo_args`` rather than at each
call site is that one flag acts in several places.
``text_encoder_cpu_offload`` picks the load-time target device, gates the FSDP
offload path, and gates the shuttle around the conditioning forward. Gating only
the loader leaves the model on the host and moves the copy to inference time,
which is where it was found: a 48 GB move, mid-generation, on a device with
about 11 GiB free.

These tests pin the gate, not the flags' defaults, which belong to whoever tunes
them.
"""
from __future__ import annotations

import dataclasses

import pytest

from fastvideo.fastvideo_args import UNIFIED_MEMORY_OFFLOAD_FLAGS, FastVideoArgs


def _args(**overrides) -> FastVideoArgs:
    kwargs: dict = dict(model_path="unused/for-this-test")
    kwargs.update(overrides)
    return FastVideoArgs(**kwargs)


@pytest.fixture
def as_discrete_cuda(monkeypatch):
    """A CUDA device with its own VRAM: the branch must not fire."""
    monkeypatch.setattr("fastvideo.platforms.current_platform.is_mps", lambda: False)
    monkeypatch.setattr("fastvideo.platforms.current_platform.has_unified_memory", lambda: False)


@pytest.fixture
def as_unified_cuda(monkeypatch):
    """A CUDA device that shares one pool with the host, such as GB10 or Jetson."""
    monkeypatch.setattr("fastvideo.platforms.current_platform.is_mps", lambda: False)
    monkeypatch.setattr("fastvideo.platforms.current_platform.has_unified_memory", lambda: True)
    monkeypatch.setattr("fastvideo.platforms.current_platform.get_device_name", lambda: "NVIDIA GB10")


def test_every_offload_flag_is_disabled(as_unified_cuda) -> None:
    args = _args(**{flag: True for flag in UNIFIED_MEMORY_OFFLOAD_FLAGS})
    args.check_fastvideo_args()

    still_on = [flag for flag in UNIFIED_MEMORY_OFFLOAD_FLAGS if getattr(args, flag)]
    assert not still_on, f"offload still enabled on unified memory: {still_on}"


def test_the_list_covers_the_flags_that_exist() -> None:
    """A new offload flag has to be added here, not just to the dataclass.

    Missing one is silent: the run works and quietly pays twice, which is how
    ``text_encoder_cpu_offload`` survived the first pass at this.
    """
    declared = {field.name for field in dataclasses.fields(FastVideoArgs) if field.name.endswith("_offload")}
    assert declared == set(UNIFIED_MEMORY_OFFLOAD_FLAGS)


@pytest.mark.parametrize("flag", UNIFIED_MEMORY_OFFLOAD_FLAGS)
def test_each_flag_individually(as_unified_cuda, flag: str) -> None:
    args = _args(**{flag: True})
    args.check_fastvideo_args()

    assert getattr(args, flag) is False


def test_discrete_device_is_untouched(as_discrete_cuda) -> None:
    args = _args(dit_layerwise_offload=True, text_encoder_cpu_offload=True, vae_cpu_offload=True)
    args.check_fastvideo_args()

    assert args.dit_layerwise_offload is True
    assert args.text_encoder_cpu_offload is True
    assert args.vae_cpu_offload is True
    # Unchanged behaviour: layerwise offload still wins over the other two DiT
    # settings on a device where offloading is worth doing.
    assert args.dit_cpu_offload is False
    assert args.use_fsdp_inference is False


def test_already_off_stays_off(as_unified_cuda) -> None:
    args = _args(**{flag: False for flag in UNIFIED_MEMORY_OFFLOAD_FLAGS})
    args.check_fastvideo_args()

    still_on = [flag for flag in UNIFIED_MEMORY_OFFLOAD_FLAGS if getattr(args, flag)]
    assert not still_on


def test_mps_path_is_unchanged(monkeypatch) -> None:
    """MPS is unified memory too, but it has its own branch and reaches this one
    through ``elif``. It must keep disabling FSDP inference, which the unified
    branch deliberately leaves alone."""
    monkeypatch.setattr("fastvideo.platforms.current_platform.is_mps", lambda: True)
    args = _args(dit_layerwise_offload=True, use_fsdp_inference=True)
    args.check_fastvideo_args()

    assert args.dit_layerwise_offload is False
    assert args.use_fsdp_inference is False


def test_unified_memory_does_not_touch_fsdp_inference(as_unified_cuda) -> None:
    """Sharding across ranks is a separate decision from where the weights live,
    and a unified-memory host can have more than one of these devices."""
    args = _args(dit_layerwise_offload=False, use_fsdp_inference=True)
    args.check_fastvideo_args()

    assert args.use_fsdp_inference is True
