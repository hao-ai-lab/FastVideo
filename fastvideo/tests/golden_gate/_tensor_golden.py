# SPDX-License-Identifier: Apache-2.0
"""Named tensor references for codecs and short sampling trajectories.

Keep weight loading and forward calls in each test. This helper only controls
determinism and compares device/runtime-matched tensors; it is not a model runner.
"""

import os
from contextlib import contextmanager

import pytest
import torch

from fastvideo.tests.golden_gate._harness import DEFAULT_SEED, device_folder, env_fingerprint, resolve_golden_path


@contextmanager
def deterministic_forward(attention_backend=None):
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        pytest.skip("golden gate requires a bf16-capable CUDA GPU")
    from fastvideo.utils import _mixed_precision_state

    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    benchmark = torch.backends.cudnn.benchmark
    previous_backend = os.environ.get("FASTVIDEO_ATTENTION_BACKEND")
    had_precision_state = hasattr(_mixed_precision_state, "state")
    previous_precision_state = getattr(_mixed_precision_state, "state", None)
    try:
        # TransformerLoader sets a thread-local compute dtype even without
        # FSDP. Let each recipe choose its own construction dtype, then restore
        # the caller's policy: leaking bf16 changes later block gates' attention
        # backend despite their unchanged torch default dtype and environment.
        if had_precision_state:
            del _mixed_precision_state.state
        if attention_backend is not None:
            os.environ["FASTVIDEO_ATTENTION_BACKEND"] = attention_backend
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        yield torch.device("cuda:0")
    finally:
        if had_precision_state:
            _mixed_precision_state.state = previous_precision_state
        elif hasattr(_mixed_precision_state, "state"):
            del _mixed_precision_state.state
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
        torch.backends.cudnn.benchmark = benchmark
        if previous_backend is None:
            os.environ.pop("FASTVIDEO_ATTENTION_BACKEND", None)
        else:
            os.environ["FASTVIDEO_ATTENTION_BACKEND"] = previous_backend


def assert_tensor_golden(name, outputs, *, identity, attention_backend=None, seed=DEFAULT_SEED):
    assert outputs, "A golden must contain at least one named tensor"
    outputs = {key: value.detach().contiguous().cpu() for key, value in sorted(outputs.items())}
    for key, value in outputs.items():
        assert value.numel() and torch.isfinite(value).all(), f"Invalid golden output: {key}"
    metadata = {
        "schema": 1,
        "name": name,
        "seed": seed,
        "identity": identity,
        "env": env_fingerprint(attention_backend),
        "tensors": {key: {"shape": list(value.shape), "dtype": str(value.dtype)} for key, value in outputs.items()},
    }
    relative = f"{device_folder()}/{name}_seed{seed}.pt"
    path, exists = resolve_golden_path(relative)
    if not exists:
        # Never make a missing reference pass on the next run of candidate code.
        candidate = path.with_suffix(".candidate.pt")
        candidate.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"outputs": outputs, "metadata": metadata}, candidate)
        pytest.fail(
            f"Missing golden {relative}; candidate saved to {candidate}. "
            "Generate from the unchanged main revision, verify in two separate processes, "
            "then review and publish the reference. Candidate output is not an approved golden.",
            pytrace=False,
        )
    golden = torch.load(path, map_location="cpu", weights_only=True)
    assert golden["metadata"] == metadata, "Golden input/checkpoint/runtime identity changed; review the reference"
    assert golden["outputs"].keys() == outputs.keys()
    for key, value in outputs.items():
        torch.testing.assert_close(value, golden["outputs"][key], atol=0, rtol=0, msg=f"{name}: {key}")
