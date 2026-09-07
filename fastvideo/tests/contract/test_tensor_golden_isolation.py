# SPDX-License-Identifier: Apache-2.0
"""Golden recipes must not inherit or leak the model loader's compute policy."""

import os

import pytest
import torch

from fastvideo import utils
from fastvideo.tests.golden_gate._tensor_golden import deterministic_forward


@pytest.mark.parametrize("existing_policy", [False, True])
@pytest.mark.parametrize("fails", [False, True])
def test_tensor_golden_restores_runtime_state(monkeypatch, existing_policy, fails):
    # No CUDA work is needed to check context ownership.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    monkeypatch.setenv("FASTVIDEO_ATTENTION_BACKEND", "TORCH_SDPA")
    previous_policy = utils.MixedPrecisionState(param_dtype=torch.float16)
    if existing_policy:
        monkeypatch.setattr(utils._mixed_precision_state, "state", previous_policy, raising=False)
    else:
        monkeypatch.delattr(utils._mixed_precision_state, "state", raising=False)
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    benchmark = torch.backends.cudnn.benchmark

    class RecipeError(Exception):
        pass

    try:
        with deterministic_forward("FLASH_ATTN"):
            assert utils.get_compute_dtype() == torch.get_default_dtype()
            assert os.environ["FASTVIDEO_ATTENTION_BACKEND"] == "FLASH_ATTN"
            assert torch.are_deterministic_algorithms_enabled()
            assert not torch.backends.cudnn.benchmark
            # Same process-state mutation performed by TransformerLoader.
            utils.set_mixed_precision_policy(torch.bfloat16, torch.float32)
            assert utils.get_compute_dtype() == torch.bfloat16
            if fails:
                raise RecipeError
    except RecipeError:
        assert fails

    assert hasattr(utils._mixed_precision_state, "state") == existing_policy
    if existing_policy:
        assert utils.get_mixed_precision_state() is previous_policy
    assert os.environ["FASTVIDEO_ATTENTION_BACKEND"] == "TORCH_SDPA"
    assert torch.are_deterministic_algorithms_enabled() == deterministic
    assert torch.is_deterministic_algorithms_warn_only_enabled() == warn_only
    assert torch.backends.cudnn.benchmark == benchmark
