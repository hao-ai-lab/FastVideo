# SPDX-License-Identifier: Apache-2.0

import pytest

import fastvideo.envs as envs
from fastvideo.tests.golden_gate._harness import GateSpec, assert_env_fingerprint_matches, env_fingerprint


@pytest.mark.parametrize("raw", [None, "0", "false", "", "1"])
def test_block_fingerprint_records_effective_fa4(monkeypatch, raw):
    import torch

    if raw is None:
        monkeypatch.delenv("FASTVIDEO_FA4", raising=False)
    else:
        monkeypatch.setenv("FASTVIDEO_FA4", raw)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda index: "NVIDIA GB200")
    spec = GateSpec("test", "test/repo", lambda layer: None, lambda device, seed: {})
    assert env_fingerprint(spec)["fastvideo_fa4"] == str(int(envs.FASTVIDEO_FA4.get()))


def test_legacy_unset_fa4_matches_disabled_without_changing_reference():
    reference = {"fastvideo_fa4": "<unset>", "torch": "pinned"}
    assert_env_fingerprint_matches(reference, {"fastvideo_fa4": "0", "torch": "pinned"})
    assert reference == {"fastvideo_fa4": "<unset>", "torch": "pinned"}


@pytest.mark.parametrize("reference_fa4", ["<unset>", "0", "1"])
def test_fa4_compute_path_changes_still_fail(reference_fa4):
    current_fa4 = "0" if reference_fa4 == "1" else "1"
    with pytest.raises(AssertionError, match="fastvideo_fa4"):
        assert_env_fingerprint_matches({"fastvideo_fa4": reference_fa4}, {"fastvideo_fa4": current_fa4})


@pytest.mark.parametrize("field", ["torch", "cuda", "cudnn", "flash_attn", "device_name", "attention_backend",
                                    "tf32_matmul", "tf32_cudnn"])
def test_other_runtime_changes_still_fail(field):
    with pytest.raises(AssertionError, match=field):
        assert_env_fingerprint_matches({"fastvideo_fa4": "<unset>", field: "old"},
                                       {"fastvideo_fa4": "0", field: "new"})
