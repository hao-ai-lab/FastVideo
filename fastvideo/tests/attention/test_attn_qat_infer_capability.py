import pytest
import torch

from fastvideo.attention.backends import attn_qat_infer


@pytest.mark.parametrize(
    "capability,expected",
    [((12, 0), True), ((12, 1), True), ((10, 0), False), ((9, 0), False)],
)
def test_attn_qat_infer_availability_requires_supported_gpu(monkeypatch, capability, expected):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: capability)
    monkeypatch.setattr(attn_qat_infer, "_get_attn_qat_infer", lambda: object())

    assert attn_qat_infer.is_attn_qat_infer_available() is expected
