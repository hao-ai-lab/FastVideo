# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from fastvideo.attention.backends import attn_qat_train


@pytest.fixture(autouse=True)
def reset_attn_qat_train_import_cache(monkeypatch):
    monkeypatch.setattr(attn_qat_train, "_attn_qat_train_attention", None)
    monkeypatch.setattr(attn_qat_train, "_attn_qat_train_import_attempted", False)
    monkeypatch.setattr(attn_qat_train, "_attn_qat_train_import_error", None)


def test_external_qat_package_patches_fastvideo_triton_backward(monkeypatch):
    attention = Mock()
    attention_cls = object()
    triton_qat = SimpleNamespace(attention=attention, _attention=attention_cls)
    patch_fastvideo = Mock()
    external_qat = SimpleNamespace(patch_fastvideo=patch_fastvideo)

    def import_module(name):
        if name == "fastvideo_kernel.triton_kernels.attn_qat_train":
            return triton_qat
        if name == "qat_attn":
            return external_qat
        raise AssertionError(f"unexpected import: {name}")

    monkeypatch.setattr(attn_qat_train.importlib, "import_module", import_module)

    assert attn_qat_train._get_attn_qat_train_attention() is attention
    patch_fastvideo.assert_called_once_with(attention_cls)


def test_missing_external_qat_package_has_actionable_error(monkeypatch):
    triton_qat = SimpleNamespace(attention=Mock(), _attention=object())

    def import_module(name):
        if name == "fastvideo_kernel.triton_kernels.attn_qat_train":
            return triton_qat
        raise ImportError("No module named 'qat_attn'")

    monkeypatch.setattr(attn_qat_train.importlib, "import_module", import_module)

    with pytest.raises(ImportError, match="separately distributed qat_attn package"):
        attn_qat_train.attn_qat_train(Mock(), Mock(), Mock())


def test_external_qat_backend_advertises_supported_head_size():
    assert attn_qat_train.AttnQatTrainBackend.get_supported_head_sizes() == [128]
