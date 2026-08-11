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


def test_fastvideo_triton_kernel_is_loaded_directly(monkeypatch):
    attention = Mock()
    import_module = Mock(return_value=SimpleNamespace(attention=attention))
    monkeypatch.setattr(attn_qat_train.importlib, "import_module", import_module)

    assert attn_qat_train._get_attn_qat_train_attention() is attention
    assert attn_qat_train._get_attn_qat_train_attention() is attention
    import_module.assert_called_once_with("fastvideo_kernel.triton_kernels.attn_qat_train")


def test_local_kernel_source_is_added_to_import_path(monkeypatch):
    monkeypatch.setattr(attn_qat_train.sys, "path", [])

    attn_qat_train._ensure_kernel_paths()

    assert str(attn_qat_train._project_root) in attn_qat_train.sys.path
    assert str(attn_qat_train._kernel_python_root) in attn_qat_train.sys.path


def test_missing_fastvideo_kernel_has_actionable_error(monkeypatch):
    missing = ImportError("No module named 'fastvideo_kernel'")
    monkeypatch.setattr(
        attn_qat_train.importlib,
        "import_module",
        Mock(side_effect=missing),
    )

    with pytest.raises(ImportError, match="requires FastVideo's fastvideo-kernel"):
        attn_qat_train.attn_qat_train(Mock(), Mock(), Mock())


def test_backend_advertises_supported_head_size():
    assert attn_qat_train.AttnQatTrainBackend.get_supported_head_sizes() == [128]
