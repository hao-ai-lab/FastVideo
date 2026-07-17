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


def test_external_qat_repo_env_is_added_to_import_path(monkeypatch, tmp_path):
    external_repo = tmp_path / "custom-nvfp4-qat-attn"
    package_dir = external_repo / "qat_attn"
    package_dir.mkdir(parents=True)
    (package_dir / "__init__.py").touch()
    monkeypatch.setenv("QAT_ATTN_REPO", str(external_repo))
    monkeypatch.setattr(attn_qat_train.sys, "path", [])

    assert attn_qat_train._configured_external_qat_repo() == external_repo.resolve()
    attn_qat_train._prepend_import_path(external_repo.resolve())
    assert attn_qat_train.sys.path[0] == str(external_repo.resolve())


def test_external_qat_repo_defaults_to_fastvideo_sibling(monkeypatch, tmp_path):
    project_root = tmp_path / "FastVideo"
    external_repo = tmp_path / "nvfp4_qat_attn"
    package_dir = external_repo / "qat_attn"
    package_dir.mkdir(parents=True)
    (package_dir / "__init__.py").touch()
    monkeypatch.delenv("QAT_ATTN_REPO", raising=False)
    monkeypatch.setattr(attn_qat_train, "_project_root", project_root)

    assert attn_qat_train._sibling_external_qat_repo() == external_repo.resolve()


def test_invalid_external_qat_repo_env_fails_without_fallback(monkeypatch, tmp_path):
    invalid_repo = tmp_path / "not-a-qat-repo"
    invalid_repo.mkdir()
    monkeypatch.setenv("QAT_ATTN_REPO", str(invalid_repo))

    with pytest.raises(ImportError, match="is not an nvfp4_qat_attn repository"):
        attn_qat_train._import_external_qat()


def test_importable_qat_package_wins_over_sibling(monkeypatch):
    external_qat = SimpleNamespace(__file__="/environment/qat_attn/__init__.py")
    sibling_lookup = Mock()
    monkeypatch.delenv("QAT_ATTN_REPO", raising=False)
    monkeypatch.setattr(attn_qat_train.importlib, "import_module", Mock(return_value=external_qat))
    monkeypatch.setattr(attn_qat_train, "_sibling_external_qat_repo", sibling_lookup)

    imported, source_repo = attn_qat_train._import_external_qat()

    assert imported is external_qat
    assert source_repo is None
    sibling_lookup.assert_not_called()


def test_missing_nested_qat_dependency_is_not_treated_as_missing_package(monkeypatch):
    nested_error = ModuleNotFoundError("No module named 'ninja'", name="ninja")
    sibling_lookup = Mock()
    monkeypatch.delenv("QAT_ATTN_REPO", raising=False)
    monkeypatch.setattr(attn_qat_train.importlib, "import_module", Mock(side_effect=nested_error))
    monkeypatch.setattr(attn_qat_train, "_sibling_external_qat_repo", sibling_lookup)

    with pytest.raises(ModuleNotFoundError) as exc_info:
        attn_qat_train._import_external_qat()

    assert exc_info.value.name == "ninja"
    sibling_lookup.assert_not_called()


def test_missing_package_falls_back_to_sibling_repo(monkeypatch, tmp_path):
    sibling_repo = tmp_path / "nvfp4_qat_attn"
    external_qat = SimpleNamespace(__file__=str(sibling_repo / "qat_attn" / "__init__.py"))
    import_module = Mock(side_effect=[ModuleNotFoundError(name="qat_attn"), external_qat])
    monkeypatch.delenv("QAT_ATTN_REPO", raising=False)
    monkeypatch.setattr(attn_qat_train.importlib, "import_module", import_module)
    monkeypatch.setattr(attn_qat_train, "_sibling_external_qat_repo", Mock(return_value=sibling_repo))
    monkeypatch.setattr(attn_qat_train.sys, "path", [])

    imported, source_repo = attn_qat_train._import_external_qat()

    assert imported is external_qat
    assert source_repo == sibling_repo
    assert attn_qat_train.sys.path[0] == str(sibling_repo)


def test_missing_external_qat_package_has_actionable_error(monkeypatch):
    triton_qat = SimpleNamespace(attention=Mock(), _attention=object())

    def import_module(name):
        if name == "fastvideo_kernel.triton_kernels.attn_qat_train":
            return triton_qat
        raise ImportError("No module named 'qat_attn'")

    monkeypatch.setattr(attn_qat_train.importlib, "import_module", import_module)

    with pytest.raises(ImportError, match="QAT_ATTN_REPO=/absolute/path/to/nvfp4_qat_attn"):
        attn_qat_train.attn_qat_train(Mock(), Mock(), Mock())


def test_external_qat_backend_advertises_supported_head_size():
    assert attn_qat_train.AttnQatTrainBackend.get_supported_head_sizes() == [128]
