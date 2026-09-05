# SPDX-License-Identifier: Apache-2.0
"""Attention compile-boundary policy tests."""

from fastvideo.attention.layer import _attention_compile_disabled


def test_attention_compile_is_disabled_by_default(monkeypatch) -> None:
    monkeypatch.delenv("FASTVIDEO_DISABLE_ATTENTION_COMPILE", raising=False)

    assert _attention_compile_disabled()


def test_attention_compile_escape_hatch(monkeypatch) -> None:
    monkeypatch.setenv("FASTVIDEO_DISABLE_ATTENTION_COMPILE", "1")

    assert _attention_compile_disabled()

    monkeypatch.setenv("FASTVIDEO_DISABLE_ATTENTION_COMPILE", "0")
    assert not _attention_compile_disabled()
