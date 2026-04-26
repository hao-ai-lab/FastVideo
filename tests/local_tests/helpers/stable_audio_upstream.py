# SPDX-License-Identifier: Apache-2.0
"""Compatibility shim: previously a sys.modules stub installer for the
stable-audio-tools import chain back when its transitive deps weren't
installed. Now that the test suite expects a real
`pip install k_diffusion einops_exts alias_free_torch` (see
`tests/local_tests/pipelines/test_stable_audio_pipeline_parity.py`
docstring), no stubs are needed and `install_stubs` is a no-op kept
for callers we don't want to chase down.
"""
from __future__ import annotations


def install_stubs() -> None:  # pragma: no cover - back-compat no-op
    return
