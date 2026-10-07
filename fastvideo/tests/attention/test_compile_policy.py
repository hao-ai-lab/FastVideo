# SPDX-License-Identifier: Apache-2.0
"""Attention compile-boundary policy tests."""

import fastvideo.envs as envs
from fastvideo.attention.layer import _attention_compile_disabled


def test_attention_compile_is_disabled_by_default(env_overrides) -> None:
    env_overrides.enter_context(envs.FASTVIDEO_DISABLE_ATTENTION_COMPILE.override(None))

    assert _attention_compile_disabled()


def test_attention_compile_escape_hatch(env_overrides) -> None:
    env_overrides.enter_context(envs.FASTVIDEO_DISABLE_ATTENTION_COMPILE.override(True))

    assert _attention_compile_disabled()

    env_overrides.enter_context(envs.FASTVIDEO_DISABLE_ATTENTION_COMPILE.override(False))
    assert not _attention_compile_disabled()
