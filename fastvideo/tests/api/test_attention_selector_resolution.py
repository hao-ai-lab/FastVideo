# SPDX-License-Identifier: Apache-2.0
"""Golden tests for explicit per-component attention-backend resolution.

The selector's contract after the refactor:
  * precedence is unchanged: global force > scoped/explicit request >
    env var > layer default > platform auto — the scoped request occupies
    exactly the position the process-global force held when the train
    stack used it for per-role models;
  * every selection input is part of the resolution cache key, so no
    mutation (scope, force, env) ever needs a ``cache_clear()``;
  * an active ``_component_attention_backend_scope`` suppresses the env var unless
    ``consult_env=True`` — ``_component_attention_backend_scope(None)`` therefore
    means "automatic selection, ignore the process-wide request" (the
    dense teacher/critic case);
  * scopes are exception-safe and per-component (component identity and
    the active device index are cache-key inputs, so two components — or
    two devices with different capabilities — never share a resolution).

CPU-only: the platform is faked (same seam as the pre-existing
role-override tests); no kernels, no GPU.
"""
from __future__ import annotations

import pytest
import torch

import fastvideo.attention.selector as selector
import fastvideo.platforms as platforms
from fastvideo.platforms import AttentionBackendEnum

FLASH = AttentionBackendEnum.FLASH_ATTN
SDPA = AttentionBackendEnum.TORCH_SDPA
SAGE = AttentionBackendEnum.SAGE_ATTN

SUPPORTED = (FLASH, SDPA, SAGE)
KWARGS = {
    "head_size": 64,
    "dtype": torch.bfloat16,
    "supported_attention_backends": SUPPORTED,
}


class _FakePlatform:
    device_name = "fake"

    @classmethod
    def get_attn_backend_cls(
        cls,
        selected_backend: AttentionBackendEnum | None,
        head_size: int,
        dtype: torch.dtype,
    ) -> str:
        del head_size, dtype
        # Auto-selection (None) resolves to FLASH on this fake platform so
        # the oracle can distinguish "auto" from every explicit request.
        return (selected_backend or FLASH).name


@pytest.fixture(autouse=True)
def _fake_platform(monkeypatch):
    monkeypatch.setattr(platforms, "_current_platform", _FakePlatform())
    monkeypatch.setattr(selector, "resolve_obj_by_qualname", lambda name: name)
    monkeypatch.delenv("FASTVIDEO_ATTENTION_BACKEND", raising=False)
    selector.global_force_attn_backend(None)
    selector._cached_get_attn_backend.cache_clear()
    yield
    selector.global_force_attn_backend(None)
    # Fake-platform resolutions must not outlive the test: mutations no
    # longer flush the cache (that is the feature), so evict explicitly.
    selector._cached_get_attn_backend.cache_clear()


def _oracle(forced, requested, env, default):
    """Today's documented precedence, spelled out independently."""
    selected = forced if forced is not None else requested
    if selected is None and env is not None:
        selected = AttentionBackendEnum[env]
    if selected is None and default is not None:
        selected = default
    if selected is not None and selected not in SUPPORTED:
        selected = default if default in SUPPORTED else None
    return (selected or FLASH).name


@pytest.mark.parametrize("forced", [None, SDPA])
@pytest.mark.parametrize("scoped", [None, "unset", SAGE])
@pytest.mark.parametrize("env", [None, "TORCH_SDPA"])
@pytest.mark.parametrize("default", [None, SAGE])
def test_precedence_matrix_matches_previous_semantics(monkeypatch, forced, scoped, env, default) -> None:
    if env is not None:
        monkeypatch.setenv("FASTVIDEO_ATTENTION_BACKEND", env)
    selector.global_force_attn_backend(forced)

    if scoped == "unset":
        # No scope: env is consulted (legacy behavior preserved).
        got = selector.get_attn_backend(default_backend=default, **KWARGS)
        expected = _oracle(forced, None, env, default)
    else:
        # Scope active: env suppressed; scoped request sits in the position
        # the global force held for per-role loads.
        with selector._component_attention_backend_scope(scoped, component="dit"):
            got = selector.get_attn_backend(default_backend=default, **KWARGS)
        expected = _oracle(forced, scoped, None, default)
    assert got == expected


def test_scope_none_ignores_env_request(monkeypatch) -> None:
    """The dense teacher/critic case: auto-select, ignore the process-wide
    request — previously implemented by popping the env var and flushing
    the selector cache around the build."""
    monkeypatch.setenv("FASTVIDEO_ATTENTION_BACKEND", "SAGE_ATTN")

    assert selector.get_attn_backend(**KWARGS) == "SAGE_ATTN"
    with selector._component_attention_backend_scope(None, component="transformer"):
        assert selector.get_attn_backend(**KWARGS) == "FLASH_ATTN"  # auto
    assert selector.get_attn_backend(**KWARGS) == "SAGE_ATTN"


def test_interleaved_component_scopes_need_no_cache_clear(monkeypatch) -> None:
    """Per-role/per-component builds interleave freely; identical
    shape/dtype keys resolve per scope with zero cache management."""
    monkeypatch.setenv("FASTVIDEO_ATTENTION_BACKEND", "SAGE_ATTN")

    for _ in range(2):
        with selector._component_attention_backend_scope(SDPA, component="student"):
            assert selector.get_attn_backend(**KWARGS) == "TORCH_SDPA"
        with selector._component_attention_backend_scope(None, component="teacher"):
            assert selector.get_attn_backend(**KWARGS) == "FLASH_ATTN"
        assert selector.get_attn_backend(**KWARGS) == "SAGE_ATTN"


def test_scope_is_exception_safe(monkeypatch) -> None:
    monkeypatch.setenv("FASTVIDEO_ATTENTION_BACKEND", "SAGE_ATTN")
    with pytest.raises(RuntimeError, match="boom"):
        with selector._component_attention_backend_scope(SDPA, component="dit"):
            raise RuntimeError("boom")
    assert selector._active_component_attention_backend_scope() is None
    assert selector.get_attn_backend(**KWARGS) == "SAGE_ATTN"


def test_global_force_wins_without_cache_clear() -> None:
    with selector._component_attention_backend_scope(SAGE, component="dit"):
        assert selector.get_attn_backend(**KWARGS) == "SAGE_ATTN"
        selector.global_force_attn_backend(SDPA)
        assert selector.get_attn_backend(**KWARGS) == "TORCH_SDPA"
        selector.global_force_attn_backend(None)
        assert selector.get_attn_backend(**KWARGS) == "SAGE_ATTN"


def test_env_change_takes_effect_without_cache_clear(monkeypatch) -> None:
    """Selection inputs live in the cache key, so an env change is simply a
    different key. (Previously a changed env var was silently ignored until
    someone remembered to flush the cache.)"""
    monkeypatch.setenv("FASTVIDEO_ATTENTION_BACKEND", "SAGE_ATTN")
    assert selector.get_attn_backend(**KWARGS) == "SAGE_ATTN"
    monkeypatch.setenv("FASTVIDEO_ATTENTION_BACKEND", "TORCH_SDPA")
    assert selector.get_attn_backend(**KWARGS) == "TORCH_SDPA"


def test_scope_typo_fails_fast() -> None:
    with pytest.raises(ValueError, match="Unknown attention backend"):
        with selector._component_attention_backend_scope("flash_atn", component="dit"):
            pass


def test_consult_env_scope_keeps_env_visible(monkeypatch) -> None:
    monkeypatch.setenv("FASTVIDEO_ATTENTION_BACKEND", "SAGE_ATTN")
    with selector._component_attention_backend_scope(None, component="dit", consult_env=True):
        assert selector.get_attn_backend(**KWARGS) == "SAGE_ATTN"


def test_active_device_is_part_of_the_cache_key(monkeypatch) -> None:
    """Platform auto-selection probes the *current* device's capability
    (AttnQatInferBackend, for one, resolves a different kernel per
    capability set), so two devices must not share one cached resolution.
    The fake platform here stands in for that probe."""
    current = {"index": 0}
    per_device = {0: FLASH, 1: SDPA}

    class _PerDevicePlatform:
        device_name = "fake"

        @classmethod
        def get_attn_backend_cls(cls, selected_backend, head_size, dtype) -> str:
            del head_size, dtype
            return (selected_backend or per_device[current["index"]]).name

    monkeypatch.setattr(platforms, "_current_platform", _PerDevicePlatform())
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: current["index"])

    assert selector.get_attn_backend(**KWARGS) == "FLASH_ATTN"
    current["index"] = 1
    assert selector.get_attn_backend(**KWARGS) == "TORCH_SDPA"


# ---------------------------------------------------------------------------
# The decision is carried on the component
# ---------------------------------------------------------------------------


def test_env_is_folded_into_the_typed_request_once(monkeypatch):
    """``FastVideoArgs.attention_backend`` is the parse-once adapter."""
    from fastvideo.fastvideo_args import FastVideoArgs

    monkeypatch.setenv("FASTVIDEO_ATTENTION_BACKEND", "SAGE_ATTN")
    assert FastVideoArgs(model_path="x").attention_backend == "SAGE_ATTN"

    # An explicit request always wins over the environment.
    assert FastVideoArgs(model_path="x", attention_backend="TORCH_SDPA").attention_backend == "TORCH_SDPA"


def test_unparseable_env_falls_through_instead_of_raising(monkeypatch):
    """The env var keeps its permissive parse; only explicit requests raise."""
    from fastvideo.fastvideo_args import FastVideoArgs

    monkeypatch.setenv("FASTVIDEO_ATTENTION_BACKEND", "flash_atn")
    assert FastVideoArgs(model_path="x").attention_backend is None

    with pytest.raises(ValueError, match="Unknown attention backend"):
        FastVideoArgs(model_path="x", attention_backend="flash_atn")


def test_recorded_decision_is_readable_from_the_component():
    """After load, the component carries what it resolved."""
    from fastvideo.configs.models.dits.base import DiTConfig

    config = DiTConfig()
    assert config._resolved_attention_backend is None

    with selector._component_attention_backend_scope(SAGE, component="transformer"):
        assert selector.record_resolved_attention_backend(config) == SAGE
    assert config._resolved_attention_backend == SAGE


def test_recorded_decision_is_the_narrowed_one():
    """A loader that narrows the request records the narrowed value.

    The DMD teacher/critic transformers build dense inside a nested scope
    while the run as a whole requested a quantized backend; the component
    must report what it actually resolved, not the run-wide request.
    """
    from fastvideo.configs.models.dits.base import DiTConfig

    student, teacher = DiTConfig(), DiTConfig()
    with selector._component_attention_backend_scope(SAGE, component="transformer"):
        selector.record_resolved_attention_backend(student)
        with selector._component_attention_backend_scope(None, component="transformer"):
            selector.record_resolved_attention_backend(teacher)

    assert student._resolved_attention_backend == SAGE
    assert teacher._resolved_attention_backend is None


def test_no_request_records_no_decision():
    from fastvideo.configs.models.encoders.base import TextEncoderConfig

    config = TextEncoderConfig()
    assert selector.record_resolved_attention_backend(config) is None
    assert config._resolved_attention_backend is None
