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
    def is_mps(cls) -> bool:
        # Not decoration. The autouse fixture swaps this stub in process-wide,
        # so anything that constructs FastVideoArgs under it reaches
        # check_fastvideo_args, which calls current_platform.is_mps(). Without
        # this, the two env-folding tests raise AttributeError before they
        # assert anything.
        return False

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
    selector._cached_get_attn_backend.cache_clear()
    yield
    # Fake-platform resolutions must not outlive the test: mutations no
    # longer flush the cache (that is the feature), so evict explicitly.
    selector._cached_get_attn_backend.cache_clear()


def _oracle(requested, env, default):
    """The documented precedence, spelled out independently."""
    selected = requested
    if selected is None and env is not None:
        selected = AttentionBackendEnum[env]
    if selected is None and default is not None:
        selected = default
    if selected is not None and selected not in SUPPORTED:
        selected = default if default in SUPPORTED else None
    return (selected or FLASH).name


@pytest.mark.parametrize("scoped", [None, "unset", SAGE])
@pytest.mark.parametrize("env", [None, "TORCH_SDPA"])
@pytest.mark.parametrize("default", [None, SAGE])
def test_precedence_matrix_matches_previous_semantics(monkeypatch, scoped, env, default) -> None:
    if env is not None:
        monkeypatch.setenv("FASTVIDEO_ATTENTION_BACKEND", env)

    if scoped == "unset":
        # No scope and no explicit request: env is consulted.
        got = selector.get_attn_backend(default_backend=default, **KWARGS)
        expected = _oracle(None, env, default)
    else:
        # Scope active: env suppressed, scoped request used.
        with selector._component_attention_backend_scope(scoped, component="dit"):
            got = selector.get_attn_backend(default_backend=default, **KWARGS)
        expected = _oracle(scoped, None, default)
    assert got == expected


@pytest.mark.parametrize("requested", [None, SAGE])
@pytest.mark.parametrize("env", [None, "TORCH_SDPA"])
@pytest.mark.parametrize("default", [None, SAGE])
def test_explicit_request_matches_the_same_oracle(monkeypatch, requested, env, default) -> None:
    """An explicit `requested=` resolves exactly as the same request would via
    a scope, and never consults the environment behind it."""
    if env is not None:
        monkeypatch.setenv("FASTVIDEO_ATTENTION_BACKEND", env)

    got = selector.get_attn_backend(default_backend=default, requested=requested, **KWARGS)
    assert got == _oracle(requested, None, default)


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


def test_explicit_none_means_auto_not_absence(monkeypatch) -> None:
    """`requested=None` is an answer ("this component resolved to automatic
    selection"), so the env var must NOT be consulted behind it. Omitting the
    argument is the absence, and still falls back to the env."""
    monkeypatch.setenv("FASTVIDEO_ATTENTION_BACKEND", "SAGE_ATTN")

    # Explicit "automatic selection" -> the fake platform's auto answer.
    assert selector.get_attn_backend(requested=None, **KWARGS) == "FLASH_ATTN"
    # Counterfactual: same call, no opinion -> env still wins.
    assert selector.get_attn_backend(**KWARGS) == "SAGE_ATTN"
    assert selector.get_attn_backend(requested=selector.NO_REQUEST, **KWARGS) == "SAGE_ATTN"


def test_explicit_request_outranks_the_construction_scope(monkeypatch) -> None:
    """A caller that knows its component's decision is not overridden by an
    ambient scope that happens to be active."""
    monkeypatch.setenv("FASTVIDEO_ATTENTION_BACKEND", "TORCH_SDPA")
    with selector._component_attention_backend_scope(SDPA, component="dit"):
        assert selector.get_attn_backend(**KWARGS) == "TORCH_SDPA"
        assert selector.get_attn_backend(requested=SAGE, **KWARGS) == "SAGE_ATTN"


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


# ---------------------------------------------------------------------------
# Reading the component's recorded decision back
# ---------------------------------------------------------------------------


def test_component_attention_backend_reads_the_recorded_decision():
    from fastvideo.configs.models.dits.base import DiTConfig

    class _Component:

        def __init__(self, config):
            self.config = config

    config = DiTConfig()
    with selector._component_attention_backend_scope(SAGE, component="transformer"):
        selector.record_resolved_attention_backend(config)

    assert selector.component_attention_backend(_Component(config)) == SAGE


def test_component_with_no_concrete_decision_reports_no_request(monkeypatch):
    """"Stamped with no request" and "never stamped" are the SAME stored value.

    ``ModelConfig._resolved_attention_backend`` is a field defaulting to None,
    so the attribute always exists and ``record_resolved_attention_backend``
    writes None whenever no scope is active. Neither state may suppress the env
    var at a call site that previously honoured it, so both report NO_REQUEST.
    """
    from fastvideo.configs.models.dits.base import DiTConfig

    class _Component:

        def __init__(self, config):
            self.config = config

    # (a) stamped outside any scope -> stored None -> NO_REQUEST
    stamped = DiTConfig()
    assert selector.record_resolved_attention_backend(stamped) is None
    assert selector.component_attention_backend(_Component(stamped)) is selector.NO_REQUEST

    # (b) never stamped at all -> same stored value, same answer
    untouched = DiTConfig()
    assert untouched._resolved_attention_backend is None  # the field always exists
    assert selector.component_attention_backend(_Component(untouched)) is selector.NO_REQUEST

    # (c) and it does NOT suppress the env var, which is the whole point
    monkeypatch.setenv("FASTVIDEO_ATTENTION_BACKEND", "TORCH_SDPA")
    selector._cached_get_attn_backend.cache_clear()
    passthrough = selector.get_attn_backend(requested=selector.component_attention_backend(_Component(untouched)),
                                            **KWARGS)
    assert passthrough == selector.get_attn_backend(**KWARGS)


def test_component_with_a_concrete_decision_reports_it():
    """A real recorded backend is an answer, and overrides the ambient fallback."""
    from fastvideo.configs.models.dits.base import DiTConfig

    class _Component:

        def __init__(self, config):
            self.config = config

    config = DiTConfig()
    with selector._component_attention_backend_scope(SAGE, component="transformer"):
        assert selector.record_resolved_attention_backend(config) == SAGE
    assert selector.component_attention_backend(_Component(config)) == SAGE


def test_component_without_a_recorded_decision_reports_no_request():
    """A transformer built outside a loader (or carrying a non-FastVideo
    config) has no decision, so callers keep the previous fallback."""

    class _Bare:
        pass

    class _HFStyle:

        def __init__(self):
            self.config = object()

    assert selector.component_attention_backend(_Bare()) is selector.NO_REQUEST
    assert selector.component_attention_backend(_HFStyle()) is selector.NO_REQUEST
