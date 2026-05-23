# SPDX-License-Identifier: Apache-2.0
"""Smoke tests for shared attention infra additions landed in PR #1225 slice 5.

Covers:
- ``AttentionMetadata.VSA_sparsity`` kw-only field with default 0.0
  (Attn-QAT 5/12). The field moved from per-backend metadata subclasses
  (e.g. ``VideoSparseAttentionMetadata``) up into the base so non-VSA
  backends can satisfy a uniform interface without re-declaring the field.
- ``AttentionMetadata.__getattr__`` raises a clean ``AttributeError``
  for unknown attributes (previously default object behaviour, but the
  explicit method gives consumers a stable error message they can match
  against).

These tests are CPU-only and have no GPU/flash-attn dependency.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from fastvideo.attention.backends.abstract import AttentionMetadata


@dataclass
class _DummyMetadata(AttentionMetadata):
    extra: int = 0


def test_vsa_sparsity_defaults_to_zero() -> None:
    """Base ``AttentionMetadata`` carries ``VSA_sparsity`` with default 0.0."""
    md = AttentionMetadata(current_timestep=3)
    assert md.VSA_sparsity == 0.0


def test_vsa_sparsity_kw_only_assignment() -> None:
    """``VSA_sparsity`` is settable via keyword (kw_only=True)."""
    md = AttentionMetadata(current_timestep=1, VSA_sparsity=0.75)
    assert md.VSA_sparsity == 0.75


def test_vsa_sparsity_is_kw_only_not_positional() -> None:
    """``VSA_sparsity`` cannot be passed positionally — protects against
    accidental positional misuse by subclasses that add fields."""
    with pytest.raises(TypeError):
        # Two positional args should be rejected: only `current_timestep`
        # is positional on the base class.
        AttentionMetadata(1, 0.5)  # type: ignore[call-arg]


def test_subclass_inherits_vsa_sparsity_default() -> None:
    """Subclasses inherit ``VSA_sparsity`` without re-declaring it."""
    md = _DummyMetadata(current_timestep=0, extra=42)
    assert md.VSA_sparsity == 0.0
    assert md.extra == 42


def test_subclass_can_override_vsa_sparsity_via_kw() -> None:
    md = _DummyMetadata(current_timestep=0, extra=7, VSA_sparsity=0.9)
    assert md.VSA_sparsity == pytest.approx(0.9)
    assert md.extra == 7


def test_unknown_attribute_raises_attribute_error() -> None:
    """``__getattr__`` raises ``AttributeError`` (not e.g. ``KeyError``).

    The dataclass machinery only invokes ``__getattr__`` when normal
    attribute lookup fails, so this protects callers from silent failures
    when they typo a field name.
    """
    md = AttentionMetadata(current_timestep=0)
    with pytest.raises(AttributeError, match="no attribute 'does_not_exist'"):
        _ = md.does_not_exist


def test_asdict_zerocopy_includes_vsa_sparsity() -> None:
    md = AttentionMetadata(current_timestep=2, VSA_sparsity=0.25)
    snapshot = md.asdict_zerocopy()
    assert snapshot["current_timestep"] == 2
    assert snapshot["VSA_sparsity"] == 0.25


def test_asdict_zerocopy_respects_skip_fields() -> None:
    md = AttentionMetadata(current_timestep=2, VSA_sparsity=0.5)
    snapshot = md.asdict_zerocopy(skip_fields={"VSA_sparsity"})
    assert "VSA_sparsity" not in snapshot
    assert snapshot["current_timestep"] == 2
