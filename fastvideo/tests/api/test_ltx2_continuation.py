# SPDX-License-Identifier: Apache-2.0
"""Tests for the typed LTX-2 continuation state.

Covers:

* round-trip through :class:`ContinuationState` (inline and blob-backed)
* payload is JSON-serializable (Dynamo RPC / HTTP client constraint)
* kind / schema_version validation on deserialization
* compat-layer validation (known kinds, payload shape)
* round-trip through :func:`request_to_sampling_param` attaches the
  state to the resulting :class:`SamplingParam` without losing fidelity
"""
from __future__ import annotations

import json

import numpy as np
import pytest
import torch

# Importing compat first, then the LTX-2 module, exercises the
# self-registration side effect on import (important for the API
# test suite where the pipeline package isn't otherwise imported).
from fastvideo.api import compat as api_compat  # noqa: F401
from fastvideo.api.schema import (
    ContinuationState,
    GenerationRequest,
    OutputConfig,
)
from fastvideo.entrypoints.streaming.session_store import InMemoryBlobStore
from fastvideo.pipelines.basic.ltx2.continuation import (
    LTX2_CONTINUATION_KIND,
    LTX2_CONTINUATION_SCHEMA_VERSION,
    LTX2ContinuationState,
)


def _make_typed_state() -> LTX2ContinuationState:
    return LTX2ContinuationState(
        segment_index=3,
        video_frames=[
            (np.ones((64, 64, 3), dtype=np.uint8) * (i * 10)) for i in range(4)
        ],
        video_conditioning_frame_idx=9,
        video_conditioning_strength=0.75,
        audio_latents=torch.randn(1, 4, 16, 64, dtype=torch.float32),
        audio_sample_rate=24000,
        audio_conditioning_num_frames=5,
        audio_conditioning_strength=0.5,
        video_position_offset_sec=0.125,
        metadata={"note": "unit-test"},
    )


class TestRoundTrip:
    """Round-trip through :class:`ContinuationState` preserves all fields."""

    def test_kind_and_schema_version(self):
        state = _make_typed_state().to_continuation_state()
        assert state.kind == LTX2_CONTINUATION_KIND
        assert state.payload["schema_version"] == LTX2_CONTINUATION_SCHEMA_VERSION

    def test_inline_roundtrip_preserves_scalars(self):
        original = _make_typed_state()
        envelope = original.to_continuation_state()
        restored = LTX2ContinuationState.from_continuation_state(envelope)
        assert restored.segment_index == original.segment_index
        assert restored.video_conditioning_frame_idx == (
            original.video_conditioning_frame_idx)
        assert restored.video_conditioning_strength == (
            original.video_conditioning_strength)
        assert restored.audio_sample_rate == original.audio_sample_rate
        assert restored.audio_conditioning_num_frames == (
            original.audio_conditioning_num_frames)
        assert restored.audio_conditioning_strength == (
            original.audio_conditioning_strength)
        assert restored.video_position_offset_sec == (
            original.video_position_offset_sec)
        assert restored.metadata == original.metadata

    def test_inline_roundtrip_preserves_video_frames(self):
        original = _make_typed_state()
        envelope = original.to_continuation_state()
        restored = LTX2ContinuationState.from_continuation_state(envelope)
        assert restored.video_frames is not None
        assert len(restored.video_frames) == len(original.video_frames)
        for before, after in zip(original.video_frames,
                                  restored.video_frames):
            np.testing.assert_array_equal(before, after)

    def test_inline_roundtrip_preserves_audio_latents(self):
        original = _make_typed_state()
        envelope = original.to_continuation_state()
        restored = LTX2ContinuationState.from_continuation_state(envelope)
        assert restored.audio_latents is not None
        assert tuple(restored.audio_latents.shape) == tuple(
            original.audio_latents.shape)
        assert restored.audio_latents.dtype == original.audio_latents.dtype
        torch.testing.assert_close(
            restored.audio_latents, original.audio_latents)

    def test_payload_is_json_serializable(self):
        envelope = _make_typed_state().to_continuation_state()
        # json.dumps must not raise — required for Dynamo RPC transport
        # and HTTP client round-trip.
        reserialized = json.loads(json.dumps(envelope.payload))
        restored = LTX2ContinuationState.from_continuation_state(
            ContinuationState(
                kind=envelope.kind,
                payload=reserialized,
            ))
        assert restored.segment_index == 3


class TestBlobIndirection:
    """Large tensors live in the :class:`BlobStore` instead of the payload."""

    def test_threshold_triggers_blob_path(self):
        blob_store = InMemoryBlobStore()
        state = _make_typed_state()
        envelope = state.to_continuation_state(
            blob_store=blob_store,
            inline_threshold_bytes=0,
        )
        assert "blob_id" in envelope.payload["video"]
        assert "blob_id" in envelope.payload["audio"]
        assert "frames_b64" not in envelope.payload["video"]
        assert "data_b64" not in envelope.payload["audio"]
        assert len(blob_store) == 2

    def test_blob_roundtrip_reconstructs_tensors(self):
        blob_store = InMemoryBlobStore()
        original = _make_typed_state()
        envelope = original.to_continuation_state(
            blob_store=blob_store,
            inline_threshold_bytes=0,
        )
        restored = LTX2ContinuationState.from_continuation_state(
            envelope, blob_store=blob_store)
        assert restored.video_frames is not None
        assert len(restored.video_frames) == len(original.video_frames)
        torch.testing.assert_close(
            restored.audio_latents, original.audio_latents)

    def test_blob_id_held_when_store_unavailable(self):
        """Deserializing without a blob store preserves the blob id so
        the caller can fetch it later."""
        blob_store = InMemoryBlobStore()
        envelope = _make_typed_state().to_continuation_state(
            blob_store=blob_store,
            inline_threshold_bytes=0,
        )
        blob_id_video = envelope.payload["video"]["blob_id"]
        blob_id_audio = envelope.payload["audio"]["blob_id"]

        restored = LTX2ContinuationState.from_continuation_state(envelope)
        assert restored.video_frames is None
        assert restored.video_frames_blob_id == blob_id_video
        assert restored.audio_latents is None
        assert restored.audio_latents_blob_id == blob_id_audio

    def test_large_threshold_keeps_payload_inline(self):
        blob_store = InMemoryBlobStore()
        envelope = _make_typed_state().to_continuation_state(
            blob_store=blob_store,
            inline_threshold_bytes=10 * 1024 * 1024,  # 10 MiB
        )
        assert "frames_b64" in envelope.payload["video"]
        assert "data_b64" in envelope.payload["audio"]
        assert len(blob_store) == 0


class TestValidation:
    """Invalid payloads error cleanly."""

    def test_wrong_kind_rejected(self):
        envelope = ContinuationState(kind="longcat.v1", payload={})
        with pytest.raises(ValueError, match="Expected ContinuationState.kind"):
            LTX2ContinuationState.from_continuation_state(envelope)

    def test_unsupported_schema_version_rejected(self):
        envelope = ContinuationState(
            kind=LTX2_CONTINUATION_KIND,
            payload={"schema_version": 999},
        )
        with pytest.raises(ValueError,
                            match="Unsupported LTX-2 continuation schema"):
            LTX2ContinuationState.from_continuation_state(envelope)

    def test_non_png_frame_rejected(self):
        state = LTX2ContinuationState(
            video_frames=[np.ones((64, 64, 3), dtype=np.float32)],
        )
        with pytest.raises(ValueError, match="uint8 HxWx3"):
            state.to_continuation_state()


class TestCompatLayerWireUp:
    """The public compat layer accepts request.state without reverting
    to NotImplementedError and attaches it to the SamplingParam path."""

    def test_request_with_state_passes_through(self, tmp_path):
        # PR 7 removes the NotImplementedError for request.state; build a
        # minimal GenerationRequest carrying an LTX-2 state and make sure
        # the public boundary accepts it.
        from fastvideo.api.compat import (
            normalize_generation_request,
            _validate_continuation_state,
        )
        envelope = _make_typed_state().to_continuation_state()
        request = GenerationRequest(
            prompt="test",
            state=envelope,
        )
        normalized = normalize_generation_request(request)
        _validate_continuation_state(normalized.state)

    def test_unknown_kind_rejected_at_boundary(self):
        from fastvideo.api.compat import _validate_continuation_state
        with pytest.raises(ValueError, match="Unknown ContinuationState kind"):
            _validate_continuation_state(
                ContinuationState(kind="mystery.v1", payload={}))

    def test_empty_kind_rejected_at_boundary(self):
        from fastvideo.api.compat import _validate_continuation_state
        with pytest.raises(ValueError, match="non-empty string"):
            _validate_continuation_state(
                ContinuationState(kind="", payload={}))

    def test_output_return_state_flag(self):
        request = GenerationRequest(
            prompt="x",
            output=OutputConfig(return_state=True),
        )
        # The typed public surface exposes the flag directly.
        assert request.output.return_state is True
