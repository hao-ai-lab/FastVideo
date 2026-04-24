# SPDX-License-Identifier: Apache-2.0
"""Typed continuation state for the LTX-2 streaming pipeline.

Segment N+1 conditions on segment N's trailing decoded frames and
denoised audio latents. The streaming runtime used to hold this state as
per-worker globals; lifting it into a typed, JSON-serializable object
lets clients snapshot, migrate, or round-trip it through an HTTP/RPC
boundary. The envelope ``ContinuationState(kind, payload)`` is the
shared public API; the typed class here owns the LTX-2 payload shape.

Serialization contract:

* Video frames → PNG bytes + base64, or a :class:`BlobStore` id.
* Audio latents → a self-describing safetensors blob + base64, or a
  :class:`BlobStore` id. safetensors preserves ``bfloat16``, which a
  raw-numpy round-trip cannot.
* The returned payload is always a plain JSON-serializable dict.
"""
from __future__ import annotations

import base64
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from fastvideo.api.compat import register_continuation_kind
from fastvideo.api.schema import ContinuationState

if TYPE_CHECKING:
    import numpy as np
    import torch

    from fastvideo.entrypoints.streaming.session_store import BlobStore

LTX2_CONTINUATION_KIND = "ltx2.v1"
"""Public ``ContinuationState.kind`` for LTX-2 payloads."""

LTX2_CONTINUATION_SCHEMA_VERSION = 1
"""Payload schema version carried inside ``payload.schema_version``."""

DEFAULT_INLINE_THRESHOLD_BYTES = 2 * 1024 * 1024
"""Tensors larger than this go to the blob store (if available). 2 MiB
is below typical single-JSON-message limits (Dynamo: 4 MiB, Postgres
TOAST: 1 GiB) and well above per-frame PNG payloads (~200 KiB at
512x512)."""


@dataclass
class LTX2ContinuationState:
    """Typed LTX-2 continuation state carried between streaming segments.

    ``video_frames`` hold trailing decoded RGB frames (uint8 HxWx3) from
    segment N for conditioning segment N+1 via the VAE encode path.
    ``audio_latents`` is the cached denoised audio latent tensor of shape
    ``[B, C, T, mel]`` that segment N+1 will copy into the overlap
    region of its clean-latent conditioning.

    Most fields map 1:1 onto the internal gpu_pool's per-worker state;
    the only new concept is the ``*_blob_id`` fields, which allow large
    tensors to live outside the JSON payload. See module docstring.
    """

    segment_index: int = 0
    """Index of the *just-completed* segment. Segment 0 has no history;
    state returned after segment 0 carries ``segment_index=0`` and the
    caller uses ``segment_index + 1`` as the next segment number."""

    video_frames: list[np.ndarray] | None = None
    """Trailing decoded frames, each an RGB uint8 ``np.ndarray`` shaped
    ``(H, W, 3)``. ``None`` when the state is blob-backed or unset."""

    video_frames_blob_id: str | None = None
    """Blob store id when the frames live outside the payload."""

    video_conditioning_frame_idx: int = 0
    """Target frame index inside the next segment that the trailing
    frames align with (matches the LTX-2 ``ltx2_video_conditions``
    tuple's ``frame_idx`` slot)."""

    video_conditioning_strength: float = 1.0
    """Conditioning strength in [0, 1]. Matches the ``ltx2_video_
    conditions`` tuple's strength slot."""

    audio_latents: torch.Tensor | None = None
    """Denoised audio latent tensor of shape ``[B, C, T, mel]``.
    ``None`` when the state is blob-backed or unset."""

    audio_latents_blob_id: str | None = None
    """Blob store id when audio latents live outside the payload."""

    audio_sample_rate: int | None = None
    """Sample rate for the audio side (e.g. 24000)."""

    audio_conditioning_num_frames: int = 0
    """Number of trailing audio frames that carry over as clean context
    into segment N+1."""

    audio_conditioning_strength: float = 1.0
    """Clean-latent mask value applied to the overlap region; 0.0 keeps
    the cached audio entirely, 1.0 renoises from scratch."""

    video_position_offset_sec: float = 0.0
    """Seconds by which video RoPE is shifted forward so the audio
    prefix can sit at ``t >= 0`` when audio conditioning is longer than
    video conditioning."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Opaque metadata bag for forward-compat fields that don't need
    their own typed slot yet (e.g. custom knob experiments)."""

    def to_continuation_state(
        self,
        *,
        blob_store: BlobStore | None = None,
        inline_threshold_bytes: int = DEFAULT_INLINE_THRESHOLD_BYTES,
    ) -> ContinuationState:
        """Serialize into a public :class:`ContinuationState`.

        When ``blob_store`` is given, tensors larger than
        ``inline_threshold_bytes`` are stored via
        :meth:`BlobStore.put` and referenced by id; otherwise all data
        is base64-encoded inline. The payload is always a plain
        JSON-serializable dict.
        """
        payload: dict[str, Any] = {
            "schema_version": LTX2_CONTINUATION_SCHEMA_VERSION,
            "segment_index": int(self.segment_index),
            "video_conditioning_frame_idx": int(self.video_conditioning_frame_idx),
            "video_conditioning_strength": float(self.video_conditioning_strength),
            "audio_conditioning_num_frames": int(self.audio_conditioning_num_frames),
            "audio_conditioning_strength": float(self.audio_conditioning_strength),
            "video_position_offset_sec": float(self.video_position_offset_sec),
            "metadata": dict(self.metadata),
        }
        if self.audio_sample_rate is not None:
            payload["audio_sample_rate"] = int(self.audio_sample_rate)

        video_payload = self._encode_video_frames(
            blob_store=blob_store,
            inline_threshold_bytes=inline_threshold_bytes,
        )
        if video_payload is not None:
            payload["video"] = video_payload

        audio_payload = self._encode_audio_latents(
            blob_store=blob_store,
            inline_threshold_bytes=inline_threshold_bytes,
        )
        if audio_payload is not None:
            payload["audio"] = audio_payload

        return ContinuationState(
            kind=LTX2_CONTINUATION_KIND,
            payload=payload,
        )

    @classmethod
    def from_continuation_state(
        cls,
        state: ContinuationState,
        *,
        blob_store: BlobStore | None = None,
    ) -> LTX2ContinuationState:
        """Rebuild a typed state from a public :class:`ContinuationState`.

        Raises :class:`ValueError` when the kind doesn't match or the
        schema version is unsupported.
        """
        if state.kind != LTX2_CONTINUATION_KIND:
            raise ValueError(f"Expected ContinuationState.kind={LTX2_CONTINUATION_KIND!r}, "
                             f"got {state.kind!r}")
        payload = state.payload or {}
        version = int(payload.get("schema_version", LTX2_CONTINUATION_SCHEMA_VERSION))
        if version != LTX2_CONTINUATION_SCHEMA_VERSION:
            raise ValueError(f"Unsupported LTX-2 continuation schema_version={version}; "
                             f"this build expects {LTX2_CONTINUATION_SCHEMA_VERSION}")

        out = cls(
            segment_index=int(payload.get("segment_index", 0)),
            video_conditioning_frame_idx=int(payload.get("video_conditioning_frame_idx", 0)),
            video_conditioning_strength=float(payload.get("video_conditioning_strength", 1.0)),
            audio_sample_rate=(int(payload["audio_sample_rate"]) if "audio_sample_rate" in payload else None),
            audio_conditioning_num_frames=int(payload.get("audio_conditioning_num_frames", 0)),
            audio_conditioning_strength=float(payload.get("audio_conditioning_strength", 1.0)),
            video_position_offset_sec=float(payload.get("video_position_offset_sec", 0.0)),
            metadata=dict(payload.get("metadata") or {}),
        )

        video = payload.get("video")
        if isinstance(video, Mapping):
            cls._decode_video_frames(out, video, blob_store=blob_store)

        audio = payload.get("audio")
        if isinstance(audio, Mapping):
            cls._decode_audio_latents(out, audio, blob_store=blob_store)

        return out

    # ------------------------------------------------------------------
    # Video frame helpers
    # ------------------------------------------------------------------

    def _encode_video_frames(
        self,
        *,
        blob_store: BlobStore | None,
        inline_threshold_bytes: int,
    ) -> dict[str, Any] | None:
        if self.video_frames_blob_id is not None:
            return {"blob_id": self.video_frames_blob_id}
        if not self.video_frames:
            return None

        encoded = [_encode_png(frame) for frame in self.video_frames]
        total = sum(len(b) for b in encoded)
        if blob_store is not None and total > inline_threshold_bytes:
            concatenated = _pack_frame_blobs(encoded)
            blob_id = blob_store.put(
                concatenated,
                mime="application/x-fastvideo-frames+png",
            )
            return {"blob_id": blob_id, "frame_count": len(encoded)}
        return {
            "frames_b64": [base64.b64encode(b).decode("ascii") for b in encoded],
        }

    @staticmethod
    def _decode_video_frames(
        out: LTX2ContinuationState,
        video: Mapping[str, Any],
        *,
        blob_store: BlobStore | None,
    ) -> None:
        blob_id = video.get("blob_id")
        if isinstance(blob_id, str):
            if blob_store is None:
                out.video_frames_blob_id = blob_id
                return
            raw = blob_store.get(blob_id)
            encoded = _unpack_frame_blobs(raw)
            out.video_frames = [_decode_png(b) for b in encoded]
            return
        frames_b64 = video.get("frames_b64")
        if isinstance(frames_b64, list):
            decoded = [_decode_png(base64.b64decode(b)) for b in frames_b64 if isinstance(b, str)]
            out.video_frames = decoded or None

    # ------------------------------------------------------------------
    # Audio latent helpers
    # ------------------------------------------------------------------

    def _encode_audio_latents(
        self,
        *,
        blob_store: BlobStore | None,
        inline_threshold_bytes: int,
    ) -> dict[str, Any] | None:
        if self.audio_latents_blob_id is not None:
            return {"blob_id": self.audio_latents_blob_id}
        if self.audio_latents is None:
            return None
        raw = _tensor_to_safetensors_bytes(self.audio_latents)
        if blob_store is not None and len(raw) > inline_threshold_bytes:
            blob_id = blob_store.put(
                raw,
                mime="application/x-fastvideo-tensor+safetensors",
            )
            return {"blob_id": blob_id}
        return {"safetensors_b64": base64.b64encode(raw).decode("ascii")}

    @staticmethod
    def _decode_audio_latents(
        out: LTX2ContinuationState,
        audio: Mapping[str, Any],
        *,
        blob_store: BlobStore | None,
    ) -> None:
        blob_id = audio.get("blob_id")
        if isinstance(blob_id, str):
            if blob_store is None:
                out.audio_latents_blob_id = blob_id
                return
            raw = blob_store.get(blob_id)
            out.audio_latents = _safetensors_bytes_to_tensor(raw)
            return
        data_b64 = audio.get("safetensors_b64")
        if isinstance(data_b64, str):
            out.audio_latents = _safetensors_bytes_to_tensor(base64.b64decode(data_b64))


def _encode_png(frame: np.ndarray) -> bytes:
    """Encode an ``(H, W, 3)`` uint8 RGB frame as PNG bytes."""
    import numpy as np
    from PIL import Image

    if not isinstance(frame, np.ndarray):
        raise TypeError(f"LTX2 continuation frame must be a numpy ndarray, got {type(frame).__name__}")
    if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[-1] != 3:
        raise ValueError("LTX2 continuation frame must be uint8 HxWx3 RGB; got "
                         f"dtype={frame.dtype}, shape={frame.shape}")
    import io

    buffer = io.BytesIO()
    Image.fromarray(frame).save(buffer, format="PNG")
    return buffer.getvalue()


def _decode_png(data: bytes) -> np.ndarray:
    import io

    import numpy as np
    from PIL import Image

    img = Image.open(io.BytesIO(data)).convert("RGB")
    return np.array(img, dtype=np.uint8)


def _pack_frame_blobs(encoded: list[bytes]) -> bytes:
    """Pack multiple PNG blobs into a single blob for blob-store storage.

    Format: ``[4-byte big-endian count][4-byte len][png][4-byte len][png]...``.
    """
    parts: list[bytes] = [len(encoded).to_bytes(4, "big")]
    for blob in encoded:
        parts.append(len(blob).to_bytes(4, "big"))
        parts.append(blob)
    return b"".join(parts)


def _unpack_frame_blobs(raw: bytes) -> list[bytes]:
    count = int.from_bytes(raw[:4], "big")
    out: list[bytes] = []
    cursor = 4
    for _ in range(count):
        length = int.from_bytes(raw[cursor:cursor + 4], "big")
        cursor += 4
        out.append(raw[cursor:cursor + length])
        cursor += length
    return out


def _tensor_to_safetensors_bytes(tensor: Any) -> bytes:
    """Serialize a torch tensor to a self-describing safetensors blob.

    Uses the in-memory safetensors API so the wire format preserves
    dtype (including ``bfloat16``, which a raw-numpy path cannot) and
    shape without needing sidecar metadata.
    """
    import torch
    from safetensors.torch import save as st_save

    if isinstance(tensor, torch.Tensor):
        return st_save({"t": tensor.detach().cpu()})
    import numpy as np
    if isinstance(tensor, np.ndarray):
        return st_save({"t": torch.from_numpy(np.ascontiguousarray(tensor))})
    raise TypeError("LTX2 audio_latents must be a torch.Tensor or numpy.ndarray, got "
                    f"{type(tensor).__name__}")


def _safetensors_bytes_to_tensor(raw: bytes) -> Any:
    from safetensors.torch import load as st_load
    return st_load(raw)["t"]


register_continuation_kind(LTX2_CONTINUATION_KIND)

__all__ = [
    "DEFAULT_INLINE_THRESHOLD_BYTES",
    "LTX2ContinuationState",
    "LTX2_CONTINUATION_KIND",
    "LTX2_CONTINUATION_SCHEMA_VERSION",
]
