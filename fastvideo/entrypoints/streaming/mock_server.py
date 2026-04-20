# SPDX-License-Identifier: Apache-2.0
"""Mock streaming server — a frontend dev aid.

Boots the same FastAPI app the real streaming server uses, but backs
it with :class:`InProcessGpuPool` wrapping a synthetic generator that
emits pre-baked RGB frames. No GPU or model weights required.

Use cases:

* Frontend development without a real model loaded.
* Integration tests that exercise the WS protocol end-to-end.
* Reproducing protocol bugs locally.

Launch: ``python -m fastvideo.entrypoints.streaming.mock_server``.
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from fastvideo.api.schema import (
    ContinuationState,
    GenerationRequest,
    GeneratorConfig,
    SamplingConfig,
    ServeConfig,
    StreamingConfig,
)
from fastvideo.entrypoints.streaming.server import build_app


@dataclass
class MockGenerator:
    """Generator stand-in that returns synthetic gradient frames.

    Each call produces one segment worth of frames whose pixels vary by
    a constant derived from the request seed and segment index. Latency
    is configurable via ``sleep_ms`` so the caller can exercise slow-
    generate scenarios without spinning a GPU.
    """

    sleep_ms: float = 0.0

    def generate(self, request: GenerationRequest) -> dict[str, Any]:
        if self.sleep_ms:
            time.sleep(self.sleep_ms / 1000.0)
        width = max(16, request.sampling.width)
        height = max(16, request.sampling.height)
        num_frames = max(1, request.sampling.num_frames)
        frames = [_gradient_frame(height, width, idx, seed=request.sampling.seed) for idx in range(num_frames)]
        state = ContinuationState(
            kind="ltx2.v1",
            payload={
                "schema_version": 1,
                "segment_index": 0,
                "source_prompt": request.prompt,
            },
        )
        return {
            "frames": frames,
            "audio_sample_rate": 24000,
            "state": state,
        }


def _gradient_frame(height: int, width: int, idx: int, *, seed: int) -> np.ndarray:
    base = (idx * 17 + seed * 3) % 256
    row = np.linspace(base, (base + 64) % 256, width, dtype=np.uint8)
    frame = np.tile(row, (height, 1))
    stacked = np.stack([frame, np.roll(frame, 8, axis=1), np.roll(frame, 16, axis=1)], axis=-1)
    return stacked.astype(np.uint8)


def build_mock_app(*, sleep_ms: float = 0.0):
    """Build a FastAPI app backed by :class:`MockGenerator`."""
    serve_config = ServeConfig(
        generator=GeneratorConfig(model_path="/models/mock"),
        streaming=StreamingConfig(
            session_timeout_seconds=120,
            generation_segment_cap=6,
        ),
    )
    serve_config.default_request.sampling = SamplingConfig(
        num_frames=24,
        height=256,
        width=256,
        fps=24,
        num_inference_steps=1,
    )
    return build_app(serve_config, MockGenerator(sleep_ms=sleep_ms))


def main() -> None:  # pragma: no cover - CLI entry
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--sleep-ms",
        type=float,
        default=0.0,
        help="Per-segment artificial latency for testing slow paths",
    )
    args = parser.parse_args()

    import uvicorn

    app = build_mock_app(sleep_ms=args.sleep_ms)
    uvicorn.run(app, host=args.host, port=args.port)


__all__ = [
    "MockGenerator",
    "build_mock_app",
    "main",
]

if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()
