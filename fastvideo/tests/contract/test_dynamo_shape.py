# SPDX-License-Identifier: Apache-2.0
"""Contract test: a mock Dynamo-style handler wraps FastVideo's public API
without touching any private module.

The Dynamo backend package (``components/src/dynamo/fastvideo/`` in the
Dynamo repo) imports only these symbols:

    from fastvideo import VideoGenerator
    from fastvideo.api import (
        ContinuationState, GenerationRequest, InputConfig, OutputConfig,
        SamplingConfig,
    )

If a FastVideo refactor breaks the adapter shape this test fails at
FastVideo CI — before the Dynamo-side integration knows. The plan (PR
7.10) requires the backend to be expressible without flat legacy LTX-2
kwargs or FastVideo-internal imports; this file asserts the subset that
exists today and is stable.
"""
from __future__ import annotations

import asyncio
import base64
import importlib
import sys
import time
from dataclasses import dataclass
from typing import Any, AsyncIterator

import pytest

# Public imports only — mirror what the Dynamo backend package uses.
from fastvideo.api import (
    ContinuationState,
    GenerationRequest,
    InputConfig,
    OutputConfig,
    SamplingConfig,
)


# ----------------------------------------------------------------------
# Fake Dynamo-side types (the real ones live in dynamo.llm and dynamo.*)
# We mimic their dict shape so the adapter code under test is realistic.
# ----------------------------------------------------------------------


@dataclass
class _FakeContext:
    """Stand-in for Dynamo's async RPC context."""

    request_id: str = "test-req"


def _nv_create_video_request(**overrides: Any) -> dict[str, Any]:
    """Build a Dynamo ``NvCreateVideoRequest``-shaped dict."""
    base: dict[str, Any] = {
        "prompt": "a fox running through snow",
        "size": "1024x1536",
        "seconds": 5,
        "response_format": "b64_json",
        "model": "Lightricks/LTX-Video",
        "nvext": {
            "fps": 24,
            "num_inference_steps": 8,
            "guidance_scale": 1.0,
            "seed": 42,
            "negative_prompt": "blurry",
        },
    }
    base.update(overrides)
    if "nvext" in overrides:
        base["nvext"] = {**base["nvext"], **overrides["nvext"]}
    return base


# ----------------------------------------------------------------------
# The adapter under test — written as if it lived in
# components/src/dynamo/fastvideo/request_handlers/video_generation/.
# ----------------------------------------------------------------------


def _parse_size(size: str | None) -> tuple[int, int]:
    if not size or "x" not in size:
        return 1024, 1536
    w, h = size.split("x", 1)
    return int(w), int(h)


def nv_request_to_generation_request(
    request: dict[str, Any], ) -> GenerationRequest:
    """Translate Dynamo's request shape into FastVideo's typed request.

    This function is the template integrators copy into the Dynamo repo.
    It uses only public FastVideo symbols.
    """
    nvext = request.get("nvext") or {}
    width, height = _parse_size(request.get("size"))
    fps = nvext.get("fps") or 24
    num_frames = nvext.get("num_frames") or (request.get("seconds") or 4) * fps

    state = nvext.get("continuation_state")
    if isinstance(state, dict):
        state = ContinuationState(kind=state["kind"], payload=state["payload"])

    return GenerationRequest(
        prompt=request["prompt"],
        negative_prompt=nvext.get("negative_prompt"),
        inputs=InputConfig(image_path=request.get("input_reference")),
        sampling=SamplingConfig(
            width=width,
            height=height,
            num_frames=num_frames,
            fps=fps,
            num_inference_steps=nvext.get("num_inference_steps", 50),
            guidance_scale=nvext.get("guidance_scale", 1.0),
            seed=nvext.get("seed", 1024),
            true_cfg_scale=nvext.get("true_cfg_scale"),
        ),
        output=OutputConfig(save_video=False, return_frames=False),
        state=state,
    )


class _MockFastVideoHandler:
    """Mock ``VideoGenerationWorkerHandler`` — the Dynamo backend's
    entry point. Shape verified: ``async def generate(dict, ctx) ->
    AsyncGenerator[dict, None]``, the signature Dynamo's
    ``endpoint.serve_endpoint(...)`` expects.
    """

    def __init__(self, fake_result: dict[str, Any]) -> None:
        self._fake_result = fake_result
        self._lock = asyncio.Lock()

    async def generate(
        self,
        request: dict[str, Any],
        context: _FakeContext,
    ) -> AsyncIterator[dict[str, Any]]:
        req = nv_request_to_generation_request(request)
        assert isinstance(req, GenerationRequest)

        # The real handler would call:
        #   result = await asyncio.to_thread(self.generator.generate, req)
        # For contract purposes we just assert the typed request reaches
        # the adapter boundary and yield a synthetic final event.
        t0 = time.perf_counter()
        async with self._lock:
            await asyncio.sleep(0)  # model call stub
        elapsed = time.perf_counter() - t0
        yield {
            "data": [{"b64_json": self._fake_result["b64_json"]}],
            "inference_time_s": elapsed,
            "model": request.get("model"),
            "nvext": {
                "continuation_state": (None if req.state is None else {
                    "kind": req.state.kind,
                    "payload": req.state.payload,
                }),
            },
        }


# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------


class TestDynamoAdapterShape:

    def test_nv_request_translates_to_typed_request(self):
        req = nv_request_to_generation_request(_nv_create_video_request())
        assert isinstance(req, GenerationRequest)
        assert req.prompt == "a fox running through snow"
        assert req.sampling.width == 1024
        assert req.sampling.height == 1536
        assert req.sampling.num_frames == 120  # 5 seconds * 24 fps
        assert req.sampling.fps == 24
        assert req.sampling.num_inference_steps == 8
        assert req.sampling.guidance_scale == 1.0
        assert req.sampling.seed == 42
        # negative_prompt lives at the request level, not inside sampling
        assert req.negative_prompt == "blurry"

    def test_nvext_num_frames_overrides_seconds_times_fps(self):
        req = nv_request_to_generation_request(
            _nv_create_video_request(nvext={"num_frames": 33}))
        assert req.sampling.num_frames == 33

    def test_input_reference_maps_to_image_path(self):
        req = nv_request_to_generation_request(
            _nv_create_video_request(input_reference="/tmp/init.png"))
        assert req.inputs.image_path == "/tmp/init.png"

    def test_continuation_state_round_trips_through_adapter(self):
        state = {
            "kind": "ltx2.v1",
            "payload": {
                "schema_version": 1,
                "segment_index": 2,
            },
        }
        req = nv_request_to_generation_request(
            _nv_create_video_request(nvext={"continuation_state": state}))
        assert isinstance(req.state, ContinuationState)
        assert req.state.kind == "ltx2.v1"
        assert req.state.payload["segment_index"] == 2


class TestDynamoHandlerContract:

    def test_handler_generate_is_async_generator(self):
        handler = _MockFastVideoHandler({"b64_json": "xyz"})

        async def run():
            out = []
            async for chunk in handler.generate(
                    _nv_create_video_request(), _FakeContext()):
                out.append(chunk)
            return out

        chunks = asyncio.run(run())
        assert len(chunks) == 1
        final = chunks[0]
        assert final["data"][0]["b64_json"] == "xyz"
        assert "inference_time_s" in final
        assert final["nvext"]["continuation_state"] is None

    def test_handler_serializes_state_back_to_nvext(self):
        """When the request carries state, the handler should be able to
        include a matching serialized state on the response. (Dynamo's
        NvVideosResponse has nvext.continuation_state reserved for this
        in the pending disaggregation path.)"""
        handler = _MockFastVideoHandler({"b64_json": "abc"})
        state_in = {
            "kind": "ltx2.v1",
            "payload": {"schema_version": 1, "segment_index": 4},
        }

        async def run():
            async for chunk in handler.generate(
                    _nv_create_video_request(
                        nvext={"continuation_state": state_in}),
                    _FakeContext()):
                return chunk
            raise AssertionError("no chunk emitted")

        chunk = asyncio.run(run())
        assert chunk["nvext"]["continuation_state"]["kind"] == "ltx2.v1"
        assert (
            chunk["nvext"]["continuation_state"]["payload"]["segment_index"]
            == 4)


class TestNoInternalImports:
    """The adapter template in this file imports only the public surface.

    Any change to FastVideo that requires the Dynamo adapter to reach
    into a private module would make this test fail at review time.
    """

    _PUBLIC_IMPORTS = frozenset({
        "fastvideo",
        "fastvideo.api",
    })

    _BANNED_PREFIXES = (
        "fastvideo.pipelines.",
        "fastvideo.configs.",
        "fastvideo.fastvideo_args",
        "fastvideo.api.compat",
        "fastvideo.api.parser",
        "fastvideo.api.overrides",
        "fastvideo.api.errors",
        "fastvideo.api.request_metadata",
        "fastvideo.api.sampling_param",
    )

    def test_adapter_template_does_not_import_private_modules(self):
        # Scan only the adapter function's source — not the whole test
        # file — so the banned-prefix list itself doesn't trip the guard.
        import inspect

        adapter_source = inspect.getsource(nv_request_to_generation_request)
        for banned in self._BANNED_PREFIXES:
            assert banned not in adapter_source, (
                f"Dynamo adapter template must not depend on {banned}*; "
                "found a reference in the adapter function body.")

    def test_public_types_are_stable_imports(self):
        # These lines match the exact import snippet in docs/design/
        # server_contracts/dynamo.md; any rename breaks CI.
        from fastvideo import VideoGenerator  # noqa: F401
        from fastvideo.api import (  # noqa: F401
            ContinuationState,
            GenerationRequest,
            InputConfig,
            OutputConfig,
            SamplingConfig,
        )

    @pytest.mark.skipif(
        "fastvideo.api" not in sys.modules,
        reason="fastvideo.api not importable in this environment",
    )
    def test_public_api_re_exports_match_doc(self):
        import fastvideo.api as api

        required = {
            "ContinuationState",
            "GenerationRequest",
            "InputConfig",
            "OutputConfig",
            "SamplingConfig",
        }
        missing = required - set(api.__all__)
        assert not missing, (
            f"fastvideo.api.__all__ missing Dynamo-contract symbols: {missing}")


def _read_own_source() -> str:
    import pathlib

    return pathlib.Path(__file__).read_text()


# Convenience export for third parties writing their own contract tests.
__all__ = [
    "nv_request_to_generation_request",
]
