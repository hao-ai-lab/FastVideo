"""Session-mode validation and IPC handoff without a GPU worker process."""

from __future__ import annotations

import asyncio
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

from dreamverse.generation_inputs import GenerationAsset, GenerationInputs
from dreamverse.worker_ipc import MediaChunk, MediaComplete, MediaInit


@pytest.fixture
def controller_module(monkeypatch):
    gpu_pool = ModuleType("dreamverse.gpu_pool")
    gpu_pool.GPUSlot = object
    monkeypatch.setitem(sys.modules, "dreamverse.gpu_pool", gpu_pool)
    path = Path(__file__).resolve().parents[1] / "session/controller.py"
    spec = importlib.util.spec_from_file_location("dreamverse_test_session_controller", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "ACTIVE_MODEL_ID", "full-h3")
    monkeypatch.setattr(module, "pin_generation_inputs", Mock())
    monkeypatch.setattr(module, "release_generation_inputs", Mock())
    return module


class Socket:
    def __init__(self):
        self.incoming = asyncio.Queue()
        self.outgoing = asyncio.Queue()
        self.messages = []
        self.closed = False

    async def accept(self):
        pass

    async def receive_json(self):
        return await self.incoming.get()

    async def send_json(self, payload):
        self.messages.append(payload)
        await self.outgoing.put(payload)

    async def send_bytes(self, payload):
        pass

    async def close(self, **kwargs):
        self.closed = True

    async def wait_for(self, kind):
        while True:
            payload = await asyncio.wait_for(self.outgoing.get(), 3)
            if payload["type"] == kind:
                return payload


class Slot:
    def __init__(self):
        self.shared_stream_buffer = None
        self.queue = asyncio.Queue()
        self.calls = []

    async def join_user(self, *args, **kwargs):
        pass

    def register_stream_queue(self, client_id):
        return self.queue

    def unregister_stream_queue(self, client_id):
        pass

    async def user_step(self, client_id, **kwargs):
        self.calls.append(kwargs)
        segment_idx = kwargs["segment_idx"]
        await self.queue.put(MediaInit(client_id, segment_idx, "test", "video/mp4", False))
        await self.queue.put(MediaChunk(client_id, segment_idx, "test", chunk=b"test"))
        await self.queue.put(MediaComplete(client_id, segment_idx, "test", 1))
        return {"e2e_latency_ms": 1.0}


class Pool:
    def __init__(self):
        self.slot = Slot()
        self.acquire_count = 0

    def get_status(self):
        return {"queue_size": 0, "available_gpus": 1, "total_gpus": 1}

    async def acquire(self, *args):
        self.acquire_count += 1
        return 0, self.slot

    async def release(self, *args):
        pass


def start_controller(module, socket, pool):
    enhancer = SimpleNamespace(
        resolve_rewrite_model=lambda value: "test-model",
        resolve_rewrite_system_prompt=lambda value: "test-system",
        resolve_rewrite_temperature=lambda value: 1.0,
    )
    controller = module.SessionController(socket, pool, enhancer, None, None)
    return asyncio.create_task(controller.run())


def test_invalid_initial_mode_does_not_acquire_gpu(controller_module):
    async def scenario():
        socket, pool = Socket(), Pool()
        await socket.incoming.put({"type": "session_init_v2", "generation_mode": "unknown"})
        await asyncio.wait_for(start_controller(controller_module, socket, pool), 3)
        error = next(message for message in socket.messages if message["type"] == "error")
        assert error["error_code"] == "invalid_generation_input"
        assert pool.acquire_count == 0
        assert socket.closed
        controller_module.pin_generation_inputs.assert_not_called()

    asyncio.run(scenario())


def test_new_project_replaces_conditioning_and_passes_it_to_gpu(controller_module, monkeypatch):
    first = GenerationInputs("t2va")
    second = GenerationInputs("fl2va", (GenerationAsset("first", "image", "/assets/first.png", "first_frame"),))
    monkeypatch.setattr(controller_module, "resolve_generation_inputs", Mock(side_effect=[first, second]))

    async def scenario():
        socket, pool = Socket(), Pool()
        await socket.incoming.put({
            "type": "session_init_v2", "generation_mode": "t2va", "curated_prompts": ["first prompt"],
            "enhancement_enabled": False,
        })
        task = start_controller(controller_module, socket, pool)
        try:
            await socket.wait_for("media_segment_complete")
            await socket.incoming.put({"type": "end_project_keep_session"})
            await socket.wait_for("project_idle")
            assert first in [call.args[0] for call in controller_module.release_generation_inputs.call_args_list]
            await socket.incoming.put({
                "type": "project_init_v1", "generation_mode": "fl2va", "curated_prompts": ["second prompt"],
                "enhancement_enabled": False,
            })
            await socket.wait_for("media_segment_complete")
            assert [call["generation_inputs"] for call in pool.slot.calls] == [first, second]
            assert pool.slot.calls[1]["segment_idx"] == 1
            assert pool.slot.calls[1]["reset_conditioning"]
            await socket.incoming.put({"type": "leave"})
            await asyncio.wait_for(task, 3)
        finally:
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
        assert [call.args[0] for call in controller_module.pin_generation_inputs.call_args_list] == [first, second]
        assert second in [call.args[0] for call in controller_module.release_generation_inputs.call_args_list]

    asyncio.run(scenario())


@pytest.mark.parametrize("injection", [
    {"initial_image": {"data_url": "not allowed"}},
    {"generation_mode": "ref2va"},
    {"conditioning_assets": []},
])
def test_simple_generate_cannot_replace_locked_inputs(controller_module, injection):
    async def scenario():
        socket, pool = Socket(), Pool()
        await socket.incoming.put({
            "type": "session_init_v2", "generation_mode": "t2va", "single_clip_mode": True,
            "enhancement_enabled": False,
        })
        task = start_controller(controller_module, socket, pool)
        try:
            await socket.wait_for("gpu_assigned")
            await socket.incoming.put({"type": "simple_generate", "prompt": "prompt", **injection})
            error = await socket.wait_for("error")
            assert "project" in error["message"]
            assert pool.slot.calls == []
            await socket.incoming.put({"type": "leave"})
            await asyncio.wait_for(task, 3)
        finally:
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)

    asyncio.run(scenario())


def test_disconnect_waits_for_worker_before_releasing_assets(controller_module, monkeypatch):
    async def scenario():
        socket, pool = Socket(), Pool()
        worker_started = asyncio.Event()
        worker_finished = asyncio.Event()
        proceed = asyncio.Event()

        async def slow_step(client_id, **kwargs):
            worker_started.set()
            await proceed.wait()
            worker_finished.set()
            return {"e2e_latency_ms": 1.0}

        pool.slot.user_step = slow_step
        await socket.incoming.put({
            "type": "session_init_v2", "generation_mode": "t2va", "curated_prompts": ["prompt"],
            "enhancement_enabled": False,
        })
        task = start_controller(controller_module, socket, pool)
        try:
            await asyncio.wait_for(worker_started.wait(), 3)
            await socket.incoming.put({"type": "leave"})
            await asyncio.sleep(0.07)
            assert not task.done()
            controller_module.release_generation_inputs.assert_not_called()
            proceed.set()
            await asyncio.wait_for(task, 3)
            assert worker_finished.is_set()
            controller_module.release_generation_inputs.assert_called_once()
        finally:
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)

    asyncio.run(scenario())


@pytest.fixture
def gpu_pool_module(monkeypatch):
    streaming = ModuleType("dreamverse.av_streaming")
    for name in ("StreamChunk", "StreamComplete", "StreamEvent", "StreamInit", "generate_stream_id", "stream_fmp4"):
        setattr(streaming, name, object)
    streaming.SHARED_STREAM_BUFFER_BYTES = 1024
    streaming.USE_SHARED_STREAM_BUFFER = False
    monkeypatch.setitem(sys.modules, "dreamverse.av_streaming", streaming)
    path = Path(__file__).resolve().parents[1] / "gpu_pool.py"
    spec = importlib.util.spec_from_file_location("dreamverse_test_gpu_pool", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "pin_generation_inputs", Mock())
    monkeypatch.setattr(module, "release_generation_inputs", Mock())
    return module


def test_gpu_step_timeout_keeps_assets_pinned_until_late_worker_completion(gpu_pool_module):
    from dreamverse.worker_ipc import StepComplete

    async def scenario():
        slot = gpu_pool_module.GPUSlot(0, "0")
        inputs = GenerationInputs("ref2va", (GenerationAsset("ref", "image", "/assets/ref.png", "reference"),))

        async def timeout(command, timeout):
            assert command.payload.generation_inputs == inputs
            raise asyncio.TimeoutError

        slot._send_command_tagged = timeout
        with pytest.raises(asyncio.TimeoutError):
            await slot.user_step("user", "prompt", generation_inputs=inputs)
        gpu_pool_module.pin_generation_inputs.assert_called_once_with(inputs)
        gpu_pool_module.release_generation_inputs.assert_not_called()

        def late_response(timeout):
            slot._active = False
            return StepComplete("user", 1, {})

        slot.response_queue = SimpleNamespace(get=late_response)
        slot._active = True
        await slot._response_reader()
        gpu_pool_module.release_generation_inputs.assert_called_once_with(inputs)
        assert slot._step_asset_inputs == {}

    asyncio.run(scenario())
