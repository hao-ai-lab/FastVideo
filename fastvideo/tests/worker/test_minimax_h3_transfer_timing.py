# SPDX-License-Identifier: Apache-2.0
"""CPU regressions for H3 Ray timing; no Ray cluster or CUDA device required."""

from types import SimpleNamespace

import pytest
import torch

from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
import fastvideo.worker.minimax_h3_disaggregated as runtime_module


def _state():
    return SimpleNamespace(
        request_id="request-7",
        prompt_embeds=torch.zeros(1, 3, 4),
        video_latents=torch.zeros(4, 5),
        audio_latents=torch.zeros(2, 6),
        layout=SimpleNamespace(
            position_ids=torch.zeros(9, 3, dtype=torch.float64),
            token_tags=torch.zeros(9, dtype=torch.int64),
            text_indices=torch.zeros(3, dtype=torch.int64),
            video_indices=torch.zeros(4, dtype=torch.int64),
            audio_indices=torch.zeros(2, dtype=torch.int64),
        ),
    )


@pytest.fixture
def timing(monkeypatch):
    clock = SimpleNamespace(now=0.0)
    events = []
    messages = []
    result = _state()

    def wait(refs, *, num_returns, fetch_local):
        assert num_returns == 1
        events.append(("wait", fetch_local))
        clock.now += 2.0 if fetch_local else 10.0
        return refs, []

    def get(ref):
        events.append(("get", ref))
        clock.now += 3.0
        return result

    def synchronize():
        events.append(("sync",))
        clock.now += 5.0

    monkeypatch.setattr(runtime_module.time, "perf_counter", lambda: clock.now)
    monkeypatch.setattr(runtime_module, "ray", SimpleNamespace(wait=wait, get=get))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "synchronize", synchronize)
    monkeypatch.setattr(runtime_module.envs, "FASTVIDEO_NVTX_PROFILE", False)
    monkeypatch.setattr(runtime_module.logger, "info", lambda fmt, *args: messages.append(fmt % args))
    return SimpleNamespace(clock=clock, events=events, messages=messages, result=result)


def test_payload_total_includes_every_layout_tensor(timing):
    assert runtime_module._log_h3_state("encoded", timing.result) == 536
    assert any("layout.position_ids:" in line for line in timing.messages)
    assert any("layout.token_tags:" in line for line in timing.messages)
    assert all("request_id=request-7" in line for line in timing.messages)


@pytest.mark.parametrize("direction", ["A_TO_B", "B_TO_A"])
def test_transfer_separates_production_fetch_and_completed_device_copy(timing, direction):
    ref = object()
    result = runtime_module._receive_h3_state(runtime_module._H3TransferRef(ref, "request-7"), direction)
    assert result is timing.result
    assert timing.events == [("sync",), ("wait", False), ("wait", True), ("get", ref), ("sync",)]
    line, = timing.messages
    assert f"direction={direction}" in line
    assert "request_id=request-7" in line
    assert "tensor_bytes=536" in line
    assert "source_wait_s=10.000000" in line
    assert "object_fetch_s=2.000000" in line
    assert "materialize_s=8.000000" in line
    assert "receive_s=10.000000" in line


def test_disabled_profiling_does_not_wait_synchronize_or_read_clock(timing, monkeypatch):
    def unexpected_clock():
        pytest.fail("profiling disabled must not read the clock")

    monkeypatch.setattr(runtime_module.time, "perf_counter", unexpected_clock)
    assert runtime_module._receive_h3_state(timing.result, "A_TO_B") is timing.result
    with runtime_module._h3_stage_timer("encode", "request-7", False):
        pass
    assert timing.events == []
    assert timing.messages == []


def test_stage_time_includes_gpu_completion_and_excludes_previous_work(timing):
    with runtime_module._h3_stage_timer("denoise", "request-7", True):
        timing.clock.now += 7.0
    assert timing.events == [("sync",), ("sync",)]
    assert timing.messages == ["[RAY_STAGE] request_id=request-7 stage=denoise elapsed_s=12.000000"]


def test_failed_transfer_propagates_without_logging_success(timing, monkeypatch):
    def fail(ref):
        raise RuntimeError("producer failed")

    monkeypatch.setattr(runtime_module.ray, "get", fail)
    with pytest.raises(RuntimeError, match="producer failed"):
        runtime_module._receive_h3_state(runtime_module._H3TransferRef(object(), "request-7"), "A_TO_B")
    assert timing.messages == []


def test_zero_fetch_duration_does_not_divide_by_zero(timing, monkeypatch):
    monkeypatch.setattr(runtime_module.time, "perf_counter", lambda: 0.0)
    runtime_module._receive_h3_state(runtime_module._H3TransferRef(object(), "request-7"), "A_TO_B")
    assert "object_fetch_mb_s=n/a" in timing.messages[0]


@pytest.mark.parametrize("stage,direction", [("denoise", "A_TO_B"), ("decode", "B_TO_A")])
def test_actor_receives_before_model_stage(timing, stage, direction):
    actor_class = (runtime_module._MiniMaxH3DiTActor if stage == "denoise"
                   else runtime_module._MiniMaxH3EncoderDecoderActor)
    actor = actor_class.__new__(actor_class)
    actor._profile_transfers = True

    def forward(state):
        assert state is timing.result
        assert any(f"direction={direction}" in line for line in timing.messages)
        timing.clock.now += 7.0
        return state

    actor.pipeline = SimpleNamespace(**{stage: forward})
    assert getattr(actor, stage)(runtime_module._H3TransferRef(object(), "request-7")) is timing.result
    assert any(f"stage={stage} elapsed_s=12.000000" in line for line in timing.messages)


def _fake_runtime(monkeypatch, enabled):
    events = []

    def method(name):
        def remote(*args):
            ref = SimpleNamespace(kind=name, args=args)
            events.append((name, ref))
            return ref

        return SimpleNamespace(remote=remote)

    def get(ref):
        # The driver must never materialize encode/denoise intermediates.
        assert ref.kind == "decode"
        events.append(("get", ref))
        return ref

    def wait(refs, *, num_returns, fetch_local):
        assert refs[0].kind == "denoise"
        assert num_returns == 1 and fetch_local is False
        events.append(("wait", refs[0]))
        return refs, []

    runtime = runtime_module.RayMiniMaxH3DisaggregatedRuntime.__new__(
        runtime_module.RayMiniMaxH3DisaggregatedRuntime)
    runtime._closed = False
    runtime._profile_transfers = enabled
    runtime.encoder_decoder = SimpleNamespace(encode=method("encode"), decode=method("decode"))
    runtime.dit = SimpleNamespace(denoise=method("denoise"))
    monkeypatch.setattr(runtime_module, "ray", SimpleNamespace(get=get, wait=wait))
    return runtime, events


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("stream", [False, True])
def test_runtime_preserves_request_ids_and_bounded_driver_dag(monkeypatch, enabled, stream):
    runtime, events = _fake_runtime(monkeypatch, enabled)
    batches = [ForwardBatch(data_type="video", extra={"request_id": str(i)}) for i in range(3)]
    if stream:
        assert len(list(runtime.iter_forward(batches))) == 3
        assert [name for name, _ in events] == [
            "encode", "denoise", "encode", "wait", "decode", "denoise", "get",
            "encode", "wait", "decode", "denoise", "get", "decode", "get",
        ]
    else:
        runtime.execute_forward(batches[0], request_id="explicit")
        assert [name for name, _ in events] == ["encode", "denoise", "decode", "get"]

    for name, result_ref in events:
        if name not in {"denoise", "decode"}:
            continue
        arg, = result_ref.args
        assert isinstance(arg, runtime_module._H3TransferRef) is enabled
        source_ref = arg.ref if enabled else arg
        assert source_ref.kind == ("encode" if name == "denoise" else "denoise")
        if enabled:
            expected_id = source_ref.args[1] if name == "denoise" else source_ref.args[0].request_id
            assert arg.request_id == expected_id
            if not stream:
                assert arg.request_id == "explicit"
