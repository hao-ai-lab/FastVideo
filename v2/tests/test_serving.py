"""Serving stack: transport, disaggregation, AsyncEngine, our fleet, Dynamo adapter, OpenAI server.

All async tests are sync ``test_*`` fns that call ``asyncio.run`` (so both pytest and the zero-dep
runner work, no pytest-asyncio). The HTTP/SSE tests hit a real socket via stdlib
``asyncio.open_connection`` — no httpx/framework dependency.
"""
from __future__ import annotations

import asyncio

import numpy as np

from v2._enums import WorkUnitKind
from v2.card import CostModel
from v2.deploy import (
    DynamoWorkerAdapter,
    FakeDynamoRuntime,
    LocalFleet,
    NoWorkerAvailable,
    build_deployment_card,
)
from v2.extend import InterceptorChain, ObserverBus
from v2.models import build_default_engine, build_omni_engine, build_wan21_card, build_wan_t2v_program
from v2.request import DiffusionParams, OutputSpec, SamplingParams, TaskType, make_request
from v2.runtime import AsyncEngine, DisaggregatedRunner, Engine, PoolSet, wan_t2v_disaggregated
from v2.serving import OmniOpenAIServer
from v2.transport import (
    InProcConnector,
    InProcKVConnector,
    ShmFakeConnector,
    TransferManifest,
)


# --------------------------------------------------------------------------- #
# transport
# --------------------------------------------------------------------------- #
def test_connector_readiness_and_credits():
    c = InProcConnector("enc", credit_capacity=2)
    assert c.get("k") is None and c.parked == 1          # readiness: not ready yet
    c.put("k", np.ones(4, dtype="float32"))
    assert c.chunk_ready("k") and c.get("k") is not None
    assert c.acquire_credit() and c.acquire_credit() and not c.acquire_credit()   # window exhausted
    c.release_credit()
    assert c.acquire_credit()


def test_shm_connector_copies_payload():
    a = np.ones(4, dtype="float32")
    c = ShmFakeConnector("shm")
    c.put("k", a)
    got = c.take("k")
    assert np.array_equal(got, a) and got is not a       # a real transfer (copy), not a reference


def test_kv_connector_alloc_finish():
    kv = InProcKVConnector(capacity_bytes=100)
    assert kv.alloc("a", 60) and not kv.alloc("b", 60)   # jointly over capacity
    kv.save("a", 7); assert kv.query("a") and kv.load("a") == 7
    kv.finish("a"); assert kv.alloc("b", 60)             # freed


# --------------------------------------------------------------------------- #
# disaggregation
# --------------------------------------------------------------------------- #
def test_disaggregated_output_is_bit_identical_to_inline():
    card = build_wan21_card()
    program = build_wan_t2v_program()
    req = make_request(TaskType.T2V, "wan2.1-1.3b", "a cat", diffusion=DiffusionParams(num_steps=4, seed=1))

    inline = build_default_engine().run(req)

    pools = PoolSet(wan_t2v_disaggregated(), card)
    pools.warmup()
    runner = DisaggregatedRunner(pools, program, req, observers=ObserverBus(), interceptors=InterceptorChain())
    runner.run_to_completion()
    disagg = runner.output()

    assert np.array_equal(inline.artifacts["video"].frames, disagg.artifacts["video"].frames)
    assert disagg.metrics["transfers"] == 3.0            # text_embeds, neg_text_embeds (enc→den), denoise_out (den→dec)
    assert pools.by_id["den"].capacity == 1              # jumbo denoise batch-of-1


# --------------------------------------------------------------------------- #
# AsyncEngine
# --------------------------------------------------------------------------- #
def test_async_engine_streams_lifecycle_events():
    async def run():
        ae = AsyncEngine(build_default_engine())
        req = make_request(TaskType.T2V, "wan2.1-1.3b", "x",
                           diffusion=DiffusionParams(num_steps=4, seed=1),
                           outputs=OutputSpec(stream={"video": True}))
        types = [e.type async for e in ae.submit(req)]
        assert types[0] == "request.start" and types[-1] == "request.complete"
        assert types.count("media.chunk") == 4
        assert ae.state(req.request_id) == "completed"
        assert ae.result(req.request_id).artifacts["video"].frames is not None
    asyncio.run(run())


def test_async_engine_concurrent_requests_complete():
    async def run():
        ae = AsyncEngine(build_default_engine())
        reqs = [make_request(TaskType.T2V, "wan2.1-1.3b", f"p{i}",
                             diffusion=DiffusionParams(num_steps=3, seed=i)) for i in range(4)]
        res = await ae.generate_many(reqs)
        assert all(res[r.request_id].artifacts["video"].frames is not None for r in reqs)
    asyncio.run(run())


def test_async_engine_cancellation_is_common_path():
    async def run():
        ae = AsyncEngine(build_default_engine())
        req = make_request(TaskType.T2V, "wan2.1-1.3b", "x", diffusion=DiffusionParams(num_steps=50, seed=1))
        gen = ae.submit(req)
        first = await gen.__anext__()
        ae.cancel(req.request_id)                         # cancel right after start
        types = [first.type] + [e.type async for e in gen]
        assert "request.cancelled" in types
        assert ae.state(req.request_id) == "cancelled"
    asyncio.run(run())


# --------------------------------------------------------------------------- #
# our own fleet
# --------------------------------------------------------------------------- #
def _two_workers():
    cards = [build_wan21_card()]
    A, B = AsyncEngine(build_default_engine()), AsyncEngine(build_default_engine())
    cardA = build_deployment_card("A", cards, max_concurrent=8)
    cardB = build_deployment_card("B", cards, max_concurrent=8)
    return A, B, cardA, cardB


def _req(seed=1):
    return make_request(TaskType.T2V, "wan2.1-1.3b", "x", diffusion=DiffusionParams(num_steps=3, seed=seed))


def test_fleet_least_loaded_routes_around_busy_worker():
    from v2.runtime import RequestState
    A, B, cardA, cardB = _two_workers()
    fleet = LocalFleet("least_loaded")
    fleet.register("A", A, cardA); fleet.register("B", B, cardB)
    A._states["x"] = RequestState.RUNNING; A._states["y"] = RequestState.RUNNING
    assert fleet.route(_req()).worker_id == "B"


def test_fleet_drain_reroutes_and_affinity_sticks():
    A, B, cardA, cardB = _two_workers()
    fleet = LocalFleet("least_loaded")
    fleet.register("A", A, cardA); fleet.register("B", B, cardB)
    fleet.drain("A")
    assert fleet.route(_req()).worker_id == "B"
    fleet.drain("B")
    try:
        fleet.route(_req()); assert False
    except NoWorkerAvailable:
        pass
    af = LocalFleet("affinity")
    af.register("A", A, cardA); af.register("B", B, cardB)
    w1 = af.route(_req(), affinity_key="sess-1")
    w2 = af.route(_req(2), affinity_key="sess-1")
    assert w1.worker_id == w2.worker_id                   # sticky by key


def test_fleet_cost_routing_picks_cheaper_worker():
    A, B, cardA, cardB = _two_workers()
    cardB.cost_model = CostModel(kind=WorkUnitKind.DIFFUSION_STEP, base_seconds=1e-9, per_unit_seconds=1e-12)
    fleet = LocalFleet("cost")
    fleet.register("A", A, cardA); fleet.register("B", B, cardB)
    assert fleet.route(_req()).worker_id == "B"


def test_fleet_generate_delegates_to_worker():
    async def run():
        A, B, cardA, cardB = _two_workers()
        fleet = LocalFleet("least_loaded")
        fleet.register("A", A, cardA); fleet.register("B", B, cardB)
        out = await fleet.generate(_req())
        assert out.artifacts["video"].frames is not None
    asyncio.run(run())


# --------------------------------------------------------------------------- #
# Dynamo adapter (frontable, not relied upon)
# --------------------------------------------------------------------------- #
def test_dynamo_adapter_contract_and_routing():
    async def run():
        A, B, cardA, cardB = _two_workers()
        adA, adB = DynamoWorkerAdapter(A, cardA), DynamoWorkerAdapter(B, cardB)
        reg = adA.registration()
        assert reg["engine_id"] == "A" and "Videos" in reg["model_type"] and reg["worker_type"] == "Aggregated"
        assert adA.health()["status"] == "healthy"
        dyn = FakeDynamoRuntime()
        dyn.register_worker(adA); dyn.register_worker(adB)
        out = await dyn.generate(_req())
        assert out.artifacts["video"].frames is not None
        adA.drain(); adB.drain()
        try:
            await dyn.generate(_req()); assert False
        except RuntimeError:
            pass                                          # both draining → no worker
    asyncio.run(run())


# --------------------------------------------------------------------------- #
# OpenAI server (real socket, stdlib client)
# --------------------------------------------------------------------------- #
async def _http(host, port, method, path, body: bytes = b"", *, read_all=True):
    reader, writer = await asyncio.open_connection(host, port)
    req = f"{method} {path} HTTP/1.1\r\nHost: x\r\nContent-Length: {len(body)}\r\n\r\n"
    writer.write(req.encode() + body)
    await writer.drain()
    data = await reader.read()                            # server sends Connection: close
    writer.close()
    head, _, payload = data.partition(b"\r\n\r\n")
    status = int(head.split(b"\r\n", 1)[0].split(b" ")[1])
    return status, head.decode("latin1"), payload.decode("utf-8", "replace")


def _server():
    eng = build_default_engine(); build_omni_engine(eng)
    return OmniOpenAIServer(AsyncEngine(eng), engine_id="worker-test")


def test_server_health_and_models_over_real_socket():
    import json

    async def run():
        s = _server()
        host, port = await s.serve(port=0)
        st, _, body = await _http(host, port, "GET", "/health")
        assert st == 200 and json.loads(body)["status"] == "healthy"
        st, _, body = await _http(host, port, "GET", "/v1/models")
        ids = [m["id"] for m in json.loads(body)["data"]]
        assert "wan2.1-1.3b" in ids and "cosmos3-vfm" in ids
        await s.close()
    asyncio.run(run())


def test_server_images_video_job_and_sync():
    import json

    async def run():
        s = _server()
        host, port = await s.serve(port=0)
        # images
        b = b'{"model":"bagel-mot","prompt":"a teapot","num_inference_steps":4}'
        st, _, body = await _http(host, port, "POST", "/v1/images/generations", b)
        assert st == 200 and json.loads(body)["data"][0]["type"] == "tensor"
        # async video job + poll
        b = b'{"model":"wan2.1-1.3b","prompt":"a wave","num_inference_steps":4}'
        st, _, body = await _http(host, port, "POST", "/v1/videos", b)
        assert st == 202
        jid = json.loads(body)["id"]
        status = "queued"
        for _ in range(200):
            st, _, body = await _http(host, port, "GET", f"/v1/videos/{jid}")
            status = json.loads(body)["status"]
            if status in ("completed", "failed"):
                break
            await asyncio.sleep(0.01)
        assert status == "completed" and "video" in json.loads(body)["result"]
        # sync video
        b = b'{"model":"ltx2.3-distilled","prompt":"a sunset","num_inference_steps":4}'
        st, _, body = await _http(host, port, "POST", "/v1/videos/sync", b)
        assert st == 200 and json.loads(body)["data"]["video"]["type"] == "tensor"
        await s.close()
    asyncio.run(run())


def test_server_chat_json_and_sse_stream():
    import json

    async def run():
        s = _server()
        host, port = await s.serve(port=0)
        # non-streaming chat (omni: text + video)
        b = b'{"model":"cosmos3-vfm","messages":[{"role":"user","content":"a phoenix"}]}'
        st, _, body = await _http(host, port, "POST", "/v1/chat/completions", b)
        d = json.loads(body)
        assert st == 200 and d["choices"][0]["message"]["content"].startswith("tok:")
        assert "video" in d["omni"]
        # SSE stream
        b = b'{"model":"cosmos3-vfm","messages":[{"role":"user","content":"a comet"}],"stream":true}'
        st, head, body = await _http(host, port, "POST", "/v1/chat/completions", b)
        assert "text/event-stream" in head
        data_lines = [ln[6:] for ln in body.split("\n") if ln.startswith("data: ")]
        assert data_lines[-1] == "[DONE]" and len(data_lines) >= 12
        types = [json.loads(x).get("event") for x in data_lines[:-2]]
        assert "request.start" in types and "request.complete" in types
        await s.close()
    asyncio.run(run())
