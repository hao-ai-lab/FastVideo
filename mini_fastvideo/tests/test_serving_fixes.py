"""Regression tests for the serving adversarial-review findings (previously-untested failure paths)."""
from __future__ import annotations

import asyncio

from mini_fastvideo.card import CostModel
from mini_fastvideo.deploy import build_deployment_card
from mini_fastvideo.models import build_default_engine, build_wan21_card, build_wan_t2v_program
from mini_fastvideo.request import Cancelled, DiffusionParams, OutputSpec, TaskType, make_request
from mini_fastvideo.runtime import AsyncEngine, PoolSet, wan_t2v_disaggregated
from mini_fastvideo.transport import InProcConnector


def _disagg_engine():
    ae = AsyncEngine(build_default_engine())
    card = build_wan21_card()
    pools = PoolSet(wan_t2v_disaggregated(), card)
    pools.warmup()
    ae.register_disaggregated("wd", pools, build_wan_t2v_program())
    return ae, pools


def test_pool_capacity_released_on_cancel_mid_disaggregation():
    """HIGH: cancelling mid-denoise must not brick the capacity-1 denoiser pool."""
    async def run():
        ae, pools = _disagg_engine()
        req = make_request(TaskType.T2V, "wd", "x",
                           diffusion=DiffusionParams(num_steps=80, seed=1),
                           outputs=OutputSpec(stream={"video": True}))   # emit per-step → observe occupancy
        cancelled = False
        async for ev in ae.submit(req):
            if not cancelled and pools.by_id["den"].in_flight >= 1:
                ae.cancel(req.request_id)
                cancelled = True
        assert cancelled and pools.by_id["den"].in_flight == 0           # capacity returned, not leaked
        # the pool is NOT bricked — a fresh request still runs
        out = await ae.generate(make_request(TaskType.T2V, "wd", "y", diffusion=DiffusionParams(num_steps=3, seed=2)))
        assert out.artifacts["video"].frames is not None
    asyncio.run(run())


def test_credit_and_capacity_released_on_transfer_failure():
    """A failing cross-pool transfer releases its credit AND the occupied pool (no leak)."""
    async def run():
        ae, pools = _disagg_engine()

        class Boom(InProcConnector):
            def put(self, key, value, manifest=None):
                raise RuntimeError("transfer boom")

        pools.by_id["enc"].connector = Boom("enc")
        out = await ae.generate(make_request(TaskType.T2V, "wd", "x", diffusion=DiffusionParams(num_steps=3, seed=1)))
        assert out.error is not None                                     # request-fatal isolation
        enc = pools.by_id["enc"].connector
        assert enc.credits.available == enc.credits.capacity            # credit refunded on the error path
        assert pools.by_id["den"].in_flight == 0                        # pool capacity refunded too
    asyncio.run(run())


def test_duplicate_request_id_rejected_not_deadlocked():
    async def run():
        ae = AsyncEngine(build_default_engine())
        r1 = make_request(TaskType.T2V, "wan2.1-1.3b", "x",
                          diffusion=DiffusionParams(num_steps=30, seed=1), request_id="dup")
        r2 = make_request(TaskType.T2V, "wan2.1-1.3b", "y",
                          diffusion=DiffusionParams(num_steps=30, seed=2), request_id="dup")
        g1 = ae.submit(r1)
        await g1.__anext__()                                            # r1 now in flight as "dup"
        raised = False
        try:
            await ae.submit(r2).__anext__()
        except ValueError:
            raised = True
        assert raised, "duplicate in-flight request_id must be rejected, not silently deadlock"
        async for _ in g1:
            pass
    asyncio.run(run())


def test_deployment_cards_do_not_alias_cost_model():
    card = build_wan21_card()
    a = build_deployment_card("A", [card])
    b = build_deployment_card("B", [card])
    assert a.cost_model is not b.cost_model                            # independent calibration state
    a.cost_model.per_unit_seconds = 999.0
    assert b.cost_model.per_unit_seconds != 999.0


def test_offline_runner_honors_cancellation():
    """Cancellation is common-path even on the offline (non-async) path."""
    eng = build_default_engine()
    runner = eng._make_runner(make_request(TaskType.T2V, "wan2.1-1.3b", "x",
                                           diffusion=DiffusionParams(num_steps=4, seed=1)))
    runner.cancel_scope.cancel()
    raised = False
    try:
        runner.run_to_completion()
    except Cancelled:
        raised = True
    assert raised, "offline runner must honor cancel, not run to completion"


def test_http_rejects_oversized_and_malformed_requests():
    from mini_fastvideo.runtime import AsyncEngine as AE
    from mini_fastvideo.serving import OmniOpenAIServer

    async def status_of(host, port, raw: bytes) -> int:
        r, w = await asyncio.open_connection(host, port)
        w.write(raw)
        await w.drain()
        data = await r.read()
        w.close()
        return int(data.split(b" ")[1])

    async def run():
        s = OmniOpenAIServer(AE(build_default_engine()))
        host, port = await s.serve(port=0)
        # 413: Content-Length far beyond the body cap (rejected before reading the body)
        st = await status_of(host, port,
                              b"POST /v1/images/generations HTTP/1.1\r\nHost: x\r\nContent-Length: 999999999\r\n\r\n")
        assert st == 413
        # 400: non-numeric Content-Length
        st = await status_of(host, port,
                             b"POST /v1/images/generations HTTP/1.1\r\nHost: x\r\nContent-Length: abc\r\n\r\n")
        assert st == 400
        await s.close()
    asyncio.run(run())
