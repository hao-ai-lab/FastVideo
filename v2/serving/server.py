"""OmniOpenAIServer — OpenAI-compatible worker server (vllm-omni pattern).

Fronts ONE AsyncEngine (the per-pool worker surface). Routes OpenAI-shaped endpoints to typed
OmniRequests, streams chat via SSE, and serves video as an async job (POST returns a job id;
GET polls) plus a /sync variant — mirroring vllm-omni's `/v1/videos` + `/v1/videos/sync`. The fleet
(deploy/fleet.py) or Dynamo routes across multiple such servers/engines above this layer.

Unlike vllm-omni's request-scheduled opaque DIFFUSION stage, the engine underneath is step-scheduled;
the endpoint is a thin shim over the loop scheduler.
"""
from __future__ import annotations

import asyncio
import base64
import contextlib
import itertools
import time
from typing import Any

import numpy as np

from v2.core.enums import Capability
from v2.core.request.artifacts import Output
from v2.core.request.streams import StreamChunk
from v2.core.request.tasks import TaskType
from v2.serving.http import HttpServer, Request, Response
from v2.serving.protocol import ChatCompletionRequest, ImageGenerationRequest, VideoGenerationRequest


def _json_default(o: Any) -> Any:
    """Make SSE event payloads JSON-safe (media.chunk carries a StreamChunk with an array)."""
    if isinstance(o, StreamChunk):
        shape = list(np.asarray(o.data).shape) if o.data is not None else None
        return {
            "stream_id": o.stream_id,
            "modality": o.modality,
            "seq": o.seq,
            "preview": o.preview,
            "final": o.final,
            "shape": shape
        }
    if isinstance(o, np.ndarray):
        return {"shape": list(o.shape), "dtype": str(o.dtype)}
    return str(o)


_job_ctr = itertools.count(1)
_chat_ctr = itertools.count(1)


def _payload(art: Any) -> Any:
    for attr in ("text", "frames", "samples", "tensor", "latent", "token_ids"):
        if hasattr(art, attr) and getattr(art, attr) is not None:
            return getattr(art, attr)
    return None


def _ser_artifacts(out: Output) -> dict:
    res: dict = {}
    for name, art in out.artifacts.items():
        p = _payload(art)
        if isinstance(p, str):
            res[name] = {"type": "text", "text": p}
        elif p is None:
            res[name] = {"type": "empty"}
        else:
            a = np.asarray(p)
            res[name] = {
                "type": "tensor",
                "shape": list(a.shape),
                "dtype": str(a.dtype),
                "b64_json": base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()
            }
    return res


class OmniOpenAIServer:

    def __init__(self, engine: Any, *, engine_id: str = "v2", max_jobs: int = 512):
        self.engine = engine  # an AsyncEngine
        self.engine_id = engine_id
        self.jobs: dict[str, dict] = {}  # video job store
        self.http = HttpServer(self.dispatch)
        self.served = 0
        self.max_jobs = max_jobs
        self._tasks: set = set()  # tracked video-job tasks (no fire-and-forget leak)

    def _evict_jobs(self) -> None:
        terminal = ("completed", "failed")
        for jid in list(self.jobs):
            if len(self.jobs) <= self.max_jobs:
                break
            if self.jobs[jid].get("status") in terminal:
                self.jobs.pop(jid, None)

    # --- routing ------------------------------------------------------------ #
    async def dispatch(self, req: Request) -> Response:
        m, p = req.method, req.path
        if m == "GET" and p == "/health":
            return Response.json(self._health())
        if m == "GET" and p == "/metrics":
            return Response.json(self._metrics())
        if m == "GET" and p == "/v1/models":
            return Response.json(self._models())
        if m == "POST" and p == "/v1/images/generations":
            return await self._images(req)
        if m == "POST" and p == "/v1/videos":
            return await self._video_create(req)
        if m == "POST" and p == "/v1/videos/sync":
            return await self._video_sync(req)
        if m == "GET" and p.startswith("/v1/videos/"):
            return self._video_get(p.rsplit("/", 1)[-1])
        if m == "POST" and p == "/v1/chat/completions":
            return await self._chat(req)
        return Response.json({"error": f"no route for {m} {p}"}, status=404)

    # --- metadata endpoints ------------------------------------------------- #
    def _health(self) -> dict:
        return {
            "status": "healthy",
            "engine_id": self.engine_id,
            "in_flight": self.engine.in_flight,
            "queue_depth": self.engine.queue_depth
        }

    def _metrics(self) -> dict:
        sm = self.engine.engine.metrics
        adm = self.engine.engine.admission.metrics
        return {
            "served": self.served,
            "in_flight": self.engine.in_flight,
            "stepped_units": sm.stepped_units,
            "admitted": adm.admitted,
            "deferred": adm.deferred,
            "by_kind": dict(adm.by_kind)
        }

    def _models(self) -> dict:
        ids = (list(self.engine.engine._registry) + list(self.engine._disagg) + list(self.engine.engine._workflows)
               )  # workflows are servables too
        return {"object": "list", "data": [{"id": m, "object": "model"} for m in ids]}

    def _card_for_model(self, model_id: str) -> Any | None:
        reg = self.engine.engine._registry
        if model_id in reg:
            return reg[model_id][0].card
        if model_id in self.engine._disagg:
            pools, _program = self.engine._disagg[model_id]
            return getattr(pools, "card", None)
        return None

    def _task_for_endpoint(self, model_id: str, endpoint: str, *, has_image: bool = False) -> TaskType:
        """Resolve endpoint intent against the registered card's capabilities.

        The serving boundary may choose a default task; the protocol parser itself
        must not infer tasks from model-name substrings.
        """
        card = self._card_for_model(model_id)
        caps = getattr(card, "capabilities", None)

        def has(cap: Capability) -> bool:
            return bool(caps is not None and caps.has(cap))

        if endpoint == "image":
            if has(Capability.TEXT_TO_IMAGE):
                return TaskType.T2I
            if has(Capability.TEXT_TO_VIDEO):
                return TaskType.T2V
            if has(Capability.TEXT_TO_VIDEO_SOUND):
                return TaskType.T2VS
            return TaskType.T2I

        if endpoint == "video":
            if has(Capability.TEXT_TO_VIDEO):
                return TaskType.T2V
            if has(Capability.TEXT_TO_VIDEO_SOUND):
                return TaskType.T2VS
            if has(Capability.TEXT_TO_IMAGE):
                return TaskType.T2I
            return TaskType.T2V

        if endpoint == "chat":
            if has_image and has(Capability.IMAGE_TO_VIDEO):
                return TaskType.I2V
            if has(Capability.TEXT_TO_VIDEO):
                return TaskType.T2V
            if has(Capability.TEXT_TO_IMAGE):
                return TaskType.T2I
            if has(Capability.TEXT_TO_SPEECH):
                return TaskType.T2A
            if has(Capability.REASONING_TEXT):
                return TaskType.REASON
            return TaskType.T2V

        return TaskType.T2V

    # --- images ------------------------------------------------------------- #
    async def _images(self, req: Request) -> Response:
        ig = ImageGenerationRequest.from_json(req.json())
        out = await self.engine.generate(ig.to_omni(self._task_for_endpoint(ig.model, "image")))
        self.served += 1
        arts = _ser_artifacts(out)
        primary = arts.get("image") or arts.get("video") or arts.get("latents")
        return Response.json({
            "created": int(time.time()),
            "model": ig.model,
            "data": [primary] * max(1, ig.n),
            "omni": arts,
            "metrics": {
                k: round(v, 6)
                for k, v in out.metrics.items()
            }
        })

    # --- video: async job + poll + sync ------------------------------------- #
    async def _video_create(self, req: Request) -> Response:
        vg = VideoGenerationRequest.from_json(req.json())
        job_id = f"video-{next(_job_ctr)}"
        self.jobs[job_id] = {"id": job_id, "status": "queued", "model": vg.model}
        t = asyncio.create_task(self._run_video_job(job_id, vg))
        self._tasks.add(t)
        t.add_done_callback(self._tasks.discard)  # tracked, not fire-and-forget
        self._evict_jobs()
        return Response.json({"id": job_id, "status": "queued", "model": vg.model}, status=202)

    async def _run_video_job(self, job_id: str, vg: VideoGenerationRequest) -> None:
        self.jobs[job_id]["status"] = "running"
        try:
            out = await self.engine.generate(vg.to_omni(self._task_for_endpoint(vg.model, "video")))
            self.served += 1
            self.jobs[job_id].update(status="completed",
                                     result=_ser_artifacts(out),
                                     metrics={
                                         k: round(v, 6)
                                         for k, v in out.metrics.items()
                                     })
        except Exception as e:  # noqa: BLE001
            self.jobs[job_id].update(status="failed", error=str(e))

    def _video_get(self, job_id: str) -> Response:
        job = self.jobs.get(job_id)
        if job is None:
            return Response.json({"error": f"no such job {job_id}"}, status=404)
        return Response.json(job)

    async def _video_sync(self, req: Request) -> Response:
        vg = VideoGenerationRequest.from_json(req.json())
        out = await self.engine.generate(vg.to_omni(self._task_for_endpoint(vg.model, "video")))
        self.served += 1
        return Response.json({
            "model": vg.model,
            "data": _ser_artifacts(out),
            "metrics": {
                k: round(v, 6)
                for k, v in out.metrics.items()
            }
        })

    # --- chat: SSE stream or JSON ------------------------------------------- #
    async def _chat(self, req: Request) -> Response:
        cc = ChatCompletionRequest.from_json(req.json())
        has_image = cc.has_image()
        task = self._task_for_endpoint(cc.model, "chat", has_image=has_image)
        if has_image and task not in (TaskType.I2V, TaskType.TI2V):
            return Response.json({"error": f"model {cc.model!r} does not declare image-to-video chat support"},
                                 status=400)
        omni = cc.to_omni(task)
        if cc.stream:
            return Response.sse(self._chat_sse(omni, cc.model))
        out = await self.engine.generate(omni)
        self.served += 1
        arts = _ser_artifacts(out)
        content = arts.get("text", {}).get("text", "") if "text" in arts else ""
        return Response.json({
            "id":
            f"chatcmpl-{next(_chat_ctr)}",
            "object":
            "chat.completion",
            "created":
            int(time.time()),
            "model":
            cc.model,
            "choices": [{
                "index": 0,
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": content
                }
            }],
            "omni":
            arts,
            "metrics": {
                k: round(v, 6)
                for k, v in out.metrics.items()
            }
        })

    async def _chat_sse(self, omni: Any, model: str):
        import json
        cid = f"chatcmpl-{next(_chat_ctr)}"
        async for ev in self.engine.submit(omni):  # live OmniEvents → SSE chunks
            chunk = {"id": cid, "object": "chat.completion.chunk", "model": model, "event": ev.type, "data": ev.payload}
            yield f"data: {json.dumps(chunk, default=_json_default)}\n\n"
        self.served += 1
        out = self.engine.result(omni.request_id)
        arts = _ser_artifacts(out) if out is not None else {}
        content = arts.get("text", {}).get("text", "") if "text" in arts else ""
        final = {
            "id": cid,
            "object": "chat.completion.chunk",
            "model": model,
            "choices": [{
                "index": 0,
                "delta": {
                    "role": "assistant",
                    "content": content
                },
                "finish_reason": "stop"
            }],
            "omni": arts
        }
        yield f"data: {json.dumps(final, default=_json_default)}\n\n"
        yield "data: [DONE]\n\n"

    # --- lifecycle ---------------------------------------------------------- #
    async def serve(self, host: str = "127.0.0.1", port: int = 0) -> tuple[str, int]:
        return await self.http.start(host, port)

    async def close(self) -> None:
        for t in list(self._tasks):  # cancel + drain in-flight video jobs on shutdown
            t.cancel()
        for t in list(self._tasks):
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await t
        await self.http.close()
