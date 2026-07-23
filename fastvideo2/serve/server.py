"""Online serving for fastvideo2 — one resident model behind an async-job
REST API plus a WebSocket with TRUE per-step progress.

Scoped from fastvideo-main's serving surfaces (entrypoints/openai +
entrypoints/streaming): the load-bearing core is a typed request schema, an
async job store with poll/download, and never running the blocking generator
on the event loop. Two deliberate differences from main:

  * per-step progress is REAL here — the driven loop's ``observe`` hook
    emits every denoise step (main's servers ship per-segment progress only,
    with per-step listed as a TODO);
  * every job records the LATENTS sha — served-vs-offline identity is gated
    bitwise on latents, independent of container/transport encoding.

Run:  python -m fastvideo2.serve --model fastwan-qad-fp8-1.3b --port 8010
Deps: fastapi + uvicorn (``pip install fastvideo2[serve]``).
"""
from __future__ import annotations

import asyncio
import hashlib
import os
import tempfile
import threading
import time
import uuid
from typing import Any


def build_app(model: Any) -> Any:
    """FastAPI app over a loaded :class:`fastvideo2.sdk.Model`. The model is
    shared; generations serialize on a lock in worker threads."""
    from fastapi import Body, FastAPI, HTTPException
    from starlette.websockets import WebSocketDisconnect

    from fastvideo2.engine import Request

    # the engine Request IS the schema; a plain whitelist mapping avoids any
    # pydantic v1/v2 model-detection hazards across environments
    _FIELDS = ("seed", "num_steps", "guidance_scale", "height", "width",
               "num_frames", "shift", "negative_prompt")

    def _to_request(payload: dict, request_id: str) -> Request:
        if not isinstance(payload.get("prompt"), str) or not payload["prompt"]:
            raise HTTPException(422, "prompt (non-empty string) is required")
        kwargs = {k: payload[k] for k in _FIELDS if payload.get(k) is not None}
        try:
            return Request(prompt=payload["prompt"], request_id=request_id, **kwargs)
        except TypeError as e:
            raise HTTPException(422, str(e)) from e

    app = FastAPI(title="fastvideo2", version="0.1")
    jobs: dict[str, dict] = {}
    gen_lock = threading.Lock()
    out_dir = tempfile.mkdtemp(prefix="fv2serve_")

    def _latents_sha(result: Any) -> str:
        import torch
        den = result.outputs.get("latents")
        lat = den["latents"] if isinstance(den, dict) else den
        return hashlib.sha256(
            lat.detach().to(torch.float32).cpu().numpy().tobytes()).hexdigest()[:16]

    def _run_job(job_id: str, req: Request) -> None:
        job = jobs[job_id]
        try:
            with gen_lock:
                job["status"] = "running"
                t0 = time.perf_counter()
                result = model.generate(request=req)
                seconds = time.perf_counter() - t0
            path = os.path.join(out_dir, f"{job_id}.mp4")
            result.save(path)
            job.update(status="completed", seconds=round(seconds, 3),
                       video_path=path, latents_sha=_latents_sha(result),
                       steps=len(result.trace))
        except Exception as e:  # surface, don't hide
            job.update(status="failed", error=f"{type(e).__name__}: {e}")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "model": model.model_id,
                "card_digest": model.card.digest()}

    @app.get("/v1/models")
    def models() -> dict:
        return {"data": [model.describe()]}

    @app.post("/v1/videos")
    async def create_video(payload: dict = Body(...)) -> dict:
        job_id = uuid.uuid4().hex[:12]
        req = _to_request(payload, job_id)
        jobs[job_id] = {"id": job_id, "status": "queued", "request": payload}
        threading.Thread(target=_run_job, args=(job_id, req),
                         daemon=True).start()
        return {"id": job_id, "status": "queued"}

    @app.get("/v1/videos/{job_id}")
    def get_video(job_id: str) -> dict:
        if job_id not in jobs:
            raise HTTPException(404, "unknown job")
        return {k: v for k, v in jobs[job_id].items() if k != "video_path"}

    @app.get("/v1/videos/{job_id}/content")
    def get_content(job_id: str) -> Any:
        from fastapi.responses import FileResponse
        job = jobs.get(job_id)
        if job is None or job.get("status") != "completed":
            raise HTTPException(404, "not ready")
        return FileResponse(job["video_path"], media_type="video/mp4")

    async def stream(ws):
        """One request per connection: send a request JSON, receive per-step
        progress events, then the terminal result with latents_sha and an
        MP4 download URL. Registered at the starlette level (below) — this
        environment's FastAPI websocket DI closes connections without ever
        invoking the endpoint."""
        await ws.accept()
        try:
            payload = await ws.receive_json()
            job_id = uuid.uuid4().hex[:12]
            req = _to_request(payload, job_id)
        except WebSocketDisconnect:
            return
        except Exception as e:
            import traceback
            await ws.send_json({"type": "error",
                                "error": traceback.format_exc()[-600:] or str(e)})
            await ws.close()
            return
        loop = asyncio.get_running_loop()
        q: asyncio.Queue = asyncio.Queue()

        def observe_step(label: str, seconds: float, meta: dict) -> None:
            loop.call_soon_threadsafe(
                q.put_nowait, {"type": "step", "label": label,
                               "seconds": round(seconds, 4), **{
                                   k: v for k, v in meta.items()
                                   if isinstance(v, (int, float, str))}})

        def run() -> None:
            try:
                from fastvideo2.engine import run as engine_run
                from fastvideo2.sdk import Result
                with gen_lock:
                    out = engine_run(model.instance, model.pipeline, req,
                                     on_step=observe_step)  # LIVE per-step
                result = Result(outputs=out.outputs, trace=out.trace,
                                request=req.resolve(model.card),
                                model_id=model.model_id,
                                card_digest=model.card.digest(),
                                fps=model.card.sampling_defaults.fps)
                path = os.path.join(out_dir, f"{job_id}.mp4")
                result.save(path)
                jobs[job_id] = {"id": job_id, "status": "completed",
                                "video_path": path}
                loop.call_soon_threadsafe(
                    q.put_nowait, {"type": "done", "id": job_id,
                                   "latents_sha": _latents_sha(result),
                                   "content_url": f"/v1/videos/{job_id}/content"})
            except Exception as e:
                loop.call_soon_threadsafe(
                    q.put_nowait, {"type": "error",
                                   "error": f"{type(e).__name__}: {e}"})

        threading.Thread(target=run, daemon=True).start()
        while True:
            msg = await q.get()
            await ws.send_json(msg)
            if msg["type"] in ("done", "error"):
                break
        await ws.close()

    from starlette.routing import WebSocketRoute
    app.router.routes.append(WebSocketRoute("/v1/stream", stream))
    return app


def main(argv: list[str] | None = None) -> None:
    import argparse

    import uvicorn

    import fastvideo2 as fv2

    p = argparse.ArgumentParser("fastvideo2.serve")
    p.add_argument("--model", default="wan2.1-t2v-1.3b")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8010)
    p.add_argument("--device", default=None)
    args = p.parse_args(argv)
    model = fv2.load(args.model, device=args.device)
    uvicorn.run(build_app(model), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
