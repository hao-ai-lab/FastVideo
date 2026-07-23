"""Dreamverse runtime on fastvideo2 — the realtime video session server.

Ported from ``apps/dreamverse`` (fastvideo-main): the WebSocket session
protocol (``session_init_v2`` → per-segment prompts → fMP4 fragments over
the socket), the ffmpeg fragmented-MP4 encoder (verbatim flags from
``entrypoints/streaming/stream.py``: libx264, zerolatency,
``empty_moov+default_base_moof+frag_keyframe+faststart``), and an optional
Cerebras/Groq prompt enhancer (boots with dummy keys — enhancement simply
stays off, the same bring-up shortcut the GB200 deploys used).

Deliberately re-based for v2.1 (the original is LTX2-specific — audio,
refine stage, continuation-state, LoRA stack): segments generate through the
fastvideo2 SDK on the FastWan 3-step DMD student (seconds per segment on
GB200), and per-step progress is LIVE via the engine's ``on_step`` hook
(``step_complete`` per denoise step — the original emits one terminal event
per segment). Message names follow the upstream protocol so their web client
schema maps directly; LTX2-only fields are ignored.

Run:  python -m fastvideo2.dreamverse --model fastwan-qad-fp8-1.3b --port 8009
ffmpeg: FASTVIDEO_FFMPEG_BIN, or PATH, or the imageio-ffmpeg bundled binary.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shutil
import subprocess
import threading
import urllib.request
import uuid
from typing import Any


def find_ffmpeg() -> str:
    p = os.environ.get("FASTVIDEO_FFMPEG_BIN") or shutil.which("ffmpeg")
    if p:
        return p
    try:  # the GB200 bring-up shortcut: pip-installed bundled binary
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError as e:
        raise RuntimeError("no ffmpeg (set FASTVIDEO_FFMPEG_BIN, install "
                           "ffmpeg, or `pip install imageio-ffmpeg`)") from e


def fmp4_encode(frames: Any, *, fps: int, ffmpeg: str) -> list[bytes]:
    """One segment -> fragmented-MP4 byte chunks (upstream's exact flags)."""
    t, h, w, _ = frames.shape
    args = [ffmpeg, "-hide_banner", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
            "-r", str(fps), "-i", "-",
            "-c:v", "libx264", "-preset", "ultrafast", "-tune", "zerolatency",
            "-pix_fmt", "yuv420p",
            "-movflags", "empty_moov+default_base_moof+frag_keyframe+faststart",
            "-f", "mp4", "-"]
    proc = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.DEVNULL, bufsize=0)
    out: list[bytes] = []

    def _read() -> None:
        while True:
            chunk = proc.stdout.read(65536)
            if not chunk:
                break
            out.append(chunk)

    reader = threading.Thread(target=_read, daemon=True)
    reader.start()
    for i in range(t):
        proc.stdin.write(frames[i].tobytes())
    proc.stdin.close()
    proc.wait(timeout=120)
    reader.join(timeout=30)
    return out


class PromptEnhancer:
    """Cerebras-or-Groq chat call (upstream's provider pair, gpt-oss-120b).
    Dummy/missing keys or any failure -> pass the prompt through unchanged."""

    def __init__(self) -> None:
        self.cerebras = os.environ.get("CEREBRAS_API_KEY", "")
        self.groq = os.environ.get("GROQ_API_KEY", "")
        self.enabled = any(k and k != "dummy" for k in (self.cerebras, self.groq))

    def enhance(self, prompt: str, history: list[str]) -> str:
        if not self.enabled:
            return prompt
        targets = []
        if self.cerebras and self.cerebras != "dummy":
            targets.append(("https://api.cerebras.ai/v1/chat/completions",
                            self.cerebras, "gpt-oss-120b"))
        if self.groq and self.groq != "dummy":
            targets.append(("https://api.groq.com/openai/v1/chat/completions",
                            self.groq, "openai/gpt-oss-120b"))
        system = ("Rewrite the user's next-video-segment prompt into one vivid, "
                  "concrete shot description. Prior segments: "
                  + " | ".join(history[-3:]))
        for url, key, model_name in targets:
            try:
                req = urllib.request.Request(
                    url, method="POST",
                    headers={"Authorization": f"Bearer {key}",
                             "Content-Type": "application/json"},
                    data=json.dumps({"model": model_name, "temperature": 1.0,
                                     "messages": [{"role": "system", "content": system},
                                                  {"role": "user", "content": prompt}]
                                     }).encode())
                with urllib.request.urlopen(req, timeout=20) as r:
                    return json.loads(r.read())["choices"][0]["message"]["content"].strip()
            except Exception:
                continue
        return prompt


def build_app(model: Any) -> Any:
    from fastapi import FastAPI
    from starlette.routing import WebSocketRoute
    from starlette.websockets import WebSocketDisconnect

    from fastvideo2.engine import Request
    from fastvideo2.engine import run as engine_run
    from fastvideo2.sdk import Result

    app = FastAPI(title="dreamverse-fv2", version="0.1")
    ffmpeg = find_ffmpeg()
    enhancer = PromptEnhancer()
    gen_lock = threading.Lock()

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "model": model.model_id,
                "enhancer": enhancer.enabled, "ffmpeg": ffmpeg}

    @app.get("/readyz")
    def readyz() -> dict:
        return {"ready": True}

    async def ws_session(ws) -> None:
        await ws.accept()
        try:
            init = await ws.receive_json()
        except WebSocketDisconnect:
            return
        if init.get("type") != "session_init_v2":
            await ws.send_json({"type": "error", "code": "bad_init",
                                "error": "expected session_init_v2"})
            await ws.close()
            return
        session_id = uuid.uuid4().hex[:12]
        enhancement_on = bool(init.get("enhancement", False)) and enhancer.enabled
        history: list[str] = []
        segment_idx = 0
        await ws.send_json({"type": "queue_status", "position": 0})
        await ws.send_json({"type": "gpu_assigned", "session_id": session_id})
        await ws.send_json({"type": "stream_start", "session_id": session_id,
                            "model": model.model_id})

        loop = asyncio.get_running_loop()
        while True:
            try:
                msg = await ws.receive_json()
            except WebSocketDisconnect:
                return
            mtype = msg.get("type")
            if mtype == "leave":
                await ws.send_json({"type": "stream_complete",
                                    "segments": segment_idx})
                await ws.close()
                return
            if mtype == "enhancement_updated":
                enhancement_on = bool(msg.get("enabled")) and enhancer.enabled
                continue
            if mtype != "segment_prompt_source":
                await ws.send_json({"type": "error", "code": "bad_message",
                                    "error": f"unsupported type {mtype!r}"})
                continue

            prompt = str(msg.get("prompt", ""))
            if not prompt:
                await ws.send_json({"type": "error", "code": "bad_prompt",
                                    "error": "prompt required"})
                continue
            if enhancement_on:
                prompt = await loop.run_in_executor(
                    None, enhancer.enhance, prompt, history)
            history.append(prompt)
            await ws.send_json({"type": "segment_start", "segment": segment_idx,
                                "prompt": prompt})

            q: asyncio.Queue = asyncio.Queue()

            def on_step(label: str, seconds: float, meta: dict) -> None:
                loop.call_soon_threadsafe(
                    q.put_nowait, {"type": "step_complete", "label": label,
                                   "seconds": round(seconds, 4)})

            def generate(p: str = prompt, seed: Any = msg.get("seed", 0),
                         steps: Any = msg.get("num_steps")) -> None:
                try:
                    req = Request(prompt=p, request_id=f"{session_id}-{segment_idx}",
                                  seed=int(seed), num_steps=steps)
                    with gen_lock:
                        out = engine_run(model.instance, model.pipeline, req,
                                         on_step=on_step)
                    result = Result(outputs=out.outputs, trace=out.trace,
                                    request=req.resolve(model.card),
                                    model_id=model.model_id,
                                    card_digest=model.card.digest(),
                                    fps=model.card.sampling_defaults.fps)
                    chunks = fmp4_encode(result.video, fps=result.fps,
                                         ffmpeg=ffmpeg)
                    import torch
                    sha = hashlib.sha256(result.latents.detach().to(
                        torch.float32).cpu().numpy().tobytes()).hexdigest()[:16]
                    loop.call_soon_threadsafe(
                        q.put_nowait, {"__chunks": chunks, "latents_sha": sha,
                                       "frames": int(result.video.shape[0])})
                except Exception as e:
                    loop.call_soon_threadsafe(
                        q.put_nowait, {"type": "error", "code": "generation",
                                       "error": f"{type(e).__name__}: {e}"})

            threading.Thread(target=generate, daemon=True).start()
            while True:
                ev = await q.get()
                if "__chunks" in ev:
                    await ws.send_json({"type": "media_init",
                                        "segment": segment_idx,
                                        "mime": 'video/mp4; codecs="avc1"'})
                    for chunk in ev["__chunks"]:
                        await ws.send_bytes(chunk)
                    await ws.send_json({"type": "media_segment_complete",
                                        "segment": segment_idx})
                    await ws.send_json({"type": "segment_complete",
                                        "segment": segment_idx,
                                        "frames": ev["frames"],
                                        "latents_sha": ev["latents_sha"]})
                    segment_idx += 1
                    break
                await ws.send_json(ev)
                if ev.get("type") == "error":
                    break

    app.router.routes.append(WebSocketRoute("/ws", ws_session))
    return app


def main(argv: list[str] | None = None) -> None:
    import argparse

    import uvicorn

    import fastvideo2 as fv2

    p = argparse.ArgumentParser("fastvideo2.dreamverse")
    p.add_argument("--model", default="fastwan-qad-fp8-1.3b")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8009)
    p.add_argument("--device", default=None)
    args = p.parse_args(argv)
    model = fv2.load(args.model, device=args.device)
    uvicorn.run(build_app(model), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
