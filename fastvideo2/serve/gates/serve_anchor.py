"""Serving anchor — served output is BITWISE the offline output.

Boots the server as a subprocess (fastwan-qad: 3-step DMD, fast), runs one
REST job and one WebSocket stream, and compares the served latents sha with
an offline SDK generation of the identical request. Also asserts the WS
delivered one live event per denoise step (the per-step contract).

Usage (cluster): python -m fastvideo2.serve.gates.serve_anchor
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
import urllib.request

MODEL = "fastwan-qad-fp8-1.3b"
PORT = 8017
PROMPT = ("A curious raccoon peers through a vibrant field of yellow "
          "sunflowers, its eyes wide with interest. The playful yet serene "
          "atmosphere is complemented by soft natural light filtering "
          "through the petals. Mid-shot, warm and cheerful tones.")
SEED = 1234


def _get(path: str) -> dict:
    with urllib.request.urlopen(f"http://127.0.0.1:{PORT}{path}", timeout=30) as r:
        return json.loads(r.read())


def _post(path: str, body: dict) -> dict:
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}{path}", method="POST",
        data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def main() -> int:
    import torch

    import fastvideo2 as fv2
    from fastvideo2.engine import Request

    server = subprocess.Popen(
        [sys.executable, "-m", "fastvideo2.serve", "--model", MODEL,
         "--port", str(PORT)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(360):  # model load can take a couple of minutes
            try:
                if _get("/health")["model"] == MODEL:
                    break
            except Exception:
                time.sleep(2)
        else:
            raise RuntimeError("server never became healthy")

        job = _post("/v1/videos", {"prompt": PROMPT, "seed": SEED})
        while True:
            st = _get(f"/v1/videos/{job['id']}")
            if st["status"] in ("completed", "failed"):
                break
            time.sleep(3)
        assert st["status"] == "completed", st
        served_sha = st["latents_sha"]

        # WS: live per-step events for a fresh request
        import asyncio

        import websockets

        async def ws_run() -> tuple[int, str]:
            async with websockets.connect(
                    f"ws://127.0.0.1:{PORT}/v1/stream", max_size=None) as ws:
                await ws.send(json.dumps({"prompt": PROMPT, "seed": 999}))
                steps = 0
                while True:
                    msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=600))
                    if msg["type"] == "step":
                        steps += 1
                    elif msg["type"] == "done":
                        return steps, msg["latents_sha"]
                    else:
                        raise RuntimeError(msg)

        ws_steps, _ = asyncio.run(ws_run())
    finally:
        server.terminate()
        server.wait(timeout=30)

    # offline identity on a fresh process-local model
    model = fv2.load(MODEL, device="cuda")
    result = model.generate(request=Request(prompt=PROMPT, request_id="offline",
                                            seed=SEED))
    lat = result.outputs["latents"]["latents"]
    offline_sha = hashlib.sha256(
        lat.detach().to(torch.float32).cpu().numpy().tobytes()).hexdigest()[:16]

    ok_identity = served_sha == offline_sha
    ok_steps = ws_steps == 3  # QAD: 3 DMD denoise steps
    print(f"served latents {served_sha} | offline {offline_sha} | "
          f"identity {'OK' if ok_identity else 'FAIL'}")
    print(f"ws live step events: {ws_steps} (expect 3) "
          f"{'OK' if ok_steps else 'FAIL'}")

    from fastvideo2.verify import GateResult, append_ledger, env_fingerprint
    append_ledger([GateResult(gate="anchor.serve-identity",
                              status="pass" if ok_identity and ok_steps else "fail",
                              model_id=MODEL, card_digest=model.card.digest(),
                              metrics={"identity": 0.0 if ok_identity else 1.0,
                                       "ws_steps": float(ws_steps)},
                              tolerances={"identity": 0.0},
                              env=env_fingerprint(),
                              detail=f"served {served_sha} offline {offline_sha}")])
    return 0 if ok_identity and ok_steps else 1


if __name__ == "__main__":
    sys.exit(main())
