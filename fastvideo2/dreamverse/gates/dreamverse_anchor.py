"""Dreamverse runtime anchor: boot the server with DUMMY prompt keys, drive
one full session over the protocol, and assert the streaming contract:
segment_start -> live step_complete x3 -> media_init -> binary fMP4 chunks
(first chunk carries an ISO-BMFF `ftyp` box) -> media_segment_complete ->
segment_complete with a latents sha.

Usage (cluster): python -m fastvideo2.dreamverse.gates.dreamverse_anchor
"""
from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import time
import urllib.request

MODEL = "fastwan-qad-fp8-1.3b"
PORT = 8019
PROMPT = "A raccoon in a field of sunflowers, warm light, mid-shot."


def main() -> int:
    env = dict(os.environ, CEREBRAS_API_KEY="dummy", GROQ_API_KEY="dummy")
    server = subprocess.Popen(
        [sys.executable, "-m", "fastvideo2.dreamverse", "--model", MODEL,
         "--port", str(PORT)], env=env,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(360):
            try:
                with urllib.request.urlopen(
                        f"http://127.0.0.1:{PORT}/health", timeout=5) as r:
                    if json.loads(r.read())["model"] == MODEL:
                        break
            except Exception:
                time.sleep(2)
        else:
            raise RuntimeError("server never became healthy")

        import websockets

        async def session() -> dict:
            counts = {"steps": 0, "chunks": 0, "ftyp": False}
            async with websockets.connect(
                    f"ws://127.0.0.1:{PORT}/ws", max_size=None) as ws:
                await ws.send(json.dumps({"type": "session_init_v2",
                                          "enhancement": True}))
                for expected in ("queue_status", "gpu_assigned", "stream_start"):
                    got = json.loads(await ws.recv())["type"]
                    assert got == expected, (got, expected)
                await ws.send(json.dumps({"type": "segment_prompt_source",
                                          "prompt": PROMPT, "seed": 7}))
                while True:
                    raw = await asyncio.wait_for(ws.recv(), timeout=900)
                    if isinstance(raw, bytes):
                        if counts["chunks"] == 0:
                            counts["ftyp"] = b"ftyp" in raw[:64]
                        counts["chunks"] += 1
                        continue
                    msg = json.loads(raw)
                    t = msg["type"]
                    if t == "step_complete":
                        counts["steps"] += 1
                    elif t == "segment_complete":
                        counts["latents_sha"] = msg["latents_sha"]
                        counts["frames"] = msg["frames"]
                        break
                    elif t == "error":
                        raise RuntimeError(msg)
                await ws.send(json.dumps({"type": "leave"}))
                assert json.loads(await ws.recv())["type"] == "stream_complete"
            return counts

        c = asyncio.run(session())
    finally:
        server.terminate()
        server.wait(timeout=30)

    ok = (c["steps"] == 3 and c["chunks"] >= 1 and c["ftyp"]
          and c.get("frames", 0) == 81)
    print(f"steps={c['steps']} chunks={c['chunks']} ftyp={c['ftyp']} "
          f"frames={c.get('frames')} sha={c.get('latents_sha')} "
          f"{'OK' if ok else 'FAIL'}")

    from fastvideo2.verify import GateResult, append_ledger, env_fingerprint
    append_ledger([GateResult(gate="anchor.dreamverse-runtime",
                              status="pass" if ok else "fail", model_id=MODEL,
                              card_digest="-",
                              metrics={"steps": float(c["steps"]),
                                       "chunks": float(c["chunks"]),
                                       "ftyp": 1.0 if c["ftyp"] else 0.0},
                              tolerances={}, env=env_fingerprint(),
                              detail=f"latents {c.get('latents_sha')}")])
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
