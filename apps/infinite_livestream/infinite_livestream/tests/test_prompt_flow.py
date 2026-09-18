"""A submitted viewer prompt must become ordered clips in the real engine queue.

Only provider calls and GPU readiness are substituted. Chat intake, moderation,
rewriting, admission, metadata, queue insertion, and web state all run normally.
"""

import asyncio
import json
from types import SimpleNamespace

from fastapi.testclient import TestClient
import pytest

from infinite_livestream import moderator as moderator_module
from infinite_livestream import upsampler as upsampler_module
from infinite_livestream.chat import WebChat
from infinite_livestream.config import load_model_config
from infinite_livestream.director import Director
from infinite_livestream.engine import Engine
from infinite_livestream.webapp import DemoWeb


@pytest.mark.parametrize("scene_count", [1, 3])
def test_accepted_prompt_reaches_generation_queue(app_config, monkeypatch, scene_count):
    scenes = [{"prompt": f"A lighthouse keeper feeds seagull {i}", "seconds": 8.0 + i}
              for i in range(scene_count)]
    provider_calls = []

    async def moderate(**kwargs):
        provider_calls.append(("moderation", kwargs["input"]))
        return SimpleNamespace(results=[SimpleNamespace(flagged=False)])

    async def rewrite(**kwargs):
        provider_calls.append(("rewrite", kwargs["messages"][-1]["content"]))
        content = json.dumps({"title": "The lighthouse", "scenes": scenes})
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])

    provider = SimpleNamespace(
        moderations=SimpleNamespace(create=moderate),
        chat=SimpleNamespace(completions=SimpleNamespace(create=rewrite)),
    )
    monkeypatch.setattr(moderator_module, "AsyncOpenAI", lambda **kwargs: provider)
    monkeypatch.setattr(upsampler_module, "AsyncOpenAI", lambda **kwargs: provider)

    async def run():
        engine = Engine(app_config, load_model_config(app_config.config_path))
        engine._ready.set()  # Queue operations need readiness, never GPU work.
        await engine.send_command("enqueue", {"prompt": "earlier viewer"})
        await engine.send_command("enqueue", {"prompt": "idle filler", "metadata": json.dumps({
            "group_id": "idle", "title": "idle", "author": "filler", "source": "idle",
            "scene": 1, "scenes": 1, "generated": True,
        })})
        chat = WebChat()
        web = DemoWeb(chat, app_config.hls_dir)
        engine.add_listener(web.listener)
        moderator = moderator_module.Moderator("test-key", "test-model", enabled=True)
        upsampler = upsampler_module.PromptUpsampler("test-key", "test-model", "house style", max_chunks=6)
        rejections = []
        director = Director(engine, upsampler, moderator, cooldown_s=10,
                            on_reject=lambda author, reason: rejections.append((author, reason)))
        web.cooldown_remaining = director.cooldown_remaining
        queued = asyncio.Event()

        def on_queue(kind, data):
            if kind == "queue_update" and len(data["generation"]) == scene_count + 2:
                queued.set()

        engine.add_listener(on_queue)
        # Submit before starting intake so TestClient never wakes a queue
        # waiter owned by a different event loop.
        with TestClient(web.app) as client:
            response = client.post("/chat", json={"author": "ada", "text": "a lighthouse keeper"})
        assert response.status_code == 200 and response.json() == {"ok": True}
        tasks = [asyncio.create_task(chat.run(director.submit)), asyncio.create_task(director.run())]
        try:
            await asyncio.wait_for(queued.wait(), timeout=2)
            clips = engine.generation_clips
            assert [c["prompt"] for c in clips] == ["earlier viewer", *[s["prompt"] for s in scenes], "idle filler"]
            tags = [json.loads(c["metadata"]) for c in clips[1:-1]]
            assert len({tag["group_id"] for tag in tags}) == 1
            assert [tag["scene"] for tag in tags] == list(range(1, scene_count + 1))
            assert all(tag["scenes"] == scene_count and tag["author"] == "ada"
                       and tag["raw_prompt"] == "a lighthouse keeper" and not tag["generated"] for tag in tags)
            if scene_count == 1:
                assert clips[1]["frames"] == 345, "single scenes must use the maximum clip length"
            assert [c["clip_id"] for c in web.state.generation] == [c["clip_id"] for c in clips]
            assert all(c["prompt"] == "a lighthouse keeper" for c in web.state.generation[1:-1])
            assert [name for name, _ in provider_calls] == ["moderation", "rewrite"]
            assert provider_calls[0][1] == "a lighthouse keeper"
            assert "a lighthouse keeper" in provider_calls[1][1]
            assert rejections == []
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    asyncio.run(run())
