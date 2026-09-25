"""POST /chat refuses a rate-limited viewer privately, in their own reply.

The chat feed is shared by every viewer, so a refusal must not go into it: one
person's rate-limit is not the room's business, and a page full of "not queued"
lines is noise. The sender learns about it from the status of their own
request, and their page locks the box until it lifts.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from infinite_livestream.chat import WebChat
from infinite_livestream.webapp import DemoWeb


@pytest.fixture()
def web(tmp_path) -> DemoWeb:
    return DemoWeb(WebChat("!prompt"), tmp_path)


def test_a_prompt_is_accepted_and_echoed(web: DemoWeb) -> None:
    client = TestClient(web.app)
    assert client.post("/chat", json={"author": "ada", "text": "a lighthouse"}).json() == {"ok": True}
    assert [(m["kind"], m["author"]) for m in web.state.chat] == [("viewer", "ada")]


def test_cooldown_is_refused_with_a_retry_after(web: DemoWeb) -> None:
    web.cooldown_remaining = lambda author: 4.2 if author == "ada" else 0.0
    client = TestClient(web.app)
    response = client.post("/chat", json={"author": "ada", "text": "a lighthouse"})
    assert response.status_code == 429
    assert response.json() == {"ok": False, "error": "cooldown", "retry_after": 4.2}


def test_a_refused_prompt_never_reaches_the_shared_feed(web: DemoWeb) -> None:
    web.cooldown_remaining = lambda author: 4.2
    client = TestClient(web.app)
    client.post("/chat", json={"author": "ada", "text": "a lighthouse"})
    assert list(web.state.chat) == [], "the room must not see one viewer's rate-limit"


def test_other_viewers_are_unaffected(web: DemoWeb) -> None:
    web.cooldown_remaining = lambda author: 4.2 if author == "ada" else 0.0
    client = TestClient(web.app)
    assert client.post("/chat", json={"author": "ada", "text": "x"}).status_code == 429
    assert client.post("/chat", json={"author": "grace", "text": "y"}).status_code == 200


def test_empty_prompts_are_still_rejected(web: DemoWeb) -> None:
    client = TestClient(web.app)
    assert client.post("/chat", json={"author": "ada", "text": "   "}).status_code == 400
