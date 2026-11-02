# SPDX-License-Identifier: Apache-2.0
"""Media references from HTTP callers: local roots, private addresses, size caps, cleanup."""

import asyncio
import os

import httpx
import pytest

from fastvideo.entrypoints.openai.media_policy import (
    MediaPolicy,
    MediaPolicyError,
    fetch_media,
    resolve_local_media,
)


def test_local_paths_are_rejected_without_a_configured_root(tmp_path):
    existing = tmp_path / "clip.mp4"
    existing.write_bytes(b"x")
    missing = tmp_path / "missing.mp4"
    errors = []
    for path in (existing, missing):
        with pytest.raises(MediaPolicyError) as error:
            resolve_local_media(str(path), MediaPolicy())
        errors.append(str(error.value))
    # The same message for existing and missing files: no filesystem oracle.
    assert errors[0] == errors[1]
    assert "server.media_roots" in errors[0]


def test_local_paths_inside_a_root_resolve(tmp_path):
    media = tmp_path / "media"
    media.mkdir()
    clip = media / "clip.mp4"
    clip.write_bytes(b"x")
    assert resolve_local_media(str(clip), MediaPolicy(roots=(str(media), ))) == os.path.realpath(clip)


def test_symlinks_cannot_escape_a_root(tmp_path):
    media = tmp_path / "media"
    media.mkdir()
    secret = tmp_path / "secret.txt"
    secret.write_text("x")
    (media / "link.mp4").symlink_to(secret)
    with pytest.raises(MediaPolicyError):
        resolve_local_media(str(media / "link.mp4"), MediaPolicy(roots=(str(media), )))


def test_missing_files_inside_a_root_are_reported(tmp_path):
    with pytest.raises(MediaPolicyError, match="does not exist"):
        resolve_local_media(str(tmp_path / "missing.mp4"), MediaPolicy(roots=(str(tmp_path), )))


@pytest.mark.parametrize("url", [
    "http://127.0.0.1/clip.mp4",
    "http://localhost/clip.mp4",
    "http://10.0.0.5/clip.mp4",
    "http://169.254.169.254/latest/meta-data",
    "http://[::1]/clip.mp4",
])
def test_private_addresses_are_blocked_by_default(tmp_path, url):
    transport = httpx.MockTransport(lambda request: httpx.Response(200, content=b"x"))
    with pytest.raises(MediaPolicyError, match="private"):
        asyncio.run(fetch_media(url, str(tmp_path / "out"), max_bytes=10, policy=MediaPolicy(), transport=transport))
    assert not (tmp_path / "out").exists()


def test_private_addresses_can_be_allowed(tmp_path):
    transport = httpx.MockTransport(lambda request: httpx.Response(200, content=b"abc"))
    target = tmp_path / "out"
    asyncio.run(
        fetch_media("http://127.0.0.1/clip.mp4",
                    str(target),
                    max_bytes=10,
                    policy=MediaPolicy(allow_private_urls=True),
                    transport=transport))
    assert target.read_bytes() == b"abc"


def test_redirects_to_private_addresses_are_blocked(tmp_path, monkeypatch):
    monkeypatch.setattr("fastvideo.entrypoints.openai.media_policy._resolve_addresses",
                        lambda host, port: ["93.184.216.34"] if host == "media.example" else ["127.0.0.1"])

    def handler(request):
        if request.url.host == "media.example":
            return httpx.Response(302, headers={"location": "http://internal.example/secret"})
        return httpx.Response(200, content=b"secret")

    with pytest.raises(MediaPolicyError, match="private"):
        asyncio.run(
            fetch_media("http://media.example/clip.mp4",
                        str(tmp_path / "out"),
                        max_bytes=10,
                        policy=MediaPolicy(),
                        transport=httpx.MockTransport(handler)))
    assert not (tmp_path / "out").exists()


def test_streamed_bytes_are_capped_even_without_content_length(tmp_path, monkeypatch):
    monkeypatch.setattr("fastvideo.entrypoints.openai.media_policy._resolve_addresses",
                        lambda host, port: ["93.184.216.34"])

    async def body():
        for _ in range(4):
            yield b"12345"

    transport = httpx.MockTransport(lambda request: httpx.Response(200, content=body()))
    with pytest.raises(MediaPolicyError, match="limit"):
        asyncio.run(
            fetch_media("http://media.example/clip.mp4",
                        str(tmp_path / "out"),
                        max_bytes=12,
                        policy=MediaPolicy(),
                        transport=transport))
    assert not (tmp_path / "out").exists()


def test_cancelled_downloads_remove_the_partial_file(tmp_path, monkeypatch):
    monkeypatch.setattr("fastvideo.entrypoints.openai.media_policy._resolve_addresses",
                        lambda host, port: ["93.184.216.34"])
    async def scenario():
        first_chunk = asyncio.Event()

        async def body():
            yield b"partial"
            first_chunk.set()
            await asyncio.sleep(30)
            yield b"never"

        transport = httpx.MockTransport(lambda request: httpx.Response(200, content=body()))
        target = tmp_path / "out"
        task = asyncio.create_task(
            fetch_media("http://media.example/clip.mp4",
                        str(target),
                        max_bytes=1024,
                        policy=MediaPolicy(),
                        transport=transport))
        await asyncio.wait_for(first_chunk.wait(), 5)
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        return target

    assert not asyncio.run(scenario()).exists()


def test_only_http_urls_are_fetched(tmp_path):
    with pytest.raises(MediaPolicyError, match="http"):
        asyncio.run(fetch_media("file:///etc/hosts", str(tmp_path / "out"), max_bytes=10, policy=MediaPolicy()))
