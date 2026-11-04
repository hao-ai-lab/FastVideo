# SPDX-License-Identifier: Apache-2.0
"""Rules for media references supplied by HTTP callers.

Local paths are readable only inside operator-listed ``server.media_roots``.
URLs must use http(s) and, unless ``server.allow_private_media_urls`` is set,
must resolve to public addresses on every redirect hop. Downloads are capped
by streamed bytes and leave no partial file on failure or cancellation.
"""

from __future__ import annotations

import asyncio
import contextlib
import ipaddress
import os
import socket
from dataclasses import dataclass
from urllib.parse import urljoin, urlparse

import aiofiles
import httpx

MAX_REDIRECTS = 5
READ_TIMEOUT_S = 60.0


class MediaPolicyError(ValueError):
    """A media reference is not allowed by the server's media policy."""


@dataclass(frozen=True)
class MediaPolicy:
    roots: tuple[str, ...] = ()
    allow_private_urls: bool = False


def resolve_local_media(source: str, policy: MediaPolicy) -> str:
    """Return the real path of a caller-supplied local file inside a configured root."""
    path = os.path.realpath(os.path.expanduser(source))
    for root in policy.roots:
        real_root = os.path.realpath(os.path.expanduser(root))
        if os.path.commonpath([path, real_root]) == real_root:
            if not os.path.isfile(path):
                raise MediaPolicyError(f"Media reference does not exist: {source}")
            return path
    # Checked before existence so callers cannot probe the server's filesystem.
    raise MediaPolicyError("Local media paths are not accepted from HTTP callers. Provide an http(s) URL, or list "
                           "the directory in server.media_roots in the serve config.")


def _resolve_addresses(host: str, port: int) -> list[str]:
    return [str(info[4][0]) for info in socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)]


async def _check_url(url: str, policy: MediaPolicy) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise MediaPolicyError("Media URLs must use http or https.")
    if policy.allow_private_urls:
        return
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        addresses = await asyncio.to_thread(_resolve_addresses, parsed.hostname, port)
    except OSError as error:
        raise MediaPolicyError(f"Unable to resolve media host {parsed.hostname}: {error}") from error
    for address in addresses:
        if not ipaddress.ip_address(address.split("%", 1)[0]).is_global:
            raise MediaPolicyError(f"Media URL {parsed.hostname} resolves to a private or local address. Set "
                                   "server.allow_private_media_urls for trusted internal media hosts.")


def discard(path: str) -> None:
    with contextlib.suppress(OSError):
        os.unlink(path)


async def fetch_media(
    url: str,
    target: str,
    *,
    max_bytes: int,
    policy: MediaPolicy,
    transport: httpx.AsyncBaseTransport | None = None,
) -> httpx.Headers:
    """Download ``url`` to ``target`` under ``policy``; returns the final response headers."""
    os.makedirs(os.path.dirname(os.path.abspath(target)), exist_ok=True)
    try:
        async with httpx.AsyncClient(follow_redirects=False, timeout=READ_TIMEOUT_S, transport=transport) as client:
            for _ in range(MAX_REDIRECTS + 1):
                await _check_url(url, policy)
                async with client.stream("GET", url) as response:
                    if response.is_redirect:
                        location = response.headers.get("location")
                        if not location:
                            raise MediaPolicyError("Media URL redirect has no location.")
                        url = urljoin(url, location)
                        continue
                    response.raise_for_status()
                    written = 0
                    async with aiofiles.open(target, "wb") as output:
                        async for chunk in response.aiter_bytes():
                            written += len(chunk)
                            if written > max_bytes:
                                raise MediaPolicyError(
                                    f"Media reference is larger than the {max_bytes}-byte limit; raise "
                                    "server.media_reference_max_bytes in the serve config or provide a smaller file.")
                            await output.write(chunk)
                    return response.headers
            raise MediaPolicyError(f"Media URL redirected more than {MAX_REDIRECTS} times.")
    except BaseException:
        # Includes cancellation: a client disconnect must not leave a partial upload behind.
        discard(target)
        raise
