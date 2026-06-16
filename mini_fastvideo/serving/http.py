"""A tiny framework-free async HTTP/1.1 server (stdlib asyncio only).

No FastAPI/uvicorn dependency — the design says the production OpenAI *fleet* frontend is Dynamo's
job; this is the per-engine worker server (our own version). Supports JSON responses and streamed
(SSE) responses. Deliberately minimal: one request per connection (Connection: close), enough to be
real and curl-able, not a hardened web server.
"""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Awaitable, Callable


@dataclass
class Request:
    method: str
    path: str
    headers: dict[str, str]
    body: bytes
    query: dict[str, str] = field(default_factory=dict)

    def json(self) -> dict:
        return json.loads(self.body or b"{}")


@dataclass
class Response:
    status: int = 200
    body: bytes = b""
    content_type: str = "application/json"
    stream: AsyncIterator | None = None          # if set, body is ignored and chunks are streamed

    @classmethod
    def json(cls, obj: Any, status: int = 200) -> "Response":
        return cls(status=status, body=json.dumps(obj).encode(), content_type="application/json")

    @classmethod
    def sse(cls, gen: AsyncIterator) -> "Response":
        return cls(status=200, content_type="text/event-stream", stream=gen)


_STATUS = {200: "OK", 201: "Created", 202: "Accepted", 400: "Bad Request",
           404: "Not Found", 500: "Internal Server Error"}

Handler = Callable[[Request], Awaitable[Response]]


async def _read_request(reader: asyncio.StreamReader) -> Request:
    head = await reader.readuntil(b"\r\n\r\n")
    lines = head.decode("latin1").split("\r\n")
    method, target, _ = lines[0].split(" ", 2)
    path, _, qs = target.partition("?")
    query = dict(kv.split("=", 1) for kv in qs.split("&") if "=" in kv)
    headers = {}
    for ln in lines[1:]:
        if ":" in ln:
            k, v = ln.split(":", 1)
            headers[k.strip().lower()] = v.strip()
    clen = int(headers.get("content-length", "0") or "0")
    body = await reader.readexactly(clen) if clen else b""
    return Request(method, path, headers, body, query)


async def _write(writer: asyncio.StreamWriter, resp: Response) -> None:
    if resp.stream is not None:
        head = (f"HTTP/1.1 {resp.status} {_STATUS.get(resp.status, 'OK')}\r\n"
                f"Content-Type: {resp.content_type}\r\nCache-Control: no-cache\r\n"
                f"Connection: close\r\n\r\n")
        writer.write(head.encode())
        await writer.drain()
        async for chunk in resp.stream:
            writer.write(chunk.encode() if isinstance(chunk, str) else chunk)
            await writer.drain()
    else:
        head = (f"HTTP/1.1 {resp.status} {_STATUS.get(resp.status, 'OK')}\r\n"
                f"Content-Type: {resp.content_type}\r\nContent-Length: {len(resp.body)}\r\n"
                f"Connection: close\r\n\r\n")
        writer.write(head.encode())
        writer.write(resp.body)
        await writer.drain()
    writer.close()


class HttpServer:
    def __init__(self, dispatch: Handler):
        self.dispatch = dispatch
        self._server: asyncio.AbstractServer | None = None

    async def _conn(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            req = await _read_request(reader)
            resp = await self.dispatch(req)
        except Exception as e:  # noqa: BLE001 — never let one connection kill the server
            resp = Response.json({"error": str(e)}, status=500)
        try:
            await _write(writer, resp)
        except Exception:
            try:
                writer.close()
            except Exception:
                pass

    async def start(self, host: str = "127.0.0.1", port: int = 0) -> tuple[str, int]:
        self._server = await asyncio.start_server(self._conn, host, port)
        sock = self._server.sockets[0].getsockname()
        return sock[0], sock[1]

    async def close(self) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
