"""A tiny framework-free async HTTP/1.1 server (stdlib asyncio only).

No FastAPI/uvicorn dependency — the production OpenAI *fleet* frontend is Dynamo's job; this is the
per-engine worker server. Supports JSON and streamed (SSE) responses. Deliberately minimal: one
request per connection (Connection: close), enough to be real and curl-able, not a hardened server.
"""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any
from collections.abc import AsyncIterator, Awaitable, Callable


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
    stream: AsyncIterator | None = None  # if set, body is ignored and chunks are streamed

    @classmethod
    def json(cls, obj: Any, status: int = 200) -> Response:
        return cls(status=status, body=json.dumps(obj).encode(), content_type="application/json")

    @classmethod
    def sse(cls, gen: AsyncIterator) -> Response:
        return cls(status=200, content_type="text/event-stream", stream=gen)


_STATUS = {
    200: "OK",
    201: "Created",
    202: "Accepted",
    400: "Bad Request",
    404: "Not Found",
    408: "Request Timeout",
    413: "Payload Too Large",
    500: "Internal Server Error"
}

Handler = Callable[[Request], Awaitable[Response]]


class _BadRequest(Exception):
    pass


class _TooLarge(Exception):
    pass


async def _read_request(reader: asyncio.StreamReader, max_body: int) -> Request:
    head = await reader.readuntil(b"\r\n\r\n")
    lines = head.decode("latin1").split("\r\n")
    parts = lines[0].split(" ", 2)
    if len(parts) < 2:
        raise _BadRequest("malformed request line")
    method, target = parts[0], parts[1]
    path, _, qs = target.partition("?")
    query = dict(kv.split("=", 1) for kv in qs.split("&") if "=" in kv)
    headers: dict[str, str] = {}
    for ln in lines[1:]:
        if ":" in ln:
            k, v = ln.split(":", 1)
            headers[k.strip().lower()] = v.strip()
    raw = headers.get("content-length", "0") or "0"
    try:
        clen = int(raw)
    except ValueError:
        raise _BadRequest(f"invalid Content-Length {raw!r}")
    if clen < 0 or clen > max_body:
        raise _TooLarge(f"Content-Length {clen} exceeds limit {max_body}")
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

    def __init__(self,
                 dispatch: Handler,
                 *,
                 read_timeout: float = 30.0,
                 max_body_bytes: int = 64 << 20,
                 header_limit: int = 1 << 20):
        self.dispatch = dispatch
        self.read_timeout = read_timeout  # slowloris guard
        self.max_body_bytes = max_body_bytes  # body-size cap → 413
        self.header_limit = header_limit  # StreamReader buffer for the header block
        self._server: asyncio.AbstractServer | None = None

    async def _conn(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            req = await asyncio.wait_for(_read_request(reader, self.max_body_bytes), self.read_timeout)
            resp = await self.dispatch(req)
        except asyncio.TimeoutError:
            resp = Response.json({"error": "request read timeout"}, status=408)
        except _BadRequest as e:
            resp = Response.json({"error": str(e)}, status=400)
        except _TooLarge as e:
            resp = Response.json({"error": str(e)}, status=413)
        except (asyncio.IncompleteReadError, asyncio.LimitOverrunError):
            resp = Response.json({"error": "malformed or oversized request"}, status=400)
        except Exception as e:  # noqa: BLE001 — never let one connection kill the server
            resp = Response.json({"error": str(e)}, status=500)
        try:
            await _write(writer, resp)
        except Exception:
            if resp.stream is not None:  # client gone mid-SSE → close the generator
                try:
                    await resp.stream.aclose()  # → GeneratorExit → submit() cancels the driver
                except Exception:
                    pass
            try:
                writer.close()
            except Exception:
                pass

    async def start(self, host: str = "127.0.0.1", port: int = 0) -> tuple[str, int]:
        self._server = await asyncio.start_server(self._conn, host, port, limit=self.header_limit)
        sock = self._server.sockets[0].getsockname()
        return sock[0], sock[1]

    async def close(self) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
