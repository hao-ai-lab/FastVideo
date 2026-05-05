from __future__ import annotations
# pyright: reportAttributeAccessIssue=false, reportMissingImports=false

import sys
from pathlib import Path
import types

import fastvideo.entrypoints.streaming.mock_server as mock_server


SERVER_DIR = Path(__file__).resolve().parent
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

import main as server_main  # noqa: E402


def _run_cli(module, monkeypatch, argv: list[str], entrypoint: str = "cli") -> list[dict[str, object]]:
    calls: list[dict[str, object]] = []
    uvicorn_stub = types.ModuleType("uvicorn")

    def run(app, host: str, port: int) -> None:
        calls.append(
            {
                "app": app,
                "host": host,
                "port": port,
            }
        )

    uvicorn_stub.run = run
    monkeypatch.setitem(sys.modules, "uvicorn", uvicorn_stub)
    monkeypatch.setattr(sys, "argv", argv)

    getattr(module, entrypoint)()
    return calls


def test_server_cli_defaults_to_local_web_port(monkeypatch):
    calls = _run_cli(server_main, monkeypatch, ["dreamverse-server"])

    assert calls == [
        {
            "app": server_main.app,
            "host": "0.0.0.0",
            "port": 8009,
        }
    ]


def test_server_cli_allows_explicit_host_and_port(monkeypatch):
    calls = _run_cli(
        server_main,
        monkeypatch,
        ["dreamverse-server", "--host", "127.0.0.1", "--port", "8123"],
    )

    assert calls == [
        {
            "app": server_main.app,
            "host": "127.0.0.1",
            "port": 8123,
        }
    ]


def test_mock_server_cli_defaults_to_local_web_port(monkeypatch):
    mock_app = object()
    build_calls: list[dict[str, object]] = []

    def build_mock_app(*, sleep_ms: float = 0.0):
        build_calls.append({"sleep_ms": sleep_ms})
        return mock_app

    monkeypatch.setattr(mock_server, "build_mock_app", build_mock_app)
    calls = _run_cli(
        mock_server,
        monkeypatch,
        ["dreamverse-mock-server"],
        entrypoint="main",
    )

    assert build_calls == [{"sleep_ms": 0.0}]
    assert calls == [
        {
            "app": mock_app,
            "host": "127.0.0.1",
            "port": 8000,
        }
    ]


def test_mock_server_cli_updates_latency(monkeypatch):
    mock_app = object()
    build_calls: list[dict[str, object]] = []

    def build_mock_app(*, sleep_ms: float = 0.0):
        build_calls.append({"sleep_ms": sleep_ms})
        return mock_app

    monkeypatch.setattr(mock_server, "build_mock_app", build_mock_app)
    calls = _run_cli(
        mock_server,
        monkeypatch,
        ["dreamverse-mock-server", "--sleep-ms", "321", "--port", "8111"],
        entrypoint="main",
    )

    assert build_calls == [{"sleep_ms": 321.0}]
    assert calls == [
        {
            "app": mock_app,
            "host": "127.0.0.1",
            "port": 8111,
        }
    ]
