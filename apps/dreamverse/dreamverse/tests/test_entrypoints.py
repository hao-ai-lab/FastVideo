from __future__ import annotations
# pyright: reportAttributeAccessIssue=false, reportMissingImports=false

import sys
from pathlib import Path
import types


SERVER_DIR = Path(__file__).resolve().parent
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

import main as server_main  # noqa: E402
import mock_server  # noqa: E402


def _run_cli(module, monkeypatch, argv: list[str]) -> list[dict[str, object]]:
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

    module.cli()
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
    calls = _run_cli(
        mock_server,
        monkeypatch,
        ["dreamverse-mock-server"],
    )

    assert calls == [
        {
            "app": mock_server.app,
            "host": "0.0.0.0",
            "port": 8009,
        }
    ]


def test_mock_server_cli_updates_latency(monkeypatch):
    old_latency_ms = mock_server.LATENCY_MS
    try:
        calls = _run_cli(
            mock_server,
            monkeypatch,
            ["dreamverse-mock-server", "--latency", "321", "--port", "8111"],
        )

        assert calls == [
            {
                "app": mock_server.app,
                "host": "0.0.0.0",
                "port": 8111,
            }
        ]
        assert mock_server.LATENCY_MS == 321
    finally:
        mock_server.LATENCY_MS = old_latency_ms
