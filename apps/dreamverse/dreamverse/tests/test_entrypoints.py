from __future__ import annotations
# pyright: reportAttributeAccessIssue=false, reportMissingImports=false

import sys
import types


import dreamverse.main as server_main
import dreamverse.mock_server as mock_server


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
    monkeypatch.setattr("dreamverse._deps.require_dreamverse_runtime_deps", lambda: None)
    monkeypatch.setattr(mock_server, "require_dreamverse_runtime_deps", lambda: None)
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
