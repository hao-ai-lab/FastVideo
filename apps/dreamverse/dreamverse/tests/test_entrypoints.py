from __future__ import annotations
# pyright: reportAttributeAccessIssue=false, reportMissingImports=false

import sys
import types

from fastapi import APIRouter
from fastapi.testclient import TestClient
import fastvideo.entrypoints.streaming as streaming_entrypoints
import pytest

def _install_stack03_import_stubs(monkeypatch):
    """Keep entrypoint tests focused while later-stack runtime modules are absent."""
    if not hasattr(streaming_entrypoints, "build_health_router"):
        monkeypatch.setattr(streaming_entrypoints, "build_health_router", lambda _pool=None: APIRouter(), raising=False)

    gpu_pool_stub = types.ModuleType("dreamverse.gpu_pool")

    class GPUPool:
        def __init__(self, _gpu_ids):
            pass

        async def initialize(self):
            pass

        async def shutdown(self):
            pass

    gpu_pool_stub.GPUPool = GPUPool
    gpu_pool_stub.get_available_gpus = lambda: []
    monkeypatch.setitem(sys.modules, "dreamverse.gpu_pool", gpu_pool_stub)

    session_logger_stub = types.ModuleType("dreamverse.session_logger")
    session_logger_stub.SessionEventLogger = lambda _path: object()
    monkeypatch.setitem(sys.modules, "dreamverse.session_logger", session_logger_stub)

    prompt_enhancer_stub = types.ModuleType("dreamverse.prompt_enhancer")
    prompt_enhancer_stub.PromptEnhancer = lambda: object()
    monkeypatch.setitem(sys.modules, "dreamverse.prompt_enhancer", prompt_enhancer_stub)

    prompt_safety_stub = types.ModuleType("dreamverse.prompt_safety")
    prompt_safety_stub.PromptSafetyFilter = lambda: object()
    monkeypatch.setitem(sys.modules, "dreamverse.prompt_safety", prompt_safety_stub)

    session_package_stub = types.ModuleType("dreamverse.session")
    session_package_stub.__path__ = []
    monkeypatch.setitem(sys.modules, "dreamverse.session", session_package_stub)

    controller_stub = types.ModuleType("dreamverse.session.controller")

    class SessionController:
        def __init__(self, **_kwargs):
            pass

        async def run(self):
            pass

    controller_stub.SessionController = SessionController
    monkeypatch.setitem(sys.modules, "dreamverse.session.controller", controller_stub)


def _import_server_main(monkeypatch):
    _install_stack03_import_stubs(monkeypatch)
    sys.modules.pop("dreamverse.main", None)
    import dreamverse.main as server_main

    return server_main


def _import_mock_server_or_skip():
    return pytest.importorskip("dreamverse.mock_server")


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
    if hasattr(module, "require_dreamverse_runtime_deps"):
        monkeypatch.setattr(module, "require_dreamverse_runtime_deps", lambda: None)
    monkeypatch.setattr(sys, "argv", argv)

    module.cli()
    return calls


def test_server_cli_defaults_to_local_web_port(monkeypatch):
    server_main = _import_server_main(monkeypatch)
    calls = _run_cli(server_main, monkeypatch, ["dreamverse-server"])

    assert calls == [
        {
            "app": server_main.app,
            "host": "0.0.0.0",
            "port": 8009,
        }
    ]


def test_server_cli_allows_explicit_host_and_port(monkeypatch):
    server_main = _import_server_main(monkeypatch)
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


def test_server_does_not_expose_backend_source_as_static_assets(monkeypatch):
    server_main = _import_server_main(monkeypatch)
    client = TestClient(server_main.app)

    response = client.get("/server-assets/main.py")

    assert response.status_code == 404


def test_mock_server_cli_defaults_to_local_web_port(monkeypatch):
    mock_server = _import_mock_server_or_skip()
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
    mock_server = _import_mock_server_or_skip()
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
