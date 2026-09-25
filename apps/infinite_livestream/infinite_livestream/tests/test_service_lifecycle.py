"""Service failures must reach the caller after the other tasks are stopped."""

import asyncio
from unittest.mock import AsyncMock, Mock

import pytest

from infinite_livestream import main
from infinite_livestream.backend import FastH3Backend


@pytest.mark.parametrize("failure", ["engine", "pacer", None])
def test_service_propagates_failure_and_cleans_up(app_config, monkeypatch, failure):
    error = RuntimeError(f"{failure} startup failed")

    def load():
        if failure == "engine":
            raise error

    async def wait_forever():
        await asyncio.Event().wait()

    async def run_pacer():
        if failure == "pacer":
            raise error
        await wait_forever()

    async def run_web():
        if failure is None:
            # Normal server shutdown still exits successfully.
            await asyncio.sleep(0)
            return
        await wait_forever()

    sink = Mock(stop=AsyncMock())
    web = Mock(run=run_web)
    pacer = Mock(run=run_pacer)
    monkeypatch.setattr(FastH3Backend, "load", lambda self: load())
    monkeypatch.setattr(main, "HlsSink", Mock(return_value=sink))
    monkeypatch.setattr(main, "DemoWeb", Mock(return_value=web))
    monkeypatch.setattr(main, "Pacer", Mock(return_value=pacer))
    monkeypatch.setattr(main, "PromptUpsampler", Mock())
    monkeypatch.setattr(main, "Moderator", Mock())

    async def run():
        if failure is None:
            await main.serve(app_config)
        else:
            with pytest.raises(RuntimeError) as caught:
                await main.serve(app_config)
            assert caught.value is error
        sink.stop.assert_awaited_once()
        assert asyncio.all_tasks() == {asyncio.current_task()}

    asyncio.run(run())
