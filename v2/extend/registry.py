"""Plugin registry + trust boundary.

Plugins are enabled at deploy scope only (never a per-request ``plugins=[...]`` field that
would wire third-party code selection into the public API); requests only *parameterize*
pre-enabled plugins through validated schemas.
"""
from __future__ import annotations

from typing import Any
from collections.abc import Callable

from v2.extend.base import InterceptorChain, Observer


class PluginRegistry:

    def __init__(self) -> None:
        self._interceptors: dict[str, Callable[..., Any]] = {}
        self._observers: dict[str, Callable[..., Observer]] = {}
        self._enabled: set[str] = set()  # deploy-scope enablement

    def register_interceptor(self, plugin_id: str, factory: Callable[..., Any]) -> None:
        self._interceptors[plugin_id] = factory

    def register_observer(self, plugin_id: str, factory: Callable[..., Observer]) -> None:
        self._observers[plugin_id] = factory

    def enable(self, plugin_id: str) -> None:
        if plugin_id not in self._interceptors and plugin_id not in self._observers:
            raise KeyError(f"plugin {plugin_id!r} not registered")
        self._enabled.add(plugin_id)

    def build_chain(self, params: dict[str, dict] | None = None, *, exact_mode: bool = False) -> InterceptorChain:
        params = params or {}
        chain = []
        for pid in self._enabled:
            if pid in self._interceptors:
                chain.append(self._interceptors[pid](**params.get(pid, {})))
        return InterceptorChain(chain, exact_mode=exact_mode)
