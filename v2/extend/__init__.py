"""Extension plane — observers (read-only) and interceptors (compute-altering)."""
from __future__ import annotations

from v2.extend.base import Interceptor, InterceptorChain, InterceptorConflict, Observer, ObserverBus
from v2.extend.interceptors import ResidualSkipInterceptor
from v2.extend.observers import NaNWatch
from v2.extend.registry import PluginRegistry

__all__ = [
    "Observer", "Interceptor", "ObserverBus", "InterceptorChain", "InterceptorConflict", "NaNWatch",
    "ResidualSkipInterceptor", "PluginRegistry"
]
