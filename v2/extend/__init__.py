"""Extension plane — observers (read-only) and interceptors (compute-altering) (design_v3 §11)."""
from __future__ import annotations

from .base import Interceptor, InterceptorChain, InterceptorConflict, Observer, ObserverBus
from .interceptors import ResidualSkipInterceptor
from .observers import NaNWatch, Profiler
from .registry import PluginRegistry

__all__ = ["Observer", "Interceptor", "ObserverBus", "InterceptorChain", "InterceptorConflict",
           "Profiler", "NaNWatch", "ResidualSkipInterceptor", "PluginRegistry"]
