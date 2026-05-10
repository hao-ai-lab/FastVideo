"""Runtime service singletons.

Assigned by ``main.lifespan`` at server startup and read by routes and the
session controller via attribute access (``runtime.gpu_pool``). Do NOT import
these names directly (``from runtime import gpu_pool``) — ``from``-import
copies the current binding, freezing it at ``None`` before lifespan runs.
"""
# pyright: reportMissingImports=false
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dreamverse.gpu_pool import GPUPool
    from dreamverse.session_logger import SessionEventLogger
    from dreamverse.prompt_enhancer import PromptEnhancer
    from dreamverse.prompt_safety import PromptSafetyFilter

gpu_pool: GPUPool | None = None
prompt_enhancer: PromptEnhancer | None = None
session_event_logger: SessionEventLogger | None = None
prompt_safety_filter: PromptSafetyFilter | None = None
