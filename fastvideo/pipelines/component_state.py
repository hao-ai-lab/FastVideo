# SPDX-License-Identifier: Apache-2.0
"""Runtime state of a pipeline's components: where each one was loaded from, and whether it is resident.

A pipeline owns one ``ComponentState``. ``ComposedPipelineBase`` records each component's path after loading it, and
attaches the state to every stage it registers (``PipelineStage.component_state``), so that a stage can reload a
component that it released to save memory.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ComponentState:
    """Mutable per-pipeline component bookkeeping that is not configuration."""

    # Local path that each component was loaded from, keyed by module name.
    model_paths: dict[str, str] = field(default_factory=dict)
    # Whether each releasable component is currently loaded, keyed by module name.
    model_loaded: dict[str, bool] = field(default_factory=lambda: {
        "transformer": True,
        "vae": True,
        "upsampler": True,
    })
