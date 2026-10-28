# SPDX-License-Identifier: Apache-2.0
"""Per-request LoRA selection for the streaming_v2 server.

The server owns one generator and FastVideo keeps the active adapter on its pipeline, so a request applies its
adapter under the same lock as its generation. A request carries ``{"lora": {"id": ..., "strength": ...}}``, and
strength 0 is exactly the base model, so a request without ``lora`` after one with it is served at strength 0.

@module fastvideo.entrypoints.streaming_v2.lora
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastvideo import envs
from fastvideo.api.schema import GeneratorConfig, LoRAAdapterConfig, StreamingV2LoRAConfig
from fastvideo.entrypoints.streaming_v2.messages import LoraSelection
from fastvideo.layers.quantization.nvfp4_config import H3_NVFP4_DIT_EXPORT_FILENAME
from fastvideo.logger import init_logger

logger = init_logger(__name__)


class LoraRequestError(ValueError):
    """A request's LoRA selection cannot be served; the message is the client-facing reason."""


def _adapter_capability(adapter: LoRAAdapterConfig) -> dict[str, Any]:
    """The capabilities entry of one adapter: what a client may select, never the local load path."""
    entry: dict[str, Any] = {
        "id": adapter.id,
        "default_strength": adapter.default_strength,
        "min_strength": adapter.min_strength,
        "max_strength": adapter.max_strength,
    }
    if adapter.description:
        entry["description"] = adapter.description
    if adapter.license:
        entry["license"] = adapter.license
    if adapter.source_url:
        entry["source_url"] = adapter.source_url
    return entry


def adapters_attach_as_side_branch(generator: GeneratorConfig) -> bool:
    """Whether adapters attach beside the base weights, so a request can change them on the running server.

    That holds with ``FASTVIDEO_LORA_SIDE_BRANCH=1`` and for a packed NVFP4 export, whose linears have no dense
    weight to merge into. Any other checkpoint merges the adapter into dense weights, or has them purged, and
    rejects a change.
    """
    if envs.FASTVIDEO_LORA_SIDE_BRANCH.get():
        return True
    quantization = generator.engine.quantization
    if quantization is None or quantization.transformer_quant != "NVFP4":
        return False
    return (Path(generator.model_path) / "transformer" / H3_NVFP4_DIT_EXPORT_FILENAME).is_file()


def lora_capabilities(runtime: LoraRuntime | None) -> dict[str, Any]:
    """The ``lora`` block of ``GET /v1/streamv2/capabilities``."""
    if runtime is None:
        return {"supported": False, "runtime_switching": False, "adapters": []}
    return {
        "supported": True,
        "runtime_switching": runtime.runtime_switching,
        "base_strength": 0.0,
        "adapters": [_adapter_capability(adapter) for adapter in runtime.config.adapters],
    }


@dataclass
class LoraRuntime:
    """The adapters a request may select, and the adapter the generator currently carries."""
    config: StreamingV2LoRAConfig
    runtime_switching: bool = False
    current_id: str | None = None
    current_strength: float | None = None
    # Set when a failed change left the generator's adapter state unknown; cleared by the next successful change.
    unknown_state: bool = False
    adapters: dict[str, LoRAAdapterConfig] = field(init=False)

    def __post_init__(self) -> None:
        self.adapters = {adapter.id: adapter for adapter in self.config.adapters}

    def validate(self, selection: LoraSelection | None) -> None:
        """Refuse an unknown adapter or a strength outside its range."""
        if selection is None:
            return
        adapter = self.adapters.get(selection.id)
        if adapter is None:
            known = ", ".join(sorted(self.adapters)) or "none"
            raise LoraRequestError(f"Unknown LoRA adapter {selection.id!r}; this server serves: {known}.")
        strength = adapter.default_strength if selection.strength is None else selection.strength
        if not math.isfinite(strength) or not adapter.min_strength <= strength <= adapter.max_strength:
            raise LoraRequestError(f"LoRA adapter {adapter.id!r} takes a strength from {adapter.min_strength:g} to "
                                   f"{adapter.max_strength:g}; got {strength:g}.")

    def apply(self, generator: Any, selection: LoraSelection | None) -> None:
        """Make the selection current on ``generator``; ``None`` sets the current adapter to strength 0.

        Raises ``RuntimeError`` for a request without ``lora`` when the generator's adapter state is unknown and
        cannot be brought back to the base model, so the server never renders with an adapter nobody asked for.
        """
        if selection is None:
            if self.unknown_state:
                if not self._reset_to_base(generator, list(self.adapters)):
                    raise RuntimeError("A failed LoRA adapter change left this server's adapter state unknown, and no "
                                       "adapter could be set to strength 0 to restore the base model.")
                return
            if self.current_id is None or self.current_strength == 0.0:
                return
            self._set(generator, self.current_id, 0.0)
            return
        self.validate(selection)
        adapter = self.adapters[selection.id]
        strength = adapter.default_strength if selection.strength is None else selection.strength
        if not self.unknown_state and self.current_id == adapter.id and self.current_strength == strength:
            return
        self._set(generator, adapter.id, strength)

    def _set(self, generator: Any, adapter_id: str, strength: float) -> None:
        adapter = self.adapters[adapter_id]
        logger.info("streaming_v2 applies LoRA adapter %r at strength %g", adapter_id, strength)
        previous_id = None if self.unknown_state else self.current_id
        try:
            generator.set_lora_adapter(adapter_id, adapter.path, strength=strength)
        except Exception as error:
            # The failed change may keep the previous adapter active or leave a partial one.
            self.current_id = None
            self.current_strength = None
            self.unknown_state = True
            if previous_id is not None:
                self._reset_to_base(generator, [previous_id])
            raise LoraRequestError(f"LoRA adapter {adapter_id!r} could not be applied: {error}") from error
        self.current_id = adapter_id
        self.current_strength = strength
        self.unknown_state = False

    def _reset_to_base(self, generator: Any, candidates: list[str]) -> bool:
        """Re-apply the first candidate that loads at strength 0, which is exactly the base model."""
        for adapter_id in candidates:
            try:
                generator.set_lora_adapter(adapter_id, self.adapters[adapter_id].path, strength=0.0)
            except Exception:
                logger.exception("streaming_v2 could not set LoRA adapter %r to strength 0", adapter_id)
                continue
            self.current_id = adapter_id
            self.current_strength = 0.0
            self.unknown_state = False
            return True
        return False
