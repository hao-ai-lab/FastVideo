# SPDX-License-Identifier: Apache-2.0
"""Per-request community LoRA selection for the OpenAI-compatible video API.

The server owns one generator and FastVideo keeps the active adapter on its pipeline, so a job applies its adapter
under the same lock as its generation. Strength 0 is exactly the base model, so a job without ``lora`` that follows
one with it runs at strength 0.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastvideo import envs
from fastvideo.api.schema import GeneratorConfig, LoRAAdapterConfig, ServeLoRAConfig
from fastvideo.layers.quantization.nvfp4_config import H3_NVFP4_DIT_EXPORT_FILENAME
from fastvideo.logger import init_logger

logger = init_logger(__name__)

_NAME_KEYS = ("name", "lora_name", "adapter")
_PATH_KEYS = ("path", "lora_path", "local_path")
_SCALE_KEYS = ("scale", "lora_scale")


class LoraRequestError(ValueError):
    """A request's LoRA selection cannot be served; the message is the client-facing reason."""


@dataclass(frozen=True)
class LoraSelection:
    """A validated request selection with its strength resolved."""
    id: str
    strength: float


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


def _first(body: dict[str, Any], keys: tuple[str, ...]) -> Any:
    return next((body[key] for key in keys if body.get(key) is not None), None)


def _adapter_info(adapter: LoRAAdapterConfig) -> dict[str, Any]:
    """What a client may select; never the local load path."""
    entry: dict[str, Any] = {
        "id": adapter.id,
        "default_strength": adapter.default_strength,
        "min_strength": adapter.min_strength,
        "max_strength": adapter.max_strength,
    }
    for key in ("description", "license", "source_url"):
        if getattr(adapter, key):
            entry[key] = getattr(adapter, key)
    return entry


@dataclass
class LoraRuntime:
    """The adapters a request may select, and the adapter the generator currently carries."""
    config: ServeLoRAConfig
    runtime_switching: bool = False
    current_id: str | None = None
    current_strength: float | None = None
    # Set when a failed change left the generator's adapter state unknown; cleared by the next successful change.
    unknown_state: bool = False
    adapters: dict[str, LoRAAdapterConfig] = field(init=False)

    def __post_init__(self) -> None:
        self.adapters = {adapter.id: adapter for adapter in self.config.adapters}

    def info(self) -> dict[str, Any]:
        """The ``lora`` block of ``GET /v1/model_info``."""
        return {
            "supported": True,
            "runtime_switching": self.runtime_switching,
            "base_strength": 0.0,
            "adapters": [_adapter_info(adapter) for adapter in self.config.adapters],
        }

    def resolve(self, body: dict[str, Any] | None) -> LoraSelection | None:
        """Validate a request's ``lora`` object against the allow-list."""
        if body is None:
            return None
        if _first(body, _PATH_KEYS) is not None:
            raise LoraRequestError("lora.path is not accepted; select an adapter by name from GET /v1/model_info.")
        name = _first(body, _NAME_KEYS)
        if name is None:
            raise LoraRequestError("lora must provide a name")
        adapter = self.adapters.get(str(name))
        if adapter is None:
            known = ", ".join(sorted(self.adapters)) or "none"
            raise LoraRequestError(f"Unknown LoRA adapter {str(name)!r}; this server serves: {known}.")
        scale = _first(body, _SCALE_KEYS)
        try:
            strength = adapter.default_strength if scale is None else float(scale)
        except (TypeError, ValueError) as error:
            raise LoraRequestError(f"Invalid LoRA scale {scale!r}") from error
        if not math.isfinite(strength) or not adapter.min_strength <= strength <= adapter.max_strength:
            raise LoraRequestError(f"LoRA adapter {adapter.id!r} takes a scale from {adapter.min_strength:g} to "
                                   f"{adapter.max_strength:g}; got {strength:g}.")
        return LoraSelection(adapter.id, strength)

    def apply(self, generator: Any, selection: LoraSelection | None) -> None:
        """Make the selection current on ``generator``; ``None`` sets the current adapter to strength 0.

        Raises when a failed change left the adapter state unknown and it cannot be brought back to the base
        model, so the server never renders with an adapter nobody asked for.
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
        if not self.unknown_state and self.current_id == selection.id and self.current_strength == selection.strength:
            return
        self._set(generator, selection.id, selection.strength)

    def _set(self, generator: Any, adapter_id: str, strength: float) -> None:
        logger.info("OpenAI server applies LoRA adapter %r at strength %g", adapter_id, strength)
        previous_id = None if self.unknown_state else self.current_id
        try:
            generator.set_lora_adapter(adapter_id, self.adapters[adapter_id].path, strength=strength)
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
                logger.exception("OpenAI server could not set LoRA adapter %r to strength 0", adapter_id)
                continue
            self.current_id = adapter_id
            self.current_strength = 0.0
            self.unknown_state = False
            return True
        return False
