# SPDX-License-Identifier: Apache-2.0
"""Per-request community LoRA selection for the OpenAI-compatible video API.

The server owns one generator and FastVideo keeps the active adapter on its pipeline, so a job applies its adapter
under the same lock as its generation. Strength 0 is exactly the base model, so a job without ``lora`` that follows
one with it runs at strength 0.
"""
from __future__ import annotations

import math
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastvideo import envs
from fastvideo.api.schema import GeneratorConfig, LoRAAdapterConfig, ServeConfig, ServeLoRAConfig
from fastvideo.layers.quantization.nvfp4_config import H3_NVFP4_DIT_EXPORT_FILENAME
from fastvideo.logger import init_logger
from fastvideo.models.loader.lora_patch import DenseLoRAPatch
from fastvideo.utils import maybe_download_lora, maybe_download_model

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
    weight to merge into. A Hugging Face model id is resolved to its snapshot first, fetching only the packed
    transformer file.
    """
    if envs.FASTVIDEO_LORA_SIDE_BRANCH.get():
        return True
    quantization = generator.engine.quantization
    if quantization is None or quantization.transformer_quant != "NVFP4":
        return False
    export = f"transformer/{H3_NVFP4_DIT_EXPORT_FILENAME}"
    try:
        model_dir = maybe_download_model(generator.model_path, allow_patterns=[export])
    except ValueError:
        logger.warning("Could not resolve %s to check for a packed NVFP4 export", generator.model_path, exc_info=True)
        return False
    return (Path(model_dir) / export).is_file()


def check_adapter_file(adapter: LoRAAdapterConfig) -> None:
    """Raise ``ValueError`` for an adapter the pipeline would refuse before changing any weight.

    The adapter must resolve to a safetensors file, and must carry no dense payload, which can only be applied
    when the model loads.
    """
    local_path = maybe_download_lora(adapter.path)
    if not os.path.isfile(local_path):
        raise ValueError(f"LoRA adapter {adapter.id!r} resolves to {local_path!r}, which is not a file")
    if DenseLoRAPatch.from_adapter(local_path) is not None:
        raise ValueError(f"LoRA adapter {adapter.id!r} carries dense replacement tensors; only low-rank adapters "
                         "can change on a running server")


def build_lora_runtime(serve_config: ServeConfig) -> LoraRuntime | None:
    """The runtime for the serve config's allow-list, or ``None`` without one; raises ``ValueError`` at startup.

    Per-request adapters are served only where the quantized linears take adapters as side branches, so a switch
    rewrites only the few linears a packed export leaves unquantized; the pipeline restores those from their original
    weights. Every adapter is checked here so that an unusable one stops the server.
    """
    if serve_config.lora is None or not serve_config.lora.adapters:
        return None
    if not adapters_attach_as_side_branch(serve_config.generator):
        raise ValueError("Per-request LoRA adapters need side-branch LoRA layers: serve a packed NVFP4 export "
                         "(transformer_quant: NVFP4) or set FASTVIDEO_LORA_SIDE_BRANCH=1.")
    runtime = LoraRuntime(serve_config.lora)
    for adapter in serve_config.lora.adapters:
        check_adapter_file(adapter)
    return runtime


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
    check_adapter: Callable[[LoRAAdapterConfig], None] = check_adapter_file
    current_id: str | None = None
    current_strength: float | None = None
    # Set only when clearing after a failed change failed too; the next successful change or clear resets it.
    unknown_state: bool = False
    adapters: dict[str, LoRAAdapterConfig] = field(init=False)
    loaded: set[str] = field(init=False, default_factory=set)

    def __post_init__(self) -> None:
        self.adapters = {adapter.id: adapter for adapter in self.config.adapters}

    def info(self) -> dict[str, Any]:
        """The ``lora`` block of ``GET /v1/model_info``."""
        return {
            "supported": True,
            "runtime_switching": True,
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

        Raises when the generator cannot be cleared back to the base model after a failed change, so the server
        never renders with an adapter nobody asked for.
        """
        if selection is None:
            if self.unknown_state:
                self._clear(generator)
                if self.unknown_state:
                    raise RuntimeError("A failed LoRA adapter change left this server's adapter state unknown, and "
                                       "clearing the adapter to restore the base model failed too.")
                return
            if self.current_id is None or self.current_strength == 0.0:
                return
            self._set(generator, self.current_id, 0.0)
            return
        if not self.unknown_state and self.current_id == selection.id and self.current_strength == selection.strength:
            return
        self._set(generator, selection.id, selection.strength)

    def _set(self, generator: Any, adapter_id: str, strength: float) -> None:
        if adapter_id not in self.loaded:
            try:
                self.check_adapter(self.adapters[adapter_id])
            except ValueError as error:
                logger.warning("LoRA adapter %r failed its pre-load check: %s", adapter_id, error)
                raise LoraRequestError(f"LoRA adapter {adapter_id!r} cannot be loaded on this server.") from error
        logger.info("OpenAI server applies LoRA adapter %r at strength %g", adapter_id, strength)
        try:
            generator.set_lora_adapter(adapter_id, self.adapters[adapter_id].path, strength=strength)
        except (BrokenPipeError, EOFError):
            # A dead worker: the serving engine marks itself unhealthy, and a clear would only reach the same worker.
            raise
        except Exception as error:
            logger.exception("LoRA adapter %r could not be applied", adapter_id)
            # A failed change may keep the previous adapter or leave a partial one; clearing gives the base model.
            self._clear(generator)
            raise LoraRequestError(f"LoRA adapter {adapter_id!r} could not be applied on this server.") from error
        self.loaded.add(adapter_id)
        self.current_id = adapter_id
        self.current_strength = strength
        self.unknown_state = False

    def _clear(self, generator: Any) -> None:
        self.current_id = None
        self.current_strength = None
        try:
            generator.clear_lora_adapter()
        except (BrokenPipeError, EOFError):
            self.unknown_state = True
            raise
        except Exception:
            logger.exception("OpenAI server could not clear the LoRA adapter")
            self.unknown_state = True
            return
        self.unknown_state = False
