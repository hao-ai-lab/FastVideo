from __future__ import annotations

from dataclasses import dataclass

from dreamverse.config import MODEL_REGISTRY

# Canonical upstream wire IDs. FL2VA is tracked in #1834 but not wired on Dreamverse
# streaming backends yet.
LTX_LOBBY_GENERATION_MODES = frozenset({"t2va", "ref2va"})
H3_LOBBY_GENERATION_MODES = frozenset({"t2va", "ref2va"})

LTX_LOBBY_ASPECT_RATIOS = frozenset({"21:9", "16:9", "4:3", "1:1", "3:4", "9:16"})

# Realtime FastLTX serving is validated through 1080p-class outputs; 4K is rejected
# until the runtime path is tested on Dreamverse GPUs.
LTX_LOBBY_RESOLUTIONS = frozenset({"480p", "720p", "1080p"})

# FastH3 serves a fixed 768x1344 (16:9-class) output; lobby resolution is nominal.
H3_LOBBY_ASPECT_RATIOS = frozenset({"16:9"})
H3_LOBBY_RESOLUTIONS = frozenset({"720p"})

LOBBY_DURATION_SEC = frozenset({5, 10, 15})

UNSUPPORTED_GENERATION_MODE_MESSAGES = {
    "fl2va": "First/last frame mode (FL2VA) is not supported yet.",
}


@dataclass(frozen=True)
class ModelCreationCapabilities:
    generation_modes: frozenset[str]
    aspect_ratios: frozenset[str]
    resolutions: frozenset[str]
    duration_sec: frozenset[int]
    unsupported_generation_modes: frozenset[str] = frozenset({"fl2va"})

    def as_dict(self) -> dict[str, object]:
        unsupported = {
            mode: UNSUPPORTED_GENERATION_MODE_MESSAGES[mode]
            for mode in sorted(self.unsupported_generation_modes)
            if mode in UNSUPPORTED_GENERATION_MODE_MESSAGES
        }
        return {
            "generation_modes": sorted(self.generation_modes),
            "aspect_ratios": sorted(self.aspect_ratios),
            "resolutions": sorted(self.resolutions),
            "duration_sec": sorted(self.duration_sec),
            "unsupported_generation_modes": unsupported,
            "reference_assets": {
                "mime_types": ["image/png", "image/jpeg", "image/webp"],
                "max_bytes": 15 * 1024 * 1024,
            },
        }


LTX_MODEL_CREATION_CAPABILITIES = ModelCreationCapabilities(
    generation_modes=LTX_LOBBY_GENERATION_MODES,
    aspect_ratios=LTX_LOBBY_ASPECT_RATIOS,
    resolutions=LTX_LOBBY_RESOLUTIONS,
    duration_sec=LOBBY_DURATION_SEC,
)

H3_MODEL_CREATION_CAPABILITIES = ModelCreationCapabilities(
    generation_modes=H3_LOBBY_GENERATION_MODES,
    aspect_ratios=H3_LOBBY_ASPECT_RATIOS,
    resolutions=H3_LOBBY_RESOLUTIONS,
    duration_sec=LOBBY_DURATION_SEC,
)

MODEL_CREATION_CAPABILITIES: dict[str, ModelCreationCapabilities] = {
    "fast-ltx2": LTX_MODEL_CREATION_CAPABILITIES,
    "fast-ltx23": LTX_MODEL_CREATION_CAPABILITIES,
    "fast-h3": H3_MODEL_CREATION_CAPABILITIES,
}


def capabilities_for_model(model_id: str) -> ModelCreationCapabilities:
    if model_id not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model_id: {model_id}")
    return MODEL_CREATION_CAPABILITIES.get(model_id, LTX_MODEL_CREATION_CAPABILITIES)


def lobby_capabilities_as_dict() -> dict[str, object]:
    model_ids = sorted(MODEL_REGISTRY.keys())
    models = {model_id: capabilities_for_model(model_id).as_dict() for model_id in model_ids}
    union_modes: set[str] = set()
    union_aspects: set[str] = set()
    union_resolutions: set[str] = set()
    union_durations: set[int] = set()
    for caps in MODEL_CREATION_CAPABILITIES.values():
        union_modes.update(caps.generation_modes)
        union_aspects.update(caps.aspect_ratios)
        union_resolutions.update(caps.resolutions)
        union_durations.update(caps.duration_sec)
    return {
        "model_ids": model_ids,
        "models": models,
        "generation_modes": sorted(union_modes),
        "aspect_ratios": sorted(union_aspects),
        "resolutions": sorted(union_resolutions),
        "duration_sec": sorted(union_durations),
        "unsupported_generation_modes": dict(UNSUPPORTED_GENERATION_MODE_MESSAGES),
        "reference_assets": {
            "mime_types": ["image/png", "image/jpeg", "image/webp"],
            "max_bytes": 15 * 1024 * 1024,
        },
    }


# Backward-compatible alias used in tests.
LOBBY_CREATION_CAPABILITIES = lobby_capabilities_as_dict()


def validate_lobby_creation_config(
    *,
    model_id: str,
    generation_mode: str,
    aspect_ratio: str,
    resolution: str,
    duration_sec: int,
) -> None:
    if model_id not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model_id: {model_id}")

    caps = capabilities_for_model(model_id)

    if generation_mode in caps.unsupported_generation_modes:
        raise ValueError(UNSUPPORTED_GENERATION_MODE_MESSAGES[generation_mode])
    if generation_mode not in caps.generation_modes:
        raise ValueError(f"Unsupported generation_mode: {generation_mode}")

    if aspect_ratio not in caps.aspect_ratios:
        raise ValueError(f"Unsupported aspect_ratio: {aspect_ratio}")
    if resolution not in caps.resolutions:
        raise ValueError(f"Unsupported resolution: {resolution}")
    if duration_sec not in caps.duration_sec:
        raise ValueError("duration_sec must be 5, 10, or 15.")
