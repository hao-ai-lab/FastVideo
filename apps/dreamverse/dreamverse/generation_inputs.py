"""GPU-independent validation for generation modes and ordered asset handles."""

from __future__ import annotations

from dataclasses import dataclass

from PIL import Image

from dreamverse.assets import asset_store

GENERATION_MODES = ("t2va", "fl2va", "ref2va")


@dataclass(frozen=True)
class GenerationAsset:
    asset_id: str
    kind: str
    path: str
    role: str


@dataclass(frozen=True)
class GenerationInputs:
    mode: str | None = None
    assets: tuple[GenerationAsset, ...] = ()

    @property
    def first_frame_path(self) -> str | None:
        return next((asset.path for asset in self.assets if asset.role == "first_frame"), None)

    @property
    def last_frame_path(self) -> str | None:
        return next((asset.path for asset in self.assets if asset.role == "last_frame"), None)

    @property
    def references(self) -> tuple[GenerationAsset, ...]:
        return tuple(asset for asset in self.assets if asset.role == "reference")


def supported_generation_modes(model_id: str) -> tuple[str, ...]:
    return GENERATION_MODES if model_id in ("full-h3", "mock") else ("t2va", )


def resolve_generation_inputs(payload: dict, model_id: str) -> GenerationInputs:
    mode = payload.get("generation_mode")
    raw_assets = payload.get("conditioning_assets", [])
    if mode is None and "generation_mode" not in payload:
        if raw_assets:
            raise ValueError("Select a generation mode before attaching conditioning assets.")
        return GenerationInputs()
    if not isinstance(mode, str) or mode not in GENERATION_MODES:
        raise ValueError("Unknown generation mode. Choose T2VA, FL2VA, or Ref2VA.")
    if mode not in supported_generation_modes(model_id):
        raise ValueError(f"{mode.upper()} requires the Full H3 runtime. This runtime is running {model_id}.")
    if payload.get("initial_image") is not None:
        raise ValueError("Use asset IDs for generation modes; do not combine them with the legacy initial_image field.")
    if not isinstance(raw_assets, list) or len(raw_assets) > 12:
        raise ValueError("conditioning_assets must be an ordered list with at most 12 assets.")
    if mode == "t2va" and raw_assets:
        raise ValueError("T2VA accepts text only. Remove conditioning assets or choose another mode.")
    assets: list[GenerationAsset] = []
    for item in raw_assets:
        if not isinstance(item, dict) or set(item) != {"asset_id", "role"}:
            raise ValueError("Each conditioning asset must contain only asset_id and role.")
        role = item["role"]
        if role not in ("first_frame", "last_frame", "reference"):
            raise ValueError("Asset role must be first_frame, last_frame, or reference.")
        stored = asset_store.get(item["asset_id"])
        assets.append(GenerationAsset(stored.asset_id, stored.kind, stored.path, role))
    if mode == "fl2va":
        if any(asset.kind != "image" or asset.role == "reference" for asset in assets):
            raise ValueError("FL2VA accepts only first-frame and last-frame images.")
        if sum(asset.role == "first_frame" for asset in assets) != 1:
            raise ValueError("FL2VA requires exactly one first-frame image.")
        if sum(asset.role == "last_frame" for asset in assets) > 1:
            raise ValueError("FL2VA accepts at most one last-frame image.")
    elif mode == "ref2va":
        if not assets or any(asset.role != "reference" for asset in assets):
            raise ValueError("Ref2VA requires an ordered list of reference assets, without keyframe roles.")
        if not any(asset.kind in ("image", "video") for asset in assets):
            raise ValueError("Ref2VA requires at least one image or video; audio alone is not supported.")
        for kind, limit in (("image", 9), ("video", 3), ("audio", 3)):
            if sum(asset.kind == kind for asset in assets) > limit:
                raise ValueError(f"Ref2VA accepts at most {limit} {kind} references.")
        for asset in assets:
            if asset.kind == "image":
                with Image.open(asset.path) as image:
                    if image.width > 4 * image.height or image.height > 4 * image.width:
                        raise ValueError(
                            "Ref2VA image aspect ratios must be between 1:4 and 4:1. Crop this image first.")
    return GenerationInputs(mode, tuple(assets))


def pin_generation_inputs(inputs: GenerationInputs) -> None:
    asset_store.pin([asset.asset_id for asset in inputs.assets])


def release_generation_inputs(inputs: GenerationInputs) -> None:
    asset_store.release([asset.asset_id for asset in inputs.assets])
