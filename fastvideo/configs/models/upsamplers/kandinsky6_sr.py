# SPDX-License-Identifier: Apache-2.0
"""Config of the Kandinsky6 SR latent-upscaler bank (``latent_upscaler/config.json``).

The component config is ``{"models": [{"target_scale": "4x" | "2x", "model": {...}}, ...], "scaling_factor": f}``.
Each ``model`` mapping describes one cascaded 2x+2x upsampler.  Only the architecture of the released checkpoints is
implemented, so every key that would select a different architecture must carry its released value; a value that is
not supported raises a ``ValueError`` naming the key instead of silently building something else.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from typing import Any

from fastvideo.configs.models.upsamplers.base import UpsamplerConfig

# key -> (required value, value a config that omits the key stands for).  The omitted-key values are the training
# code's defaults, so e.g. a config without ``input_skip`` means ``input_skip=True`` and is rejected.
_FIXED_KEYS: dict[str, tuple[Any, Any]] = {
    "upscale_factor": (4, 4),
    "dims": (3, 2),
    "temporal_padding": ("replicate", "zeros"),
    "upsample_mode": ("pxs_v2", "pixel_shuffle"),
    "upsample_padding_mode": ("zeros", "reflect"),
    "modulated_norm": (True, False),
    "modulated_output_proj": (True, False),
    "bare_stem": (True, False),
    "input_skip": (False, True),
    "global_skip": (False, False),
    "grn": (False, False),
    "layer_scale_init": (None, None),
    "depthwise": (False, False),
    "bottleneck_channels": (None, None),
    "motion_attention": (None, None),
    # Only read by depthwise blocks; the residual convs are always 3x3x3.
    "kernel_size": (3, 3),
}
# Fixed for entries built with the x2 entry (``enable_x2_entry=true``).
_X2_FIXED_KEYS: dict[str, tuple[Any, Any]] = {
    "x2_tail_mode": ("private_full", "shared"),
    "x2_finisher": ("pxs_residual", "none"),
}
# Accepted without effect at inference: training-time settings, and switches that only apply to the pixel-shuffle
# upsample modes (the pxs_v2 upsample has no temporal kernel and no ICNR init).
_IGNORED_KEYS = frozenset({
    "gradient_checkpointing", "stochastic_depth_rate", "loss_weight_2x", "loss_weight_4x", "temporal_mix", "icnr",
    "upsample_position"
})
_REQUIRED_INT_KEYS = ("in_channels", "hidden_channels", "num_pre_blocks", "num_mid_blocks", "num_post_blocks",
                      "expand_ratio")
_SUPPORTED_SCALES = (2, 4)


def _positive_int(where: str, key: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{where}: `{key}` must be a positive integer, got {value!r}")
    return value


def _parse_target_scale(where: str, value: Any) -> int:
    text = str(value).removesuffix("x")
    if isinstance(value, bool) or not text.isdigit() or int(text) not in _SUPPORTED_SCALES:
        raise ValueError(f"{where}: `target_scale` must be one of '2x', '4x', got {value!r}")
    return int(text)


@dataclass(frozen=True)
class Kandinsky6SRLatentUpscalerEntryConfig:
    """One bank entry: a cascaded 2x+2x upsampler serving ``target_scale``.

    ``stage_channels`` are the widths at the 1x, 2x and 4x latent grids.  With ``enable_x2_entry`` the entry also has
    a private x2 path (its own input stem, ``x2_adapter_blocks`` residual blocks at 1x, and private copies of the mid
    stage and second stage) that upsamples by 2 instead of 4.
    """

    target_scale: int
    in_channels: int
    hidden_channels: int
    stage_channels: tuple[int, int, int]
    num_pre_blocks: int
    num_mid_blocks: int
    num_post_blocks: int
    expand_ratio: int
    enable_x2_entry: bool = False
    x2_adapter_blocks: int = 0

    @classmethod
    def from_dict(cls, spec: Any, index: int = 0) -> Kandinsky6SRLatentUpscalerEntryConfig:
        """Validate one ``models[index]`` entry of ``latent_upscaler/config.json``."""
        where = f"latent upscaler models[{index}]"
        if not isinstance(spec, Mapping) or set(spec) != {"target_scale", "model"
                                                          } or not isinstance(spec["model"], Mapping):
            raise ValueError(f"{where} must be a mapping with exactly `target_scale` and a `model` mapping, "
                             f"got {spec!r}")
        target_scale = _parse_target_scale(where, spec["target_scale"])
        model = dict(spec["model"])

        if model.get("architecture") != "multi_scale":
            raise ValueError(f"{where}: `architecture` must be 'multi_scale', got {model.get('architecture')!r}")
        enable_x2_entry = model.get("enable_x2_entry", False)
        if not isinstance(enable_x2_entry, bool):
            raise ValueError(f"{where}: `enable_x2_entry` must be a boolean, got {enable_x2_entry!r}")

        fixed = dict(_FIXED_KEYS)
        if enable_x2_entry:
            fixed.update(_X2_FIXED_KEYS)
        for key, (required, omitted) in fixed.items():
            value = model.get(key, omitted)
            if value != required or type(value) is not type(required):
                raise ValueError(f"{where}: unsupported `{key}`={value!r}"
                                 f"{' (omitted)' if key not in model else ''}; only {required!r} is implemented")

        x2_defaults = {
            "x2_adapter_blocks": 0,
            "x2_adapter_sources": None,
            "x2_tail_mode": "shared",
            "x2_finisher": "none"
        }
        known = {
            "architecture", "enable_x2_entry", "stage_channels", *_REQUIRED_INT_KEYS, *fixed, *x2_defaults,
            *_IGNORED_KEYS
        }
        unknown = sorted(set(model) - known)
        if unknown:
            raise ValueError(f"{where}: unknown keys {unknown}")
        if not enable_x2_entry:
            present = sorted(key for key, default in x2_defaults.items() if model.get(key, default) != default)
            if present:
                raise ValueError(f"{where}: {present} require `enable_x2_entry`=true")
        if target_scale == 2 and not enable_x2_entry:
            raise ValueError(f"{where}: `target_scale`='2x' requires `enable_x2_entry`=true")

        missing = [key for key in _REQUIRED_INT_KEYS if key not in model]
        if missing:
            raise ValueError(f"{where}: missing keys {missing}")
        ints = {key: _positive_int(where, key, model[key]) for key in _REQUIRED_INT_KEYS}
        hidden = ints["hidden_channels"]

        raw_stages = model.get("stage_channels")
        if raw_stages is None:
            stage_channels = (hidden, hidden, hidden)
        else:
            if not isinstance(raw_stages, list | tuple) or len(raw_stages) != 3:
                raise ValueError(f"{where}: `stage_channels` must list 3 widths, got {raw_stages!r}")
            first, second, third = (_positive_int(where, "stage_channels", width) for width in raw_stages)
            stage_channels = (first, second, third)
            if stage_channels[0] != hidden:
                raise ValueError(f"{where}: `hidden_channels` ({hidden}) must equal `stage_channels[0]` "
                                 f"({stage_channels[0]})")

        x2_adapter_blocks = 0
        if enable_x2_entry:
            x2_adapter_blocks = _positive_int(where, "x2_adapter_blocks", model.get("x2_adapter_blocks", 0))
            # Training warm-starts the adapter from these pre_blocks; only their consistency matters here.
            sources = model.get("x2_adapter_sources")
            if sources is not None and (not isinstance(sources, list | tuple) or len(sources) != x2_adapter_blocks
                                        or any(not isinstance(i, int) or not 0 <= i < ints["num_pre_blocks"]
                                               for i in sources)):
                raise ValueError(f"{where}: `x2_adapter_sources` must list {x2_adapter_blocks} indices in "
                                 f"[0, num_pre_blocks={ints['num_pre_blocks']}), got {sources!r}")

        return cls(target_scale=target_scale,
                   stage_channels=stage_channels,
                   enable_x2_entry=enable_x2_entry,
                   x2_adapter_blocks=x2_adapter_blocks,
                   **ints)


def _as_entry(entry: Any, index: int) -> Kandinsky6SRLatentUpscalerEntryConfig:
    if isinstance(entry, Kandinsky6SRLatentUpscalerEntryConfig):
        return entry
    # ``PipelineConfig.dump_to_json`` writes entries flattened by ``dataclasses.asdict``; read them back as such.
    if isinstance(entry, Mapping) and "model" not in entry and set(entry) == {
            item.name
            for item in fields(Kandinsky6SRLatentUpscalerEntryConfig)
    }:
        return Kandinsky6SRLatentUpscalerEntryConfig(**{**entry, "stage_channels": tuple(entry["stage_channels"])})
    return Kandinsky6SRLatentUpscalerEntryConfig.from_dict(entry, index)


@dataclass
class Kandinsky6SRLatentUpscalerConfig(UpsamplerConfig):
    """``latent_upscaler/config.json``: one entry per served scale, plus the VAE latent ``scaling_factor``.

    ``models`` accepts the raw config.json entries and is normalised to
    :class:`Kandinsky6SRLatentUpscalerEntryConfig` in ``__post_init__`` (also re-run by ``update_model_config``).
    """

    models: list[Kandinsky6SRLatentUpscalerEntryConfig] = field(default_factory=list)
    scaling_factor: float = 1.0
    # Index order used by re-keyed ModuleList weights. The released re-keying
    # script uses (2, 4) when the legacy models config has no scales field.
    scales: tuple[int, ...] = (2, 4)

    def __post_init__(self) -> None:
        if not isinstance(self.models, list | tuple):
            raise ValueError(f"latent upscaler `models` must be a list of entries, got {self.models!r}")
        entries = [_as_entry(entry, index) for index, entry in enumerate(self.models)]
        scales = [entry.target_scale for entry in entries]
        if len(set(scales)) != len(scales):
            raise ValueError(f"latent upscaler `models` has duplicate target scales: {scales}")
        self.models = entries
        self.scales = tuple(self.scales)
        if (not self.scales or len(set(self.scales)) != len(self.scales)
                or any(type(scale) is not int or scale not in _SUPPORTED_SCALES for scale in self.scales)):
            raise ValueError(f"latent upscaler `scales` must contain unique values from (2, 4), got {self.scales}")
        self.scaling_factor = float(self.scaling_factor)
