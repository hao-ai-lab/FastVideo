"""Configuration: one YAML file, plus secrets from the environment.

`configs/infinite_livestream.yaml` holds everything the app is configured with: what the
checkpoint is asked for, how it is hosted, and how the deployment behaves.
Point at a copy of it with `--config`.

API keys stay in the environment, because a key in a version-controlled file is
a key that leaks. `LIVESTREAM_WEIGHTS_PATH` is there too, being a property of
the machine rather than of the deployment.

`load_config` is the only reader of either; nothing else touches `os.environ`
or parses YAML.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from . import clip_plan

# ---------------------------------------------------------------- presets


class PresetError(ValueError):
    """A preset file is missing or malformed."""


# Inside the package, so it survives installation: the app ships as part of
# fastvideo, and anything beside the package rather than in it is not
# packaged.
DEFAULT_CONFIG = Path(__file__).parent / "configs" / "infinite_livestream.yaml"

# Where the playlist goes when the config does not say. A relative default
# would write into whatever directory the server was started from, which for a
# source checkout is the repo root. Mirrors how `apps/dreamverse` picks its
# state root.
_STATE_ROOT = Path(os.environ.get("XDG_STATE_HOME") or Path.home() / ".local/state") / "fastvideo-livestream"
DEFAULT_HLS_DIR = _STATE_ROOT / "hls"
DEFAULT_FILLERS_DIR = Path(__file__).parent / "presets"
PRESET_FILE = "fillers.json"


def load_preset(directory: str | Path) -> dict:
    """Load and validate the style and idle prompts the stream runs on.

    `directory` holds `fillers.json`: the `style` every rewritten scene is
    written in, and the `idle_prompts` that keep the stream fed when nobody is
    typing. An empty prompt list disables the filler. Other keys are ignored,
    so the file can carry its own notes.
    """
    path = Path(directory) / PRESET_FILE
    if not path.is_file():
        raise PresetError(f"no {PRESET_FILE} in {directory}")
    try:
        preset = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise PresetError(f"{path} is not valid JSON: {error}") from None
    style = preset.get("style")
    prompts = preset.get("idle_prompts")
    if not isinstance(style, str) or not style.strip():
        raise PresetError(f"{path} needs a non-empty string `style`")
    if not isinstance(prompts, list) or not all(isinstance(p, str) for p in prompts):
        raise PresetError(f"{path} needs `idle_prompts` as a list of strings")
    return {
        "style": style.strip(),
        "idle_prompts": [p.strip() for p in prompts if p.strip()],
    }


# ------------------------------------------------------------ model config

# Component directories the T2VA pipeline loads. Missing weights must kill
# startup, not surface as a loader traceback on the first clip.
REQUIRED_COMPONENTS = (
    "transformer",
    "text_encoder",
    "tokenizer",
    "processor",
    "vae",
    "audio_vae",
    "scheduler",
    "audio_scheduler",
)


@dataclass(frozen=True)
class ModelConfig:
    """Everything the engine YAML configures, validated once at load.

    The top-level fields are what the queues and the clip planner need;
    ``inference`` and ``runtime`` are the raw blocks, which the backend reads
    its engine knobs (attention kernels, compile flags, parallelism, offload
    policy) straight out of.
    """

    aspect: str
    clip_frames: int
    seed: int
    num_inference_steps: int
    queue_size: int
    generation_queue_size: int
    warmup_aspects: tuple[str, ...]
    warmup_frames: tuple[int, ...]
    inference: dict[str, Any]
    runtime: dict[str, Any]


def load_model_config(config_path: Path | None = None) -> ModelConfig:
    """Read the `inference` and `runtime` blocks into a validated `ModelConfig`.

    The same file `Config.load` reads. Split out because the queues and the
    backend need the checkpoint's shape, and nothing else in the file.

    Raises:
        ValueError: If the configured aspect is not one the checkpoint offers,
            or a queue size is not positive.
    """
    path = config_path or DEFAULT_CONFIG
    document: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    inference: dict[str, Any] = document.get("inference") or {}
    runtime: dict[str, Any] = document.get("runtime") or {}

    aspect = str(inference.get("aspect", "16:9"))
    if aspect not in clip_plan.ASPECT_CHOICES:
        raise ValueError(f"inference.aspect must be one of {list(clip_plan.ASPECT_CHOICES)}, got {aspect!r}")

    queue_size = int(inference.get("queue_size", 10))
    if queue_size < 1:
        raise ValueError(f"inference.queue_size must be positive, got {queue_size}")

    generation_queue_size = int(inference.get("generation_queue_size", 20))
    if generation_queue_size < 1:
        raise ValueError(f"inference.generation_queue_size must be positive, got {generation_queue_size}")

    clip_frames = clip_plan.frames_for_seconds(float(inference.get("clip_seconds", clip_plan.MAX_SECONDS)))

    return ModelConfig(
        aspect=aspect,
        clip_frames=clip_frames,
        seed=int(inference.get("seed", 1000)),
        num_inference_steps=int(inference.get("num_inference_steps", 5)),
        queue_size=queue_size,
        generation_queue_size=generation_queue_size,
        warmup_aspects=tuple(str(a) for a in (inference.get("warmup_aspects") or [aspect])),
        warmup_frames=_parse_warmup_lengths(inference.get("warmup_lengths"), clip_frames),
        inference=inference,
        runtime=runtime,
    )


def _parse_warmup_lengths(raw: Any, clip_frames: int) -> tuple[int, ...]:
    """Resolve ``inference.warmup_lengths`` to the frame counts load() warms.

    ``"default"`` (or nothing) warms only the configured clip length;
    ``"all"`` warms every length the checkpoint can generate; a list of
    seconds warms those, snapped to legal lengths. The default length is
    always included -- it is the shape every plain enqueue uses.
    """
    if raw in (None, "", "default"):
        return (clip_frames, )
    if raw == "all":
        frames = set(clip_plan.legal_frame_counts())
    elif isinstance(raw, list | tuple):
        frames = {clip_plan.frames_for_seconds(float(seconds)) for seconds in raw}
    else:
        raise ValueError(f'inference.warmup_lengths must be "default", "all", or a list of seconds, got {raw!r}')
    frames.add(clip_frames)
    return tuple(sorted(frames))


def resolve_model_path(config: ModelConfig, weights_root: Path) -> Path:
    """The checkpoint directory under the weights path; "." means the path itself."""
    subdir = str(config.runtime.get("checkpoint_dir", "."))
    if subdir in ("", "."):
        return weights_root
    return weights_root / subdir


def require_weights(root: Path, model_path: Path) -> None:
    """Fail startup loudly when the weights are incomplete."""
    problems: list[str] = []
    if not model_path.is_dir():
        problems.append(f"checkpoint directory is missing: {model_path}")
    else:
        index = model_path / "modular_model_index.json"
        if not index.is_file():
            problems.append(f"modular_model_index.json is missing: {index}")
        for component in REQUIRED_COMPONENTS:
            if not (model_path / component).is_dir():
                problems.append(f"component directory is missing: {model_path / component}")
    if problems:
        raise FileNotFoundError(f"FastH3 weights under {root} are incomplete:\n  " + "\n  ".join(problems))


# -------------------------------------------------------------- app config


@dataclass(frozen=True)
class Config:
    """One immutable snapshot of everything the app is configured with."""

    # The engine: where the weights live and which YAML shapes it
    weights_path: Path
    config_path: Path

    # Upsampling
    openai_api_key: str
    openai_base_url: str | None
    openai_model: str
    max_chunks: int
    # Filler always wears the preset's style; a viewer's own request may pick
    # whatever look suits it. Set 0 to put every clip in the house style.
    viewer_free_style: bool

    # The style every scene is written in, and the idle prompts
    style: str
    idle_prompts: tuple[str, ...]

    # Moderation (its own endpoint: the upsampling gateway may not expose
    # /moderations, so this can point at api.openai.com while upsampling
    # goes elsewhere)
    moderation_enabled: bool
    moderation_api_key: str
    moderation_base_url: str | None
    moderation_model: str

    # Idle filler
    idle_queue_target: int

    # Output: the HLS playlist the page plays, written by `sink.py`.
    hls_dir: str
    video_bitrate_k: int
    hls_retention_s: int

    # The watch page: video, chat and the queue on one HTTP origin, so a
    # single tunnel publishes the whole thing.
    web_host: str
    web_port: int

    # Chat
    chat_command: str
    chat_cooldown_s: float

    @staticmethod
    def load(argv: list[str] | None = None) -> Config:
        """Read the config file and the environment, and validate the result."""
        parser = argparse.ArgumentParser(description="Chat-driven FastH3 livestream (see README.md).")
        parser.add_argument("--config", default=None, help=f"config YAML (default {DEFAULT_CONFIG})")
        parser.add_argument("--weights", default=None, help="override LIVESTREAM_WEIGHTS_PATH")
        parser.add_argument("--port", default=None, type=int, help="override web.port")
        args = parser.parse_args(argv)

        path = Path(args.config).expanduser() if args.config else DEFAULT_CONFIG
        if not path.is_file():
            raise SystemExit(f"config not found: {path}")
        document: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        upsampler = document.get("upsampler") or {}
        moderation = document.get("moderation") or {}
        director = document.get("director") or {}
        output = document.get("output") or {}
        web = document.get("web") or {}

        weights = args.weights or os.environ.get("LIVESTREAM_WEIGHTS_PATH", "")
        openai_key = os.environ.get("OPENAI_API_KEY", "")
        fillers = director.get("fillers")
        try:
            preset = load_preset(Path(fillers).expanduser() if fillers else DEFAULT_FILLERS_DIR)
        except PresetError as error:
            raise SystemExit(str(error)) from None

        config = Config(
            weights_path=Path(weights).expanduser() if weights else Path(),
            config_path=path,
            openai_api_key=openai_key,
            openai_base_url=upsampler.get("base_url") or None,
            openai_model=str(upsampler.get("model", "gpt-4o-mini")),
            max_chunks=max(1, int(upsampler.get("max_chunks", 6))),
            viewer_free_style=bool(upsampler.get("viewer_free_style", True)),
            style=preset["style"],
            idle_prompts=tuple(preset["idle_prompts"]),
            moderation_enabled=bool(moderation.get("enabled", True)),
            # Falls back to the upsampling credentials, which is right when one
            # endpoint serves both.
            moderation_api_key=os.environ.get("MODERATION_API_KEY") or openai_key,
            moderation_base_url=moderation.get("base_url") or upsampler.get("base_url") or None,
            moderation_model=str(moderation.get("model", "omni-moderation-latest")),
            idle_queue_target=int(director.get("idle_queue_target", 6)),
            hls_dir=str(output.get("hls_dir") or DEFAULT_HLS_DIR),
            video_bitrate_k=int(output.get("video_bitrate_k", 4500)),
            hls_retention_s=int(output.get("hls_retention_s", 120)),
            web_host=str(web.get("host", "0.0.0.0")),
            web_port=args.port or int(web.get("port", 8081)),
            chat_command=str(director.get("chat_command", "!prompt")).strip(),
            chat_cooldown_s=float(director.get("chat_cooldown_s", 10)),
        )
        config.validate()
        return config

    def validate(self) -> None:
        """Fail fast on contradictions instead of half-starting."""
        if not str(self.weights_path) or self.weights_path == Path():
            raise SystemExit("Set LIVESTREAM_WEIGHTS_PATH, or pass --weights, pointing at the FastH3 weights.")
        if not self.openai_api_key:
            raise SystemExit(
                "Set OPENAI_API_KEY. Prompt rewriting runs for the idle filler too, so the stream does not start without it."
            )
        if self.hls_retention_s < 6:
            raise SystemExit("output.hls_retention_s must be at least 6 seconds.")
        if not self.chat_command.startswith("!"):
            raise SystemExit("director.chat_command should start with '!' (e.g. !prompt).")


__all__ = [
    "Config",
    "ModelConfig",
    "PresetError",
    "load_model_config",
    "load_preset",
    "require_weights",
    "resolve_model_path",
]
