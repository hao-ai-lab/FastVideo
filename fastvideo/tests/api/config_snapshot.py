# SPDX-License-Identifier: Apache-2.0
"""Snapshot cases for the final configuration objects, compared against golden JSON files.

Each case builds the configuration objects that runtime code reads (``FastVideoArgs`` with its ``PipelineConfig``,
and a ``SamplingParam`` where the case has a request) through one of the public entry paths:

- ``registry``: every registered model path, through ``FastVideoArgs.from_kwargs`` and ``SamplingParam.from_pretrained``.
- ``yaml``: every repository config file with a ``generator`` section, through ``load_generator_config_from_file``,
  ``resolve_inference_config``, and ``generator_config_to_fastvideo_args``, with the resolution decisions; its ``request`` or ``default_request`` through the ``fastvideo generate`` and
  ``fastvideo serve`` loaders and ``request_to_sampling_param``.
- ``kwargs``: ``VideoGenerator.from_pretrained`` keywords, through ``from_pretrained_kwargs_to_config`` and the same
  resolution.
- ``config``: typed ``GeneratorConfig`` mappings, as ``VideoGenerator.from_config`` receives them, through the same
  resolution.
- ``cli``: argparse flags, through ``FastVideoArgs.add_cli_args`` and ``FastVideoArgs.from_cli_args``.
- ``environment``: environment variables that ``FastVideoArgs.__post_init__`` folds into fields.
- ``request``: typed ``GenerationRequest`` values, through ``request_to_sampling_param``.

Every case runs with all registered environment variables unset (except the ones an ``env`` case sets) and with model
index downloads blocked, so the snapshot depends only on the source tree.

Regenerate the golden files after an intended behavior change:

    python -m fastvideo.tests.api.config_snapshot

``fastvideo/tests/api/test_config_snapshot_matches_golden.py`` compares every case with its golden file.
"""
from __future__ import annotations

import argparse
import contextlib
import dataclasses
import enum
import json
import math
import re
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
GOLDEN_DIR = Path(__file__).resolve().parent / "config_snapshot_goldens"

# Static model-definition tables (weight-name regex maps and FSDP/compile predicates). They describe how checkpoints
# load, not how arguments resolve, and they would make up most of the golden size.
EXCLUDED_FIELDS = frozenset({
    "param_names_mapping",
    "reverse_param_names_mapping",
    "lora_param_names_mapping",
    "_fsdp_shard_conditions",
    "_compile_conditions",
})

WAN_T2V = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
WAN22_T2V = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
LTX2 = "FastVideo/LTX2-Distilled-Diffusers"


class OfflineResolutionError(RuntimeError):
    """A case needed a network download to resolve its model path."""


@dataclasses.dataclass(frozen=True)
class SnapshotCase:
    """One entry path into the configuration, identified by ``category/name``."""

    category: str
    name: str
    build: Callable[[], dict[str, Any]]

    @property
    def case_id(self) -> str:
        return f"{self.category}/{self.name}"

    @property
    def golden_path(self) -> Path:
        return GOLDEN_DIR / self.category / f"{re.sub(r'[^A-Za-z0-9._-]+', '__', self.name)}.json"


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

_ADDRESS = re.compile(r" at 0x[0-9a-fA-F]+")


def _normalize_text(text: str) -> str:
    """Remove the checkout location, the home directory, and object addresses from a string."""
    text = text.replace(str(REPO_ROOT), "<repo>").replace(str(Path.home()), "<home>")
    return _ADDRESS.sub("", text)


def to_jsonable(value: Any) -> Any:
    """Convert a configuration value into stable JSON data.

    Dataclasses become dicts of their fields (minus ``EXCLUDED_FIELDS``), enums their values, tuples and sets lists,
    callables and classes their ``module.qualname``, tensors and dtypes a short description, and any other object a
    dict of its attributes tagged with its class name.
    """
    import torch

    if value is None or isinstance(value, bool | int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else repr(value)
    if isinstance(value, str):
        return _normalize_text(value)
    if isinstance(value, enum.Enum):
        return to_jsonable(value.value)
    if isinstance(value, Path):
        return _normalize_text(str(value))
    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, torch.Tensor):
        return f"<tensor shape={list(value.shape)} dtype={value.dtype}>"
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: to_jsonable(getattr(value, field.name))
            for field in dataclasses.fields(value) if field.name not in EXCLUDED_FIELDS
        }
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, list | tuple):
        return [to_jsonable(item) for item in value]
    if isinstance(value, set | frozenset):
        return sorted((to_jsonable(item) for item in value), key=lambda item: json.dumps(item, sort_keys=True))
    if isinstance(value, type) or callable(value) and hasattr(value, "__qualname__"):
        return f"{getattr(value, '__module__', '?')}.{value.__qualname__}"
    if hasattr(value, "__dict__"):
        attributes = {key: item for key, item in vars(value).items() if key not in EXCLUDED_FIELDS}
        return {"__class__": f"{type(value).__module__}.{type(value).__qualname__}", **to_jsonable(attributes)}
    return _normalize_text(repr(value))


def snapshot_fastvideo_args(fastvideo_args: Any) -> dict[str, Any]:
    """FastVideoArgs fields, with its PipelineConfig as a separate top-level entry."""
    fields = {
        field.name: to_jsonable(getattr(fastvideo_args, field.name))
        for field in dataclasses.fields(fastvideo_args) if field.name != "pipeline_config"
    }
    pipeline_config = fastvideo_args.pipeline_config
    return {
        "fastvideo_args": fields,
        "pipeline_config_class": f"{type(pipeline_config).__module__}.{type(pipeline_config).__qualname__}",
        "pipeline_config": to_jsonable(pipeline_config),
    }


def snapshot_resolved_fastvideo_args(generator_config: Any) -> dict[str, Any]:
    """Resolve a GeneratorConfig the way VideoGenerator.from_config does, and record the resolution decisions."""
    from fastvideo.api.compat import generator_config_to_fastvideo_args
    from fastvideo.api.inference_resolution import resolve_inference_config

    resolved = resolve_inference_config(generator_config)
    snapshot = snapshot_fastvideo_args(generator_config_to_fastvideo_args(resolved))
    snapshot["resolution_decisions"] = to_jsonable(resolved.decisions)
    return snapshot


def snapshot_sampling_param(sampling_param: Any) -> dict[str, Any]:
    return {
        "sampling_param_class": f"{type(sampling_param).__module__}.{type(sampling_param).__qualname__}",
        "sampling_param": to_jsonable(sampling_param),
    }


def describe_error(error: BaseException) -> str:
    return f"{type(error).__name__}: {_normalize_text(str(error))}"


def snapshot_request(request: Any, model_path: str) -> dict[str, Any]:
    """The values VideoGenerator derives from a GenerationRequest; a rejected request is recorded as a value."""
    from fastvideo.api.compat import request_to_batch_extra, request_to_sampling_param

    from fastvideo.api.request_resolution import resolve_request

    snapshot = {
        "request_batch_extra": to_jsonable(request_to_batch_extra(request)),
        "request_resolution_decisions": to_jsonable(resolve_request(request, model_path=model_path).decisions),
    }
    try:
        snapshot.update(snapshot_sampling_param(request_to_sampling_param(request, model_path=model_path)))
    except (ValueError, NotImplementedError) as error:
        snapshot["sampling_param_error"] = describe_error(error)
    return snapshot


# ---------------------------------------------------------------------------
# Isolation: no inherited environment variables, no downloads
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def isolated_environment(env_values: dict[str, Any] | None = None) -> Iterator[None]:
    """Unset every registered FastVideo environment variable, set ``env_values``, and block model-index downloads.

    The registry's ``override`` context managers restore the previous values on exit. Downloads are blocked by
    replacing ``fastvideo.registry.maybe_download_model_index``, the only network call on the configuration path.
    """
    import fastvideo.envs as envs
    import fastvideo.registry as registry

    def refuse_download(model_path: str, *args: Any, **kwargs: Any) -> dict[str, Any]:
        raise OfflineResolutionError(f"resolving {model_path!r} needs a model_index.json download")

    values = env_values or {}
    with contextlib.ExitStack() as stack:
        for name, field in envs.environment_variables.items():
            stack.enter_context(field.override(values.get(name)))
        original = registry.maybe_download_model_index
        registry.maybe_download_model_index = refuse_download
        stack.callback(setattr, registry, "maybe_download_model_index", original)
        yield


# ---------------------------------------------------------------------------
# Case builders
# ---------------------------------------------------------------------------


def _from_kwargs(**kwargs: Any) -> dict[str, Any]:
    from fastvideo.fastvideo_args import FastVideoArgs

    return snapshot_fastvideo_args(FastVideoArgs.from_kwargs(**kwargs))


def _registry_case(model_path: str) -> Callable[[], dict[str, Any]]:

    def build() -> dict[str, Any]:
        from fastvideo.api.sampling_param import SamplingParam

        snapshot = _from_kwargs(model_path=model_path)
        snapshot.update(snapshot_sampling_param(SamplingParam.from_pretrained(model_path)))
        return snapshot

    return build


def _yaml_case(path: Path) -> Callable[[], dict[str, Any]]:

    def build() -> dict[str, Any]:
        """Load the file the way VideoGenerator.from_file and the CLI do."""
        from fastvideo.api.compat import load_generator_config_from_file

        generator_config = load_generator_config_from_file(path)
        snapshot = snapshot_resolved_fastvideo_args(generator_config)
        snapshot.update(snapshot_request(_yaml_request(path), generator_config.model_path))
        return snapshot

    return build


def _yaml_request(path: Path) -> Any:
    """The request a config file carries: ``request`` for ``fastvideo generate``, ``default_request`` for serve."""
    from fastvideo.entrypoints.cli.inference_config import build_generate_run_config, build_serve_config

    namespace = argparse.Namespace(config=str(path))
    if "request" in yaml.safe_load(path.read_text()):
        return build_generate_run_config(namespace).request
    return build_serve_config(namespace).default_request


def _kwargs_case(model_path: str, kwargs: dict[str, Any]) -> Callable[[], dict[str, Any]]:

    def build() -> dict[str, Any]:
        from fastvideo.api.compat import from_pretrained_kwargs_to_config

        return snapshot_resolved_fastvideo_args(from_pretrained_kwargs_to_config(model_path, kwargs))

    return build


def _config_case(raw: dict[str, Any]) -> Callable[[], dict[str, Any]]:

    def build() -> dict[str, Any]:
        from fastvideo.api.compat import normalize_generator_config

        return snapshot_resolved_fastvideo_args(normalize_generator_config(raw))

    return build


def _cli_case(argv: list[str]) -> Callable[[], dict[str, Any]]:

    def build() -> dict[str, Any]:
        from fastvideo.fastvideo_args import FastVideoArgs
        from fastvideo.utils import FlexibleArgumentParser

        parser = FlexibleArgumentParser()
        FastVideoArgs.add_cli_args(parser)
        return snapshot_fastvideo_args(FastVideoArgs.from_cli_args(parser.parse_args(argv)))

    return build


def _env_case(env_values: dict[str, Any], **kwargs: Any) -> Callable[[], dict[str, Any]]:
    """A FastVideoArgs.from_kwargs case run with ``env_values`` set (applied by ``run_case``)."""

    def build() -> dict[str, Any]:
        return _from_kwargs(**kwargs)

    build.env_values = env_values  # type: ignore[attr-defined]
    return build


def _request_case(model_path: str, raw_request: dict[str, Any]) -> Callable[[], dict[str, Any]]:

    def build() -> dict[str, Any]:
        from fastvideo.api.compat import normalize_generation_request

        return snapshot_request(normalize_generation_request(raw_request), model_path)

    return build


def _repository_config_files() -> list[Path]:
    """Every YAML file outside the tests that has a ``generator`` mapping at the top level.

    Files with ``runtime: mlx`` are skipped: ``fastvideo/entrypoints/openai/mlx_server.py`` reads them with its own
    schema, and they never reach ``FastVideoArgs``.
    """
    files = []
    for path in sorted(REPO_ROOT.rglob("*.y*ml")):
        relative = path.relative_to(REPO_ROOT).as_posix()
        if relative.startswith(("fastvideo/tests/", ".", "fastvideo-kernel/")) or "/." in relative:
            continue
        try:
            raw = yaml.safe_load(path.read_text())
        except yaml.YAMLError:
            continue
        if isinstance(raw, dict) and isinstance(raw.get("generator"), dict) and raw.get("runtime") != "mlx":
            files.append(path)
    return files


def collect_cases() -> list[SnapshotCase]:
    """All snapshot cases, in a stable order."""
    from fastvideo.registry import get_registered_model_paths

    cases = [SnapshotCase("registry", path, _registry_case(path)) for path in get_registered_model_paths()]
    cases += [
        SnapshotCase("yaml",
                     path.relative_to(REPO_ROOT).as_posix(), _yaml_case(path)) for path in _repository_config_files()
    ]
    # One case per kind of branch in from_pretrained_kwargs_to_config: flat names of typed fields and compile sub-keys.
    kwargs_cases = {
        "typed_offload_and_parallelism": (WAN_T2V, {
            "num_gpus": 2,
            "sp_size": 2,
            "dit_cpu_offload": False,
            "vae_cpu_offload": False,
        }),
        "torch_compile_kwargs_typed_and_extra": (WAN_T2V, {
            "enable_torch_compile": True,
            "torch_compile_kwargs": {
                "backend": "inductor",
                "mode": "max-autotune",
                "fullgraph": True,
                "options": {
                    "triton.cudagraphs": False
                },
            },
        }),
        "disable_autocast": (WAN_T2V, {
            "disable_autocast": True
        }),
    }
    cases += [
        SnapshotCase("kwargs", name, _kwargs_case(model, kwargs)) for name, (model, kwargs) in kwargs_cases.items()
    ]
    # Typed configs for the settings that only from_config accepts: typed fields, keys left in
    # pipeline.experimental, LTX-2 refine and VAE tile fields, and a pipeline config JSON path.
    pipeline_json = str(REPO_ROOT / "fastvideo/configs/fasthunyuan_t2v.json")
    config_cases = {
        "attention_precision_and_flow_shift": {
            "model_path": WAN_T2V,
            "engine": {
                "attention": {
                    "backend": "TORCH_SDPA",
                    "vsa_sparsity": 0.5
                },
                "precision": {
                    "dit": "fp32"
                }
            },
            "pipeline": {
                "flow_shift": 5.0
            },
        },
        "keys_without_typed_fields": {
            "model_path": WAN_T2V,
            "pipeline": {
                "experimental": {
                    "master_port": 29600,
                    "refine_enabled": True,
                    "boundary_ratio": 0.5
                }
            },
        },
        "ltx2_refine_lora_path": {
            "model_path": LTX2,
            "pipeline": {
                "ltx2": {
                    "refine": {
                        "lora_path": "/checkpoints/refine_lora.safetensors"
                    }
                }
            },
        },
        "ltx2_refine_lora_disabled": {
            "model_path": LTX2,
            "pipeline": {
                "ltx2": {
                    "refine": {
                        "lora_path": ""
                    }
                }
            },
        },
        "pipeline_config_json_path": {
            "model_path": "FastVideo/FastHunyuan-diffusers",
            "pipeline": {
                "components": {
                    "pipeline_config_path": pipeline_json
                }
            },
        },
        "boundary_ratio": {
            "model_path": WAN22_T2V,
            "pipeline": {
                "experimental": {
                    "boundary_ratio": 0.8
                }
            },
        },
        "vae_tiling_typed": {
            "model_path": LTX2,
            "pipeline": {
                "vae_tiling": False
            },
        },
        "ltx2_vae_tile_sizes": {
            "model_path": LTX2,
            "pipeline": {
                "ltx2": {
                    "vae_spatial_tile_size_in_pixels": 512,
                    "vae_temporal_tile_size_in_frames": 64
                }
            },
        },
    }
    cases += [SnapshotCase("config", name, _config_case(raw)) for name, raw in config_cases.items()]
    # argparse defaults differ from the dataclass defaults, so model_path_only is the CLI baseline.
    cli_cases = {
        "model_path_only": ["--model-path", WAN_T2V],
        "offload_and_parallelism": [
            "--model-path", WAN_T2V, "--num-gpus", "2", "--sp-size", "2", "--dit-cpu-offload", "true",
            "--vae-cpu-offload", "false"
        ],
        "attention_backend_and_disable_autocast":
        ["--model-path", WAN_T2V, "--attention-backend", "TORCH_SDPA", "--disable-autocast"],
        "pipeline_config_flags": [
            "--model-path", WAN_T2V, "--flow-shift", "5.0", "--dit-precision", "fp32", "--embedded-cfg-scale", "4.5",
            "--vae-tiling", "false", "--vae-sp", "false"
        ],
        "vae_config_flags": [
            "--model-path", WAN_T2V, "--vae-config.load-encoder", "false", "--vae-config.use-tiling", "true",
            "--vae-config.tile-sample-min-height", "128"
        ],
        "dit_config_flags": ["--model-path", WAN_T2V, "--dit-config.prefix", "Probe"],
    }
    cases += [SnapshotCase("cli", name, _cli_case(argv)) for name, argv in cli_cases.items()]
    # Environment variables that FastVideoArgs.__post_init__ folds into fields.
    env_cases = {
        "attention_backend": {
            "FASTVIDEO_ATTENTION_BACKEND": "TORCH_SDPA"
        },
        "attention_backend_unknown_name": {
            "FASTVIDEO_ATTENTION_BACKEND": "NOT_A_BACKEND"
        },
        "inference_torch_compile": {
            "FASTVIDEO_INFERENCE_TORCH_COMPILE": True
        },
        "vae_parallel": {
            "FASTVIDEO_VAE_PARALLEL_DECODE": True,
            "FASTVIDEO_VAE_PARALLEL_ENCODE": True,
            "FASTVIDEO_VAE_PARALLEL_DECODE_STRATEGY": "all_gather",
        },
    }
    cases += [
        SnapshotCase("environment", name, _env_case(env_values, model_path=WAN_T2V))
        for name, env_values in env_cases.items()
    ]
    cases.append(
        SnapshotCase("environment", "explicit_argument_wins_over_attention_backend",
                     _env_case({"FASTVIDEO_ATTENTION_BACKEND": "TORCH_SDPA"},
                               model_path=WAN_T2V,
                               attention_backend="FLASH_ATTN")))
    # Requests cover sampling fields, LTX-2 stage overrides, extensions, and output flags.
    request_cases = {
        "sampling_values": (WAN_T2V, {
            "prompt": "a cat",
            "negative_prompt": "blurry",
            "sampling": {
                "num_inference_steps": 12,
                "guidance_scale": 4.0,
                "seed": 7,
                "height": 720,
                "width": 1280,
            },
        }),
        "stage_overrides_refine": (LTX2, {
            "prompt": "a cat",
            "stage_overrides": {
                "refine": {
                    "num_inference_steps": 2,
                    "guidance_scale": 1.5
                }
            },
        }),
        "extensions_batch_extra_and_pipeline_override": (WAN_T2V, {
            "prompt": "a cat",
            "extensions": {
                "vsa_mode": "dense",
                "embedded_cfg_scale": 3.0
            },
        }),
        "output_return_state": (LTX2, {
            "prompt": "a cat",
            "output": {
                "return_state": True
            }
        }),
    }
    cases += [
        SnapshotCase("request", name, _request_case(model, raw)) for name, (model, raw) in request_cases.items()
    ]
    return cases


def run_case(case: SnapshotCase) -> dict[str, Any]:
    """Build one case's snapshot in an isolated environment; an error other than a needed download is the snapshot."""
    with isolated_environment(getattr(case.build, "env_values", None)):
        try:
            return case.build()
        except OfflineResolutionError:
            raise
        except Exception as error:
            return {"error": describe_error(error)}


# ---------------------------------------------------------------------------
# Golden files and diffs
# ---------------------------------------------------------------------------


def dump_json(snapshot: dict[str, Any]) -> str:
    return json.dumps(snapshot, indent=1, sort_keys=True) + "\n"


def flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    """Map every leaf of nested JSON data to its dotted path, with list indices in brackets."""
    if isinstance(value, dict) and value:
        leaves: dict[str, Any] = {}
        for key, item in value.items():
            leaves.update(flatten(item, f"{prefix}.{key}" if prefix else str(key)))
        return leaves
    if isinstance(value, list) and value:
        leaves = {}
        for index, item in enumerate(value):
            leaves.update(flatten(item, f"{prefix}[{index}]"))
        return leaves
    return {prefix: value}


def describe_diff(golden: dict[str, Any], current: dict[str, Any], limit: int = 40) -> str:
    """One line per differing leaf: path, golden value, current value."""
    golden_leaves, current_leaves = flatten(golden), flatten(current)
    lines = []
    for path in sorted(set(golden_leaves) | set(current_leaves)):
        old = golden_leaves.get(path, "<absent>")
        new = current_leaves.get(path, "<absent>")
        if old != new:
            lines.append(f"  {path}: golden={old!r} current={new!r}")
    if len(lines) > limit:
        lines = lines[:limit] + [f"  … {len(lines) - limit} more differences"]
    return "\n".join(lines)


def regenerate() -> None:
    """Rewrite every golden file and remove golden files that no case produces."""
    written = set()
    skipped = []
    for case in collect_cases():
        try:
            snapshot = run_case(case)
        except OfflineResolutionError as error:
            skipped.append((case.case_id, str(error)))
            continue
        if "error" in snapshot:
            print(f"error recorded for {case.case_id}: {snapshot['error']}")
        case.golden_path.parent.mkdir(parents=True, exist_ok=True)
        case.golden_path.write_text(dump_json(snapshot))
        written.add(case.golden_path)
    for stale in GOLDEN_DIR.rglob("*.json"):
        if stale not in written:
            stale.unlink()
    print(f"wrote {len(written)} golden files under {GOLDEN_DIR}")
    for case_id, reason in skipped:
        print(f"skipped {case_id}: {reason}")


if __name__ == "__main__":
    regenerate()
