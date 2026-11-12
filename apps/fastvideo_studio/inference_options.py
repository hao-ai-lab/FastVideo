# SPDX-License-Identifier: Apache-2.0
"""Studio advice from the repository's existing serving recipes.

Recipes are suggestions, not hardware support claims or replacements for core
checkpoint validation. Reading them never downloads weights or overrides jobs.
"""
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SAMPLING_FIELDS = {"num_inference_steps", "num_frames", "height", "width", "guidance_scale", "fps"}


@lru_cache(maxsize=1)
def _recipes() -> dict[str, dict[str, Any]]:
    result = {}
    for path in sorted((_REPO_ROOT / "examples/serving").glob("openai_*.yaml")):
        config = yaml.safe_load(path.read_text())
        generator = config.get("generator", {})
        model_id = generator.get("model_path")
        if not model_id or model_id in result:
            continue
        pipeline = generator.get("pipeline", {})
        if pipeline.get("workload_type", "t2v") != "t2v" or pipeline.get("components", {}).get("lora_path"):
            continue
        experimental = pipeline.get("experimental", {})
        if (not experimental.get("attention_backend") or generator.get("engine", {}).get("quantization")
                or pipeline.get("components")):
            continue
        sampling = config.get("default_request", {}).get("sampling", {})
        defaults = {k: v for k, v in sampling.items() if k in _SAMPLING_FIELDS}
        experimental = pipeline.get("experimental", {})
        for source, target in (("attention_backend", "attention_backend"), ("VSA_sparsity", "vsa_sparsity"),
                               ("VSA_tile_size", "vsa_tile_size")):
            if source in experimental:
                defaults[target] = experimental[source]
        result[model_id] = {"defaults": defaults, "source": str(path.relative_to(_REPO_ROOT))}
    return result


def resolve_attention_defaults(
    model_id: str,
    workload_type: str,
    backend: str | None,
    tile_size: int | None,
    *,
    use_recipe: bool = True,
) -> tuple[str | None, int | None]:
    """Fill only unset attention fields from a matching self-contained recipe.

    The caller resolves explicit job/server overrides first. Unknown models,
    non-T2V workloads and pipeline overrides keep the core selection path.
    Sampling parameters and process-wide environment are never changed here.
    """
    defaults = _recipes().get(model_id, {}).get("defaults", {}) if use_recipe and workload_type == "t2v" else {}
    recipe_backend = defaults.get("attention_backend")
    resolved_backend = backend if backend is not None else recipe_backend
    if tile_size is None and recipe_backend is not None and resolved_backend == recipe_backend:
        tile_size = defaults.get("vsa_tile_size")
    return resolved_backend, tile_size


def inference_options(model_id: str, workload_type: str = "t2v") -> dict[str, Any]:
    from fastvideo.platforms import AttentionBackendEnum
    from fastvideo.attention.selector import get_env_variable_attn_backend
    recipe = _recipes().get(model_id) if workload_type == "t2v" else None
    server_backend = get_env_variable_attn_backend()
    backend, tile = resolve_attention_defaults(model_id, workload_type, server_backend.name if server_backend else None,
                                               None)
    return {
        "attention_backends": [backend.name for backend in AttentionBackendEnum],
        **(recipe or {
            "defaults": {},
            "source": None
        }),
        "auto_attention_backend": backend,
        "auto_vsa_tile_size": tile,
    }
