"""Pipeline config dataclasses (vendored from fastvideo).

The per-model config classes live in submodules (``wan``, ``ltx2_pipeline_configs``, ``matrixgame2``,
``flux_2``, …) — import them directly. ``get_pipeline_config_cls_from_name`` lives in
``v2._vendor.configs.pipeline_registry`` (imported lazily by ``base`` where used, to avoid an import cycle:
the registry imports ``base``, so ``base`` must not pull the registry at module load)."""
from v2._vendor.configs.pipelines.base import PipelineConfig

__all__ = ["PipelineConfig"]
