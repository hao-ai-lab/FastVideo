from fastvideo.eval.models import ensure_checkpoint, get_cache_dir


def _redirect_third_party_caches() -> None:
    """Point libraries that respect env vars at the eval cache root.

    Run before metric modules import torch.hub (and friends) so the
    redirect actually takes effect. We intentionally leave ``HF_HOME``
    alone — HF's default cache (``~/.cache/huggingface/hub``) is widely
    shared with other ML projects, and isolating it for eval would force
    users to re-download already-cached transformers weights.

    Other libraries with non-standard caches (CLIP, pyiqa) don't honour
    env vars at all; their callsites in metric.py files pass
    ``download_root=str(get_cache_dir() / "<library>")`` directly.
    """
    import os
    root = get_cache_dir()
    os.environ.setdefault("TORCH_HOME", str(root / "torch"))


_redirect_third_party_caches()


from fastvideo.eval.types import MetricResult  # noqa: E402
from fastvideo.eval.metrics.base import BaseMetric  # noqa: E402
from fastvideo.eval.registry import register, list_metrics, get_metric  # noqa: E402
from fastvideo.eval.api import evaluate  # noqa: E402
from fastvideo.eval.evaluator import Evaluator, create_evaluator  # noqa: E402

# Trigger metric auto-discovery
import fastvideo.eval.metrics  # noqa: F401, E402

__all__ = [
    "evaluate",
    "Evaluator",
    "create_evaluator",
    "MetricResult",
    "BaseMetric",
    "register",
    "list_metrics",
    "get_metric",
    "ensure_checkpoint",
    "get_cache_dir",
]
