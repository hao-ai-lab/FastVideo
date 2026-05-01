"""Model checkpoint resolution and caching for eval metrics.

Single public function — :func:`ensure_checkpoint` — that hands a metric
a local path to its weights, downloading on miss. Three source kinds:

* an existing local path → returned as-is;
* an HTTP(S) URL → downloaded to ``get_cache_dir() / name`` via
  ``huggingface_hub.http_get`` (resume + retries built in);
* an HF repo id (``"org/repo"``) → :func:`huggingface_hub.hf_hub_download`
  if *filename* is given, else :func:`huggingface_hub.snapshot_download`.
  *name* is ignored in HF mode — HF manages its own cache key under
  ``~/.cache/huggingface/hub``.

All download paths are filelock-safe (cooperative across threads,
processes, and SLURM ranks) via :func:`fastvideo.utils.get_lock`.
"""

from __future__ import annotations

import os
from pathlib import Path

from fastvideo import envs
from fastvideo.utils import get_lock


def get_cache_dir() -> Path:
    """Eval cache root.

    Layout::

        get_cache_dir() / models /   ← URL-fetched checkpoints (LAION head,
                                       AMT, GRiT, …)
        get_cache_dir() / torch  /   ← redirected ``TORCH_HOME`` (DINO etc.)
        get_cache_dir() / clip   /   ← passed as ``download_root`` to
                                       ``clip.load(...)`` callsites
        ~/.cache/huggingface/hub /   ← left at HF's default; widely shared
                                       with other ML projects

    Override priority: ``FASTVIDEO_EVAL_CACHE`` > ``WM_EVAL_CACHE``
    (legacy) > ``${FASTVIDEO_CACHE_ROOT}/eval``.

    Metric authors writing new code: when wrapping a third-party loader
    that has its own cache convention (CLIP's ``download_root``, pyiqa's
    ``cache_dir``, etc.), pass ``str(get_cache_dir() / "<library>")`` so
    users get a single ``FASTVIDEO_EVAL_CACHE`` knob to redirect them all.
    """
    return Path(
        os.environ.get(
            "FASTVIDEO_EVAL_CACHE",
            os.environ.get(
                "WM_EVAL_CACHE",
                os.path.join(envs.FASTVIDEO_CACHE_ROOT, "eval"),
            ),
        )
    )


def ensure_checkpoint(
    name: str,
    source: str,
    filename: str | None = None,
) -> str:
    """Resolve a model checkpoint path, downloading on miss.

    See module docstring for the full source contract. *name* is used
    only as the local cache filename for URL sources; ignored otherwise.
    """
    if os.path.exists(source):
        return source

    if source.startswith(("http://", "https://")):
        return _ensure_url(name, source)

    if "/" in source:
        return _ensure_hf(source, filename)

    raise ValueError(
        f"Cannot resolve checkpoint: source {source!r} is neither a "
        "path, URL, nor HF repo id"
    )


def _ensure_url(name: str, url: str) -> str:
    local = get_cache_dir() / "models" / name
    if local.exists():
        return str(local)

    local.parent.mkdir(parents=True, exist_ok=True)
    with get_lock(url):
        if local.exists():        # racing process won; reuse its result
            return str(local)
        from huggingface_hub.file_download import http_get
        tmp = local.with_suffix(local.suffix + ".tmp")
        with open(tmp, "wb") as f:
            http_get(url, f)
        tmp.rename(local)
    return str(local)


def _ensure_hf(repo_id: str, filename: str | None) -> str:
    from huggingface_hub import hf_hub_download, snapshot_download
    with get_lock(f"{repo_id}/{filename or '*'}"):
        if filename:
            return hf_hub_download(repo_id=repo_id, filename=filename)
        return snapshot_download(repo_id=repo_id)
