# SPDX-License-Identifier: Apache-2.0
"""Env-gated attention shape logger.

Set ``FASTVIDEO_ATTN_SHAPE_LOG=/path/to/shapes.jsonl`` to record every unique
attention shape executed by the backends that call :func:`record`. Each unique
(backend, batch, heads, seq lens, head_dim, dtype, causal) combination becomes
one JSONL line with an occurrence count plus first-seen metadata (torch
version, device name, SM arch).

Purpose: collect the real attention workloads FastVideo runs so kernel authors
(e.g. FlashInfer SageAttention3 on DGX Spark / SM121) can tune against actual
shapes instead of guesses.

Workers are separate processes, so each process writes its own file: the
configured path is suffixed with ``.rank<N>`` when ``WORLD_SIZE>1``, else
``.pid<N>``. The file is rewritten on flush (counts mutate, so append-only
JSONL would go stale); flushes happen every ``_FLUSH_EVERY`` new unique shapes
and at interpreter exit.

Unset env var = zero overhead: call sites guard on the module-level
``enabled`` bool and never enter this module's code.
"""

from __future__ import annotations

import atexit
import json
import os
import threading

import torch

_LOG_PATH: str | None = os.environ.get("FASTVIDEO_ATTN_SHAPE_LOG")
enabled: bool = _LOG_PATH is not None

_FLUSH_EVERY = 1  # rewrite the file every N new unique shapes (crash-safety)

_lock = threading.Lock()
_entries: dict[tuple, dict] = {}
_new_since_flush = 0
_atexit_registered = False


def _log_path() -> str:
    assert _LOG_PATH is not None
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        return f"{_LOG_PATH}.rank{os.environ.get('RANK', '0')}"
    return f"{_LOG_PATH}.pid{os.getpid()}"


def _device_meta(device: torch.device) -> dict:
    meta: dict = {"torch": torch.__version__, "device": str(device), "sm": None}
    if device.type == "cuda" and torch.cuda.is_available():
        cap = torch.cuda.get_device_capability(device)
        meta["device"] = torch.cuda.get_device_name(device)
        meta["sm"] = f"{cap[0]}.{cap[1]}"
    return meta


def _flush_locked() -> None:
    """Rewrite the whole snapshot; atomic via os.replace."""
    path = _log_path()
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        for entry in _entries.values():
            f.write(json.dumps(entry) + "\n")
    os.replace(tmp, path)


def _flush_at_exit() -> None:
    with _lock:
        if _entries:
            _flush_locked()


def record(backend: str,
           q: torch.Tensor,
           k: torch.Tensor,
           v: torch.Tensor,
           *,
           causal: bool,
           sm_scale: float | None,
           layout: str = "bshd") -> None:
    """Record one attention call.

    ``layout`` describes what the calling backend sees: ``"bshd"`` for
    [batch, seq, heads, head_dim] (the FastVideo impl.forward convention) or
    ``"bhsd"`` for [batch, heads, seq, head_dim] (backends that transpose in
    ``preprocess_qkv``, e.g. SAGE_ATTN_THREE).
    """
    global _new_since_flush, _atexit_registered
    if not enabled:
        return
    if layout == "bhsd":
        batch, num_q_heads, seq_len_q, head_dim = q.shape
        num_kv_heads, seq_len_kv = k.shape[1], k.shape[2]
    else:
        batch, seq_len_q, num_q_heads, head_dim = q.shape
        num_kv_heads, seq_len_kv = k.shape[2], k.shape[1]
    key = (backend, batch, num_q_heads, num_kv_heads, seq_len_q, seq_len_kv, head_dim, str(q.dtype), bool(causal))
    with _lock:
        entry = _entries.get(key)
        if entry is not None:
            entry["count"] += 1
            return
        _entries[key] = {
            "backend": backend,
            "batch": batch,
            "num_q_heads": num_q_heads,
            "num_kv_heads": num_kv_heads,
            "seq_len_q": seq_len_q,
            "seq_len_kv": seq_len_kv,
            "head_dim": head_dim,
            "dtype": str(q.dtype),
            "causal": bool(causal),
            "sm_scale": float(sm_scale) if sm_scale is not None else None,
            "count": 1,
            **_device_meta(q.device),
        }
        if not _atexit_registered:
            atexit.register(_flush_at_exit)
            _atexit_registered = True
        _new_since_flush += 1
        if _new_since_flush >= _FLUSH_EVERY:
            _flush_locked()
            _new_since_flush = 0
