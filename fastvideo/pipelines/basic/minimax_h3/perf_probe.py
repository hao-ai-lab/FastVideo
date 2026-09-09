# SPDX-License-Identifier: Apache-2.0
"""Scalar correctness checkpoints for the FastH3 pipeline, recorded by `probe`.

Off unless ``PROBE=1`` (and ``PROBE_OUT`` set); every helper returns on its
first line then, so the timed run pays nothing. When on, each call reduces a
tensor to a few fp32 scalars on the device (no sync) and hands them to
``probe.record``; the ``.item()`` sync happens once at ``flush()``.

Tags are ``req<N>/<name>`` where N counts ``generate`` requests seen by this
worker (0 = the warm-up request), so all requests of one launch are recorded and
files pair by tag across launches. One file per rank (``PROBE_OUT`` gets a
``.rank<N>`` infix from probe itself).
"""
from __future__ import annotations

import torch

try:
    import probe as _probe
except ImportError:  # pragma: no cover - probe is only installed for the campaign
    _probe = None

_request = -1

# Default relative tolerance; `probe derive` on three unmodified launches
# decides the real one, and compare uses the baseline file's tolerance anyway.
_RTOL = 2e-2


def enabled() -> bool:
    return _probe is not None and _probe.enabled()


def begin_request() -> None:
    """Called once per request at the first pipeline stage."""
    global _request
    if not enabled():
        return
    _request += 1


def _tag(name: str) -> str:
    return f"req{_request}/{name}"


def tensor(name: str, t: torch.Tensor | None, *, rtol: float = _RTOL, exact: bool = False) -> None:
    """Record shape, dtype, finiteness and absmean/std/absmax of ``t``."""
    if not enabled():
        return
    if t is None:
        _probe.record(_tag(f"{name}.none"), True)
        return
    _probe.record(_tag(f"{name}.shape"), str(tuple(t.shape)))
    _probe.record(_tag(f"{name}.dtype"), str(t.dtype))
    if t.numel() == 0:
        return
    x = t.detach().float()
    r = 0.0 if exact else rtol
    _probe.record(_tag(f"{name}.finite"), torch.isfinite(x).all())
    _probe.record(_tag(f"{name}.absmean"), x.abs().mean(), rtol=r)
    _probe.record(_tag(f"{name}.std"), x.std(), rtol=r)
    _probe.record(_tag(f"{name}.absmax"), x.abs().max(), rtol=r)


def video(name: str, t: torch.Tensor | None, *, rtol: float = _RTOL) -> None:
    """A decoded video [B, C, T, H, W]: global stats plus per-frame means and a 4x4 spatial grid."""
    if not enabled():
        return
    tensor(name, t, rtol=rtol)
    if t is None or t.numel() == 0 or t.ndim != 5:
        return
    x = t.detach().float()
    frame_mean = x.mean(dim=(0, 1, 3, 4))  # [T]
    frame_std = x.std(dim=(1, 3, 4)).mean(dim=0)  # [T]
    for i in range(int(frame_mean.shape[0])):
        _probe.record(_tag(f"{name}.frame{i:03d}.mean"), frame_mean[i], rtol=rtol, atol=2e-3)
        _probe.record(_tag(f"{name}.frame{i:03d}.std"), frame_std[i], rtol=rtol)
    h, w = x.shape[-2], x.shape[-1]
    for gy in range(4):
        for gx in range(4):
            cell = x[..., gy * h // 4:(gy + 1) * h // 4, gx * w // 4:(gx + 1) * w // 4]
            _probe.record(_tag(f"{name}.cell{gy}{gx}.mean"), cell.mean(), rtol=rtol, atol=2e-3)


def scalar(name: str, value, *, rtol: float = 0.0) -> None:
    if not enabled():
        return
    _probe.record(_tag(name), value, rtol=rtol)


def flush() -> None:
    """Write the snapshot so far. Called at the end of every request."""
    if not enabled():
        return
    _probe.flush()
