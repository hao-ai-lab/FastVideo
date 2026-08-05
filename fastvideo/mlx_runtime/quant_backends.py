# SPDX-License-Identifier: Apache-2.0
"""MLX block-scaled quantization backends (affine INT8, MXFP8/4, NVFP4).

Isolated experiment module: probes which ``mx.quantize`` modes the installed
MLX build supports and exposes a thin wrapper around native quantized matmul.
Depends only on ``mlx.core`` and the standard library — do not import the rest
of FastVideo from here.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Final, Mapping

import mlx.core as mx

# Probe matrix side length: must be divisible by every mode's group size
# (affine g64, mxfp* g32, nvfp4 g16).
_PROBE_DIM: Final[int] = 64


class QuantBackend(str, Enum):
    """Named MLX quantization backends evaluated for M5 Neural Accelerators."""

    AFFINE_INT8_G64 = "affine_int8_g64"
    MXFP8 = "mxfp8"
    MXFP4 = "mxfp4"
    NVFP4 = "nvfp4"


BACKENDS: Final[tuple[str, ...]] = tuple(b.value for b in QuantBackend)

# Backend name -> kwargs for mx.quantize / mx.quantized_matmul.
# Affine baseline matches FastVideo DiT load path (INT8, group size 64).
# MX/NV block-scaled modes use MLX defaults (see mx.quantize docs).
_BACKEND_KWARGS: Final[Mapping[str, Mapping[str, object]]] = {
    QuantBackend.AFFINE_INT8_G64.value: {
        "mode": "affine",
        "bits": 8,
        "group_size": 64,
    },
    QuantBackend.MXFP8.value: {
        "mode": "mxfp8",
        "bits": None,
        "group_size": None,
    },
    QuantBackend.MXFP4.value: {
        "mode": "mxfp4",
        "bits": None,
        "group_size": None,
    },
    QuantBackend.NVFP4.value: {
        "mode": "nvfp4",
        "bits": None,
        "group_size": None,
    },
}

_SUPPORT_CACHE: dict[str, bool] = {}
_SUPPORT_ERROR_CACHE: dict[str, str | None] = {}
_BYTES_CACHE: dict[str, float] = {}


@dataclass(frozen=True)
class QuantizedWeight:
    """Packed quantized weight plus scales/biases for one backend."""

    weight: mx.array
    scales: mx.array
    biases: mx.array | None
    backend: str
    mode: str
    bits: int | None
    group_size: int | None
    # Original (rows, cols) of the fp weight, used for bytes-per-element.
    orig_shape: tuple[int, int]


def _normalize_backend(backend: str) -> str:
    name = backend.strip().lower()
    if name not in _BACKEND_KWARGS:
        known = ", ".join(BACKENDS)
        raise ValueError(f"Unknown quant backend {backend!r}. Expected one of: {known}")
    return name


def _kwargs_for(backend: str) -> dict[str, object]:
    return dict(_BACKEND_KWARGS[_normalize_backend(backend)])


def support_error(backend: str) -> str | None:
    """Return the support-probe error message, or ``None`` if supported.

    Wraps ``mx.quantize`` + ``mx.quantized_matmul`` in try/except so missing
    modes on older MLX builds surface as a string rather than an exception.
    """
    name = _normalize_backend(backend)
    if name in _SUPPORT_ERROR_CACHE:
        return _SUPPORT_ERROR_CACHE[name]

    kwargs = _kwargs_for(name)
    try:
        w = mx.zeros((_PROBE_DIM, _PROBE_DIM), dtype=mx.float16)
        quantized = mx.quantize(
            w,
            group_size=kwargs["group_size"],  # type: ignore[arg-type]
            bits=kwargs["bits"],  # type: ignore[arg-type]
            mode=str(kwargs["mode"]),
        )
        w_q = quantized[0]
        scales = quantized[1]
        biases = quantized[2] if len(quantized) == 3 else None
        x = mx.zeros((1, _PROBE_DIM), dtype=mx.float16)
        y = mx.quantized_matmul(
            x,
            w_q,
            scales,
            biases,
            transpose=True,
            group_size=kwargs["group_size"],  # type: ignore[arg-type]
            bits=kwargs["bits"],  # type: ignore[arg-type]
            mode=str(kwargs["mode"]),
        )
        mx.eval(y)
        _SUPPORT_ERROR_CACHE[name] = None
        _SUPPORT_CACHE[name] = True
    except Exception as exc:  # noqa: BLE001 - MLX raises varied types per mode/version.
        msg = f"{type(exc).__name__}: {exc}"
        _SUPPORT_ERROR_CACHE[name] = msg
        _SUPPORT_CACHE[name] = False
    return _SUPPORT_ERROR_CACHE[name]


def is_supported(backend: str) -> bool:
    """Return True if the installed MLX build can quantize/matmul with ``backend``."""
    name = _normalize_backend(backend)
    if name not in _SUPPORT_CACHE:
        support_error(name)
    return _SUPPORT_CACHE[name]


def quantize_weight(w: mx.array, backend: str) -> QuantizedWeight:
    """Quantize a 2D weight with the native MLX API for ``backend``.

    Raises:
        ValueError: unknown backend, non-2D input, or last dim not divisible
            by the mode's group size.
        RuntimeError: backend is not supported by the installed MLX build.
    """
    name = _normalize_backend(backend)
    err = support_error(name)
    if err is not None:
        mlx_version = getattr(mx, "__version__", "unknown")
        raise RuntimeError(
            f"Quant backend {name!r} is not supported by installed mlx "
            f"({mlx_version}): {err}"
        )

    if w.ndim != 2:
        raise ValueError(f"quantize_weight expects a 2D weight, got shape {tuple(w.shape)}")

    rows, cols = int(w.shape[0]), int(w.shape[1])
    kwargs = _kwargs_for(name)
    group_size = kwargs["group_size"]
    # When group_size is None, MLX applies the mode default; only check when set.
    if isinstance(group_size, int) and cols % group_size != 0:
        raise ValueError(
            f"Weight last dim {cols} must be divisible by group_size={group_size} "
            f"for backend {name!r}"
        )

    quantized = mx.quantize(
        w,
        group_size=kwargs["group_size"],  # type: ignore[arg-type]
        bits=kwargs["bits"],  # type: ignore[arg-type]
        mode=str(kwargs["mode"]),
    )
    w_q = quantized[0]
    scales = quantized[1]
    biases = quantized[2] if len(quantized) == 3 else None
    eval_args = [w_q, scales] if biases is None else [w_q, scales, biases]
    mx.eval(*eval_args)

    return QuantizedWeight(
        weight=w_q,
        scales=scales,
        biases=biases,
        backend=name,
        mode=str(kwargs["mode"]),
        bits=kwargs["bits"] if isinstance(kwargs["bits"], int) else None,
        group_size=group_size if isinstance(group_size, int) else None,
        orig_shape=(rows, cols),
    )


def quantized_matmul(x: mx.array, qw: QuantizedWeight) -> mx.array:
    """Compute ``x @ w.T`` in the quantized domain via ``mx.quantized_matmul``."""
    return mx.quantized_matmul(
        x,
        qw.weight,
        qw.scales,
        qw.biases,
        transpose=True,
        group_size=qw.group_size,
        bits=qw.bits,
        mode=qw.mode,
    )


def _artifact_nbytes(qw: QuantizedWeight) -> int:
    total = int(qw.weight.nbytes) + int(qw.scales.nbytes)
    if qw.biases is not None:
        total += int(qw.biases.nbytes)
    return total


def bytes_per_weight(backend: str) -> float:
    """Effective stored bytes per weight element (packed data + scales/biases).

    Quantizes a probe matrix and divides total artifact nbytes by the number of
    original weight elements — not a hard-coded formula.
    """
    name = _normalize_backend(backend)
    if name in _BYTES_CACHE:
        return _BYTES_CACHE[name]

    err = support_error(name)
    if err is not None:
        mlx_version = getattr(mx, "__version__", "unknown")
        raise RuntimeError(
            f"Cannot measure bytes_per_weight for unsupported backend {name!r} "
            f"(mlx {mlx_version}): {err}"
        )

    probe = mx.zeros((_PROBE_DIM, _PROBE_DIM), dtype=mx.float16)
    qw = quantize_weight(probe, name)
    n_elem = qw.orig_shape[0] * qw.orig_shape[1]
    value = _artifact_nbytes(qw) / float(n_elem)
    _BYTES_CACHE[name] = value
    return value
