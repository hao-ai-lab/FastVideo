# SPDX-License-Identifier: Apache-2.0
"""Layer-0 grad-norm regression for the per-method training smoke tests.

Phase 2 / 5a-ii: layers a device-keyed grad-norm check on top of the
finite/non-zero grad assertions established in 5a-i. After one
``single_train_step`` + ``backward``, the L2 norm of transformer block 0's
trainable gradients is compared against a reference value pinned per GPU in
``grad_norm_refs.json`` (next to this module).

Determinism: the harness seeds both the global RNG and the method's
``cuda_generator`` via ``method.on_train_start()`` (``training.data.seed`` in the
fixture), and the synthetic ``raw_batch`` is built *after* that call, so the
forward/backward is reproducible within bf16 reduction noise on a given GPU.

Why device-keyed: grad norms differ across GPU architectures (kernels,
accumulation order), so a single golden value can't cover every runner. CI runs
this suite on L40S; B200/GB200 is our local dev GPU.

To seed (or refresh) the reference for the current GPU, run the test with
``FASTVIDEO_GRADNORM_UPDATE=1`` — it records the measured norm into
``grad_norm_refs.json`` and skips the assertion for that run.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch

_REFS_PATH = Path(__file__).resolve().parent / "grad_norm_refs.json"
_UPDATE_ENV = "FASTVIDEO_GRADNORM_UPDATE"

# bf16 single-step smoke: catch gross breakage (wrong wiring, dead grads,
# scale regressions), not micro-drift from reduction nondeterminism.
_DEFAULT_RTOL = 0.10

# GPU-name substring -> reference key. First match wins.
_DEVICE_MAPPINGS: tuple[tuple[str, str], ...] = (
    ("L40S", "L40S"),
    ("GB200", "GB200"),
    ("B200", "GB200"),
    ("H100", "H100"),
    ("H200", "H200"),
    ("A100", "A100"),
)


def _device_name() -> str:
    if not torch.cuda.is_available():
        return "CPU"
    return torch.cuda.get_device_name(0)


def resolve_device_key(device_name: str | None = None) -> str | None:
    """Map a CUDA device name to its reference key, or None if unsupported."""
    name = device_name if device_name is not None else _device_name()
    for pattern, key in _DEVICE_MAPPINGS:
        if pattern in name:
            return key
    return None


def layer0_grad_norm(transformer) -> float:
    """Global L2 norm of transformer block 0's trainable gradients.

    Block 0 is the reference surface 5a-i already isolates: its grad is the
    *last* one produced during backprop, so a healthy value implies the whole
    forward + chain-rule path is intact.
    """
    blocks = getattr(transformer, "blocks", None)
    assert blocks is not None and len(blocks) > 0, (
        "transformer is expected to expose a non-empty ``.blocks``")
    sq_sum = 0.0
    for p in blocks[0].parameters():
        if p.requires_grad and p.grad is not None:
            sq_sum += p.grad.detach().float().pow(2).sum().item()
    return sq_sum**0.5


def _load_refs() -> dict[str, dict[str, float]]:
    if _REFS_PATH.exists():
        return json.loads(_REFS_PATH.read_text())
    return {}


def _save_refs(refs: dict[str, dict[str, float]]) -> None:
    _REFS_PATH.write_text(
        json.dumps(refs, indent=2, sort_keys=True) + "\n")


def check_grad_norm_regression(
    test_name: str,
    transformer,
    *,
    rtol: float = _DEFAULT_RTOL,
) -> None:
    """Assert block-0 grad norm matches the device-keyed reference within rtol.

    - Skips when the current GPU has no reference (unsupported device, or not
      yet seeded) so a new runner never hard-fails before its golden exists.
    - With ``FASTVIDEO_GRADNORM_UPDATE=1`` records/updates the reference for the
      current device instead of asserting.
    """
    norm = layer0_grad_norm(transformer)
    device_key = resolve_device_key()

    if os.environ.get(_UPDATE_ENV) == "1":
        if device_key is None:
            pytest.skip(
                f"{_UPDATE_ENV}=1 but GPU '{_device_name()}' has no reference "
                "key; add it to _DEVICE_MAPPINGS first")
        refs = _load_refs()
        refs.setdefault(test_name, {})[device_key] = round(norm, 4)
        _save_refs(refs)
        pytest.skip(
            f"recorded grad-norm reference {test_name}[{device_key}] = "
            f"{norm:.4f} (assertion skipped under {_UPDATE_ENV}=1)")

    ref = _load_refs().get(test_name, {}).get(device_key) \
        if device_key is not None else None
    if ref is None:
        pytest.skip(
            f"no grad-norm reference for {test_name} on '{_device_name()}' "
            f"(device_key={device_key}); run with {_UPDATE_ENV}=1 to seed it")

    rel = abs(norm - ref) / (abs(ref) + 1e-12)
    assert rel <= rtol, (
        f"{test_name}[{device_key}] grad-norm regression: got {norm:.4f}, "
        f"reference {ref:.4f}, relative error {rel:.3%} exceeds rtol "
        f"{rtol:.0%}. If this is an intentional change, refresh the reference "
        f"with {_UPDATE_ENV}=1 and explain why in the PR.")
