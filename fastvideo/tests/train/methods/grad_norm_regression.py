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
accumulation order), so a single golden value can't cover every runner. The
JSON currently carries refs for the two GPUs we actually run on — ``L40S`` (CI)
and ``GB200`` (our Blackwell dev box; ``B200`` maps to the same key).

Seeding a reference for the current device:

- **CI / L40S** — invoke ``modal run`` against ``seed_grad_norm_references`` in
  ``fastvideo/tests/modal/pr_test.py`` (pinned to ``gpu="L40S:1"``), then copy
  the recorded value from the log into ``grad_norm_refs.json``.
- **Local / non-L40S GPUs** — on that workstation::

      FASTVIDEO_GRADNORM_UPDATE=1 \\
          pytest fastvideo/tests/train/methods -vs -rs

  The harness writes the measured norm into ``grad_norm_refs.json`` under the
  device's key and skips the assertion for that run. Append a new substring
  entry to ``_DEVICE_MAPPINGS`` first for any device not already listed.
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

# GPU-name substring -> reference key. First match wins. Only devices with
# seeded references in ``grad_norm_refs.json`` are listed here — to add a new
# GPU, append an entry, then seed the reference (see module docstring).
_DEVICE_MAPPINGS: tuple[tuple[str, str], ...] = (
    ("L40S", "L40S"),
    ("GB200", "GB200"),
    ("B200", "GB200"),  # same Blackwell arch as GB200
    ("H200", "H200"),
)


def _device_name() -> str:
    if not torch.cuda.is_available():
        return "CPU"
    return torch.cuda.get_device_name(0)


def resolve_device_key(device_name: str | None = None) -> str | None:
    """Map a CUDA device name to its reference key, or None if unsupported.

    The substring match is case-insensitive so it survives driver/environment
    differences in how ``torch.cuda.get_device_name`` capitalizes the model.
    """
    name = device_name if device_name is not None else _device_name()
    name_lower = name.lower()
    for pattern, key in _DEVICE_MAPPINGS:
        if pattern.lower() in name_lower:
            return key
    return None


# Transformer block-list attribute names across model families. Wan / causal
# Wan / Matrix-Game expose ``.blocks``; Cosmos (diffusers-derived) exposes
# ``.transformer_blocks``. First non-empty match wins.
_BLOCK_LIST_ATTRS: tuple[str, ...] = ("blocks", "transformer_blocks")


def resolve_blocks(transformer):
    """Return transformer block 0's containing ModuleList, or None.

    Block 0 is the reference surface for the grad-norm check: its grad is the
    *last* one produced during backprop, so a healthy value implies the whole
    forward + chain-rule path is intact. Different model families name the
    list differently (see ``_BLOCK_LIST_ATTRS``).
    """
    for attr in _BLOCK_LIST_ATTRS:
        blocks = getattr(transformer, attr, None)
        if blocks is not None and len(blocks) > 0:
            return blocks
    return None


def layer0_grad_norm(transformer) -> float:
    """Global L2 norm of transformer block 0's trainable gradients.

    Block 0 is the reference surface 5a-i already isolates: its grad is the
    *last* one produced during backprop, so a healthy value implies the whole
    forward + chain-rule path is intact.

    Accumulates the squared sums on the GPU and does a single CPU-GPU sync
    (``.item()``) at the end, rather than one per parameter.
    """
    blocks = resolve_blocks(transformer)
    assert blocks is not None and len(blocks) > 0, (
        "transformer is expected to expose a non-empty block list "
        f"(one of {_BLOCK_LIST_ATTRS})")
    grads = [
        p.grad for p in blocks[0].parameters()
        if p.requires_grad and p.grad is not None
    ]
    if not grads:
        return 0.0
    # Some loaders (e.g. Cosmos via fsdp_load) keep parameters/grads as
    # DTensors even at world_size=1; localize so the reduction stays on plain
    # tensors. ``to_local`` is a no-op for ordinary tensors at ws=1.
    grads = [g.to_local() if hasattr(g, "to_local") else g for g in grads]
    sq_sum = torch.zeros((), device=grads[0].device, dtype=torch.float32)
    for g in grads:
        sq_sum += g.detach().float().pow(2).sum()
    return sq_sum.sqrt().item()


def _load_refs() -> dict[str, dict[str, float]]:
    if _REFS_PATH.exists():
        return json.loads(_REFS_PATH.read_text(encoding="utf-8"))
    return {}


def _save_refs(refs: dict[str, dict[str, float]]) -> None:
    _REFS_PATH.write_text(
        json.dumps(refs, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")


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
