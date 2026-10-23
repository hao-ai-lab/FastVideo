# SPDX-License-Identifier: Apache-2.0
"""Resolve MLX H3 VSA settings against a FastH3 inference contract.

Mirrors ``MiniMaxH3PipelineConfig.resolve_checkpoint_settings``: an unset value
takes the trained one, a different tile size raises (the checkpoint's gates and
weights were trained for one tile geometry), and a different sparsity or keep
rate runs with a warning.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fastvideo.logger import init_logger
from fastvideo.mlx_runtime.minimax_h3_vsa import VSA_H3_TILE_SHAPES, validate_ref_policy

logger = init_logger(__name__)

VSA_ATTENTION_BACKEND = "VIDEO_SPARSE_ATTN_H3"
DEFAULT_VSA_SPARSITY = 0.9
DEFAULT_VSA_TILE_SIZE = 64


@dataclass(frozen=True)
class ResolvedVSASettings:
    enabled: bool
    sparsity: float
    tile_size: int
    ref_policy: str | None
    ref_keep_rate: float | None


def contract_declares_vsa(contract: dict[str, Any] | None) -> bool:
    """True for a contract trained with H3 VSA (it names the backend and a tile size)."""
    return (contract is not None and contract.get("attention_backend") == VSA_ATTENTION_BACKEND
            and "vsa_tile_size" in contract)


def resolve_vsa_settings(
    contract: dict[str, Any] | None,
    *,
    vsa: bool | None = None,
    sparsity: float | None = None,
    tile_size: int | None = None,
    ref_policy: str | None = None,
    ref_keep_rate: float | None = None,
    checkpoint_capable: bool = True,
) -> ResolvedVSASettings:
    """``vsa=None`` means on for a VSA contract (when the checkpoint can) and off otherwise."""
    declared = contract_declares_vsa(contract)
    trained = contract if declared and contract is not None else {}
    if vsa is None:
        enabled = declared
        if enabled and not checkpoint_capable:
            logger.warning("The FastH3 contract is trained for VSA but this MLX checkpoint is dense-only "
                           "(no to_gate_compress weights); running dense. Reconvert with --include-vsa.")
            enabled = False
    else:
        enabled = bool(vsa)

    trained_tile = trained.get("vsa_tile_size")
    if tile_size is None:
        tile_size = int(trained_tile) if trained_tile is not None else DEFAULT_VSA_TILE_SIZE
    elif enabled and trained_tile is not None and tile_size != trained_tile:
        raise ValueError(f"This FastH3 checkpoint was trained with vsa_tile_size={trained_tile}; this run requests "
                         f"{tile_size}. Leave it unset or pass the trained value.")
    if enabled and tile_size not in VSA_H3_TILE_SHAPES:
        raise ValueError(f"VSA tile_size must be one of {sorted(VSA_H3_TILE_SHAPES)}, got {tile_size}.")

    trained_sparsity = trained.get("vsa_sparsity")
    if sparsity is None:
        sparsity = float(trained_sparsity) if trained_sparsity is not None else DEFAULT_VSA_SPARSITY
    elif enabled and trained_sparsity is not None and sparsity != trained_sparsity:
        logger.warning("FastH3 checkpoint was trained with vsa_sparsity=%s; this run uses %s.", trained_sparsity,
                       sparsity)

    trained_policy = trained.get("vsa_ref_policy")
    if ref_policy is None:
        ref_policy = trained_policy
    elif enabled and trained_policy is not None and ref_policy != trained_policy:
        raise ValueError(f"This FastH3 checkpoint was trained with vsa_ref_policy={trained_policy!r}; this run "
                         f"requests {ref_policy!r}.")
    trained_keep = trained.get("vsa_ref_keep_rate")
    if ref_keep_rate is None:
        ref_keep_rate = float(trained_keep) if ref_policy is not None and trained_keep is not None else None
    elif enabled and trained_keep is not None and ref_keep_rate != trained_keep:
        logger.warning("FastH3 checkpoint was trained with vsa_ref_keep_rate=%s; this run uses %s.", trained_keep,
                       ref_keep_rate)
    if enabled:
        validate_ref_policy(ref_policy, ref_keep_rate)
    else:
        ref_policy, ref_keep_rate = None, None
    return ResolvedVSASettings(enabled, float(sparsity), int(tile_size), ref_policy, ref_keep_rate)
