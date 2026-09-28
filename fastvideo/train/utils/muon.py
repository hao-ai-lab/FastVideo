# SPDX-License-Identifier: Apache-2.0
"""Muon optimizer (FSDP2 / DTensor-aware) with an auxiliary AdamW group.

Muon (MomentUm Orthogonalized by Newton-schulz, Keller Jordan 2024) replaces
the raw momentum update of a 2-D weight matrix with its nearest semi-orthogonal
matrix, computed by a few Newton-Schulz iterations. It is applied to the hidden
weight matrices of the transformer (attention q/k/v/out projections, MLP fc
layers) while embeddings, the output head, and all 1-D parameters (norm/bias)
stay on AdamW — the standard Muon recipe.

FSDP2 note: under ``fully_shard`` the parameters are ``DTensor``s sharded along
dim-0, but Newton-Schulz needs the *full* 2-D matrix. We keep the momentum
buffer sharded (memory-efficient, like the parameter) and only gather each
matrix transiently for the orthogonalization step (``full_tensor()``), then
re-shard the update back to the parameter's placement (gather-per-matrix). Peak
extra memory is one full matrix at a time, not the whole replicated optimizer
state.
"""

from __future__ import annotations

import torch

try:
    from torch.distributed.tensor import DTensor, distribute_tensor
    _HAS_DTENSOR = True
except Exception:  # pragma: no cover - older torch
    DTensor = ()  # type: ignore[assignment]
    _HAS_DTENSOR = False


def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int) -> torch.Tensor:
    """Newton-Schulz iteration to orthogonalize ``G`` (Keller Jordan quintic).

    Returns a matrix with the same shape as ``G`` whose singular values are
    pushed toward 1. Runs in bf16; the quintic coefficients are tuned so the
    iteration converges from a spectral-norm-normalized start.
    """
    assert G.ndim == 2, f"expected a 2-D matrix, got shape {tuple(G.shape)}"
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    transposed = False
    if X.size(0) > X.size(1):
        X = X.T
        transposed = True
    # Normalize so the spectral norm is <= 1 before the iteration.
    X = X / (X.norm() + 1e-7)
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X.to(G.dtype)


def _is_dtensor(t: torch.Tensor) -> bool:
    return _HAS_DTENSOR and isinstance(t, DTensor)


def _full(t: torch.Tensor) -> torch.Tensor:
    """Gather a (possibly sharded) tensor to its full local form."""
    return t.full_tensor() if _is_dtensor(t) else t


def _like_param(update_full: torch.Tensor, param: torch.Tensor) -> torch.Tensor:
    """Re-shard a full update tensor to match ``param``'s DTensor placement."""
    if _is_dtensor(param):
        return distribute_tensor(update_full, param.device_mesh, param.placements)
    return update_full


def split_params_for_muon(
    named_params: list[tuple[str, torch.nn.Parameter]], ) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter]]:
    """Partition trainable params into (muon, aux-adam) groups.

    Muon eligibility: exactly 2-D weight matrices that are *not* embeddings or
    the final output head. Everything else — 1-D params (norms/biases), >2-D
    conv/patch-embed weights, embeddings, and the output projection — goes to
    AdamW. ``out_proj`` / ``to_out`` (attention output projections) ARE hidden
    matmuls and stay on Muon; only the model's final head (``proj_out`` /
    ``final_layer`` / ``*head``) is excluded.
    """
    _EXCLUDE = ("embed", "embedder", "proj_out", "final_layer", "head")
    muon: list[torch.nn.Parameter] = []
    aux: list[torch.nn.Parameter] = []
    for name, p in named_params:
        if not p.requires_grad:
            continue
        lname = name.lower()
        if p.ndim == 2 and not any(tok in lname for tok in _EXCLUDE):
            muon.append(p)
        else:
            aux.append(p)
    return muon, aux


class MuonWithAuxAdam(torch.optim.Optimizer):
    """Single optimizer running Muon on group 0 and AdamW on the aux group.

    Param groups carry a ``use_muon`` flag. The Muon group uses
    ``lr/momentum/weight_decay/ns_steps``; the aux group uses
    ``lr/betas/eps/weight_decay``. Keeping both in one optimizer means the
    existing single-optimizer / single-scheduler trainer plumbing is unchanged.
    """

    def __init__(
        self,
        muon_params: list[torch.nn.Parameter],
        aux_params: list[torch.nn.Parameter],
        *,
        lr: float,
        momentum: float = 0.95,
        weight_decay: float = 0.0,
        ns_steps: int = 5,
        nesterov: bool = True,
        aux_lr: float | None = None,
        aux_betas: tuple[float, float] = (0.9, 0.999),
        aux_eps: float = 1e-8,
        aux_weight_decay: float | None = None,
    ) -> None:
        groups = []
        if muon_params:
            groups.append({
                "params": muon_params,
                "use_muon": True,
                "lr": float(lr),
                "momentum": float(momentum),
                "weight_decay": float(weight_decay),
                "ns_steps": int(ns_steps),
                "nesterov": bool(nesterov),
            })
        if aux_params:
            groups.append({
                "params": aux_params,
                "use_muon": False,
                "lr": float(aux_lr if aux_lr is not None else lr),
                "betas": tuple(aux_betas),
                "eps": float(aux_eps),
                "weight_decay": float(aux_weight_decay if aux_weight_decay is not None else weight_decay),
            })
        if not groups:
            raise ValueError("MuonWithAuxAdam got no parameters")
        super().__init__(groups, {})

    @torch.no_grad()
    def step(self, closure=None):  # type: ignore[override]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            if group["use_muon"]:
                self._muon_step(group)
            else:
                self._adam_step(group)
        return loss

    def _muon_step(self, group) -> None:
        lr = group["lr"]
        momentum = group["momentum"]
        wd = group["weight_decay"]
        ns_steps = group["ns_steps"]
        nesterov = group["nesterov"]
        for p in group["params"]:
            if p.grad is None:
                continue
            g = p.grad
            state = self.state[p]
            if "momentum_buffer" not in state:
                state["momentum_buffer"] = torch.zeros_like(p)
            buf = state["momentum_buffer"]
            # Sharded momentum update (DTensor ops stay on local shards).
            buf.mul_(momentum).add_(g)
            g_eff = g.add(buf, alpha=momentum) if nesterov else buf
            # Gather the full matrix only for the orthogonalization.
            update_full = zeropower_via_newtonschulz5(_full(g_eff), ns_steps)
            # Keller-Jordan RMS-matching scale for non-square matrices.
            rows, cols = update_full.shape
            scale = max(1.0, rows / cols)**0.5
            if wd != 0.0:
                p.mul_(1.0 - lr * wd)
            p.add_(_like_param(update_full, p), alpha=-lr * scale)

    def _adam_step(self, group) -> None:
        lr = group["lr"]
        beta1, beta2 = group["betas"]
        eps = group["eps"]
        wd = group["weight_decay"]
        for p in group["params"]:
            if p.grad is None:
                continue
            g = p.grad
            state = self.state[p]
            if "step" not in state:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(p)
                state["exp_avg_sq"] = torch.zeros_like(p)
            state["step"] += 1
            t = state["step"]
            exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
            exp_avg.mul_(beta1).add_(g, alpha=1.0 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(g, g, value=1.0 - beta2)
            bias1 = 1.0 - beta1**t
            bias2 = 1.0 - beta2**t
            denom = (exp_avg_sq.sqrt() / (bias2**0.5)).add_(eps)
            if wd != 0.0:
                p.mul_(1.0 - lr * wd)
            p.addcdiv_(exp_avg, denom, value=-lr / bias1)
