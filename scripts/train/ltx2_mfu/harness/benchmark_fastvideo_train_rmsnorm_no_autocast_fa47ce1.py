#!/usr/bin/env python3
"""Scratch candidate: run base-stage LTX RMSNorm outside CUDA autocast."""

from __future__ import annotations

import json
import os

import torch

from benchmark_fastvideo_train_pack_d016 import main


def _install_candidate() -> None:
    import fastvideo.models.dits.ltx2 as ltx2

    original_dispatch = ltx2._rms_norm_dispatch
    original_stage_aware_forward = ltx2.StageAwareRMSNorm.forward

    def rms_norm_no_autocast(
        x: torch.Tensor,
        eps: float,
        weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        with torch.autocast(device_type="cuda", enabled=False):
            return torch.nn.functional.rms_norm(
                x,
                (x.shape[-1], ),
                weight=weight,
                eps=eps,
            ).to(x.dtype)

    def dispatch(
        x: torch.Tensor,
        eps: float,
        weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if ltx2._is_ltx2_refine_stage():
            return original_dispatch(x, eps=eps, weight=weight)
        return rms_norm_no_autocast(x, eps=eps, weight=weight)

    def stage_aware_forward(
        self: ltx2.StageAwareRMSNorm,
        x: torch.Tensor,
    ) -> torch.Tensor:
        if ltx2._is_ltx2_refine_stage():
            return original_stage_aware_forward(self, x)
        return rms_norm_no_autocast(x, eps=self.eps, weight=self.weight)

    if not torch.cuda.is_available():
        raise RuntimeError("RMSNorm candidate requires CUDA")

    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))
    generator = torch.Generator(device=device).manual_seed(20260721)
    x = torch.randn(
        (2, 3, 32),
        device=device,
        dtype=torch.bfloat16,
        generator=generator,
    )
    weight = torch.linspace(
        0.5,
        1.5,
        x.shape[-1],
        device=x.device,
        dtype=x.dtype,
    )
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        old_unweighted = original_dispatch(x, eps=1e-6)
        old_norm = ltx2.StageAwareRMSNorm(x.shape[-1], eps=1e-6).to(
            device=x.device,
            dtype=x.dtype,
        )
        with torch.no_grad():
            old_norm.weight.copy_(weight)
        old_weighted = original_stage_aware_forward(old_norm, x)

    ltx2._rms_norm_dispatch = dispatch
    ltx2.StageAwareRMSNorm.forward = stage_aware_forward

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        new_unweighted = ltx2._rms_norm_dispatch(x, eps=1e-6)
        new_norm = ltx2.StageAwareRMSNorm(x.shape[-1], eps=1e-6).to(
            device=x.device,
            dtype=x.dtype,
        )
        with torch.no_grad():
            new_norm.weight.copy_(weight)
        new_weighted = new_norm(x)
    with torch.autocast(device_type="cuda", enabled=False):
        native_unweighted = torch.nn.functional.rms_norm(
            x,
            (x.shape[-1], ),
            eps=1e-6,
        ).to(x.dtype)
        native_weighted = torch.nn.functional.rms_norm(
            x,
            (x.shape[-1], ),
            weight=weight,
            eps=1e-6,
        ).to(x.dtype)

    if new_unweighted.dtype != x.dtype or new_weighted.dtype != x.dtype:
        raise RuntimeError(
            f"RMSNorm candidate changed dtype: input={x.dtype} "
            f"unweighted={new_unweighted.dtype} weighted={new_weighted.dtype}"
        )
    torch.testing.assert_close(new_unweighted, native_unweighted, rtol=0, atol=0)
    torch.testing.assert_close(new_weighted, native_weighted, rtol=0, atol=0)
    torch.testing.assert_close(new_unweighted, old_unweighted, rtol=0.02, atol=0.02)
    torch.testing.assert_close(new_weighted, old_weighted, rtol=0.02, atol=0.02)
    if torch.equal(new_unweighted, new_weighted):
        raise RuntimeError("weighted StageAwareRMSNorm ignored its learned weight")

    print(
        "RMSNORM_SEMANTICS "
        + json.dumps({
            "candidate_installed": True,
            "input_dtype": str(x.dtype),
            "output_dtype": str(new_weighted.dtype),
            "native_unweighted_exact": torch.equal(new_unweighted, native_unweighted),
            "native_weighted_exact": torch.equal(new_weighted, native_weighted),
            "old_unweighted_max_abs_diff": float((new_unweighted - old_unweighted).abs().max()),
            "old_weighted_max_abs_diff": float((new_weighted - old_weighted).abs().max()),
            "stage_aware_weight_effective": not torch.equal(new_unweighted, new_weighted),
            "refine_path": "delegates_to_original",
        }, sort_keys=True),
        flush=True,
    )


if __name__ == "__main__":
    _install_candidate()
    main()
