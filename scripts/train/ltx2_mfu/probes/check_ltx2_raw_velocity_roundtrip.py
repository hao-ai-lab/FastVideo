#!/usr/bin/env python3
"""Quantify LTX-2's avoidable BF16 velocity -> x0 -> velocity error.

The modular fine-tuning target is raw flow velocity ``noise - clean``.  This
script feeds that exact target through FastVideo's real ``_to_denoised``
helper and the current modular adapter's inverse, then compares the result
with returning the raw transformer velocity directly.
"""

from __future__ import annotations

import argparse
import json
import math
from typing import Any

import torch

from fastvideo.models.dits.ltx2 import _to_denoised


DEFAULT_SIGMAS = (1.0, 0.5, 0.1, 0.01, 0.001)


def _metrics(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    actual = actual.float().flatten()
    expected = expected.float().flatten()
    error = actual - expected
    expected_rms = torch.sqrt(torch.mean(expected.square()))
    error_rms = torch.sqrt(torch.mean(error.square()))
    denominator = torch.linalg.vector_norm(actual) * torch.linalg.vector_norm(expected)
    cosine = torch.dot(actual, expected) / denominator
    return {
        "mse": float(torch.mean(error.square())),
        "rmse": float(error_rms),
        "relative_rmse": float(error_rms / expected_rms),
        "max_abs": float(error.abs().max()),
        "cosine": float(cosine),
        "exact_fraction": float(torch.mean((actual == expected).float())),
        "zero_fraction": float(torch.mean((actual == 0).float())),
    }


def _run_sigma(
    clean_bf16: torch.Tensor,
    noise_bf16: torch.Tensor,
    sigma_value: float,
) -> dict[str, Any]:
    sigma = torch.tensor(
        sigma_value,
        device=clean_bf16.device,
        dtype=torch.float32,
    ).view(1, 1, 1, 1, 1)

    # These expressions match LTX2Model._prepare_dit_inputs and FineTuneMethod:
    # x_t is accumulated in fp32 and stored in bf16, while the target is the
    # bf16 subtraction noise - clean.
    noisy_bf16 = (
        (1.0 - sigma) * clean_bf16.float() + sigma * noise_bf16.float()
    ).to(torch.bfloat16)
    target_bf16 = noise_bf16 - clean_bf16

    # Current path: the raw DiT output is converted to bf16 x0, only for the
    # modular adapter to reconstruct fp32 velocity by subtracting and dividing.
    denoised_bf16 = _to_denoised(
        noisy_bf16,
        target_bf16,
        sigma,
    )
    reconstructed_velocity = (
        noisy_bf16.float() - denoised_bf16.float()
    ) / sigma

    # Proposed training path: use the transformer's raw velocity as-is.
    direct_velocity = target_bf16.float()

    # FP32 reference separates unavoidable floating-point cancellation from
    # the extra bf16 x0 materialization in the current path.
    clean_fp32 = clean_bf16.float()
    noise_fp32 = noise_bf16.float()
    target_fp32 = noise_fp32 - clean_fp32
    noisy_fp32 = (1.0 - sigma) * clean_fp32 + sigma * noise_fp32
    denoised_fp32 = _to_denoised(
        noisy_fp32,
        target_fp32,
        sigma,
    )
    reconstructed_fp32 = (noisy_fp32 - denoised_fp32) / sigma

    return {
        "sigma": sigma_value,
        "direct_raw_velocity": _metrics(direct_velocity, target_bf16),
        "current_bf16_roundtrip": _metrics(
            reconstructed_velocity,
            target_bf16,
        ),
        "fp32_roundtrip_control": _metrics(
            reconstructed_fp32,
            target_fp32,
        ),
        "denoised_x0_vs_clean": _metrics(denoised_bf16, clean_bf16),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "cuda"),
    )
    parser.add_argument("--elements", type=int, default=262_144)
    parser.add_argument("--seed", type=int, default=20260721)
    args = parser.parse_args()

    if args.elements <= 0:
        parser.error("--elements must be positive")
    device = (
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu"
        if args.device == "auto"
        else args.device
    )
    if device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable")

    generator = torch.Generator(device=device).manual_seed(args.seed)
    shape = (1, 1, 1, 1, args.elements)
    clean_bf16 = torch.randn(
        shape,
        generator=generator,
        device=device,
        dtype=torch.float32,
    ).to(torch.bfloat16)
    noise_bf16 = torch.randn(
        shape,
        generator=generator,
        device=device,
        dtype=torch.float32,
    ).to(torch.bfloat16)

    results = [
        _run_sigma(clean_bf16, noise_bf16, sigma)
        for sigma in DEFAULT_SIGMAS
    ]
    direct_mses = [item["direct_raw_velocity"]["mse"] for item in results]
    current_mses = [item["current_bf16_roundtrip"]["mse"] for item in results]
    small_sigma_relative_rmse = results[-1]["current_bf16_roundtrip"]["relative_rmse"]

    if any(value != 0.0 for value in direct_mses):
        raise AssertionError(f"direct raw velocity was not exact: {direct_mses}")
    if not all(math.isfinite(value) for value in current_mses):
        raise AssertionError(f"non-finite current-path MSE: {current_mses}")
    if current_mses[-1] <= current_mses[2] * 100.0:
        raise AssertionError(
            "expected sigma=1e-3 BF16 round-trip MSE to exceed sigma=0.1 "
            f"by >100x, got {current_mses[-1]} vs {current_mses[2]}"
        )
    if small_sigma_relative_rmse <= 0.1:
        raise AssertionError(
            "expected material small-sigma error, got relative RMSE "
            f"{small_sigma_relative_rmse}"
        )

    print(
        "LTX2_RAW_VELOCITY_PROOF " + json.dumps({
            "device": device,
            "dtype": "torch.bfloat16",
            "elements": args.elements,
            "seed": args.seed,
            "training_target": "noise - clean (raw flow velocity)",
            "source_evidence": [
                "fastvideo/train/methods/fine_tuning/finetune.py:99-100",
                "fastvideo/training/ltx2_training_pipeline.py:435-438",
                "fastvideo/models/schedulers/scheduling_self_forcing_flow_match.py:140-141",
            ],
            "results": results,
        }, sort_keys=True),
        flush=True,
    )


if __name__ == "__main__":
    main()
