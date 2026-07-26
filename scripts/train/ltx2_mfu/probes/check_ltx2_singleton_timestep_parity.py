#!/usr/bin/env python3
"""CPU proof that uniform LTX-2 token timesteps may be embedded once."""

from __future__ import annotations

import argparse
import copy
import json

import torch

from fastvideo.models.dits.ltx2 import AdaLayerNormSingle, _to_denoised


# These are norm-based numerical gates, not bitwise-parity claims. Changing
# the number of rows presented to a GEMM may change its reduction schedule.
_FLOAT32_FORWARD_MAX_ABS = 1e-6
_FLOAT32_FORWARD_RELATIVE_L2 = 2e-6
_FLOAT32_GRADIENT_RELATIVE_L2 = 2e-6
_SIGMA_GRADIENT_RELATIVE_L2 = 5e-6
_GRADIENT_COSINE = 0.99999999
_FLOAT64_FORWARD_MAX_ABS = 1e-12
_FLOAT64_FORWARD_RELATIVE_L2 = 1e-13
_FLOAT64_GRADIENT_RELATIVE_L2 = 1e-12


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    left = left.detach().double().flatten()
    right = right.detach().double().flatten()
    denominator = left.norm() * right.norm()
    return float(torch.dot(left, right) / denominator) if denominator else 1.0


def _comparison(left: torch.Tensor, right: torch.Tensor) -> dict[str, float]:
    left = left.detach().double()
    right = right.detach().double()
    difference = left - right
    difference_norm = torch.linalg.vector_norm(difference)
    reference_norm = torch.linalg.vector_norm(left)
    return {
        "max_abs": float(difference.abs().max()),
        "rmse": float(torch.sqrt(torch.mean(difference.square()))),
        "relative_l2": float(difference_norm / max(reference_norm, torch.finfo(torch.float64).eps)),
        "relative_to_max": float(
            difference.abs().max() / max(left.abs().max(), torch.finfo(torch.float64).eps)
        ),
        "cosine": _cosine(left, right),
        "exact_fraction": float(torch.mean((left == right).double())),
    }


def _forward_stages(
    adaln: AdaLayerNormSingle,
    sigma: torch.Tensor,
    hidden_dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    time_projection = adaln.emb.time_proj(sigma)
    timestep_linear_1 = adaln.emb.timestep_embedder.linear_1(
        time_projection.to(dtype=hidden_dtype)
    )
    timestep_silu = adaln.emb.timestep_embedder.act(timestep_linear_1)
    embedding = adaln.emb.timestep_embedder.linear_2(timestep_silu)
    modulation_silu = adaln.silu(embedding)
    modulation = adaln.linear(modulation_silu)
    return {
        "time_projection": time_projection,
        "timestep_linear_1": timestep_linear_1,
        "timestep_silu": timestep_silu,
        "embedding": embedding,
        "modulation_silu": modulation_silu,
        "modulation": modulation,
    }


def _run_case(
    coefficient: int,
    dtype: torch.dtype,
    token_count: int,
) -> dict[str, object]:
    torch.manual_seed(20260721 + coefficient)
    torch.set_num_threads(1)

    batch_size, hidden_size = 2, 32
    expanded_adaln = AdaLayerNormSingle(hidden_size, embedding_coefficient=coefficient).float().to(dtype)
    singleton_adaln = copy.deepcopy(expanded_adaln)

    sigma_values = torch.rand(batch_size, 1, dtype=torch.float32).to(dtype)
    expanded_sigma = sigma_values.detach().clone().requires_grad_(True)
    singleton_sigma = expanded_sigma.detach().clone().requires_grad_(True)
    expanded_input = expanded_sigma.expand(batch_size, token_count).contiguous().flatten()
    singleton_input = singleton_sigma.flatten()

    expanded_modulation, expanded_embedding = expanded_adaln(expanded_input, hidden_dtype=dtype)
    singleton_modulation, singleton_embedding = singleton_adaln(singleton_input, hidden_dtype=dtype)
    expanded_modulation = expanded_modulation.view(batch_size, token_count, coefficient, hidden_size)
    singleton_modulation = singleton_modulation.view(batch_size, 1, coefficient, hidden_size)
    expanded_embedding = expanded_embedding.view(batch_size, token_count, hidden_size)
    singleton_embedding = singleton_embedding.view(batch_size, 1, hidden_size)

    singleton_modulation_broadcast = singleton_modulation.expand_as(expanded_modulation)
    singleton_embedding_broadcast = singleton_embedding.expand_as(expanded_embedding)
    modulation_comparison = _comparison(expanded_modulation, singleton_modulation_broadcast)
    embedding_comparison = _comparison(expanded_embedding, singleton_embedding_broadcast)

    with torch.no_grad():
        expanded_stages = _forward_stages(expanded_adaln, expanded_input, dtype)
        singleton_stages = _forward_stages(singleton_adaln, singleton_input, dtype)
    stage_comparisons: dict[str, dict[str, float]] = {}
    expanded_repeat_consistency: dict[str, dict[str, float]] = {}
    for name, expanded_stage in expanded_stages.items():
        singleton_stage = singleton_stages[name]
        expanded_stage = expanded_stage.view(batch_size, token_count, -1)
        singleton_stage = singleton_stage.view(batch_size, 1, -1)
        stage_comparisons[name] = _comparison(
            expanded_stage,
            singleton_stage.expand_as(expanded_stage),
        )
        expanded_repeat_consistency[name] = _comparison(
            expanded_stage,
            expanded_stage[:, :1].expand_as(expanded_stage),
        )

    # A distinct upstream gradient per token covers every downstream broadcast
    # consumer: block Ada values and the final output modulation.
    modulation_grad = torch.randn(
        expanded_modulation.shape,
        dtype=torch.float32,
    ).to(dtype)
    embedding_grad = torch.randn(
        expanded_embedding.shape,
        dtype=torch.float32,
    ).to(dtype)
    expanded_loss = (expanded_modulation * modulation_grad).sum() + (expanded_embedding * embedding_grad).sum()
    singleton_loss = ((singleton_modulation * modulation_grad).sum()
                      + (singleton_embedding * embedding_grad).sum())
    expanded_loss.backward()
    singleton_loss.backward()

    parameter_checks: dict[str, dict[str, float]] = {}
    worst_gradient_max_abs = 0.0
    worst_gradient_relative_l2 = 0.0
    worst_gradient_relative_to_max = 0.0
    worst_gradient_cosine = 1.0
    for (expanded_name, expanded_parameter), (singleton_name, singleton_parameter) in zip(
            expanded_adaln.named_parameters(), singleton_adaln.named_parameters(), strict=True):
        assert expanded_name == singleton_name
        assert expanded_parameter.grad is not None and singleton_parameter.grad is not None
        comparison = _comparison(expanded_parameter.grad, singleton_parameter.grad)
        parameter_checks[expanded_name] = comparison
        worst_gradient_max_abs = max(worst_gradient_max_abs, comparison["max_abs"])
        worst_gradient_relative_l2 = max(worst_gradient_relative_l2, comparison["relative_l2"])
        worst_gradient_relative_to_max = max(
            worst_gradient_relative_to_max,
            comparison["relative_to_max"],
        )
        worst_gradient_cosine = min(worst_gradient_cosine, comparison["cosine"])

    assert expanded_sigma.grad is not None and singleton_sigma.grad is not None
    sigma_gradient_comparison = _comparison(expanded_sigma.grad, singleton_sigma.grad)

    # The wrapper's final denoising conversion also broadcasts [B, 1] over
    # [B, tokens, channels].
    sample = torch.randn(batch_size, token_count, hidden_size, dtype=torch.float32).to(dtype)
    velocity = torch.randn_like(sample)
    expanded_denoised = _to_denoised(
        sample,
        velocity,
        expanded_sigma.detach().expand(batch_size, token_count),
        calc_dtype=dtype,
    )
    singleton_denoised = _to_denoised(
        sample,
        velocity,
        singleton_sigma.detach(),
        calc_dtype=dtype,
    )
    denoised_comparison = _comparison(expanded_denoised, singleton_denoised)

    return {
        "coefficient": coefficient,
        "dtype": str(dtype),
        "token_count": token_count,
        "modulation": modulation_comparison,
        "embedding": embedding_comparison,
        "denoised": denoised_comparison,
        "forward_stages": stage_comparisons,
        "expanded_repeat_consistency": expanded_repeat_consistency,
        "worst_gradient_max_abs": worst_gradient_max_abs,
        "worst_gradient_relative_l2": worst_gradient_relative_l2,
        "worst_gradient_relative_to_max": worst_gradient_relative_to_max,
        "worst_gradient_cosine": worst_gradient_cosine,
        "sigma_gradient": sigma_gradient_comparison,
        "parameter_checks": parameter_checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagnostic-only", action="store_true")
    parser.add_argument("--token-count", type=int, default=4290)
    args = parser.parse_args()
    if args.token_count <= 0:
        parser.error("--token-count must be positive")

    results = [
        _run_case(coefficient, dtype, args.token_count)
        for dtype in (torch.float32, torch.float64)
        for coefficient in (6, 9)
    ]
    print(
        "LTX2_SINGLETON_PARITY " + json.dumps(results, sort_keys=True),
        flush=True,
    )
    if args.diagnostic_only:
        return

    for result in results:
        forward_max_abs = (
            _FLOAT32_FORWARD_MAX_ABS
            if result["dtype"] == "torch.float32"
            else _FLOAT64_FORWARD_MAX_ABS
        )
        forward_relative_l2 = (
            _FLOAT32_FORWARD_RELATIVE_L2
            if result["dtype"] == "torch.float32"
            else _FLOAT64_FORWARD_RELATIVE_L2
        )
        for comparison in result["forward_stages"].values():
            assert comparison["max_abs"] < forward_max_abs
            assert comparison["relative_l2"] < forward_relative_l2
        for comparison in result["expanded_repeat_consistency"].values():
            assert comparison["max_abs"] < forward_max_abs
            assert comparison["relative_l2"] < forward_relative_l2

        if result["dtype"] == "torch.float32":
            assert result["worst_gradient_relative_l2"] < _FLOAT32_GRADIENT_RELATIVE_L2
        else:
            assert result["worst_gradient_relative_l2"] < _FLOAT64_GRADIENT_RELATIVE_L2
        assert result["worst_gradient_cosine"] > _GRADIENT_COSINE
        # get_timestep_embedding() explicitly computes in float32 even when
        # the surrounding AdaLN is float64. Its sigma-gradient reduction is
        # therefore expected to use the float32 tolerance in both cases.
        assert result["sigma_gradient"]["relative_l2"] < _SIGMA_GRADIENT_RELATIVE_L2
        assert result["sigma_gradient"]["cosine"] > _GRADIENT_COSINE
        assert result["denoised"]["max_abs"] == 0.0


if __name__ == "__main__":
    main()
