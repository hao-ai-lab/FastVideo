# SPDX-License-Identifier: Apache-2.0

import pytest

from fastvideo.benchmarks.mlx_fastwan_bench import (
    ALLOWED_MODES,
    BENCHMARK_PRESETS,
    _html_grid,
    _load_prompt_cases,
    _mode_to_dtype_quant,
    _parse_list,
)


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        ("fp16", ("fp16", None)),
        ("bf16", ("bf16", None)),
        ("int8", ("fp16", "int8")),
        ("int4", ("fp16", "int4")),
        ("mxfp8", ("fp16", "mxfp8")),
        ("mxfp4", ("fp16", "mxfp4")),
        ("nvfp4", ("fp16", "nvfp4")),
    ],
)
def test_benchmark_modes_map_to_runtime_quantization(mode: str, expected: tuple[str, str | None]) -> None:
    assert mode in ALLOWED_MODES
    assert _mode_to_dtype_quant(mode) == expected


def test_benchmark_rejects_unknown_modes() -> None:
    with pytest.raises(ValueError, match="Unsupported modes"):
        _parse_list("fp16,not_a_mode", ALLOWED_MODES, "modes")


def test_load_prompt_cases_from_plain_text(tmp_path) -> None:
    prompt_file = tmp_path / "prompts.txt"
    prompt_file.write_text("\n# comment\nA fox runs through a forest.\nA raccoon walks in sunflowers.\n")

    cases = _load_prompt_cases("unused", prompt_file)

    assert [case.id for case in cases] == ["prompt-001", "prompt-002"]
    assert [case.prompt for case in cases] == [
        "A fox runs through a forest.",
        "A raccoon walks in sunflowers.",
    ]


def test_load_prompt_cases_from_jsonl(tmp_path) -> None:
    prompt_file = tmp_path / "prompts.jsonl"
    prompt_file.write_text('{"id": "Fox Forest", "prompt": "A fox runs."}\n{"name": "clock", "caption": "A clock burns."}\n')

    cases = _load_prompt_cases("unused", prompt_file)

    assert [case.id for case in cases] == ["fox-forest", "clock"]
    assert [case.prompt for case in cases] == ["A fox runs.", "A clock burns."]


def test_load_builtin_prompt_set() -> None:
    cases = _load_prompt_cases("unused", None, "motion7")
    assert len(cases) == 7
    assert cases[0].id == "beach-sunset"


def test_benchmark_presets_include_memory_tiers() -> None:
    assert BENCHMARK_PRESETS["mac-16gb"].modes == "int8"
    assert BENCHMARK_PRESETS["mac-16gb"].decoders == "taehv"
    assert BENCHMARK_PRESETS["mac-16gb"].mlx_memory_limit_gib == 16.0
    assert BENCHMARK_PRESETS["mac-64gb"].decoders == "taehv,wan-vae"


def test_html_grid_includes_video_and_sync_controls() -> None:
    rendered = _html_grid([
        {
            "prompt_id": "fox",
            "prompt": "A fox runs.",
            "mode": "int8",
            "decoder": "taehv",
            "status": "ok",
            "video_path": "fox/video_int8_taehv.mp4",
            "total_s": 12.3,
            "denoise_s": 10.0,
            "decode_s": 1.0,
            "peak_gib": 4.5,
        }
    ])

    assert "Restart + play all" in rendered
    assert "fox/video_int8_taehv.mp4" in rendered
    assert "A fox runs." in rendered


def test_denoise_dmd_on_device_runs_tiny_dit_and_reports_step_times(distributed_setup) -> None:
    mx = pytest.importorskip("mlx.core", reason="MLX is required for the on-device denoise test")
    import numpy as np
    import torch

    from fastvideo.benchmarks.mlx_fastwan_bench import denoise_dmd_on_device
    from fastvideo.mlx_runtime.sampling import MLXDMDSchedule, dmd_step
    from fastvideo.models.schedulers.scheduling_flow_match_euler_discrete import (
        FlowMatchEulerDiscreteScheduler,
    )
    from fastvideo.tests.mlx.tiny_wan import (
        build_hf_config,
        build_inputs,
        build_tiny_wan_config,
        build_torch_model,
        mlx_dit_from_torch_model,
        mlx_rotary_embeddings,
    )

    dit = mlx_dit_from_torch_model(build_torch_model(), build_hf_config(build_tiny_wan_config()))
    hidden_states, encoder_hidden_states, _ = build_inputs()
    schedule = MLXDMDSchedule.from_torch_scheduler(FlowMatchEulerDiscreteScheduler(shift=8.0))

    timesteps = [1000, 757, 522]
    generator = torch.Generator(device="cpu").manual_seed(7)
    latents_seed = torch.randn(hidden_states.shape, generator=generator, dtype=torch.float32).numpy()
    renoise_by_step = [
        torch.randn(hidden_states.shape, generator=generator, dtype=torch.float32).numpy()
        for _ in range(len(timesteps) - 1)
    ]

    latents_np, step_times = denoise_dmd_on_device(
        mx=mx,
        dit=dit,
        latents=mx.array(latents_seed),
        encoder_hidden_states=mx.array(encoder_hidden_states.numpy()),
        freqs_cis=mlx_rotary_embeddings(hidden_states),
        timesteps=timesteps,
        renoise_by_step=renoise_by_step,
        schedule=schedule,
        dmd_step=dmd_step,
        mx_dtype=mx.float32,
    )

    assert latents_np.shape == tuple(hidden_states.shape)
    assert np.isfinite(latents_np).all()
    assert len(step_times) == len(timesteps)
    assert all(t > 0 for t in step_times)
