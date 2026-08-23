# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from fastvideo.models.schedulers.scheduling_minimax_h3 import MiniMaxH3Scheduler


REPO_ROOT = Path(__file__).resolve().parents[3]
EXAMPLE_PATH = REPO_ROOT / "examples" / "inference" / "basic" / "basic_fasth3.py"


def _load_example():
    spec = importlib.util.spec_from_file_location("basic_fasth3_contract", EXAMPLE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fasth3 = _load_example()


def _args(*overrides: str):
    return fasth3.parse_args(["--prompt", "a test prompt", *overrides])


def test_default_all_profile_matches_fastest_contract(tmp_path):
    args = _args()
    config = fasth3.build_generator_config(args)
    request = fasth3.build_request(args, tmp_path / "result.mp4", args.seed)
    environment = fasth3.profile_environment(args)

    assert args.profile == "all"
    assert args.num_gpus == 4
    assert args.vsa_sparsity == 0.9
    assert args.vsa_tile_size == 64
    assert args.vsa_kernel == "sm100a"
    assert args.seed == 1000
    assert args.warmup_seed == 999
    assert args.warmup is True
    assert args.repeats == 3
    assert args.inference_torch_compile is True
    assert args.ulysses_a2a == "off"

    assert config.engine.num_gpus == 4
    assert config.engine.parallelism.tp_size == 1
    assert config.engine.parallelism.sp_size == 4
    assert config.engine.use_fsdp_inference is False
    assert config.engine.offload.dit is False
    assert config.engine.offload.text_encoder is True
    assert config.engine.offload.vae is True
    assert config.engine.offload.pin_cpu_memory is True
    assert config.engine.compile.enabled is False
    assert config.engine.compile.vae_enabled is True
    assert config.pipeline.experimental == {
        "attention_backend": "VIDEO_SPARSE_ATTN_H3",
        "VSA_sparsity": 0.9,
        "VSA_tile_size": 64,
        "inference_torch_compile": True,
        "vae_parallel_decode": True,
        "vae_parallel_decode_strategy": "gather",
    }

    assert environment["FASTVIDEO_VSA_SM100A"] == "1"
    assert environment["FASTVIDEO_VSA_CUTEDSL"] == "0"
    assert environment["FASTVIDEO_DISABLE_ATTENTION_COMPILE"] == "0"
    assert environment["FASTVIDEO_FA4"] == "1"
    assert environment["FASTVIDEO_NVFP4_FA4"] == "0"
    assert environment["FASTVIDEO_MINIMAX_H3_FA4_PACKED_VARLEN"] == "0"
    assert environment["FASTVIDEO_MINIMAX_H3_FUSIONS"] == "all"
    assert environment["FASTVIDEO_INFERENCE_TORCH_COMPILE"] == "1"
    assert environment["FASTVIDEO_VAE_PARALLEL_DECODE"] == "1"
    assert environment["FASTVIDEO_VAE_PARALLEL_DECODE_STRATEGY"] == "gather"
    assert environment["FASTVIDEO_ULYSSES_A2A"] == "off"
    assert environment["FASTVIDEO_H3_VSA_PROBE"] is None

    assert request.sampling.num_inference_steps == 5
    assert request.sampling.seed == 1000
    assert request.sampling.guidance_scale == 1.0
    assert request.sampling.batch_cfg is False
    assert request.output.output_path == str(tmp_path / "result.mp4")


@pytest.mark.parametrize("num_frames", (124, 243, 345))
def test_fast_profile_supports_measured_durations_without_separate_scripts(tmp_path, num_frames):
    args = _args("--num-frames", str(num_frames))
    config = fasth3.build_generator_config(args)
    request = fasth3.build_request(args, tmp_path / f"result_{num_frames}.mp4", args.seed)

    assert args.inference_torch_compile is True
    assert args.ulysses_a2a == "off"
    assert config.pipeline.experimental["inference_torch_compile"] is True
    assert request.sampling.num_frames == num_frames


def test_compile_mode_requires_regional_compile_opt_out(capsys):
    with pytest.raises(SystemExit):
        _args("--compile-mode", "reduce-overhead")
    assert "--compile-mode cannot be combined with regional compile" in capsys.readouterr().err

    args = _args("--compile-mode", "reduce-overhead", "--no-inference-torch-compile")
    assert args.compile_mode == "reduce-overhead"
    assert args.inference_torch_compile is False


def test_default_sigma_grid_runs_exactly_four_dit_forwards():
    scheduler = MiniMaxH3Scheduler(shift=12.0)
    scheduler.set_timesteps(5)

    assert scheduler.sigmas is not None and len(scheduler.sigmas) == 5
    assert scheduler.timesteps is not None and len(scheduler.timesteps) == 4


def test_strict_profile_changes_only_non_parity_fusions():
    all_args = _args("--profile", "all")
    strict_args = _args("--profile", "strict")
    all_environment = fasth3.profile_environment(all_args)
    strict_environment = fasth3.profile_environment(strict_args)

    changed = {key for key in all_environment if all_environment[key] != strict_environment[key]}
    assert changed == {"FASTVIDEO_MINIMAX_H3_FUSIONS"}
    assert all_environment["FASTVIDEO_MINIMAX_H3_FUSIONS"] == "all"
    assert strict_environment["FASTVIDEO_MINIMAX_H3_FUSIONS"] == "0"
    assert fasth3.build_generator_config(all_args) == fasth3.build_generator_config(strict_args)


def test_opt_outs_override_inherited_environment(monkeypatch):
    args = _args(
        "--vsa-kernel",
        "triton",
        "--no-fa4",
        "--no-h3-fusions",
        "--no-compile-vae",
        "--no-parallel-vae",
        "--no-replicated-dit",
        "--no-pin-cpu-memory",
        "--no-inference-torch-compile",
    )
    expected = fasth3.profile_environment(args)
    for name in expected:
        monkeypatch.setenv(name, "inherited-experiment-value")

    fasth3.configure_environment(args)
    config = fasth3.build_generator_config(args)

    assert "FASTVIDEO_H3_VSA_PROBE" not in fasth3.os.environ
    assert fasth3.os.environ["FASTVIDEO_VSA_SM100A"] == "0"
    assert fasth3.os.environ["FASTVIDEO_VSA_CUTEDSL"] == "0"
    assert fasth3.os.environ["FASTVIDEO_DISABLE_ATTENTION_COMPILE"] == "0"
    assert fasth3.os.environ["FASTVIDEO_FA4"] == "0"
    assert fasth3.os.environ["FASTVIDEO_NVFP4_FA4"] == "0"
    assert fasth3.os.environ["FASTVIDEO_MINIMAX_H3_FUSIONS"] == "0"
    assert fasth3.os.environ["FASTVIDEO_MINIMAX_H3_FA4_PACKED_VARLEN"] == "0"
    assert fasth3.os.environ["FASTVIDEO_INFERENCE_TORCH_COMPILE"] == "0"
    assert fasth3.os.environ["FASTVIDEO_VAE_PARALLEL_DECODE"] == "0"
    assert fasth3.os.environ["FASTVIDEO_ULYSSES_A2A"] == "off"
    assert config.engine.compile.vae_enabled is False
    assert config.pipeline.experimental["vae_parallel_decode"] is False
    assert config.engine.use_fsdp_inference is True
    assert config.engine.offload.pin_cpu_memory is False


def test_selected_fast_profile_requires_its_optional_routes(monkeypatch):
    monkeypatch.setattr(fasth3, "_fa4_is_installed", lambda: False)
    monkeypatch.setattr(fasth3, "_sm100a_kernel_is_installed", lambda: False)

    with pytest.raises(RuntimeError, match="flash-attn-4"):
        fasth3.validate_profile_dependencies(_args())
    with pytest.raises(RuntimeError, match="fastvideo-kernel 0.3.4"):
        fasth3.validate_profile_dependencies(_args("--no-fa4"))

    fasth3.validate_profile_dependencies(_args("--no-fa4", "--vsa-kernel", "triton"))

    monkeypatch.setattr(fasth3, "_fa4_is_installed", lambda: True)
    monkeypatch.setattr(fasth3, "_sm100a_kernel_is_installed", lambda: True)
    fasth3.validate_profile_dependencies(_args())


def test_run_excludes_warmup_and_uses_distinct_measured_outputs(monkeypatch, tmp_path, capsys):
    calls = []

    class FakeGenerator:
        shutdown_called = False

        def generate(self, request):
            calls.append(request)
            Path(request.output.output_path).touch()
            return SimpleNamespace(
                video_path=request.output.output_path,
                generation_time=1.25,
                logging_info=SimpleNamespace(stages={"denoising": {"execution_time": 2.5}}),
            )

        def shutdown(self):
            self.shutdown_called = True

    fake_generator = FakeGenerator()

    class FakeVideoGenerator:

        @classmethod
        def from_config(cls, config):
            assert config == fasth3.build_generator_config(args)
            return fake_generator

    for name in fasth3.profile_environment(_args()):
        monkeypatch.setenv(name, "inherited-experiment-value")
    monkeypatch.setattr(fasth3, "validate_profile_dependencies", lambda args: None)
    monkeypatch.setattr(fasth3, "VideoGenerator", FakeVideoGenerator)
    clock = iter((10.0, 15.0, 20.0, 26.0, 30.0, 37.0, 40.0, 48.0))
    monkeypatch.setattr(fasth3.time, "perf_counter", lambda: next(clock))
    args = _args("--output", str(tmp_path))

    measured = fasth3.run(args)

    assert measured == [6.0, 7.0, 8.0]
    assert [request.sampling.seed for request in calls] == [999, 1000, 1000, 1000]
    assert [Path(request.output.output_path).name for request in calls] == [
        "_fasth3_warmup.mp4",
        "fasth3_all_run_01.mp4",
        "fasth3_all_run_02.mp4",
        "fasth3_all_run_03.mp4",
    ]
    assert (tmp_path / "_fasth3_warmup.mp4").is_file()
    assert fake_generator.shutdown_called is True
    output = capsys.readouterr().out
    assert "[warmup] wall=5.000s (excluded)" in output
    assert f"Warmup output written to: {tmp_path / '_fasth3_warmup.mp4'}" in output
    assert output.count("Output written to:") == 3
    assert output.count("Denoising time: 2.500s") == 3
    assert "Measured E2E wall times (n=3, warmup excluded): [6.0, 7.0, 8.0]" in output
    assert "Median E2E wall time: 7.000s" in output
    assert "Median denoising time: 2.500s" in output
