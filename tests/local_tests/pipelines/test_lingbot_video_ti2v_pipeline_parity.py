# SPDX-License-Identifier: Apache-2.0
"""Exact one-step, 40-step, and decoded-pixel LingBot-Video TI2V parity."""

from __future__ import annotations

import gc
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, cast

import pytest
import torch

os.environ.setdefault("FASTVIDEO_ATTENTION_BACKEND", "TORCH_SDPA")
os.environ.setdefault("LINGBOT_MOE_EXPERT_BACKEND", "grouped_mm")
os.environ.setdefault("LINGBOT_MOE_PAD_BACKEND", "loop")
os.environ.setdefault("LINGBOT_MOE_REORDER_BACKEND", "sort")
os.environ.setdefault("LINGBOT_MOE_RESTORE_BACKEND", "scatter")

from fastvideo.configs.models.dits.lingbot_video import LingBotVideoConfig
from fastvideo.models.dits.lingbot_video import LingBotVideoTransformer3DModel
from fastvideo.models.loader.fsdp_load import maybe_load_fsdp_model
from tests.local_tests.lingbot_video.hf_assets import (
    FASTVIDEO_DENSE,
    FASTVIDEO_MOE,
    download_components,
    materialize_component_view,
)

BASELINE_DIR = os.environ.get("LINGBOT_VIDEO_TI2V_BASELINE_DIR")
CASE_DIR = os.environ.get("LINGBOT_VIDEO_TI2V_CASE_DIR")
RESULT_DIR = os.environ.get("LINGBOT_VIDEO_TI2V_RESULT_DIR")
VARIANT = os.environ.get("LINGBOT_VIDEO_TI2V_VARIANT", "dense")


def _configure_backends(_worker: Any | None = None) -> dict[str, bool]:
    """Apply the baseline's deterministic CUDA policy in parent and worker processes."""
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.enabled = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.enable_cudnn_sdp(False)
    return {
        "deterministic": torch.are_deterministic_algorithms_enabled(),
        "cudnn": torch.backends.cudnn.enabled,
        "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_tf32": torch.backends.cudnn.allow_tf32,
        "cudnn_sdp": torch.backends.cuda.cudnn_sdp_enabled(),
    }


def _install_worker_dit_capture(worker: Any, capture_path: str) -> str:
    """Capture the first production text-encoder and DiT calls."""
    pipeline = worker.pipeline
    pipeline.post_init()
    text_encoder = pipeline.get_module("text_encoder")
    transformer = pipeline.get_module("transformer")
    text_encoder_calls: list[dict[str, torch.Tensor]] = []
    dit_calls: list[dict[str, torch.Tensor]] = []

    def capture_text_encoder_call(
        _module: Any,
        _args: tuple[Any, ...],
        kwargs: dict[str, Any],
        _output: Any,
    ) -> None:
        """Retain the exact processor tensors supplied to both CFG encoder calls."""
        if len(text_encoder_calls) < 2:
            text_encoder_calls.append({
                name: value.detach().cpu()
                for name, value in kwargs.items()
                if torch.is_tensor(value)
            })

    def capture_call(_module: Any, args: tuple[Any, ...], kwargs: dict[str, Any], output: Any) -> None:
        """Write the two one-step sequential-CFG calls after both have completed."""
        if len(dit_calls) >= 2:
            return
        output_tensor = output[0] if isinstance(output, tuple) else output.sample
        dit_calls.append({
            "hidden_states": args[0].detach().float().cpu(),
            "timestep": args[1].detach().float().cpu(),
            "encoder_hidden_states": args[2].detach().cpu(),
            "encoder_attention_mask": kwargs["encoder_attention_mask"].detach().cpu(),
            "output": output_tensor.detach().float().cpu(),
        })
        if len(dit_calls) == 2:
            torch.save({"text_encoder": text_encoder_calls, "dit": dit_calls}, capture_path)

    worker._lingbot_ti2v_text_encoder_capture_handle = text_encoder.register_forward_hook(
        capture_text_encoder_call,
        with_kwargs=True,
    )
    worker._lingbot_ti2v_capture_handle = transformer.register_forward_hook(capture_call, with_kwargs=True)
    return capture_path


def _require_inputs() -> tuple[Path, Path, Path]:
    """Require explicit baseline, example-input, and result directories."""
    if os.environ.get("LINGBOT_VIDEO_RUN_GPU_TESTS") != "1":
        pytest.skip("set LINGBOT_VIDEO_RUN_GPU_TESTS=1 on an allocated H200")
    if not torch.cuda.is_available():
        pytest.skip("LingBot-Video TI2V parity requires CUDA")
    if VARIANT not in {"dense", "moe"}:
        raise ValueError(f"unsupported TI2V parity variant: {VARIANT}")
    if not BASELINE_DIR or not CASE_DIR or not RESULT_DIR:
        raise ValueError(
            "set LINGBOT_VIDEO_TI2V_BASELINE_DIR, LINGBOT_VIDEO_TI2V_CASE_DIR, "
            "and LINGBOT_VIDEO_TI2V_RESULT_DIR"
        )
    return Path(BASELINE_DIR), Path(CASE_DIR), Path(RESULT_DIR)


def _model_root(scratch: Path) -> Path:
    """Materialize the converted base-generation components for the selected variant."""
    components = ("scheduler", "text_encoder", "tokenizer", "transformer", "vae")
    if VARIANT == "dense":
        return download_components(FASTVIDEO_DENSE, *components)
    return materialize_component_view(FASTVIDEO_MOE, scratch / "fastvideo_moe_base", *components)


def _load_native_transformer(transformer_dir: Path, device: torch.device) -> torch.nn.Module:
    """Strict-load one converted Dense or MoE DiT through FastVideo's production loader."""
    hf_config = json.loads((transformer_dir / "config.json").read_text(encoding="utf-8"))
    weight_files = sorted(str(path) for path in transformer_dir.glob("*.safetensors"))
    return maybe_load_fsdp_model(
        model_cls=LingBotVideoTransformer3DModel,
        init_params={"config": LingBotVideoConfig(), "hf_config": hf_config},
        weight_dir_list=weight_files,
        device=device,
        hsdp_replicate_dim=1,
        hsdp_shard_dim=1,
        default_dtype=torch.bfloat16,
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
        strict=True,
        cpu_offload=False,
        fsdp_inference=False,
        training_mode=False,
        pin_cpu_memory=False,
    ).eval()


def _run_native_dit(model_root: Path, baseline: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Run the native DiT on the exact conditional and unconditional reference inputs."""
    device = torch.device("cuda:0")
    model = _load_native_transformer(model_root / "transformer", device)
    hidden_states = baseline["dit_hidden_states"].to(device)
    timestep = baseline["dit_timestep"].to(device)
    results: dict[str, torch.Tensor] = {}
    try:
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for branch in ("conditional", "unconditional"):
                embeds = baseline[f"dit_{'prompt' if branch == 'conditional' else 'negative'}_embeds"].to(device)
                mask = baseline[f"dit_{'prompt' if branch == 'conditional' else 'negative'}_mask"].to(device)
                output = model(
                    hidden_states,
                    timestep,
                    embeds,
                    encoder_attention_mask=mask,
                    return_dict=False,
                )[0]
                results[branch] = output.detach().float().cpu()
        results["guided"] = results["unconditional"] + 3.0 * (
            results["conditional"] - results["unconditional"]
        )
        return results
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()


def _as_channel_first(value: Any) -> torch.Tensor:
    """Normalize FastVideo latent or pixel output to a CPU float tensor."""
    tensor = value if torch.is_tensor(value) else torch.as_tensor(value)
    if tensor.ndim == 5 and tensor.shape[-1] == 3:
        tensor = tensor.permute(0, 4, 1, 2, 3)
    return tensor.detach().float().cpu()


def _run_fastvideo(
    model_root: Path,
    case_dir: Path,
    manifest: dict[str, Any],
    baseline: dict[str, torch.Tensor],
    initial_latents: torch.Tensor,
    result_dir: Path,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run one-step and 40-step native TI2V calls without reloading the model."""
    from fastvideo import VideoGenerator
    from lingbot_video.pipeline_lingbot_video import DEFAULT_NEGATIVE_PROMPT

    generator = VideoGenerator.from_pretrained(
        str(model_root),
        workload_type="i2v",
        num_gpus=1,
        sp_size=1,
        use_fsdp_inference=False,
        dit_cpu_offload=False,
        dit_layerwise_offload=False,
        vae_cpu_offload=True,
        text_encoder_cpu_offload=True,
        pin_cpu_memory=False,
        output_type="np",
        refine_enabled=False,
    )
    common = {
        "prompt": manifest["prompt"],
        "negative_prompt": DEFAULT_NEGATIVE_PROMPT,
        "image_path": str(case_dir / "first_frame.png"),
        "output_path": str(result_dir),
        "save_video": False,
        "return_frames": True,
        "return_trajectory_latents": True,
        "height": manifest["height"],
        "width": manifest["width"],
        "num_frames": manifest["num_frames"],
        "guidance_scale": manifest["guidance_scale"],
        "batch_cfg": False,
        "seed": manifest["seed"],
    }
    try:
        states = generator.executor.collective_rpc(_configure_backends)
        assert all(state == _configure_backends() for state in states)
        capture_path = result_dir / "pipeline_dit_calls.pt"
        generator.executor.collective_rpc(
            _install_worker_dit_capture,
            kwargs={"capture_path": str(capture_path)},
        )
        one_step = cast(
            dict[str, Any],
            generator.generate_video(num_inference_steps=1, latents=initial_latents.clone(), **common),
        )
        calls = torch.load(capture_path, map_location="cpu", weights_only=True)
        for index, branch in enumerate(("conditional", "unconditional")):
            expected_inputs = baseline["text_encoder_calls"][index]
            actual_inputs = calls["text_encoder"][index]
            assert actual_inputs.keys() == expected_inputs.keys()
            for name in actual_inputs:
                metrics = _metrics(actual_inputs[name], expected_inputs[name])
                print(f"pipeline_{branch}_text_encoder_{name}={json.dumps(metrics, sort_keys=True)}")
                assert metrics["equal"]
        for index, branch in enumerate(("conditional", "unconditional")):
            expected_prefix = "prompt" if branch == "conditional" else "negative"
            comparisons = {
                f"pipeline_{branch}_hidden_states": (calls["dit"][index]["hidden_states"], baseline["dit_hidden_states"]),
                f"pipeline_{branch}_timestep": (calls["dit"][index]["timestep"], baseline["dit_timestep"]),
                f"pipeline_{branch}_embeds": (
                    calls["dit"][index]["encoder_hidden_states"],
                    baseline[f"dit_{expected_prefix}_embeds"],
                ),
                f"pipeline_{branch}_mask": (
                    calls["dit"][index]["encoder_attention_mask"],
                    baseline[f"dit_{expected_prefix}_mask"],
                ),
                f"pipeline_{branch}_output": (calls["dit"][index]["output"], baseline[f"dit_{branch}"]),
            }
            for name, (actual, expected) in comparisons.items():
                metrics = _metrics(actual, expected)
                print(f"{name}={json.dumps(metrics, sort_keys=True)}")
                assert metrics["equal"]
        full = cast(
            dict[str, Any],
            generator.generate_video(num_inference_steps=40, latents=initial_latents.clone(), **common),
        )
        one_trajectory = cast(torch.Tensor, one_step["trajectory"])
        full_trajectory = cast(torch.Tensor, full["trajectory"])
        return (
            one_trajectory[:, 0].detach().float().cpu(),
            full_trajectory.detach().float().cpu(),
            _as_channel_first(full["samples"]),
        )
    finally:
        generator.shutdown()


def _metrics(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, int | float | bool]:
    """Return exact-equality and absolute-drift metrics for one tensor boundary."""
    difference = (actual - expected).abs()
    return {
        "equal": torch.equal(actual, expected),
        "differing": int(torch.count_nonzero(actual != expected).item()),
        "max_abs": float(difference.max().item()),
        "mean_abs": float(difference.float().mean().item()),
    }


def test_lingbot_video_ti2v_exact_parity() -> None:
    """Require exact actual-input DiT, one-step, 40-step, and pixel parity."""
    baseline_dir, case_dir, result_dir = _require_inputs()
    result_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((baseline_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest["variant"] != VARIANT:
        raise ValueError(f"baseline variant {manifest['variant']} does not match requested {VARIANT}")
    image_sha256 = hashlib.sha256((case_dir / "first_frame.png").read_bytes()).hexdigest()
    if image_sha256 != manifest["image_sha256"]:
        raise ValueError("condition image does not match the reference baseline")
    baseline = torch.load(baseline_dir / manifest["baseline"], map_location="cpu", weights_only=True)
    _configure_backends()
    with tempfile.TemporaryDirectory(dir=result_dir) as temporary:
        model_root = _model_root(Path(temporary))
        dit = _run_native_dit(model_root, baseline)
        results = {
            "dit_conditional": _metrics(dit["conditional"], baseline["dit_conditional"]),
            "dit_unconditional": _metrics(dit["unconditional"], baseline["dit_unconditional"]),
            "dit_guided": _metrics(dit["guided"], baseline["dit_guided"]),
        }
        for name, metrics in results.items():
            print(f"{name}={json.dumps(metrics, sort_keys=True)}")
            assert metrics["equal"]
        one_step, trajectory, pixels = _run_fastvideo(
            model_root,
            case_dir,
            manifest,
            baseline,
            baseline["initial_latents"],
            result_dir,
        )
    conditioned_reference = baseline["trajectory_before_condition"].clone()
    condition_frames = baseline["condition_latent"].shape[2]
    conditioned_reference[:, :, :, :condition_frames] = baseline["condition_latent"][:, None]
    results.update({
        "one_step_latent": _metrics(one_step, baseline["one_step_latent"]),
        "forty_step_trajectory": _metrics(trajectory, conditioned_reference),
        "decoded_pixels": _metrics(pixels, baseline["pixels"]),
    })
    (result_dir / "result.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    for name in ("one_step_latent", "forty_step_trajectory", "decoded_pixels"):
        print(f"{name}={json.dumps(results[name], sort_keys=True)}")
        assert results[name]["equal"]
