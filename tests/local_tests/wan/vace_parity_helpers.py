# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for Wan-VACE Diffusers parity tests."""

from __future__ import annotations

import contextlib
import inspect
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import torch

CALL_ARG_KEYS = (
    "hidden_states",
    "encoder_hidden_states",
    "timestep",
    "control_hidden_states",
    "control_hidden_states_scale",
)


def resolve_model_dir(env_name: str) -> Path:
    raw = os.getenv(env_name)
    if raw is None:
        pytest.skip(f"Set {env_name} to a local Wan-VACE snapshot")
    model_dir = Path(raw)
    if not (model_dir / "model_index.json").is_file():
        pytest.fail(f"{env_name} does not contain model_index.json: {model_dir}")
    return model_dir


def video_generator_kwargs() -> dict[str, Any]:
    return dict(
        num_gpus=1,
        tp_size=1,
        sp_size=1,
        use_fsdp_inference=False,
        dit_cpu_offload=False,
        text_encoder_cpu_offload=True,
        vae_cpu_offload=False,
        pin_cpu_memory=False,
        lazy_module_load=False,
        output_type="latent",
        attention_backend="TORCH_SDPA",
    )


def load_official_fp32_vae(model_dir: Path, device: torch.device) -> Any:
    from diffusers import AutoencoderKLWan

    return AutoencoderKLWan.from_pretrained(str(model_dir / "vae"),
                                            torch_dtype=torch.float32,
                                            local_files_only=True).to(device).eval()


def load_official_fp32_text_encoder(model_dir: Path, device: torch.device) -> Any:
    from transformers import UMT5EncoderModel

    return UMT5EncoderModel.from_pretrained(str(model_dir / "text_encoder"),
                                            torch_dtype=torch.float32,
                                            local_files_only=True).to(device).eval()


def prepare_official_pipeline(model_dir: Path, device: torch.device) -> Any:
    from diffusers import WanVACEPipeline as DiffusersWanVACEPipeline

    official = DiffusersWanVACEPipeline.from_pretrained(str(model_dir),
                                                      torch_dtype=torch.bfloat16,
                                                      local_files_only=True).to(device)
    official.scheduler = type(official.scheduler).from_config(official.scheduler.config, flow_shift=16.0)
    official.vae = load_official_fp32_vae(model_dir, device)
    official.text_encoder = load_official_fp32_text_encoder(model_dir, device)
    return official


def load_official_transformer(model_dir: Path, device: torch.device, dtype: torch.dtype = torch.bfloat16) -> Any:
    from diffusers.models.transformers.transformer_wan_vace import WanVACETransformer3DModel

    return WanVACETransformer3DModel.from_pretrained(str(model_dir),
                                                     subfolder="transformer",
                                                     torch_dtype=dtype,
                                                     local_files_only=True).to(device).eval()


def run_official_on_call(transformer: Any, call_args: dict[str, Any], device: torch.device) -> torch.Tensor:
    """Re-run the Diffusers DiT on one recorded FastVideo call (teacher forcing)."""
    kwargs = {}
    for key, value in call_args.items():
        value = value[0] if isinstance(value, list) else value  # FastVideo passes encoder states as a list
        if torch.is_tensor(value):
            kwargs[key] = value.to(device)
    scale = torch.as_tensor(call_args.get("control_hidden_states_scale", 1.0), device=device)
    num_layers = len(transformer.config.vace_layers)
    kwargs["control_hidden_states_scale"] = scale.to(transformer.dtype).reshape(-1).expand(num_layers)
    with torch.inference_mode():
        return transformer(**kwargs).sample.detach().cpu()


def load_fv_transformer(model_dir: Path,
                        device: torch.device,
                        config_cls: Any,
                        dtype: torch.dtype = torch.bfloat16) -> Any:
    from fastvideo.fastvideo_args import FastVideoArgs
    from fastvideo.models.loader.component_loader import TransformerLoader

    pipeline_config = config_cls()
    pipeline_config.dit_precision = "fp32" if dtype == torch.float32 else "bf16"
    args = FastVideoArgs(model_path=str(model_dir), pipeline_config=pipeline_config)
    args.device = device
    transformer = TransformerLoader().load(str(model_dir / "transformer"), args)
    return transformer.to(device).eval()


def set_parity_cuda_flags() -> dict[str, bool]:
    previous = {
        "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "allow_cudnn_tf32": torch.backends.cudnn.allow_tf32,
    }
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return previous


def restore_parity_cuda_flags(previous: dict[str, bool]) -> None:
    torch.backends.cuda.matmul.allow_tf32 = previous["allow_tf32"]
    torch.backends.cudnn.allow_tf32 = previous["allow_cudnn_tf32"]


def assert_bf16_bitwise_equal(actual: torch.Tensor, expected: torch.Tensor, label: str) -> None:
    actual_bf16 = actual.detach().cpu().to(torch.bfloat16)
    expected_bf16 = expected.detach().cpu().to(torch.bfloat16)
    if actual_bf16.equal(expected_bf16):
        return
    diff = (actual_bf16.float() - expected_bf16.float()).abs()
    raise AssertionError(
        f"{label}: bf16 not bitwise equal; max abs={diff.max().item():.6g}, "
        f"mismatched={(actual_bf16 != expected_bf16).sum().item()}/{actual_bf16.numel()}")


def _bind_forward_args(module: Any, positional: tuple[Any, ...], keyword: dict[str, Any]) -> inspect.BoundArguments:
    bound = inspect.signature(module.forward).bind(*positional, **keyword)
    bound.apply_defaults()
    return bound


def _serialize_call_arg(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return [_serialize_call_arg(item) for item in value]
    if torch.is_tensor(value):
        return value.detach().cpu()
    return value


def capture_call_args(module: Any, positional: tuple[Any, ...], keyword: dict[str, Any]) -> dict[str, Any]:
    bound = _bind_forward_args(module, positional, keyword)
    return {key: _serialize_call_arg(bound.arguments[key]) for key in CALL_ARG_KEYS if key in bound.arguments}


def register_transformer_capture(transformer: Any, captured: dict[str, Any]) -> list[Any]:
    def capture_inputs(module: Any, positional: tuple[Any, ...], keyword: dict[str, Any] | None = None) -> None:
        call_args = capture_call_args(module, positional, keyword or {})
        captured.setdefault("call_args", []).append(call_args)
        captured["model_inputs"].append(call_args["hidden_states"].detach().cpu())

    def capture_output(module: Any, positional: tuple[Any, ...], keyword: dict[str, Any], output: Any) -> None:
        prediction = output[0] if isinstance(output, tuple) else getattr(output, "sample", output)
        captured["noise_preds"].append(prediction.detach().cpu())

    return [
        transformer.register_forward_pre_hook(capture_inputs, with_kwargs=True),
        transformer.register_forward_hook(capture_output, with_kwargs=True),
    ]


def register_sp_layer_capture(transformer: Any, captured: dict[str, Any]) -> list[Any]:
    """Capture VACE/main block boundaries for local SP divergence diagnosis."""
    names = ("patch_embedding", "vace_patch_embedding", "condition_embedder", "norm_out", "proj_out")
    first_block_parts = ("proj_in", "norm1", "to_q", "to_k", "to_v", "norm_q", "norm_k", "attn1",
                         "to_out", "attn2", "ffn.fc_in", "ffn.act", "ffn.fc_out", "ffn", "proj_out")
    detailed = {f"{block}.{part}" for block in ("vace_blocks.0", "blocks.0") for part in first_block_parts}
    selected = [(name, module) for name, module in transformer.named_modules()
                if name in names or name in detailed or
                (name.startswith(("vace_blocks.", "blocks.")) and name.count(".") == 1)]
    captured["layers"] = {}
    handles = []
    for name, module in selected:
        def capture(_module: Any, _inputs: tuple[Any, ...], output: Any, layer_name: str = name) -> None:
            value = output[0] if isinstance(output, tuple) else output
            captured["layers"].setdefault(layer_name, []).append(value.detach().cpu())

        handles.append(module.register_forward_hook(capture))
    return handles


def run_fastvideo_stages(worker_wrapper: Any,
                         model_path: str,
                         request: dict[str, Any],
                         capture_decoded: bool = False,
                         trace_sp_layers: bool = False) -> dict[str, Any]:
    from fastvideo.api.sampling_param import SamplingParam
    from fastvideo.pipelines.pipeline_batch_info import ForwardBatch
    from fastvideo.utils import shallow_asdict

    previous_cuda_flags = set_parity_cuda_flags()
    worker = worker_wrapper.worker
    pipeline = worker.pipeline
    args = worker.fastvideo_args
    if not pipeline.post_init_called:
        pipeline.post_init()
    sampling = SamplingParam.from_pretrained(model_path)
    for key, value in request.items():
        setattr(sampling, key, value)
    sampling.__post_init__()
    latent_frames = (sampling.num_frames - 1) // 4 + 1
    batch = ForwardBatch(**shallow_asdict(sampling),
                         eta=0.0,
                         n_tokens=latent_frames * (sampling.height // 8) * (sampling.width // 8),
                         VSA_sparsity=args.VSA_sparsity)
    captured: dict[str, Any] = {"model_inputs": [], "noise_preds": []}
    transformer = pipeline.get_module("transformer")
    capture_handles = register_transformer_capture(transformer, captured)
    if trace_sp_layers:
        capture_handles.extend(register_sp_layer_capture(transformer, captured))
    try:
        with torch.no_grad():
            for stage in pipeline.stages:
                batch = stage(batch, args)
                name = stage._pipeline_stage_name
                if name == "prompt_encoding_stage":
                    captured["prompt_embeds"] = batch.prompt_embeds[0].detach().cpu()
                elif name == "vace_input_stage":
                    captured["video"] = batch.video_latent.detach().cpu()
                    captured["mask"] = batch.mask_video.detach().cpu() if batch.mask_video is not None else None
                    captured["references"] = [image.detach().cpu() for image in batch.vace_reference_images or []]
                elif name == "timestep_preparation_stage":
                    captured["timesteps"] = batch.timesteps.detach().cpu()
                elif name == "vace_context_stage":
                    captured["control"] = batch.vace_control_latents.detach().cpu()
                elif name == "latent_preparation_stage":
                    captured["initial_latents"] = batch.latents.detach().cpu()
                elif name == "denoising_stage" and capture_decoded:
                    captured["denoised_latents"] = batch.latents.detach().cpu()
            captured["latents"] = batch.output.detach().cpu()
    finally:
        for capture_handle in capture_handles:
            capture_handle.remove()
        restore_parity_cuda_flags(previous_cuda_flags)
    return captured


@contextlib.contextmanager
def single_gpu_parity_runtime(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Single-rank distributed env with SDPA attention and strict CUDA matmul flags."""
    from fastvideo.distributed import cleanup_dist_env_and_memory, maybe_init_distributed_environment_and_model_parallel
    from fastvideo.utils import get_open_port

    from fastvideo import envs

    monkeypatch.setenv("MASTER_ADDR", "localhost")
    monkeypatch.setenv("MASTER_PORT", str(get_open_port()))
    tf32_previous = set_parity_cuda_flags()
    with envs.FASTVIDEO_ATTENTION_BACKEND.override("TORCH_SDPA"):
        maybe_init_distributed_environment_and_model_parallel(1, 1)
        try:
            yield
        finally:
            restore_parity_cuda_flags(tf32_previous)
            cleanup_dist_env_and_memory()


def prepare_vace_inputs(mode: str, tmp_path: Path) -> tuple[dict[str, Any], dict[str, Any], int]:
    """Write tiny VACE inputs; return (FastVideo kwargs, Diffusers kwargs, reference count)."""
    import imageio
    import numpy as np
    from PIL import Image

    if mode == "t2v":
        return {}, {}, 0
    if mode == "reference":
        image_path = tmp_path / "reference.png"
        Image.new("RGB", (64, 32), (240, 32, 24)).save(image_path)
        return {"references": [str(image_path)]}, {"reference_images": [Image.open(image_path).convert("RGB")]}, 1

    video_path = tmp_path / "video.mp4"
    mask_path = tmp_path / "mask.mp4"
    video_frames = np.stack([np.full((64, 64, 3), (i * 24, 64, 160), dtype=np.uint8) for i in range(5)])
    mask_frames = np.zeros((5, 64, 64, 3), dtype=np.uint8)
    mask_frames[:, :, 32:] = 255
    imageio.mimwrite(video_path, video_frames, fps=16, codec="libx264")
    imageio.mimwrite(mask_path, mask_frames, fps=16, codec="libx264")
    from fastvideo.models.vision_utils import load_video

    decoded_video, _ = load_video(str(video_path), return_fps=True)
    decoded_mask, _ = load_video(str(mask_path), return_fps=True)
    return ({"video_path": str(video_path), "mask_path": str(mask_path)},
            {"video": decoded_video, "mask": decoded_mask}, 0)
