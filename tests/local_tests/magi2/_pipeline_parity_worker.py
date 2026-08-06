# SPDX-License-Identifier: Apache-2.0
"""Capture strict MAGI-2 pipeline boundaries from one implementation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
from typing import Any

import numpy as np
import torch
import torch.distributed as dist


REPO_ROOT = Path(__file__).resolve().parents[3]
OFFICIAL_ROOT = Path(
    os.environ.get("MAGI2_OFFICIAL_REF_DIR", REPO_ROOT.parent / "MAGI-2-preview")
)
WEIGHTS_ROOT = Path(
    os.environ.get("MAGI2_LOCAL_WEIGHTS_DIR", REPO_ROOT / "official_weights" / "magi2")
)
CONVERTED_ROOT = Path(
    os.environ.get("MAGI2_CONVERTED_WEIGHTS_DIR", REPO_ROOT / "converted_weights" / "magi2")
)
WORLD_SIZE = 8
SEED = 42


def _parse_args() -> argparse.Namespace:
    """Parse the implementation, denoising counts, and requested modalities."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--implementation",
        choices=("official", "fastvideo"),
        required=True,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--preview-steps", type=int, default=100)
    parser.add_argument("--refiner-steps", type=int, default=5)
    parser.add_argument(
        "--cases",
        nargs="+",
        choices=("t2v", "i2v"),
        default=("t2v", "i2v"),
    )
    return parser.parse_args()


def _seed_all() -> None:
    """Reset every random-number source used by the release pipeline."""
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)


def _enable_determinism() -> None:
    """Apply the deterministic controls exposed by the official entry point."""
    os.environ["MAGI2_DETERMINISTIC"] = "1"
    os.environ["MAGI_ATTENTION_DETERMINISTIC_MODE"] = "1"
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    _seed_all()
    torch.use_deterministic_algorithms(True)


def _hash_array(array: np.ndarray) -> str:
    """Hash the exact bytes of one contiguous NumPy array without a byte copy."""
    contiguous_array = np.ascontiguousarray(array)
    return hashlib.sha256(memoryview(contiguous_array).cast("B")).hexdigest()


def _tensor_digest(tensor: torch.Tensor | None) -> dict[str, Any] | None:
    """Describe and hash a tensor while preserving its source metadata."""
    if tensor is None:
        return None
    source_tensor = tensor.detach()
    byte_array = source_tensor.to(device="cpu").contiguous().view(torch.uint8).numpy()
    return {
        "shape": list(source_tensor.shape),
        "dtype": str(source_tensor.dtype),
        "stride": list(source_tensor.stride()),
        "sha256": _hash_array(byte_array),
    }


def _array_digest(array: np.ndarray | None) -> dict[str, Any] | None:
    """Describe and hash a NumPy array while preserving its source metadata."""
    if array is None:
        return None
    source_array = np.asarray(array)
    return {
        "shape": list(source_array.shape),
        "dtype": str(source_array.dtype),
        "stride": list(source_array.strides),
        "sha256": _hash_array(source_array),
    }


def _case_inputs(case_name: str) -> tuple[str, str | None]:
    """Return the official release prompt and optional reference image path."""
    if case_name == "t2v":
        prompt_path = OFFICIAL_ROOT / "assets" / "sample_enhanced_t2v.json"
        return prompt_path.read_text(encoding="utf-8").strip(), None
    if case_name == "i2v":
        prompt_path = OFFICIAL_ROOT / "assets" / "sample_000.txt"
        image_path = OFFICIAL_ROOT / "assets" / "sample_000.jpeg"
        return prompt_path.read_text(encoding="utf-8").strip(), str(image_path)
    raise ValueError(f"Unknown MAGI-2 parity case: {case_name}")


def _capture_official_case(
    engine: Any,
    case_name: str,
    preview_steps: int,
    refiner_steps: int,
    is_capture_rank: bool,
) -> dict[str, Any]:
    """Run the official engine and capture every externally visible stage boundary."""
    prompt, image_path = _case_inputs(case_name)
    capture: dict[str, Any] = {}
    text_call_count = 0

    original_encode_images = engine._encode_images
    original_get_text_embedding = engine.get_text_embedding
    original_get_special_token = engine.get_special_token
    original_preview_sample = engine.sampler.sample
    original_refiner_forward = engine._forward_magi2_refiner
    original_refiner_sample = engine.evaluate_magi2_refiner_with_latent

    def capture_images(image: Any, height: int, width: int):
        """Record the Wan image-encoder output and its figure identity."""
        encoded_images = original_encode_images(image, height, width)
        if is_capture_rank:
            reference_latent, reference_length, reference_ids = encoded_images
            capture["reference_latent"] = _tensor_digest(reference_latent)
            capture["reference_length"] = _tensor_digest(reference_length)
            capture["reference_ids"] = _tensor_digest(reference_ids)
        return encoded_images

    def capture_text(prompt_text: str) -> torch.Tensor:
        """Record positive and negative Qwen3.5 conditioning in call order."""
        nonlocal text_call_count
        text_embedding = original_get_text_embedding(prompt_text)
        if is_capture_rank:
            if text_call_count == 0:
                capture["conditioned_prompt"] = prompt_text
                capture["positive_text"] = _tensor_digest(text_embedding)
            elif text_call_count == 1:
                capture["negative_prompt"] = prompt_text
                capture["negative_text"] = _tensor_digest(text_embedding)
            else:
                raise RuntimeError("MAGI-2 encoded more than two prompts in one request")
        text_call_count += 1
        return text_embedding

    def capture_special_token(
        prompt_text: str,
        figure_ids: torch.Tensor,
        text_feature: torch.Tensor,
    ) -> torch.Tensor:
        """Record the pooled text embedding that prefixes image patches."""
        special_token = original_get_special_token(
            prompt_text,
            figure_ids,
            text_feature,
        )
        if is_capture_rank:
            capture["special_tokens"] = _tensor_digest(special_token.unsqueeze(0))
        return special_token

    def capture_preview(sampler_input: Any) -> tuple[torch.Tensor, torch.Tensor]:
        """Record the initial noise and the completed preview latents."""
        if is_capture_rank:
            capture["initial_video_noise"] = _tensor_digest(sampler_input.latent)
            capture["initial_audio_noise"] = _tensor_digest(
                sampler_input.audio_latent
            )
        preview_video, preview_audio = original_preview_sample(sampler_input)
        if is_capture_rank:
            capture["preview_video"] = _tensor_digest(preview_video)
            capture["preview_audio"] = _tensor_digest(preview_audio)
        return preview_video, preview_audio

    def capture_refiner_forward(
        latent_video: torch.Tensor,
        latent_audio: torch.Tensor,
        txt_feat: torch.Tensor,
        ref_audio_feat: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Record the noise-injected 1080p latent entering the refiner."""
        if is_capture_rank and "refiner_input" not in capture:
            capture["refiner_input"] = _tensor_digest(latent_video)
        return original_refiner_forward(
            latent_video,
            latent_audio,
            txt_feat,
            ref_audio_feat,
        )

    def capture_refiner_sample(**kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
        """Record the video latent that the refiner passes to the decoder."""
        refined_video, refined_audio = original_refiner_sample(**kwargs)
        if is_capture_rank:
            capture["refined_video"] = _tensor_digest(refined_video)
        return refined_video, refined_audio

    engine._encode_images = capture_images
    engine.get_text_embedding = capture_text
    engine.get_special_token = capture_special_token
    engine.sampler.sample = capture_preview
    engine._forward_magi2_refiner = capture_refiner_forward
    engine.evaluate_magi2_refiner_with_latent = capture_refiner_sample
    _seed_all()
    try:
        video, audio = engine.evaluate(
            prompt=prompt,
            image=image_path,
            eval_task_type="text2video" if image_path is None else "image2video",
            seconds=10.0,
            preview_width=896,
            preview_height=512,
            br_num_inference_steps=preview_steps,
            refiner_width=1920,
            refiner_height=1088,
            magi2_refiner_num_inference_steps=refiner_steps,
        )
    finally:
        engine._encode_images = original_encode_images
        engine.get_text_embedding = original_get_text_embedding
        engine.get_special_token = original_get_special_token
        engine.sampler.sample = original_preview_sample
        engine._forward_magi2_refiner = original_refiner_forward
        engine.evaluate_magi2_refiner_with_latent = original_refiner_sample

    if is_capture_rank:
        capture.setdefault("special_tokens", None)
        capture["decoded_video"] = _array_digest(video)
        capture["decoded_audio"] = _array_digest(audio)
    return capture


def _load_official_engine() -> Any:
    """Initialize the official eight-rank runtime and load every release component."""
    sys.path.insert(0, str(OFFICIAL_ROOT))
    from inference.common.magi2_config import load_config
    from inference.infra.checkpoint.load_checkpoint import (
        load_magi2_model,
        load_magi2_refiner,
    )
    from inference.infra.distributed import (
        initialize_expert_parallel,
        initialize_model_parallel,
    )

    config = load_config(str(OFFICIAL_ROOT / "configs" / "magi2_refiner.json"))
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    if dist.get_world_size() != WORLD_SIZE:
        raise RuntimeError(
            f"MAGI-2 pipeline parity requires {WORLD_SIZE} ranks, "
            f"received {dist.get_world_size()}"
        )
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    initialize_model_parallel(cp_size=WORLD_SIZE)
    initialize_expert_parallel(ep_size=WORLD_SIZE)

    from inference.pipeline.inference_engine import Magi2InferenceEngine

    preview_model = load_magi2_model(config)
    refiner_model = load_magi2_refiner(config)
    return Magi2InferenceEngine(
        model=preview_model,
        config=config.evaluation_config,
        device=f"cuda:{local_rank}",
        weight_dtype=torch.bfloat16,
        magi2_refiner=refiner_model,
    )


def _canonical_fastvideo_video(video: torch.Tensor | None) -> np.ndarray | None:
    """Convert FastVideo's BCHWT float output into the release's THWC bytes."""
    if video is None:
        return None
    if video.shape[0] != 1:
        raise ValueError(f"MAGI-2 parity expects one decoded video, received {video.shape}")
    return video[0].permute(1, 2, 3, 0).mul(255).numpy().astype(np.uint8)


def _capture_fastvideo_case(
    pipeline: Any,
    fastvideo_args: Any,
    case_name: str,
    preview_steps: int,
    refiner_steps: int,
    is_capture_rank: bool,
) -> dict[str, Any]:
    """Run FastVideo stages and capture the boundaries used by the official worker."""
    from fastvideo.fastvideo_args import WorkloadType
    from fastvideo.pipelines.pipeline_batch_info import ForwardBatch

    prompt, image_path = _case_inputs(case_name)
    fastvideo_args.workload_type = (
        WorkloadType.T2V if case_name == "t2v" else WorkloadType.I2V
    )
    batch = ForwardBatch(
        data_type="video",
        prompt=prompt,
        negative_prompt=pipeline.negative_prompt,
        image_path=image_path,
        seed=SEED,
        num_frames=249,
        height=1088,
        width=1920,
        fps=25,
        num_inference_steps=preview_steps,
        num_inference_steps_sr=refiner_steps,
    )
    capture: dict[str, Any] = {}
    refiner_stage = pipeline.refiner_stage
    original_refiner_predict = refiner_stage._predict_velocity

    def capture_refiner_predict(
        video_latent: torch.Tensor,
        audio_latent: torch.Tensor,
        text_context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Record the noise-injected latent before the first refiner forward."""
        if is_capture_rank and "refiner_input" not in capture:
            capture["refiner_input"] = _tensor_digest(video_latent)
        return original_refiner_predict(video_latent, audio_latent, text_context)

    refiner_stage._predict_velocity = capture_refiner_predict
    _seed_all()
    try:
        for stage_name, stage in pipeline._stage_name_mapping.items():
            batch = stage(batch, fastvideo_args)
            if not is_capture_rank:
                continue
            if stage_name == "reference_image_stage":
                capture["conditioned_prompt"] = batch.prompt
                capture["reference_latent"] = _tensor_digest(
                    batch.magi2_ref_image_feat
                )
                capture["reference_length"] = _tensor_digest(
                    batch.magi2_ref_image_feat_len
                )
                reference_ids = (
                    None
                    if batch.magi2_ref_image_feat is None
                    else torch.tensor([[1]], dtype=torch.long)
                )
                capture["reference_ids"] = _tensor_digest(reference_ids)
            elif stage_name == "text_encoding_stage":
                capture["negative_prompt"] = batch.negative_prompt
                capture["positive_text"] = _tensor_digest(batch.magi2_text_context)
                capture["negative_text"] = _tensor_digest(
                    batch.magi2_negative_context
                )
                capture["special_tokens"] = _tensor_digest(
                    batch.magi2_ref_image_special_tokens
                )
            elif stage_name == "latent_preparation_stage":
                capture["initial_video_noise"] = _tensor_digest(batch.latents)
                capture["initial_audio_noise"] = _tensor_digest(
                    batch.audio_latents
                )
            elif stage_name == "preview_denoising_stage":
                capture["preview_video"] = _tensor_digest(batch.latents)
                capture["preview_audio"] = _tensor_digest(batch.audio_latents)
            elif stage_name == "refiner_stage":
                capture["refined_video"] = _tensor_digest(batch.latents)
            elif stage_name == "video_decoding_stage":
                canonical_video = _canonical_fastvideo_video(batch.output)
                capture["decoded_video"] = _array_digest(canonical_video)
            elif stage_name == "audio_decoding_stage":
                capture["decoded_audio"] = _array_digest(batch.extra.get("audio"))
    finally:
        refiner_stage._predict_velocity = original_refiner_predict
    return capture


def _load_fastvideo_pipeline() -> tuple[Any, Any]:
    """Initialize FastVideo's eight-rank runtime and load the converted checkpoint."""
    sys.path.insert(0, str(REPO_ROOT))
    from fastvideo.fastvideo_args import FastVideoArgs, WorkloadType
    from fastvideo.pipelines.basic.magi2.magi2_pipeline import Magi2Pipeline
    from fastvideo.pipelines.basic.magi2.pipeline_configs import (
        Magi2PreviewPipelineConfig,
    )
    from fastvideo.pipelines.basic.magi2.presets import MAGI2_NEGATIVE_PROMPT

    fastvideo_args = FastVideoArgs(
        model_path=str(CONVERTED_ROOT),
        workload_type=WorkloadType.T2V,
        num_gpus=WORLD_SIZE,
        tp_size=1,
        sp_size=WORLD_SIZE,
        pipeline_config=Magi2PreviewPipelineConfig(),
        deterministic=True,
        dit_cpu_offload=True,
        dit_layerwise_offload=False,
        text_encoder_cpu_offload=True,
        image_encoder_cpu_offload=True,
        vae_cpu_offload=True,
        enable_stage_verification=True,
    )
    pipeline = Magi2Pipeline(str(CONVERTED_ROOT), fastvideo_args)
    pipeline.negative_prompt = MAGI2_NEGATIVE_PROMPT
    pipeline.post_init()
    if dist.get_world_size() != WORLD_SIZE:
        raise RuntimeError(
            f"MAGI-2 pipeline parity requires {WORLD_SIZE} ranks, "
            f"received {dist.get_world_size()}"
        )
    return pipeline, fastvideo_args


def _write_capture(
    output_dir: Path,
    implementation: str,
    preview_steps: int,
    refiner_steps: int,
    captures: dict[str, Any],
) -> None:
    """Atomically write the rank-zero digest manifest for one implementation."""
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact = {
        "schema_version": 1,
        "implementation": implementation,
        "world_size": WORLD_SIZE,
        "preview_steps": preview_steps,
        "refiner_steps": refiner_steps,
        "cases": captures,
    }
    artifact_path = output_dir / "capture.json"
    temporary_path = output_dir / "capture.json.tmp"
    temporary_path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(temporary_path, artifact_path)


def main() -> None:
    """Load one implementation, capture all cases, and persist rank-zero digests."""
    args = _parse_args()
    if args.preview_steps <= 0 or args.refiner_steps <= 0:
        raise ValueError("MAGI-2 parity step counts must be positive")
    os.environ["MAGI2_CKPT_ROOT"] = str(WEIGHTS_ROOT)
    os.environ.pop("MAGI2_SAVE_LATENT_PATH", None)
    os.environ.pop("NEGATIVE_PROMPT", None)
    os.environ.pop("SKIP_LOAD_MODEL", None)
    _enable_determinism()

    if args.implementation == "official":
        engine = _load_official_engine()
        is_capture_rank = dist.get_rank() == 0
        captures = {
            case_name: _capture_official_case(
                engine,
                case_name,
                args.preview_steps,
                args.refiner_steps,
                is_capture_rank,
            )
            for case_name in args.cases
        }
    else:
        pipeline, fastvideo_args = _load_fastvideo_pipeline()
        is_capture_rank = dist.get_rank() == 0
        captures = {
            case_name: _capture_fastvideo_case(
                pipeline,
                fastvideo_args,
                case_name,
                args.preview_steps,
                args.refiner_steps,
                is_capture_rank,
            )
            for case_name in args.cases
        }

    if is_capture_rank:
        _write_capture(
            args.output_dir,
            args.implementation,
            args.preview_steps,
            args.refiner_steps,
            captures,
        )
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
