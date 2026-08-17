# SPDX-License-Identifier: Apache-2.0
"""Encode one Crush-Smol audio-video-caption record for MiniMax H3 supervised fine-tuning (SFT).

The Parquet row keeps the synchronized video target, audio target, and caption
conditioning together for the text-to-video-and-audio (T2VA) dataloader.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import shutil
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from fastvideo.configs.pipelines.minimax_h3 import MiniMaxH3PipelineConfig
from fastvideo.dataset.dataloader.schema import pyarrow_schema_t2va
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.models.loader.component_loader import PipelineComponentLoader
from fastvideo.pipelines import ForwardBatch
from fastvideo.pipelines.basic.minimax_h3.packing import MINIMAX_H3_FPS
from fastvideo.pipelines.basic.minimax_h3.reference import (
    decode_reference_video,
    prepare_reference_frames,
    prepare_reference_waveform,
    resample_reference_frames,
)
from fastvideo.pipelines.basic.minimax_h3.stages.minimax_h3_conditioning import MiniMaxH3ConditioningStage
from fastvideo.pipelines.basic.minimax_h3.stages.minimax_h3_input_preparation import MINIMAX_H3_KEYFRAMES_KEY
from fastvideo.utils import verify_model_config_and_directory

# Match the H3 validation geometry and media rates, and limit both streams to
# the same maximum duration used by validation.
NUM_FRAMES = 124
VIDEO_HEIGHT = 768
VIDEO_WIDTH = 1344
AUDIO_SAMPLE_RATE = 32000
DATA_DIR = Path("data/crush-smol")
OUTPUT_DIR = Path("data/crush-smol_h3_t2va_single_sample_preprocessed")
MODEL_PATH = Path("data/models/MiniMax-H3")
TRAINING_VIDEO_NAME = "1gGQy4nxyUo-Scene-016.mp4"


def _init_single_process_distributed() -> None:
    """Initialize the one-rank process groups required by component loaders."""
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29531")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    from fastvideo.distributed import maybe_init_distributed_environment_and_model_parallel

    maybe_init_distributed_environment_and_model_parallel(1, 1)


def load_training_media(path: Path) -> tuple[np.ndarray, torch.Tensor]:
    """Decode paired targets at the geometry and rates consumed by H3.

    The video uses exactly 124 frames, and the audio is capped at the duration
    of those frames before variational autoencoder (VAE) encoding.
    """
    decoded_frames, source_fps, soundtrack = decode_reference_video(path)
    resampled_frames = resample_reference_frames(decoded_frames, source_fps)
    if resampled_frames.shape[0] < NUM_FRAMES:
        raise ValueError(f"Training video {path} has {resampled_frames.shape[0]} frames at {MINIMAX_H3_FPS} FPS; "
                         f"{NUM_FRAMES} are required")
    frames = prepare_reference_frames(resampled_frames, NUM_FRAMES)
    if frames.shape != (NUM_FRAMES, VIDEO_HEIGHT, VIDEO_WIDTH, 3):
        raise ValueError(f"Training video must resolve to {(NUM_FRAMES, VIDEO_HEIGHT, VIDEO_WIDTH, 3)}, "
                         f"got {tuple(frames.shape)}")
    if soundtrack is None:
        raise ValueError(f"Training video {path} requires a soundtrack")
    waveform, source_sample_rate = soundtrack
    waveform = prepare_reference_waveform(
        waveform,
        source_sample_rate,
        AUDIO_SAMPLE_RATE,
        max_duration=NUM_FRAMES / MINIMAX_H3_FPS,
    )
    return frames, waveform


def load_crush_smol_training_sample(
    manifest_path: Path,
    video_dir: Path,
) -> tuple[Path, str]:
    """Resolve the selected Crush-Smol video and its sole dataset caption.

    The pinned dataset manifest owns the caption. Selecting the sample by file
    name keeps the training input stable if the manifest order changes.
    """
    manifest = json.loads(manifest_path.read_text())
    if not isinstance(manifest, list):
        raise ValueError(f"Manifest {manifest_path} must contain a list of records")
    matching_records = [
        record for record in manifest if isinstance(record, dict) and record.get("path") == TRAINING_VIDEO_NAME
    ]
    if len(matching_records) != 1:
        raise ValueError(f"Manifest {manifest_path} must contain exactly one {TRAINING_VIDEO_NAME} record")
    captions = matching_records[0].get("cap")
    if not isinstance(captions, list) or len(captions) != 1 or not isinstance(captions[0], str):
        raise ValueError(f"The {TRAINING_VIDEO_NAME} record must contain exactly one caption string")
    caption = captions[0].strip()
    if not caption:
        raise ValueError(f"The {TRAINING_VIDEO_NAME} caption must be nonempty")
    video_path = video_dir / TRAINING_VIDEO_NAME
    if not video_path.is_file():
        raise FileNotFoundError(f"Crush-Smol training video is missing at {video_path}")
    return video_path, caption


def _load_component(
    name: str,
    model_path: Path,
    model_index: dict[str, Any],
    fastvideo_args: FastVideoArgs,
) -> Any:
    """Load one checkpoint component through the inference component registry.

    Using the registered loader keeps preprocessing aligned with the component
    classes and precision policy that produce H3 inference conditioning.
    """
    # Diffusers modular manifests can append loading metadata after the
    # provider and architecture fields defined by the component contract.
    transformers_or_diffusers, _ = model_index[name][:2]
    return PipelineComponentLoader.load_module(
        module_name=name,
        component_model_path=str(model_path / name),
        transformers_or_diffusers=transformers_or_diffusers,
        fastvideo_args=fastvideo_args,
    )


def encode_video_latents(
    frames: np.ndarray,
    model_path: Path,
    model_index: dict[str, Any],
    fastvideo_args: FastVideoArgs,
) -> torch.Tensor:
    """Encode normalized ``[24, T, H, W]`` causal video VAE targets.

    The video variational autoencoder (VAE) produces a channel-first latent
    layout that feeds H3's checkpoint-compatible token
    packing order ``(C, patch_t, patch_h, patch_w)`` without reordering latent
    channels into patch-major features.
    """
    print("Loading MiniMax H3 video VAE")
    vae = _load_component("vae", model_path, model_index, fastvideo_args)
    # Feed [B, C, T, H, W] pixels and retain [C, T, H, W] latents; H3 later
    # flattens every video token in (C, patch_t, patch_h, patch_w) order.
    pixels = torch.from_numpy(frames.copy()).permute(3, 0, 1, 2)[None]
    pixels = pixels.to(device=torch.device("cuda:0"), dtype=torch.float32).div_(255.0)
    generator = torch.Generator("cpu").manual_seed(42)
    with torch.no_grad():
        posterior = vae.encode(vae.normalize_pixels(pixels)).latent_dist
        latents = vae.normalize_latents(posterior.sample(generator=generator))
        normalized_latents = latents.squeeze(0).float().cpu().contiguous()
    del posterior, latents, pixels, vae
    gc.collect()
    torch.cuda.empty_cache()
    print(f"Video latent shape: {tuple(normalized_latents.shape)}")
    return normalized_latents


def encode_audio_latents(
    waveform: torch.Tensor,
    model_path: Path,
    model_index: dict[str, Any],
    fastvideo_args: FastVideoArgs,
) -> torch.Tensor:
    """Encode normalized ``[2, 32, T]`` stereo targets with the mono audio VAE.

    Treating the stereo axis as the VAE batch axis applies the same mono
    encoder to both synchronized channels while preserving channel identity.
    """
    print("Loading MiniMax H3 audio VAE")
    audio_vae = _load_component("audio_vae", model_path, model_index, fastvideo_args)
    waveform = waveform.to(device=torch.device("cuda:0"), dtype=torch.float32)
    with torch.no_grad():
        posterior = audio_vae.encode(waveform[:, None]).latent_dist
        latents = audio_vae.normalize_latents(posterior.mode())
        normalized_latents = latents.float().cpu().contiguous()
    del posterior, latents, waveform, audio_vae
    gc.collect()
    torch.cuda.empty_cache()
    print(f"Audio latent shape: {tuple(normalized_latents.shape)}")
    return normalized_latents


def encode_text_embedding(
    caption: str,
    model_path: Path,
    model_index: dict[str, Any],
    fastvideo_args: FastVideoArgs,
) -> torch.Tensor:
    """Encode the caption through the H3 Qwen3-VL layer-50 inference path.

    Reusing ``MiniMaxH3ConditioningStage`` keeps training tokenization and the
    selected hidden-state layer identical to validation conditioning.
    """
    print("Loading MiniMax H3 tokenizer, processor, and Qwen3-VL encoder")
    tokenizer = _load_component("tokenizer", model_path, model_index, fastvideo_args)
    processor = _load_component("processor", model_path, model_index, fastvideo_args)
    conditioner = _load_component("text_encoder", model_path, model_index, fastvideo_args)
    stage = MiniMaxH3ConditioningStage(
        conditioner=conditioner,
        tokenizer=tokenizer,
        processor=processor,
    )
    batch = ForwardBatch(data_type="video", prompt=caption)
    batch.extra[MINIMAX_H3_KEYFRAMES_KEY] = []
    batch = stage.forward(batch, fastvideo_args)
    if not batch.prompt_embeds:
        raise RuntimeError("MiniMax H3 conditioning returned no prompt embedding")
    text_embedding = batch.prompt_embeds[0].squeeze(0).float().cpu().contiguous()
    del batch, stage, conditioner, processor, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    print(f"Text embedding shape: {tuple(text_embedding.shape)}")
    return text_embedding


def build_parquet_record(
    *,
    file_name: str,
    caption: str,
    video_latents: torch.Tensor,
    audio_latents: torch.Tensor,
    text_embedding: torch.Tensor,
) -> dict[str, Any]:
    """Serialize one synchronized H3 sample for the Parquet schema collator.

    ``collate_rows_from_parquet_schema`` reconstructs every tensor from a
    bytes/shape/dtype triplet and reads the byte payload as float32, so this
    boundary stores contiguous float32 tensors for lossless reconstruction.
    """
    tensors = {
        "vae_latent": video_latents.float().contiguous(),
        "audio_latent": audio_latents.float().contiguous(),
        "text_embedding": text_embedding.float().contiguous(),
    }
    record: dict[str, Any] = {"id": Path(file_name).stem}
    for name, tensor in tensors.items():
        record[f"{name}_bytes"] = tensor.numpy().tobytes()
        record[f"{name}_shape"] = list(tensor.shape)
        record[f"{name}_dtype"] = "float32"
    record.update({
        "file_name": file_name,
        "caption": caption,
        "media_type": "video_with_audio",
        "width": VIDEO_WIDTH,
        "height": VIDEO_HEIGHT,
        "num_frames": NUM_FRAMES,
        "duration_sec": NUM_FRAMES / MINIMAX_H3_FPS,
        "fps": float(MINIMAX_H3_FPS),
        "audio_sample_rate": AUDIO_SAMPLE_RATE,
    })
    return record


def write_parquet(record: dict[str, Any], output_dir: Path) -> Path:
    """Write the sole overfit row after removing other Parquet shards.

    The training dataloader scans every Parquet shard in ``output_dir``;
    replacing those shards preserves the one-sample overfit contract.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    for parquet_path in output_dir.glob("*.parquet"):
        parquet_path.unlink()
    # The map-style cache stores Parquet file metadata and row counts, so a
    # replacement shard set requires cache reconstruction.
    shutil.rmtree(output_dir / "map_style_cache", ignore_errors=True)
    table = pa.table(
        {name: [record[name]]
         for name in pyarrow_schema_t2va.names},
        schema=pyarrow_schema_t2va,
    )
    output_path = output_dir / "data_00000.parquet"
    pq.write_table(table, output_path)
    return output_path


def validate_preprocessed_training_data(
    output_dir: Path = OUTPUT_DIR,
    manifest_path: Path = DATA_DIR / "videos2caption.json",
    video_dir: Path = DATA_DIR / "videos",
) -> None:
    """Verify that the H3 dataset contains only the selected Crush-Smol record.

    The launch path calls this function before Slurm submission so that stale
    Parquet shards or mismatched captions cannot enter the overfit run.
    """
    parquet_paths = sorted(output_dir.glob("*.parquet"))
    expected_path = output_dir / "data_00000.parquet"
    if parquet_paths != [expected_path]:
        raise ValueError(f"Expected only {expected_path}, found {parquet_paths}")
    table = pq.read_table(expected_path)
    if table.num_rows != 1:
        raise ValueError(f"Expected one training row in {expected_path}, found {table.num_rows}")
    video_path, expected_caption = load_crush_smol_training_sample(manifest_path, video_dir)
    record = table.to_pylist()[0]
    if record["file_name"] != video_path.name:
        raise ValueError(f"Training row file_name must be {video_path.name!r}, got {record['file_name']!r}")
    if record["caption"] != expected_caption:
        raise ValueError("Training row caption does not match the pinned Crush-Smol manifest")
    print(f"Validated one Crush-Smol H3 training row in {expected_path}")


def main() -> None:
    """Encode the selected Crush-Smol record with each H3 component in sequence.

    Releasing each component before loading the next component keeps video,
    audio, and text preprocessing within one GPU's memory.
    """
    _init_single_process_distributed()
    resolved_model_path = MODEL_PATH.resolve()
    if not resolved_model_path.is_dir():
        raise FileNotFoundError(f"Filtered MiniMax H3 model directory is missing at {resolved_model_path}")
    model_index = verify_model_config_and_directory(str(resolved_model_path))
    video_path, caption = load_crush_smol_training_sample(
        DATA_DIR / "videos2caption.json",
        DATA_DIR / "videos",
    )
    frames, waveform = load_training_media(video_path)
    pipeline_config = MiniMaxH3PipelineConfig()
    fastvideo_args = FastVideoArgs(
        model_path=str(resolved_model_path),
        pipeline_config=pipeline_config,
        num_gpus=1,
        tp_size=1,
        sp_size=1,
        hsdp_shard_dim=1,
        use_fsdp_inference=False,
        vae_cpu_offload=False,
        text_encoder_cpu_offload=False,
    )
    video_latents = encode_video_latents(frames, resolved_model_path, model_index, fastvideo_args)
    audio_latents = encode_audio_latents(waveform, resolved_model_path, model_index, fastvideo_args)
    text_embedding = encode_text_embedding(caption, resolved_model_path, model_index, fastvideo_args)
    record = build_parquet_record(
        file_name=TRAINING_VIDEO_NAME,
        caption=caption,
        video_latents=video_latents,
        audio_latents=audio_latents,
        text_embedding=text_embedding,
    )
    output_path = write_parquet(record, OUTPUT_DIR)
    print(f"Wrote one Crush-Smol MiniMax H3 T2VA record to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-only", action="store_true")
    cli_args = parser.parse_args()
    if cli_args.validate_only:
        validate_preprocessed_training_data()
    else:
        main()
