# SPDX-License-Identifier: Apache-2.0
"""Encode raw Ref2VA media for packed MiniMax-H3 DMD/PDD training.

This output stores clean reference rows. It is distinct from the pre-noised
anchor schema used by ``preprocess_minimax_h3_ref2va`` and Ref2VA SFT.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from numbers import Integral
from pathlib import Path
import shutil
from typing import Any

import numpy as np
from PIL import Image
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from fastvideo.dataset.dataloader.record_schema import ref2va_record_creator
from fastvideo.dataset.dataloader.schema import pyarrow_schema_ref2va
from fastvideo.dataset.utils import collate_rows_from_parquet_schema
from fastvideo.pipelines.basic.minimax_h3.packing import (
    MINIMAX_H3_AUDIO_LATENTS_PER_SECOND,
    MINIMAX_H3_FPS,
    MINIMAX_H3_KEYFRAME_ENCODE_SEED,
    audio_latent_num_frames,
    video_latent_num_frames,
)
from fastvideo.pipelines.basic.minimax_h3.ref2va_manifest import (
    MiniMaxH3Ref2VARawSample,
    _records_from_json_document,
    build_minimax_h3_references,
    parse_minimax_h3_ref2va_raw_sample,
)
from fastvideo.pipelines.basic.minimax_h3.reference import (
    decode_reference_video,
    prepare_reference,
    prepare_reference_waveform,
    resample_reference_frames,
)
from fastvideo.pipelines.preprocess import preprocess_minimax_h3_ref2va as encoding
from fastvideo.utils import verify_model_config_and_directory

# Source audio kept past the clip end so resampling edge effects fall outside it.
_RESAMPLE_MARGIN_SECONDS = 0.1


@dataclass(frozen=True)
class TargetGeometry:
    width: int = 1344
    height: int = 768
    num_frames: int = 124

    def __post_init__(self) -> None:
        for name in ("width", "height", "num_frames"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
                raise ValueError(f"Target {name} must be a positive integer")
        if self.width % 32 or self.height % 32:
            raise ValueError("Target width and height must be divisible by 32")
        video_latent_num_frames(self.num_frames)

    @property
    def bucket(self) -> str:
        return f"bucket={self.width}x{self.height}-{self.num_frames}f"


def load_samples(path: Path, default_geometry: TargetGeometry) -> list[tuple[MiniMaxH3Ref2VARawSample, TargetGeometry]]:
    """Read the public raw schema with an optional per-record target_geometry."""
    path = path.expanduser().resolve()
    text = path.read_text(encoding="utf-8")
    try:
        records = _records_from_json_document(json.loads(text), path)
    except json.JSONDecodeError:
        records = [json.loads(line) for line in text.splitlines() if line.strip()]
    if not records:
        raise ValueError("The input manifest contains no samples")
    samples = []
    ids: set[str] = set()
    for record in records:
        if not isinstance(record, dict):
            raise TypeError("Every raw sample must be a JSON object")
        raw = dict(record)
        geometry_fields = raw.pop("target_geometry", None)
        if geometry_fields is None:
            geometry = default_geometry
        else:
            if not isinstance(geometry_fields, dict) or set(geometry_fields) != {"width", "height", "num_frames"}:
                raise ValueError("target_geometry must contain exactly width, height and num_frames")
            geometry = TargetGeometry(**geometry_fields)
        sample = parse_minimax_h3_ref2va_raw_sample(raw, manifest_path=path)
        if sample.sample_id in ids:
            raise ValueError(f"Duplicate sample id: {sample.sample_id!r}")
        ids.add(sample.sample_id)
        if not any(ref.media_type in {"image", "video", "video_audio"} for ref in sample.references):
            raise ValueError("Packed Ref2VA training requires at least one visual reference")
        samples.append((sample, geometry))
    return samples


def load_target_media(path: Path, geometry: TargetGeometry) -> tuple[np.ndarray, torch.Tensor]:
    """Resample to 24 fps, trim, and resize to an explicitly declared target canvas."""
    frames, source_fps, soundtrack = decode_reference_video(path)
    frames = resample_reference_frames(frames, source_fps)
    if len(frames) < geometry.num_frames:
        raise ValueError(f"Target {path} has {len(frames)} frames at 24 fps; {geometry.num_frames} required")
    frames = frames[:geometry.num_frames]
    if frames.shape[1:3] != (geometry.height, geometry.width):
        frames = np.stack([
            np.asarray(Image.fromarray(frame).resize((geometry.width, geometry.height), Image.Resampling.LANCZOS))
            for frame in frames
        ])
    if soundtrack is None:
        raise ValueError(f"Target {path} requires a synchronized soundtrack")
    waveform, source_rate = soundtrack
    duration = geometry.num_frames / MINIMAX_H3_FPS
    # Resample a little past the clip and cut at the VAE rate: cutting at the
    # source rate first floors twice and loses a sample at some rates, and the
    # margin keeps resampling edge effects out of the clip.
    waveform = prepare_reference_waveform(waveform,
                                          source_rate,
                                          encoding.AUDIO_SAMPLE_RATE,
                                          max_duration=duration + _RESAMPLE_MARGIN_SECONDS)
    # The audio VAE pads to whole latents. Never hand it more audio than the
    # clip's audio_latent_num_frames covers, which would add one latent.
    samples_per_latent = encoding.AUDIO_SAMPLE_RATE // MINIMAX_H3_AUDIO_LATENTS_PER_SECOND
    expected_samples = min(int(duration * encoding.AUDIO_SAMPLE_RATE),
                           audio_latent_num_frames(geometry.num_frames) * samples_per_latent)
    shortfall = expected_samples - waveform.shape[-1]
    if shortfall > math.ceil(encoding.AUDIO_SAMPLE_RATE / source_rate):
        raise ValueError(f"Target soundtrack is shorter than the requested {geometry.num_frames} frames")
    if shortfall > 0:
        # Less than one source sample short: a rounding artifact, not missing audio.
        waveform = torch.nn.functional.pad(waveform, (0, shortfall))
    return frames, waveform[:, :expected_samples]


def build_record(sample: MiniMaxH3Ref2VARawSample, geometry: TargetGeometry, *, model_path: Path,
                 model_index: dict[str, Any], fastvideo_args: Any, patch_size: tuple[int, int, int]) -> dict[str, Any]:
    frames, waveform = load_target_media(sample.target_video_path, geometry)
    references = [
        prepare_reference(ref, geometry.num_frames, encoding.AUDIO_SAMPLE_RATE)
        for ref in build_minimax_h3_references(sample.references)
    ]
    for raw, prepared in zip(sample.references, references, strict=True):
        if prepared.has_audio != (raw.media_type in {"audio", "video_audio"}):
            raise ValueError("Decoded reference audio does not match its declared reference type")
    video = encoding.encode_video_latents(frames, model_path, model_index, fastvideo_args)
    audio = encoding.encode_audio_latents(waveform, model_path, model_index, fastvideo_args)
    visual_rows = encoding.encode_ref_visual_anchor(references,
                                                    model_path,
                                                    model_index,
                                                    fastvideo_args,
                                                    patch_size,
                                                    augment=False)
    audio_rows = encoding.encode_ref_audio_anchor(references, model_path, model_index, fastvideo_args)
    text, tags = encoding.encode_ref2va_conditioning(sample.caption, references, model_path, model_index,
                                                     fastvideo_args)
    metadata = encoding._canonical_prepared_references(references)
    return ref2va_record_creator(
        sample.sample_id,
        sample.caption,
        file_name=sample.target_file,
        width=geometry.width,
        height=geometry.height,
        num_frames=geometry.num_frames,
        text_embedding=text.numpy(),
        text_token_tags=tags.numpy(),
        reference_video_latent_rows=visual_rows.numpy(),
        reference_audio_latent_rows=audio_rows.numpy(),
        reference_media_types=[ref["media_type"] for ref in metadata],
        reference_has_audio=[ref["has_audio"] for ref in metadata],
        reference_num_latent_frames=[ref["num_latent_frames"] for ref in metadata],
        reference_latent_heights=[ref["latent_height"] for ref in metadata],
        reference_latent_widths=[ref["latent_width"] for ref in metadata],
        reference_num_audio_latents=[ref["num_audio_latents"] for ref in metadata],
        reference_noise_seed=MINIMAX_H3_KEYFRAME_ENCODE_SEED,
        vae_latent=video.numpy(),
        audio_latent=audio.numpy(),
    )


def validate_record(record: dict[str, Any], geometry: TargetGeometry, max_presentation_tokens: int) -> None:
    expected = {
        "vae_latent_shape":
        [24, video_latent_num_frames(geometry.num_frames), geometry.height // 16, geometry.width // 16],
        "audio_latent_shape": [2, 32, audio_latent_num_frames(geometry.num_frames)],
    }
    for key, shape in expected.items():
        if record.get(key) != shape:
            raise ValueError(f"{key} must be {shape}, got {record.get(key)}")
    for key in ("width", "height", "num_frames"):
        if record.get(key) != getattr(geometry, key):
            raise ValueError(f"Record {key} differs from its declared target geometry")
    batch = collate_rows_from_parquet_schema([record],
                                             pyarrow_schema_ref2va,
                                             text_padding_length=None,
                                             max_presentation_tokens=max_presentation_tokens)
    for key, value in batch.items():
        if isinstance(value, torch.Tensor) and not bool(torch.isfinite(value).all()):
            raise ValueError(f"Non-finite tensor in encoded field {key}")


def preprocess(*,
               manifest_path: Path,
               model_path: Path,
               output_dir: Path,
               geometry: TargetGeometry,
               max_presentation_tokens: int = 65536) -> list[Path]:
    """Stage all rows and publish only after every encoded row validates."""
    samples = load_samples(manifest_path, geometry)
    if isinstance(max_presentation_tokens,
                  bool) or not isinstance(max_presentation_tokens, int) or max_presentation_tokens <= 0:
        raise ValueError("max_presentation_tokens must be a positive integer")
    model_path, output_dir = model_path.expanduser().resolve(), output_dir.expanduser().resolve()
    encoding._validate_output_destination(output_dir, replace_existing=False)
    if output_dir.is_relative_to(model_path) or model_path.is_relative_to(output_dir):
        raise ValueError("Model and preprocessing output directories must not overlap")
    if not model_path.is_dir():
        raise FileNotFoundError(f"Model directory does not exist: {model_path}")
    model_index = verify_model_config_and_directory(str(model_path))
    required = {"vae", "audio_vae", "tokenizer", "processor", "text_encoder", "transformer_ref"}
    if missing := required - model_index.keys():
        raise ValueError(f"Checkpoint is missing components: {sorted(missing)}")
    encoding._init_single_process_distributed()
    fastvideo_args, patch_size = encoding._build_fastvideo_args(model_path)
    staging = encoding._new_staging_directory(output_dir)
    relative_paths = []
    try:
        for index, (sample, target_geometry) in enumerate(samples):
            record = build_record(sample,
                                  target_geometry,
                                  model_path=model_path,
                                  model_index=model_index,
                                  fastvideo_args=fastvideo_args,
                                  patch_size=patch_size)
            validate_record(record, target_geometry, max_presentation_tokens)
            relative = Path(target_geometry.bucket) / f"data_{index:08d}.parquet"
            path = staging / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(pa.Table.from_pylist([record], schema=pyarrow_schema_ref2va), path)
            # Read through ParquetFile to avoid Hive partition-column inference.
            roundtrip = pq.ParquetFile(path).read().to_pylist()[0]
            validate_record(roundtrip, target_geometry, max_presentation_tokens)
            relative_paths.append(relative)
        encoding._promote_staged_dataset(staging, output_dir, replace_existing=False)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return [output_dir / relative for relative in relative_paths]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--width", type=int, default=1344)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--num-frames", type=int, default=124)
    parser.add_argument("--max-presentation-tokens", type=int, default=65536)
    parser.add_argument("--validate-manifest-only", action="store_true")
    args = parser.parse_args()
    geometry = TargetGeometry(args.width, args.height, args.num_frames)
    if args.validate_manifest_only:
        samples = load_samples(args.manifest, geometry)
        print(f"Validated {len(samples)} packed Ref2VA inputs")
        return
    paths = preprocess(manifest_path=args.manifest,
                       model_path=args.model_path,
                       output_dir=args.output_dir,
                       geometry=geometry,
                       max_presentation_tokens=args.max_presentation_tokens)
    print(f"Prepared {len(paths)} packed Ref2VA training rows in {args.output_dir}")


if __name__ == "__main__":
    main()
