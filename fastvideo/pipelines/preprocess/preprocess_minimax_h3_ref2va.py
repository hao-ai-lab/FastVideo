# SPDX-License-Identifier: Apache-2.0
"""Precompute a raw MiniMax H3 Ref2VA manifest into training Parquet shards."""

from __future__ import annotations

import argparse
import gc
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any
import uuid

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from fastvideo.configs.pipelines.minimax_h3 import MiniMaxH3PipelineConfig
from fastvideo.dataset.minimax_h3_ref2va_dataset import (
    MINIMAX_H3_REF2VA_AUDIO_ROW_WIDTH,
    MINIMAX_H3_REF2VA_SCHEMA_VERSION,
    MINIMAX_H3_REF2VA_VISUAL_ROW_WIDTH,
    collate_minimax_h3_ref2va_rows,
    pyarrow_schema_minimax_h3_ref2va,
)
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.models.schedulers.scheduling_minimax_h3 import MiniMaxH3Scheduler
from fastvideo.pipelines import ForwardBatch
from fastvideo.pipelines.basic.minimax_h3.packing import (
    MINIMAX_H3_KEYFRAME_ENCODE_SEED,
    MINIMAX_H3_KEYFRAME_NOISE_AUG,
    audio_latent_num_frames,
    keyframe_condition_noise,
    patchify_video_latents,
    video_latent_num_frames,
)
from fastvideo.pipelines.basic.minimax_h3.ref2va_manifest import (
    MiniMaxH3RawReference,
    MiniMaxH3Ref2VARawSample,
    build_minimax_h3_references,
    load_minimax_h3_ref2va_raw_samples,
)
from fastvideo.pipelines.basic.minimax_h3.reference import (
    MiniMaxH3PreparedReference,
    prepare_reference,
    trim_reference_num_frames,
)
from fastvideo.pipelines.basic.minimax_h3.stages.minimax_h3_conditioning import (
    MINIMAX_H3_TEXT_TOKEN_TAGS_KEY,
    MiniMaxH3ConditioningStage,
)
from fastvideo.pipelines.basic.minimax_h3.stages.minimax_h3_input_preparation import MINIMAX_H3_KEYFRAMES_KEY
from fastvideo.pipelines.preprocess.preprocess_minimax_h3_overfit import (
    AUDIO_SAMPLE_RATE,
    MODEL_PATH,
    NUM_FRAMES,
    VIDEO_HEIGHT,
    VIDEO_WIDTH,
    _init_single_process_distributed,
    _load_component,
    build_parquet_record,
    encode_audio_latents,
    encode_video_latents,
    load_training_media,
)
from fastvideo.utils import verify_model_config_and_directory

DEFAULT_MANIFEST = Path("examples/training/finetune/minimax-h3/synthetic/train.jsonl")
DEFAULT_OUTPUT_DIR = Path("data/synthetic_h3_ref2va_single_sample_preprocessed")
_PATCH_SIZE = (1, 2, 2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--validate-manifest-only", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument(
        "--replace-existing",
        action="store_true",
        help=("Replace a non-empty output directory after the staged dataset "
              "validates; retain the previous directory as a backup."),
    )
    return parser.parse_args()


def _sample_visual_posterior(posterior: Any) -> torch.Tensor:
    generator = torch.Generator("cpu").manual_seed(MINIMAX_H3_KEYFRAME_ENCODE_SEED)
    return posterior.sample(generator=generator)


def encode_ref_visual_anchor(
    references: list[MiniMaxH3PreparedReference],
    model_path: Path,
    model_index: dict[str, Any],
    fastvideo_args: FastVideoArgs,
    patch_size: tuple[int, int, int],
) -> torch.Tensor:
    """Encode and cache official 0.999-noised ordered visual condition rows."""
    visual_references = [reference for reference in references if reference.media_type != "audio"]
    if not visual_references:
        return torch.empty((0, MINIMAX_H3_REF2VA_VISUAL_ROW_WIDTH), dtype=torch.float32)

    print("Loading MiniMax H3 video VAE for Ref2VA visual anchors")
    vae = _load_component("vae", model_path, model_index, fastvideo_args)
    device = torch.device("cuda:0")
    clean_rows: list[torch.Tensor] = []
    latent_channels = int(vae.latent_channels)
    with torch.no_grad():
        for reference in references:
            if reference.media_type == "audio":
                continue
            if reference.media_type == "image":
                if reference.image is None:
                    raise ValueError("Prepared image reference is missing pixels")
                pixels = torch.from_numpy(np.asarray(reference.image).copy()).permute(2, 0, 1)[None, :, None]
                pixels = pixels.to(device=device, dtype=torch.float32).div_(255.0)
                posterior = vae.encode_keyframe(vae.normalize_pixels(pixels)).latent_dist
            else:
                if reference.frames is None:
                    raise ValueError("Prepared video reference is missing frames")
                frames = reference.frames[:trim_reference_num_frames(reference.frames.shape[0])]
                pixels = torch.from_numpy(frames.copy()).permute(3, 0, 1, 2)[None]
                pixels = pixels.to(device=device, dtype=torch.float32).div_(255.0)
                posterior = vae.encode(vae.normalize_pixels(pixels)).latent_dist

            # The fp16 round trip before latent normalization is part of the
            # released Ref2VA condition encoding path.
            latents = vae.normalize_latents(_sample_visual_posterior(posterior).to(torch.float16).float()).cpu()
            if latents.ndim != 5 or latents.shape[0] != 1 or latents.shape[1] != latent_channels:
                raise ValueError(f"Unexpected reference visual latent shape: {tuple(latents.shape)}")
            reference.num_latent_frames = int(latents.shape[2])
            reference.latent_height = int(latents.shape[3])
            reference.latent_width = int(latents.shape[4])
            clean_rows.append(patchify_video_latents(latents, patch_size).float().contiguous())
            del posterior, latents, pixels

        clean_anchor = torch.cat(clean_rows).to(device=device, dtype=torch.float32)
        shapes = tuple((reference.num_latent_frames, reference.latent_height, reference.latent_width)
                       for reference in references if reference.media_type != "audio")
        noise_generator = torch.Generator("cpu").manual_seed(MINIMAX_H3_KEYFRAME_ENCODE_SEED)
        noise = keyframe_condition_noise(
            shapes,
            patch_size,
            latent_channels,
            generator=noise_generator,
            device=device,
            dtype=torch.float32,
        )
        anchor = MiniMaxH3Scheduler(shift=12.0).scale_noise(
            clean_anchor,
            MINIMAX_H3_KEYFRAME_NOISE_AUG,
            noise,
        ).float().cpu().contiguous()

    expected_width = latent_channels * int(np.prod(patch_size))
    if expected_width != MINIMAX_H3_REF2VA_VISUAL_ROW_WIDTH or anchor.shape[1] != expected_width:
        raise ValueError(
            f"Ref2VA visual anchor width must be {MINIMAX_H3_REF2VA_VISUAL_ROW_WIDTH}, got {anchor.shape[1]}")
    del clean_anchor, clean_rows, noise, vae
    gc.collect()
    torch.cuda.empty_cache()
    print(f"Ref2VA visual anchor shape: {tuple(anchor.shape)} at fixed clean-time 0.999")
    return anchor


def encode_ref_audio_anchor(
    references: list[MiniMaxH3PreparedReference],
    model_path: Path,
    model_index: dict[str, Any],
    fastvideo_args: FastVideoArgs,
) -> torch.Tensor:
    """Encode clean channel-major audio anchors in ordered-reference order."""
    if not any(reference.has_audio for reference in references):
        return torch.empty((0, MINIMAX_H3_REF2VA_AUDIO_ROW_WIDTH), dtype=torch.float32)

    print("Loading MiniMax H3 audio VAE for Ref2VA audio anchors")
    audio_vae = _load_component("audio_vae", model_path, model_index, fastvideo_args)
    device = torch.device("cuda:0")
    latent_channels = int(audio_vae.latent_channels)
    if int(audio_vae.sampling_rate) != AUDIO_SAMPLE_RATE:
        raise ValueError(f"Audio VAE sampling rate must be {AUDIO_SAMPLE_RATE}, got {audio_vae.sampling_rate}")
    rows: list[torch.Tensor] = []
    with torch.no_grad():
        for reference in references:
            if not reference.has_audio:
                continue
            if reference.waveform is None:
                raise ValueError("Audio-bearing reference is missing its prepared waveform")
            posterior = audio_vae.encode(reference.waveform.to(device=device, dtype=torch.float32)[:, None]).latent_dist
            latents = audio_vae.normalize_latents(posterior.mode().float()).cpu().transpose(1, 2)
            if latents.ndim != 3 or latents.shape[0] != 2 or latents.shape[2] != latent_channels:
                raise ValueError(f"Unexpected reference audio latent shape: {tuple(latents.shape)}")
            reference.num_audio_latents = int(latents.shape[1])
            rows.append(latents.reshape(-1, latent_channels).float().contiguous())
            del posterior, latents
    anchor = torch.cat(rows).float().contiguous()
    if latent_channels != MINIMAX_H3_REF2VA_AUDIO_ROW_WIDTH or anchor.shape[1] != latent_channels:
        raise ValueError(
            f"Ref2VA audio anchor width must be {MINIMAX_H3_REF2VA_AUDIO_ROW_WIDTH}, got {anchor.shape[1]}")
    del rows, audio_vae
    gc.collect()
    torch.cuda.empty_cache()
    print(f"Ref2VA audio anchor shape: {tuple(anchor.shape)}")
    return anchor


def encode_ref2va_conditioning(
    caption: str,
    references: list[MiniMaxH3PreparedReference],
    model_path: Path,
    model_index: dict[str, Any],
    fastvideo_args: FastVideoArgs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode the exact ordered presentation without padding or truncation."""
    print("Loading MiniMax H3 tokenizer, processor, and Qwen3-VL encoder")
    tokenizer = _load_component("tokenizer", model_path, model_index, fastvideo_args)
    processor = _load_component("processor", model_path, model_index, fastvideo_args)
    conditioner = _load_component("text_encoder", model_path, model_index, fastvideo_args)
    stage = MiniMaxH3ConditioningStage(
        conditioner=conditioner,
        tokenizer=tokenizer,
        processor=processor,
        ref2va=bool(references),
    )
    batch = ForwardBatch(data_type="video", prompt=caption, references=references)
    if not references:
        # The Ref stage intentionally rejects an empty list. Prompt-only rows
        # use the exact T2VA tokenizer path and contain only text tags.
        batch.extra[MINIMAX_H3_KEYFRAMES_KEY] = []
    batch = stage.forward(batch, fastvideo_args)
    if len(batch.prompt_embeds) != 1:
        raise RuntimeError("MiniMax H3 conditioning must return exactly one embedding")
    text_embedding = batch.prompt_embeds[0].squeeze(0).float().cpu().contiguous()
    text_token_tags = batch.extra.get(MINIMAX_H3_TEXT_TOKEN_TAGS_KEY)
    if not isinstance(text_token_tags, torch.Tensor):
        raise RuntimeError("MiniMax H3 conditioning did not return text token tags")
    text_token_tags = text_token_tags.to(dtype=torch.long, device="cpu").contiguous()
    if text_embedding.ndim != 2 or text_embedding.shape[1] != 5120 or text_embedding.shape[0] == 0:
        raise ValueError(f"Unexpected Qwen embedding shape: {tuple(text_embedding.shape)}")
    if text_token_tags.shape != text_embedding.shape[:1]:
        raise ValueError("Qwen text token tags do not align with its hidden states")
    if not bool(((text_token_tags == 0) | (text_token_tags == 1)).all()):
        raise ValueError("Qwen text token tags may contain only vision=0 and text=1")

    dynamic_length = int(text_embedding.shape[0])
    print(f"Qwen Ref2VA conditioning shape: {tuple(text_embedding.shape)}; preserving all {dynamic_length} tokens")
    del batch, stage, conditioner, processor, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    return text_embedding, text_token_tags


def _serialize_float32_tensor(record: dict[str, Any], name: str, tensor: torch.Tensor) -> None:
    tensor = tensor.detach().float().cpu().contiguous()
    record[f"{name}_bytes"] = tensor.numpy().tobytes()
    record[f"{name}_shape"] = list(tensor.shape)
    record[f"{name}_dtype"] = "float32"


def _canonical_prepared_references(references: list[MiniMaxH3PreparedReference]) -> list[dict[str, Any]]:
    canonical: list[dict[str, Any]] = []
    for reference in references:
        is_audio = reference.media_type == "audio"
        canonical.append({
            "media_type": reference.media_type,
            "has_audio": bool(reference.has_audio),
            # MiniMaxH3PreparedReference defaults num_latent_frames to 1, so
            # standalone audio must be canonicalized explicitly to zero visual
            # geometry rather than copying dataclass defaults.
            "num_latent_frames": 0 if is_audio else int(reference.num_latent_frames),
            "latent_height": 0 if is_audio else int(reference.latent_height),
            "latent_width": 0 if is_audio else int(reference.latent_width),
            "num_audio_latents": int(reference.num_audio_latents),
        })
    return canonical


def build_ref2va_parquet_record(
    *,
    file_name: str,
    caption: str,
    video_latents: torch.Tensor,
    audio_latents: torch.Tensor,
    text_embedding: torch.Tensor,
    text_token_tags: torch.Tensor,
    ref_visual_anchor: torch.Tensor,
    ref_audio_anchor: torch.Tensor,
    references: list[MiniMaxH3PreparedReference],
) -> dict[str, Any]:
    record = build_parquet_record(
        file_name=file_name,
        caption=caption,
        video_latents=video_latents,
        audio_latents=audio_latents,
        text_embedding=text_embedding,
    )
    record["schema_version"] = MINIMAX_H3_REF2VA_SCHEMA_VERSION
    record["text_token_tags"] = text_token_tags.to(dtype=torch.long, device="cpu").tolist()
    _serialize_float32_tensor(record, "ref_visual_anchor", ref_visual_anchor)
    _serialize_float32_tensor(record, "ref_audio_anchor", ref_audio_anchor)
    record["references"] = _canonical_prepared_references(references)
    return record


def validate_record_contract(record: dict[str, Any]) -> None:
    missing = [name for name in pyarrow_schema_minimax_h3_ref2va.names if name not in record]
    if missing:
        raise ValueError(f"Ref2VA record is missing schema fields: {missing}")
    expected_target_shapes = {
        "vae_latent_shape": [24, video_latent_num_frames(NUM_FRAMES), VIDEO_HEIGHT // 16, VIDEO_WIDTH // 16],
        "audio_latent_shape": [2, 32, audio_latent_num_frames(NUM_FRAMES)],
    }
    for name, expected in expected_target_shapes.items():
        if record[name] != expected:
            raise ValueError(f"{name} must be {expected}, got {record[name]}")
    # Reuse the actual training collator as the authoritative nested-reference,
    # empty-anchor, dtype, row-count, and dynamic-text validation gate.
    collate_minimax_h3_ref2va_rows([record])


def _prepare_references(references: tuple[MiniMaxH3RawReference, ...], ) -> list[MiniMaxH3PreparedReference]:
    raw_references = build_minimax_h3_references(references)
    prepared = [prepare_reference(reference, NUM_FRAMES, AUDIO_SAMPLE_RATE) for reference in raw_references]
    for index, (raw, resolved) in enumerate(zip(references, prepared, strict=True)):
        expected_audio = raw.media_type in {"audio", "video_audio"}
        if resolved.has_audio != expected_audio:
            raise ValueError(f"Reference {index} resolved has_audio={resolved.has_audio}, expected {expected_audio}; "
                             "check the declared reference type and soundtrack")
    return prepared


def _build_fastvideo_args(model_path: Path) -> tuple[FastVideoArgs, tuple[int, int, int]]:
    pipeline_config = MiniMaxH3PipelineConfig()
    patch_size = tuple(pipeline_config.dit_config.arch_config.patch_size)
    if patch_size != _PATCH_SIZE:
        raise ValueError(f"MiniMax H3 Ref2VA requires patch_size={_PATCH_SIZE}, got {patch_size}")
    return (
        FastVideoArgs(
            model_path=str(model_path),
            pipeline_config=pipeline_config,
            num_gpus=1,
            tp_size=1,
            sp_size=1,
            hsdp_shard_dim=1,
            use_fsdp_inference=False,
            vae_cpu_offload=False,
            text_encoder_cpu_offload=False,
        ),
        patch_size,
    )


def _build_record(
    sample: MiniMaxH3Ref2VARawSample,
    *,
    model_path: Path,
    model_index: dict[str, Any],
    fastvideo_args: FastVideoArgs,
    patch_size: tuple[int, int, int],
) -> dict[str, Any]:
    target_frames, target_waveform = load_training_media(sample.target_video_path)
    references = _prepare_references(sample.references)

    # The established single-sample helpers deliberately release each large
    # component before the next one is loaded, keeping this multi-sample path
    # within the same one-GPU memory envelope.
    video_latents = encode_video_latents(target_frames, model_path, model_index, fastvideo_args)
    audio_latents = encode_audio_latents(target_waveform, model_path, model_index, fastvideo_args)
    ref_visual_anchor = encode_ref_visual_anchor(
        references,
        model_path,
        model_index,
        fastvideo_args,
        patch_size,
    )
    ref_audio_anchor = encode_ref_audio_anchor(references, model_path, model_index, fastvideo_args)
    text_embedding, text_token_tags = encode_ref2va_conditioning(
        sample.caption,
        references,
        model_path,
        model_index,
        fastvideo_args,
    )
    record = build_ref2va_parquet_record(
        file_name=sample.target_file,
        caption=sample.caption,
        video_latents=video_latents,
        audio_latents=audio_latents,
        text_embedding=text_embedding,
        text_token_tags=text_token_tags,
        ref_visual_anchor=ref_visual_anchor,
        ref_audio_anchor=ref_audio_anchor,
        references=references,
    )
    record["id"] = sample.sample_id
    validate_record_contract(record)
    return record


def _validate_output_destination(output_dir: Path, *, replace_existing: bool) -> None:
    if not output_dir.exists():
        return
    if not output_dir.is_dir():
        raise NotADirectoryError(f"Preprocessing output exists but is not a directory: {output_dir}")
    if any(output_dir.iterdir()) and not replace_existing:
        raise FileExistsError(f"Refusing to replace non-empty preprocessing output {output_dir}. "
                              "Pass --replace-existing to stage and validate a replacement.")


def _new_staging_directory(output_dir: Path) -> Path:
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=str(output_dir.parent)))


def _promote_staged_dataset(
    staging_dir: Path,
    output_dir: Path,
    *,
    replace_existing: bool,
) -> Path | None:
    """Promote one validated sibling directory and retain any old dataset."""
    _validate_output_destination(output_dir, replace_existing=replace_existing)
    backup_dir: Path | None = None
    if output_dir.exists() and any(output_dir.iterdir()):
        backup_dir = output_dir.with_name(f".{output_dir.name}.backup-{uuid.uuid4().hex}")
        os.replace(output_dir, backup_dir)
    try:
        # A sibling rename is atomic when the destination is absent (or an
        # existing empty directory). The old non-empty dataset was moved, not
        # deleted, and is restored if promotion fails.
        os.replace(staging_dir, output_dir)
    except BaseException:
        if backup_dir is not None and backup_dir.exists() and not output_dir.exists():
            os.replace(backup_dir, output_dir)
        raise
    return backup_dir


def _write_record(record: dict[str, Any], output_dir: Path, index: int) -> Path:
    table = pa.table(
        {name: [record[name]]
         for name in pyarrow_schema_minimax_h3_ref2va.names},
        schema=pyarrow_schema_minimax_h3_ref2va,
    )
    output_path = output_dir / f"data_{index:05d}.parquet"
    pq.write_table(table, output_path)
    return output_path


def validate_preprocessed_dataset(
    *,
    manifest_path: Path,
    output_dir: Path,
) -> None:
    samples = load_minimax_h3_ref2va_raw_samples(manifest_path)
    expected_paths = [output_dir / f"data_{index:05d}.parquet" for index in range(len(samples))]
    parquet_paths = sorted(output_dir.glob("*.parquet"))
    if parquet_paths != expected_paths:
        raise ValueError(f"Expected Parquet shards {expected_paths}, found {parquet_paths}")

    for sample, parquet_path in zip(samples, parquet_paths, strict=True):
        table = pq.read_table(parquet_path)
        if table.num_rows != 1:
            raise ValueError(f"Expected one row in {parquet_path}, found {table.num_rows}")
        if not table.schema.equals(pyarrow_schema_minimax_h3_ref2va, check_metadata=False):
            raise ValueError(f"Unexpected Parquet schema in {parquet_path}: {table.schema}")
        record = table.to_pylist()[0]
        if record["id"] != sample.sample_id:
            raise ValueError(f"Row id {record['id']!r} does not match manifest id {sample.sample_id!r}")
        if record["file_name"] != sample.target_file:
            raise ValueError(f"Row file_name {record['file_name']!r} does not match target {sample.target_file!r}")
        if record["caption"] != sample.caption:
            raise ValueError(f"Row caption does not match manifest sample {sample.sample_id!r}")
        validate_record_contract(record)

    print(f"Validated {len(samples)} MiniMax H3 Ref2VA training sample(s) in {output_dir}")


def preprocess(
    *,
    manifest_path: Path,
    model_path: Path,
    output_dir: Path,
    replace_existing: bool = False,
) -> list[Path]:
    samples = load_minimax_h3_ref2va_raw_samples(manifest_path)
    model_path = model_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    _validate_output_destination(output_dir, replace_existing=replace_existing)
    if output_dir.is_relative_to(model_path) or model_path.is_relative_to(output_dir):
        raise ValueError("Preprocessing output and MiniMax H3 model directories must not overlap")
    if not model_path.is_dir():
        raise FileNotFoundError(f"MiniMax H3 model directory is missing at {model_path}")
    _init_single_process_distributed()
    model_index = verify_model_config_and_directory(str(model_path))
    required_components = {"vae", "audio_vae", "tokenizer", "processor", "text_encoder", "transformer_ref"}
    missing_components = sorted(required_components - set(model_index))
    if missing_components:
        raise ValueError(f"MiniMax H3 Ref2VA checkpoint is missing components: {missing_components}")
    fastvideo_args, patch_size = _build_fastvideo_args(model_path)

    staging_dir = _new_staging_directory(output_dir)
    try:
        staged_paths: list[Path] = []
        for index, sample in enumerate(samples):
            print(f"Preprocessing sample {index + 1}/{len(samples)}: {sample.sample_id}")
            record = _build_record(
                sample,
                model_path=model_path,
                model_index=model_index,
                fastvideo_args=fastvideo_args,
                patch_size=patch_size,
            )
            output_path = _write_record(record, staging_dir, index)
            staged_paths.append(output_path)
            print(f"Staged {sample.sample_id} at {output_path}")

        validate_preprocessed_dataset(manifest_path=manifest_path, output_dir=staging_dir)
        backup_dir = _promote_staged_dataset(
            staging_dir,
            output_dir,
            replace_existing=replace_existing,
        )
        if backup_dir is not None:
            print(f"Retained previous preprocessing output at {backup_dir}")
        return [output_dir / path.name for path in staged_paths]
    finally:
        if staging_dir.exists():
            shutil.rmtree(staging_dir, ignore_errors=True)


def main() -> None:
    args = parse_args()
    manifest_path = args.manifest.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if args.validate_manifest_only:
        samples = load_minimax_h3_ref2va_raw_samples(manifest_path)
        print(f"Validated {len(samples)} raw MiniMax H3 Ref2VA sample(s) in {manifest_path}")
        return
    if args.validate_only:
        validate_preprocessed_dataset(manifest_path=manifest_path, output_dir=output_dir)
        return
    output_paths = preprocess(
        manifest_path=manifest_path,
        model_path=args.model_path,
        output_dir=output_dir,
        replace_existing=bool(args.replace_existing),
    )
    print(f"Prepared {len(output_paths)} MiniMax H3 Ref2VA training shard(s) in {output_dir}")


if __name__ == "__main__":
    main()
