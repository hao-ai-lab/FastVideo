# SPDX-License-Identifier: Apache-2.0
"""Precompute a raw MiniMax H3 Ref2VA manifest into training Parquet shards."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from fastvideo.configs.pipelines.minimax_h3 import MiniMaxH3PipelineConfig
from fastvideo.dataset.minimax_h3_ref2va_dataset import pyarrow_schema_minimax_h3_ref2va
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.pipelines.basic.minimax_h3.ref2va_manifest import (
    MiniMaxH3RawReference,
    MiniMaxH3Ref2VARawSample,
    build_minimax_h3_references,
    load_minimax_h3_ref2va_raw_samples,
)
from fastvideo.pipelines.basic.minimax_h3.reference import MiniMaxH3PreparedReference, prepare_reference
from fastvideo.pipelines.preprocess.preprocess_minimax_h3_overfit import (
    AUDIO_SAMPLE_RATE,
    MODEL_PATH,
    NUM_FRAMES,
    _init_single_process_distributed,
    encode_audio_latents,
    encode_video_latents,
    load_training_media,
)
from fastvideo.pipelines.preprocess.preprocess_minimax_h3_ref2va_overfit import (
    build_ref2va_parquet_record,
    encode_ref2va_conditioning,
    encode_ref_audio_anchor,
    encode_ref_visual_anchor,
    validate_record_contract,
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
    return parser.parse_args()


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
    record["file_name"] = sample.target_file
    validate_record_contract(record)
    return record


def _reset_output_shards(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for parquet_path in output_dir.glob("*.parquet"):
        parquet_path.unlink()
    shutil.rmtree(output_dir / "map_style_cache", ignore_errors=True)


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
) -> list[Path]:
    samples = load_minimax_h3_ref2va_raw_samples(manifest_path)
    _init_single_process_distributed()
    model_path = model_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if not model_path.is_dir():
        raise FileNotFoundError(f"MiniMax H3 model directory is missing at {model_path}")
    model_index = verify_model_config_and_directory(str(model_path))
    required_components = {"vae", "audio_vae", "tokenizer", "processor", "text_encoder", "transformer_ref"}
    missing_components = sorted(required_components - set(model_index))
    if missing_components:
        raise ValueError(f"MiniMax H3 Ref2VA checkpoint is missing components: {missing_components}")
    fastvideo_args, patch_size = _build_fastvideo_args(model_path)

    _reset_output_shards(output_dir)
    output_paths: list[Path] = []
    for index, sample in enumerate(samples):
        print(f"Preprocessing sample {index + 1}/{len(samples)}: {sample.sample_id}")
        record = _build_record(
            sample,
            model_path=model_path,
            model_index=model_index,
            fastvideo_args=fastvideo_args,
            patch_size=patch_size,
        )
        output_path = _write_record(record, output_dir, index)
        output_paths.append(output_path)
        print(f"Wrote {sample.sample_id} to {output_path}")

    validate_preprocessed_dataset(manifest_path=manifest_path, output_dir=output_dir)
    return output_paths


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
    )
    print(f"Prepared {len(output_paths)} MiniMax H3 Ref2VA training shard(s) in {output_dir}")


if __name__ == "__main__":
    main()
