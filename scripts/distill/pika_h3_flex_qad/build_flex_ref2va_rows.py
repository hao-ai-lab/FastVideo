# SPDX-License-Identifier: Apache-2.0
"""Pika H3 Flex Ref2VA rows from the OmniRef v4 encoded rows: Flex reference sizing plus the inputs v4 lacks.

The OmniRef v4 rows (the FastH3 OmniRef QAD data) carry every image reference at a 2048 short edge and every video
reference on the 768p canvas. Pika H3 Flex sizes them by tier (1024 / 480p canvas below 768p, 2048 / 768p canvas at
768p) and accepts input mixes v4 never encoded. This builds packed Ref2VA rows (``pyarrow_schema_ref2va``) from the
encoded v4 rows alone; the source media is not needed. Each reference is decoded from its stored latents with the
H3 VAE / audio VAE, then prepared and re-encoded with the released Ref2VA encode path at Flex's tier sizing:

``retier``             a v4 row's own references and prompt; same case and id as the v4 row, so a manifest listed
                       before v4 replaces it (``load_manifest_groups`` keeps the first entry per id and resolution).
``audio_reference``    first-frame rows plus the clip's own soundtrack as a standalone audio reference.
``mixed_image_video``  continuation rows plus the target's first frame as an image ahead of the preceding video.
``many_images``        storyboard rows plus image references of other rows of the same asset, 6-9 images.
``multi_video``        continuation rows plus a sibling clip's preceding video (silent) from the same asset.

Target video and audio latents are copied from the v4 row (QAD regresses on the teacher; they carry geometry).

    build_flex_ref2va_rows.py plan --v4-manifest <v4 MANIFEST.jsonl> --plan-config <QAD yaml> \\
        --plan-max-frames 345 --eval-manifest <held-out manifest> --output-dir <out>
    CUDA_VISIBLE_DEVICES=<g> build_flex_ref2va_rows.py encode --model-path <merged Flex> --output-dir <out> \\
        --shard <g> --num-shards <n>
    build_flex_ref2va_rows.py finalize --output-dir <out> --eval-manifest <held-out manifest>
"""
from __future__ import annotations

import argparse
import functools
import hashlib
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

V4_CASES = ("first_frame", "first_last_frame", "storyboard", "continue_scene", "continue_shot")
IMAGE_CASES = ("first_frame", "first_last_frame", "storyboard")
CONTINUE_CASES = ("continue_scene", "continue_shot")
NEW_CASES = ("audio_reference", "mixed_image_video", "many_images", "multi_video")
MAX_IMAGES = 9
MAX_VIDEO_SECONDS = 15.0
FPS = 24
_ASSET = re.compile(r"(asset-[0-9a-f]{32})")
_PATCH = (1, 2, 2)


# ---------------------------------------------------------------------- planning (CPU)
def load_v4(manifest: Path) -> dict[tuple[str, str], dict[str, Any]]:
    """Encoded v4 entries keyed by (id, resolution)."""
    entries = {}
    for line in manifest.read_text().splitlines():
        entry = json.loads(line)
        entries[(entry["id"], entry["resolution"])] = entry
    return entries


def asset_of(entry_id: str) -> str:
    match = _ASSET.search(entry_id)
    if match is None:
        raise ValueError(f"{entry_id}: no asset id")
    return match.group(1)


def qad_plan_ids(config: str, overrides: list[str]) -> set[tuple[str, str]]:
    """(id, resolution) of every training and held-out row the QAD config reads from the v4 manifest."""
    from fastvideo.train.models.minimax_h3.omniref_data import (OMNIREF_CASES, limit_frames, load_eval_rows,
                                                                load_manifest_groups, select_rows, top_up_eval_rows)
    from fastvideo.train.utils.config import load_run_config

    cfg = load_run_config(config, overrides=overrides).raw["models"]["student"]["data"]
    cases = tuple(cfg.get("cases", OMNIREF_CASES))
    resolutions = tuple(cfg.get("eval_resolutions", ("480p", )))
    eval_per_group = cfg.get("eval_per_group", 2)
    groups = load_manifest_groups(list(cfg["manifests"]), cases, int(cfg.get("max_frames", 243)))
    eval_rows = load_eval_rows(cfg["eval_manifest"], cases, resolutions, eval_per_group,
                               int(cfg.get("eval_max_frames", 0)))
    if isinstance(eval_per_group, dict):
        eval_groups = limit_frames({k: v for k, v in groups.items() if k[1] in resolutions},
                                   int(cfg.get("eval_max_frames", 0)))
        eval_rows = top_up_eval_rows(eval_rows, eval_groups, int(cfg.get("eval_seed", 20261008)), dict(eval_per_group))
    plan = select_rows(groups, int(cfg.get("seed", 20261007)), dict(cfg.get("per_group", {"480p": 400})),
                       frozenset(row.source for row in eval_rows))
    return {(v4_id(row.case, row.source), row.resolution) for row in [*plan, *eval_rows]}


def v4_id(case: str, source: str) -> str:
    """The v4 manifest id of (case, source): image cases are ``<case>:<clip>``, continuation ids are the clip."""
    return f"{case}:{source}" if case in IMAGE_CASES else source


def _subject(prompt: str, added: str, *, before: bool = False) -> str:
    """Add one definition to the prompt's ``subject_definitions:`` paragraph."""
    head, sep, rest = prompt.partition("subject_definitions:")
    if not sep:
        raise ValueError("prompt has no subject_definitions section")
    body, gap, tail = rest.partition("\n\n")
    body = f" {added}{body}" if before else f"{body.rstrip()} {added}"
    return f"{head}{sep}{body}{gap}{tail}"


def spec(case: str, base: dict[str, Any], refs: list[dict[str, Any]], prompt_edits: list[tuple[str, bool]],
         **extra: Any) -> dict[str, Any]:
    """Row spec: ``base``'s target and prompt, ``refs`` (stored-latent sources), prompt definitions to add."""
    source = base["id"].split(":", 1)[-1]
    return {
        "id": base["id"] if case == "retier" else f"{case}:{source}",
        "case": base["case"] if case == "retier" else case,
        "source_case": base["case"],
        "resolution": base["resolution"],
        "base_parquet": base["parquet"],
        "references": refs,
        "prompt_edits": prompt_edits,
        **extra,
    }


def own_refs(entry: dict[str, Any], count: int, *, audio: bool = True) -> list[dict[str, Any]]:
    """``count`` stored references of ``entry`` (their embedded audio kept unless ``audio`` is False)."""
    return [{"kind": "stored", "parquet": entry["parquet"], "index": i, "audio": audio} for i in range(count)]


def plan_rows(v4: dict[tuple[str, str], dict[str, Any]], retier_ids: set[tuple[str, str]], extra: dict[str, int],
              seed: int, exclude_sources: set[str], ref_count: Any) -> list[dict[str, Any]]:
    """Every spec: the 480p retier rows plus ``extra[resolution]`` rows per new case (held-out sources excluded).

    ``ref_count(parquet)`` is the number of references of a stored row.
    """
    specs = []
    for key in sorted(retier_ids):
        if key in v4 and key[1] == "480p":
            entry = v4[key]
            specs.append(spec("retier", entry, own_refs(entry, ref_count(entry["parquet"])), []))
    by_asset: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for (entry_id, res), entry in v4.items():
        kind = "continue" if entry["case"] in CONTINUE_CASES else "image"
        by_asset[(asset_of(entry_id), res, kind)].append(entry)
    sources = {"audio_reference": ("first_frame", ), "mixed_image_video": CONTINUE_CASES,
               "many_images": ("storyboard", ), "multi_video": CONTINUE_CASES}
    for case in NEW_CASES:
        for res, count in sorted(extra.items()):
            pool = sorted((e for (i, r), e in v4.items()
                           if r == res and e["case"] in sources[case] and i.split(":", 1)[-1] not in exclude_sources),
                          key=lambda e: e["id"])
            rng = random.Random(f"{seed}:{case}:{res}")
            rng.shuffle(pool)
            made = 0
            for entry in pool:
                if made >= count:
                    break
                row = new_case_spec(case, entry, rng, by_asset, ref_count)
                if row is not None:
                    specs.append(row)
                    made += 1
    return specs


def new_case_spec(case: str, entry: dict[str, Any], rng: random.Random,
                  by_asset: dict[tuple[str, str, str], list[dict[str, Any]]], ref_count: Any) -> dict[str, Any] | None:
    count = ref_count(entry["parquet"])
    asset, res = asset_of(entry["id"]), entry["resolution"]
    if case == "audio_reference":
        refs = [*own_refs(entry, count), {"kind": "target_audio", "parquet": entry["parquet"]}]
        return spec(case, entry, refs, [("<Audio 1> is the soundtrack of the target video.", False)])
    if case == "mixed_image_video":
        refs = [{"kind": "target_frame", "parquet": entry["parquet"]}, *own_refs(entry, count)]
        return spec(case, entry, refs, [("<Picture 1> is the exact first frame of the target video.", True)])
    if case == "many_images":
        others = [e for e in by_asset[(asset, res, "image")] if e["id"] != entry["id"]]
        others = rng.sample(others, min(len(others), 12))
        want = rng.randint(max(6, count + 1), MAX_IMAGES)
        pool = [(e, i) for e in others for i in range(ref_count(e["parquet"]))]
        if len(pool) < want - count:
            return None
        picks = rng.sample(pool, want - count)
        refs = [*own_refs(entry, count), *({"kind": "stored", "parquet": e["parquet"], "index": i, "audio": False}
                                           for e, i in picks)]
        edits = [(f"<Picture {n}> is another frame from the same source video.", False)
                 for n in range(count + 1, want + 1)]
        return spec(case, entry, refs, edits)
    if case == "multi_video":
        siblings = [e for e in by_asset[(asset, res, "continue")] if e["id"] != entry["id"]]
        if not siblings:
            return None
        other = rng.choice(siblings)
        refs = [*own_refs(entry, count), {"kind": "stored", "parquet": other["parquet"], "index": 0, "audio": False}]
        return spec(case, entry, refs, [("<Video 2> is an earlier shot from the same scene, without sound.", False)])
    raise ValueError(f"unknown case {case!r}")


def apply_prompt_edits(prompt: str, edits: list[tuple[str, bool]]) -> str:
    for added, before in edits:
        prompt = _subject(prompt, added, before=before)
    return prompt


# ---------------------------------------------------------------------- stored-latent decoding (GPU)
_COLUMNS = ("caption", "vae_latent_bytes", "vae_latent_shape", "audio_latent_bytes", "audio_latent_shape",
            "reference_video_latent_rows_bytes", "reference_video_latent_rows_shape",
            "reference_audio_latent_rows_bytes", "reference_audio_latent_rows_shape", "reference_media_types",
            "reference_has_audio", "reference_num_latent_frames", "reference_latent_heights",
            "reference_latent_widths", "reference_num_audio_latents")


@functools.cache
def reference_count(parquet: str) -> int:
    import pyarrow.parquet as pq
    return len(pq.ParquetFile(parquet).read(columns=["reference_media_types"]).to_pylist()[0]["reference_media_types"])


@functools.lru_cache(maxsize=16)
def stored_row(parquet: str) -> dict[str, Any]:
    import pyarrow.parquet as pq
    return pq.ParquetFile(parquet).read(columns=list(_COLUMNS)).to_pylist()[0]


def _array(row: dict[str, Any], name: str) -> Any:
    import numpy as np
    import torch

    shape = [int(v) for v in row[f"{name}_shape"]]
    return torch.from_numpy(np.frombuffer(row[f"{name}_bytes"], dtype=np.float32).reshape(shape).copy())


def reference_latents(row: dict[str, Any], index: int) -> tuple[str, Any, Any]:
    """(media type, normalized visual latents ``[1, 24, T, H, W]`` or None, audio latents ``[2, 32, T]`` or None)."""
    from fastvideo.pipelines.basic.minimax_h3.packing import unpack_audio_tokens, unpatchify_video_tokens

    types = row["reference_media_types"]
    visual_offset = audio_offset = 0
    for i, media in enumerate(types):
        tokens = 0
        if media != "audio":
            frames, height, width = (int(row["reference_num_latent_frames"][i]), int(row["reference_latent_heights"][i]),
                                     int(row["reference_latent_widths"][i]))
            tokens = frames * (height // _PATCH[1]) * (width // _PATCH[2])
        audio_rows = 2 * int(row["reference_num_audio_latents"][i]) if row["reference_has_audio"][i] else 0
        if i == index:
            visual = audio = None
            if tokens:
                rows = _array(row, "reference_video_latent_rows")[visual_offset:visual_offset + tokens]
                visual = unpatchify_video_tokens(rows, frames, height, width, 24, _PATCH)
            if audio_rows:
                rows = _array(row, "reference_audio_latent_rows")[audio_offset:audio_offset + audio_rows]
                audio = unpack_audio_tokens(rows, audio_rows // 2)
            return media, visual, audio
        visual_offset += tokens
        audio_offset += audio_rows
    raise IndexError(f"reference {index} of {len(types)}")


def decode_visual(latents: Any, vae: Any) -> Any:
    """Normalized latents ``[1, 24, T, H, W]`` -> uint8 frames ``[T', H*16, W*16, 3]``."""
    import torch

    with torch.no_grad():
        z = vae.denormalize_latents(latents.to("cuda", torch.float32))
        sample = vae.decode(z, return_dict=False)[0]
        pixels = vae.denormalize_pixels(sample).clamp(0, 1)
    return (pixels[0].permute(1, 2, 3, 0) * 255).round().to(torch.uint8).cpu().numpy()


def decode_audio(latents: Any, audio_vae: Any) -> Any:
    """Normalized audio latents ``[2, 32, T]`` -> waveform ``[2, samples]`` at the audio VAE's rate."""
    import torch

    with torch.no_grad():
        waveform = audio_vae.decode(audio_vae.denormalize_latents(latents.to("cuda", torch.float32)),
                                    return_dict=False)[0]
    return waveform.reshape(2, -1).float().cpu()


def build_references(row_spec: dict[str, Any], vae: Any, audio_vae: Any) -> list[Any]:
    """``MiniMaxH3Reference`` list in declared order, every medium decoded from stored latents."""
    from PIL import Image

    from fastvideo.pipelines.basic.minimax_h3.reference import MiniMaxH3Reference

    rate = int(audio_vae.sampling_rate)
    refs = []
    for ref in row_spec["references"]:
        row = stored_row(ref["parquet"])
        if ref["kind"] == "target_audio":
            audio = _array(row, "audio_latent")
            refs.append(MiniMaxH3Reference(source=decode_audio(audio, audio_vae), media_type="audio", sample_rate=rate))
        elif ref["kind"] == "target_frame":
            first = _array(row, "vae_latent")[None, :, :1]
            refs.append(MiniMaxH3Reference(source=Image.fromarray(decode_visual(first, vae)[0])))
        else:
            media, visual, audio = reference_latents(row, int(ref["index"]))
            if media == "image":
                refs.append(MiniMaxH3Reference(source=Image.fromarray(decode_visual(visual, vae)[0])))
            elif media == "video":
                frames = decode_visual(visual, vae)
                soundtrack = decode_audio(audio, audio_vae) if audio is not None and ref["audio"] else None
                refs.append(
                    MiniMaxH3Reference(source=frames,
                                       media_type="video",
                                       fps=float(FPS),
                                       soundtrack=soundtrack,
                                       sample_rate=rate if soundtrack is not None else None))
            else:
                refs.append(MiniMaxH3Reference(source=decode_audio(audio, audio_vae), media_type="audio",
                                               sample_rate=rate))
    return refs


def check_limits(references: list[Any]) -> None:
    from fastvideo.pipelines.basic.minimax_h3.reference import (MINIMAX_H3_MAX_REFERENCE_AUDIOS,
                                                                MINIMAX_H3_MAX_REFERENCE_IMAGES,
                                                                MINIMAX_H3_MAX_REFERENCE_VIDEOS,
                                                                MINIMAX_H3_MAX_REFERENCES)
    counts: dict[str, int] = defaultdict(int)
    for ref in references:
        counts[ref.media_type] += 1
    if (counts["image"] > MINIMAX_H3_MAX_REFERENCE_IMAGES or counts["video"] > MINIMAX_H3_MAX_REFERENCE_VIDEOS
            or counts["audio"] > MINIMAX_H3_MAX_REFERENCE_AUDIOS or len(references) > MINIMAX_H3_MAX_REFERENCES):
        raise ValueError(f"reference counts {dict(counts)} exceed Flex's limits")
    video_seconds = sum(ref.frames.shape[0] / FPS for ref in references if ref.media_type == "video")
    if video_seconds > MAX_VIDEO_SECONDS + 1e-6:
        raise ValueError(f"combined reference video is {video_seconds:.2f} s (> {MAX_VIDEO_SECONDS} s)")


_COMPONENTS: dict[str, Any] = {}


@functools.cache
def _encoders(model_path: str) -> tuple[Any, Any, Any]:
    """``(encoding module, model_index, fastvideo_args)``; every component loads once and stays resident.

    The released encode functions load their component per call; routing ``_load_component`` through a cache keeps
    the VAE, audio VAE and Qwen3-VL resident across rows without changing their numerics.
    """
    from fastvideo.pipelines.preprocess import preprocess_minimax_h3_ref2va as encoding
    from fastvideo.utils import verify_model_config_and_directory

    encoding._init_single_process_distributed()
    fastvideo_args, _ = encoding._build_fastvideo_args(Path(model_path))
    loader = encoding._load_component

    def cached(name: str, path: Path, index: dict[str, Any], args: Any) -> Any:
        if name not in _COMPONENTS:
            _COMPONENTS[name] = loader(name, path, index, args)
        return _COMPONENTS[name]

    encoding._load_component = cached
    return encoding, verify_model_config_and_directory(model_path), fastvideo_args


def encode_spec(row_spec: dict[str, Any], model_path: str) -> dict[str, Any]:
    """The base v4 row with this spec's references and prompt, conditioning re-encoded at Flex's tier sizing."""
    import pyarrow.parquet as pq

    from fastvideo.pipelines.basic.minimax_h3.reference import prepare_reference
    from fastvideo.pipelines.basic.pika_h3_flex.buckets import PIKA_TIERS, tier_for_canvas

    encoding, model_index, fastvideo_args = _encoders(model_path)
    path = Path(model_path)
    vae = encoding._load_component("vae", path, model_index, fastvideo_args)
    audio_vae = encoding._load_component("audio_vae", path, model_index, fastvideo_args)
    record = pq.ParquetFile(row_spec["base_parquet"]).read().to_pylist()[0]
    tier = PIKA_TIERS[tier_for_canvas(int(record["height"]), int(record["width"]))]
    prepared = [
        prepare_reference(ref,
                          int(record["num_frames"]),
                          encoding.AUDIO_SAMPLE_RATE,
                          reference_short_edge=tier.reference_short_edge,
                          reference_canvas_short_edge=tier.short_edge,
                          reference_canvas_max_pixels=tier.max_pixels)
        for ref in build_references(row_spec, vae, audio_vae)
    ]
    check_limits(prepared)
    prompt = apply_prompt_edits(record["caption"], [tuple(edit) for edit in row_spec["prompt_edits"]])
    visual = encoding.encode_ref_visual_anchor(prepared, path, model_index, fastvideo_args, _PATCH, augment=False)
    audio = encoding.encode_ref_audio_anchor(prepared, path, model_index, fastvideo_args)
    text, tags = encoding.encode_ref2va_conditioning(prompt, prepared, path, model_index, fastvideo_args)
    metadata = encoding._canonical_prepared_references(prepared)
    for name, tensor in (("text_embedding", text), ("text_token_tags", tags), ("reference_video_latent_rows", visual),
                         ("reference_audio_latent_rows", audio)):
        # Every packed tensor is stored as float32 (the readers decode tags as float32 and round).
        array = tensor.detach().float().cpu().contiguous()
        record[f"{name}_bytes"] = array.numpy().tobytes()
        record[f"{name}_shape"] = list(array.shape)
        record[f"{name}_dtype"] = "float32"
    for field, key in (("reference_media_types", "media_type"), ("reference_has_audio", "has_audio"),
                       ("reference_num_latent_frames", "num_latent_frames"),
                       ("reference_latent_heights", "latent_height"), ("reference_latent_widths", "latent_width"),
                       ("reference_num_audio_latents", "num_audio_latents")):
        record[field] = [ref[key] for ref in metadata]
    record["id"] = row_spec["id"]
    record["caption"] = prompt
    return record


# ---------------------------------------------------------------------- drivers
def run_plan(args: argparse.Namespace, overrides: list[str]) -> None:
    from fastvideo.train.models.minimax_h3.omniref_data import load_eval_rows

    v4 = load_v4(Path(args.v4_manifest))
    retier: set[tuple[str, str]] = set()
    for config in args.plan_config:
        retier |= qad_plan_ids(config, overrides)
        for frames in args.plan_max_frames:
            retier |= qad_plan_ids(config, [*overrides, "--models.student.data.max_frames", str(frames)])
    extra = {item.split("=")[0]: int(item.split("=")[1]) for item in args.extra_per_group}
    exclude = {row.source for row in load_eval_rows(args.eval_manifest, V4_CASES, ("480p", "768p"), 99, 0)}
    specs = plan_rows(v4, retier, extra, args.seed, exclude, reference_count)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "plan.json").write_text(json.dumps(specs))
    summary: dict[str, int] = defaultdict(int)
    for row_spec in specs:
        label = row_spec["case"] if row_spec["case"] in NEW_CASES else f"retier/{row_spec['case']}"
        summary[f"{label}/{row_spec['resolution']}"] += 1
    print(f"{len(specs)} rows: {dict(sorted(summary.items()))}")


def run_encode(args: argparse.Namespace) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    from fastvideo.dataset.dataloader.schema import pyarrow_schema_ref2va

    out = Path(args.output_dir)
    specs = json.loads((out / "plan.json").read_text())[args.shard::args.num_shards]
    manifest = out / f"MANIFEST-shard{args.shard:02d}.jsonl"
    done = set()
    if manifest.exists():
        done = {(e["id"], e["resolution"]) for e in map(json.loads, manifest.read_text().splitlines())}
    for index, row_spec in enumerate(specs):
        if (row_spec["id"], row_spec["resolution"]) in done:
            continue
        try:
            record = encode_spec(row_spec, args.model_path)
        except Exception as error:  # one bad source row must not stop the shard; it is reported and skipped
            print(f"SKIP {row_spec['id']} {row_spec['resolution']}: {type(error).__name__}: {error}", flush=True)
            with (out / f"skipped-shard{args.shard:02d}.jsonl").open("a") as handle:
                handle.write(json.dumps({"id": row_spec["id"], "resolution": row_spec["resolution"],
                                         "error": f"{type(error).__name__}: {error}"}) + "\n")
            continue
        bucket = re.search(r"bucket=[^/]+", row_spec["base_parquet"])
        if bucket is None:
            raise ValueError(f"{row_spec['base_parquet']} has no bucket directory")
        digest = hashlib.sha256(f"{row_spec['id']}|{row_spec['resolution']}".encode()).hexdigest()[:24]
        path = out / "parquet" / row_spec["case"] / bucket.group(0) / f"{digest}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist([record], schema=pyarrow_schema_ref2va), path)
        pq.ParquetFile(path).read()  # round trip
        entry = {"case": row_spec["case"], "id": row_spec["id"], "parquet": str(path),
                 "resolution": row_spec["resolution"], "source_case": row_spec["source_case"]}
        with manifest.open("a") as handle:
            handle.write(json.dumps(entry) + "\n")
        print(f"encoded {index + 1}/{len(specs)} {row_spec['case']} {row_spec['resolution']} {row_spec['id']}",
              flush=True)


def run_finalize(args: argparse.Namespace) -> None:
    """Merge the shard manifests; with ``--eval-manifest``, also write the held-out manifest with retiered rows."""
    out = Path(args.output_dir)
    lines = []
    for path in sorted(out.glob("MANIFEST-shard*.jsonl")):
        lines += [line for line in path.read_text().splitlines() if line.strip()]
    (out / "MANIFEST.jsonl").write_text("\n".join(lines) + "\n")
    skipped = sum(len(p.read_text().splitlines()) for p in out.glob("skipped-shard*.jsonl"))
    print(f"MANIFEST.jsonl: {len(lines)} rows ({skipped} skipped)")
    if args.eval_manifest:
        encoded = {(e["id"], e["resolution"]): e["parquet"] for e in map(json.loads, lines)}
        items = json.loads(Path(args.eval_manifest).read_text())
        replaced = 0
        for item in items:
            key = (v4_id(item["case"], item["source"]), item["resolution"])
            if key in encoded:
                item["parquet"] = encoded[key]
                replaced += 1
        (out / "eval-manifest.json").write_text(json.dumps(items, indent=1))
        print(f"eval-manifest.json: {replaced} of {len(items)} held-out rows retiered")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=("plan", "encode", "finalize"))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--v4-manifest", help="OmniRef v4 encoded MANIFEST.jsonl")
    parser.add_argument("--plan-config", nargs="*", default=[], help="QAD configs whose 480p rows to retier")
    parser.add_argument("--plan-max-frames", nargs="*", type=int, default=[], help="extra max_frames plans to cover")
    parser.add_argument("--eval-manifest", help="held-out manifest; its sources are excluded from new cases")
    parser.add_argument("--extra-per-group", nargs="*", default=["480p=80", "768p=20"])
    parser.add_argument("--seed", type=int, default=20261030)
    parser.add_argument("--model-path", help="merged Flex Ref2VA model (encode)")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    args, overrides = parser.parse_known_args()
    if args.stage == "plan":
        run_plan(args, overrides)
    elif args.stage == "encode":
        run_encode(args)
    else:
        run_finalize(args)


if __name__ == "__main__":
    main()
