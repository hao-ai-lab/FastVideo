# SPDX-License-Identifier: Apache-2.0
"""Pika H3 Flex Ref2VA rows from the OmniRef v4 sources: Flex reference sizing plus the inputs v4 lacks.

The OmniRef v4 encoded rows (the FastH3 OmniRef QAD data) size every image reference at a 2048 short edge.
Pika H3 Flex uses 1024 below its 768p tier and 2048 at 768p, and it accepts input mixes v4 never encoded. This
builds packed Ref2VA rows (``pyarrow_schema_ref2va``) for the Flex QAD from the v4 raw manifests and media:

``retier``             the v4 row's own references and prompt, re-encoded with Flex's tier sizing; same case name
                       and id as the v4 row, so a manifest listed before v4 replaces it (``load_manifest_groups``
                       keeps the first entry per id and resolution).
``audio_reference``    first-frame rows plus the target's soundtrack as a standalone audio reference.
``mixed_image_video``  continuation rows plus the target's first frame as an image ahead of the preceding video.
``many_images``        storyboard rows extended with frames of the target to 6-9 image references.
``multi_video``        continuation rows plus a second, silent preceding clip of the same asset.

Only references and the Qwen3-VL conditioning are encoded (with the released Ref2VA encode path of
``preprocess_minimax_h3_ref2va``; the VAE, audio VAE and text encoder stay resident). Target video and audio latents
are copied from the v4 row of the same target: QAD regresses on the teacher, so they only carry the geometry.

    python scripts/distill/pika_h3_flex_qad/build_flex_ref2va_rows.py plan --v4-root <v4 run dir> \\
        --plan-config <QAD yaml>... --extra-per-group 480p=80 768p=20 --output-dir <out>
    CUDA_VISIBLE_DEVICES=<g> python .../build_flex_ref2va_rows.py encode --model-path <merged Flex> \\
        --output-dir <out> --shard <g> --num-shards <n>
    python .../build_flex_ref2va_rows.py finalize --output-dir <out>
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
FPS = 24.0
_ASSET = re.compile(r"(asset-[0-9a-f]{32})")


# ---------------------------------------------------------------------- planning (CPU)
def load_v4(v4_root: Path) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[tuple[str, str], dict[str, Any]]]:
    """``(raw, encoded)`` v4 entries keyed by (id, resolution)."""
    raw: dict[tuple[str, str], dict[str, Any]] = {}
    for path in sorted((v4_root / "manifests").glob("shard-*.jsonl")):
        for line in path.read_text().splitlines():
            entry = json.loads(line)
            raw[(entry["id"], entry["resolution"])] = entry
    encoded: dict[tuple[str, str], dict[str, Any]] = {}
    for line in (v4_root / "encoded" / "full" / "MANIFEST.jsonl").read_text().splitlines():
        entry = json.loads(line)
        encoded[(entry["id"], entry["resolution"])] = entry
    return raw, encoded


def qad_plan_ids(config: str, overrides: list[str]) -> set[tuple[str, str]]:
    """(id, resolution) of every training and held-out row the QAD config would read from the v4 manifest."""
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
    ids = set()
    for row in [*plan, *eval_rows]:
        entry_id = f"{row.case}:{row.source}" if row.case in IMAGE_CASES else row.source
        ids.add((entry_id, row.resolution))
    return ids


def _keyframe_source(path: str) -> str:
    """The 768p copy of a keyframe (same frame, more pixels) when it exists; Flex upsamples 480p sources."""
    hi = path.replace("/keyframes/480p/", "/keyframes/768p/")
    return hi if hi != path and Path(hi).is_file() else path


def target_path(entry: dict[str, Any], v4_assets: Path) -> str:
    clip = entry["target_clip_id"]
    if clip.endswith("__target"):
        clip = clip[:-len("__target")]
        asset = _ASSET.search(clip).group(1)  # type: ignore[union-attr]
        return str(v4_assets / "continuation_clips" / entry["resolution"] / asset / clip / "target.mp4")
    return str(v4_assets / "target_clips" / entry["resolution"] / f"{clip}.mp4")


def _subject(prompt: str, added: str, *, before: bool = False) -> str:
    """Add one definition to the prompt's ``subject_definitions:`` paragraph."""
    head, sep, rest = prompt.partition("subject_definitions:")
    if not sep:
        raise ValueError("prompt has no subject_definitions section")
    body, gap, tail = rest.partition("\n\n")
    body = f" {added}{body}" if before else f"{body.rstrip()} {added}"
    return f"{head}{sep}{body}{gap}{tail}"


def _ref(kind: str, **fields: Any) -> dict[str, Any]:
    return {"kind": kind, **fields}


def v4_references(entry: dict[str, Any], v4_assets: Path) -> list[dict[str, Any]]:
    """The v4 entry's references as builder specs (image crops and video-frame indices kept)."""
    refs = []
    for ref in entry["references"]:
        if ref["kind"] == "image":
            refs.append(_ref("image", path=_keyframe_source(ref["path"]), crop=[ref.get("crop_left", 0),
                                                                              ref.get("crop_right", 0)]))
        elif ref["kind"] == "video_frame":
            refs.append(_ref("frame", path=ref["path"], frame=int(ref["frame_index"]),
                             crop=[ref.get("crop_left", 0), ref.get("crop_right", 0)]))
        elif ref["kind"] == "video":
            refs.append(_ref("video", path=ref["path"], audio=bool(ref.get("use_embedded_audio", False))))
        else:
            raise ValueError(f"{entry['id']}: unsupported v4 reference kind {ref['kind']!r}")
    return refs


def variant(case: str, entry: dict[str, Any], v4_assets: Path, rng: random.Random,
            siblings: list[dict[str, Any]]) -> dict[str, Any] | None:
    """One output row spec for ``case`` built from v4 ``entry`` (None when the entry cannot carry it)."""
    refs = v4_references(entry, v4_assets)
    prompt = entry["prompt"]
    target = target_path(entry, v4_assets)
    frames = int(entry["target"]["num_frames"])
    if case == "retier":
        pass
    elif case == "audio_reference":
        refs = [*refs, _ref("audio", path=target)]
        prompt = _subject(prompt, "<Audio 1> is the soundtrack of the target video.")
    elif case == "mixed_image_video":
        refs = [_ref("frame", path=target, frame=0, crop=[0, 0]), *refs]
        prompt = _subject(prompt, "<Picture 1> is the exact first frame of the target video.", before=True)
    elif case == "many_images":
        have = len(refs)
        want = rng.randint(max(6, have + 1), MAX_IMAGES)
        picks = sorted(rng.sample(range(1, frames - 1), want - have))
        refs = [*refs, *(_ref("frame", path=target, frame=f, crop=[0, 0]) for f in picks)]
        for index, frame in enumerate(picks, start=have + 1):
            prompt = _subject(prompt, f"<Picture {index}> is a frame of the target video at {frame / FPS:.2f} s.")
    elif case == "multi_video":
        own = refs[0]["path"]
        others = [s for s in siblings if s["references"][0]["path"] != own]
        if not others:
            return None
        other = rng.choice(others)
        refs = [*refs, _ref("video", path=other["references"][0]["path"], audio=False)]
        prompt = _subject(prompt, "<Video 2> is an earlier shot from the same scene, without sound.")
    else:
        raise ValueError(f"unknown case {case!r}")
    source_id = entry["id"].split(":", 1)[-1]
    out_case = entry["case"] if case == "retier" else case
    return {
        "id": entry["id"] if case == "retier" else f"{case}:{source_id}",
        "case": out_case,
        "source_case": entry["case"],
        "v4_id": entry["id"],
        "resolution": entry["resolution"],
        "prompt": prompt,
        "references": refs,
        "target": entry["target"],
    }


def plan_rows(v4_root: Path, v4_assets: Path, retier_ids: set[tuple[str, str]], extra: dict[str, int],
              seed: int, exclude_sources: set[str]) -> list[dict[str, Any]]:
    raw, encoded = load_v4(v4_root)
    specs = [variant("retier", raw[key], v4_assets, random.Random(0), []) for key in sorted(retier_ids)
             if key in raw and key[1] == "480p" and raw[key]["case"] in IMAGE_CASES]
    by_asset: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for (entry_id, res), entry in raw.items():
        if entry["case"] in CONTINUE_CASES:
            by_asset[(_ASSET.search(entry_id).group(1), res)].append(entry)  # type: ignore[union-attr]
    sources = {"audio_reference": ("first_frame", ), "mixed_image_video": CONTINUE_CASES,
               "many_images": ("storyboard", ), "multi_video": CONTINUE_CASES}
    for case in NEW_CASES:
        for res, count in sorted(extra.items()):
            pool = sorted((e for (i, r), e in raw.items()
                           if r == res and e["case"] in sources[case] and (i, r) in encoded
                           and i.split(":", 1)[-1] not in exclude_sources), key=lambda e: e["id"])
            rng = random.Random(f"{seed}:{case}:{res}")
            rng.shuffle(pool)
            made = 0
            for entry in pool:
                if made >= count:
                    break
                asset = _ASSET.search(entry["id"]).group(1)  # type: ignore[union-attr]
                spec = variant(case, entry, v4_assets, rng, by_asset.get((asset, res), []))
                if spec is not None:
                    specs.append(spec)
                    made += 1
    for spec in specs:
        spec["v4_parquet"] = encoded[(spec["v4_id"], spec["resolution"])]["parquet"]
    return specs


# ---------------------------------------------------------------------- encoding (GPU)
def _crop(image: Any, crop: list[int]) -> Any:
    left, right = int(crop[0]), int(crop[1])
    return image if not (left or right) else image.crop((left, 0, image.width - right, image.height))


@functools.lru_cache(maxsize=1)
def _decoded(path: str) -> Any:
    from fastvideo.pipelines.basic.minimax_h3.reference import decode_reference_video
    return decode_reference_video(path)


def _frame(path: str, index: int) -> Any:
    from PIL import Image

    frames = _decoded(path)[0]
    return Image.fromarray(frames[min(index, len(frames) - 1)])


def build_references(spec: dict[str, Any]) -> list[Any]:
    """``MiniMaxH3Reference`` list in declared order; a silent video drops its container soundtrack."""
    from PIL import Image

    from fastvideo.pipelines.basic.minimax_h3.reference import MiniMaxH3Reference

    refs = []
    for ref in spec["references"]:
        if ref["kind"] == "image":
            refs.append(MiniMaxH3Reference(source=_crop(Image.open(ref["path"]).convert("RGB"), ref["crop"])))
        elif ref["kind"] == "frame":
            refs.append(MiniMaxH3Reference(source=_crop(_frame(ref["path"], ref["frame"]), ref["crop"])))
        elif ref["kind"] == "video" and ref["audio"]:
            refs.append(MiniMaxH3Reference(source=ref["path"], media_type="video"))
        elif ref["kind"] == "video":
            frames, fps, _ = _decoded(ref["path"])
            refs.append(MiniMaxH3Reference(source=frames, media_type="video", fps=fps))
        elif ref["kind"] == "audio":
            _, _, soundtrack = _decoded(ref["path"])
            if soundtrack is None:
                raise ValueError(f"{ref['path']} has no soundtrack")
            refs.append(MiniMaxH3Reference(source=soundtrack[0], media_type="audio", sample_rate=soundtrack[1]))
        else:
            raise ValueError(f"unknown reference kind {ref['kind']!r}")
    return refs


def check_limits(references: list[Any]) -> None:
    from fastvideo.pipelines.basic.minimax_h3.reference import (MINIMAX_H3_MAX_REFERENCE_AUDIOS,
                                                                MINIMAX_H3_MAX_REFERENCE_IMAGES,
                                                                MINIMAX_H3_MAX_REFERENCE_VIDEOS,
                                                                MINIMAX_H3_MAX_REFERENCES)
    counts = defaultdict(int)
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


def encode_spec(spec: dict[str, Any], model_path: str) -> dict[str, Any]:
    """The v4 row of the same target with this spec's references, prompt and Flex-sized conditioning."""
    import pyarrow.parquet as pq

    from fastvideo.pipelines.basic.minimax_h3.reference import prepare_reference
    from fastvideo.pipelines.basic.pika_h3_flex.buckets import PIKA_TIERS, tier_for_canvas

    encoding, model_index, fastvideo_args = _encoders(model_path)
    target = spec["target"]
    tier = PIKA_TIERS[tier_for_canvas(int(target["height"]), int(target["width"]))]
    prepared = [
        prepare_reference(ref,
                          int(target["num_frames"]),
                          encoding.AUDIO_SAMPLE_RATE,
                          reference_short_edge=tier.reference_short_edge,
                          reference_canvas_short_edge=tier.short_edge,
                          reference_canvas_max_pixels=tier.max_pixels) for ref in build_references(spec)
    ]
    check_limits(prepared)
    path = Path(model_path)
    visual = encoding.encode_ref_visual_anchor(prepared, path, model_index, fastvideo_args, (1, 2, 2), augment=False)
    audio = encoding.encode_ref_audio_anchor(prepared, path, model_index, fastvideo_args)
    text, tags = encoding.encode_ref2va_conditioning(spec["prompt"], prepared, path, model_index, fastvideo_args)
    metadata = encoding._canonical_prepared_references(prepared)
    record = pq.ParquetFile(spec["v4_parquet"]).read().to_pylist()[0]
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
    record["id"] = spec["id"]
    record["caption"] = spec["prompt"]
    return record


def run_encode(args: argparse.Namespace) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    from fastvideo.dataset.dataloader.schema import pyarrow_schema_ref2va

    out = Path(args.output_dir)
    specs = json.loads((out / "plan.json").read_text())[args.shard::args.num_shards]
    manifest = out / f"MANIFEST-shard{args.shard:02d}.jsonl"
    done = {json.loads(line)["id"] + "|" + json.loads(line)["resolution"]
            for line in manifest.read_text().splitlines()} if manifest.exists() else set()
    for index, spec in enumerate(specs):
        if f"{spec['id']}|{spec['resolution']}" in done:
            continue
        try:
            record = encode_spec(spec, args.model_path)
        except Exception as error:  # a bad source clip must not stop the shard; it is reported and skipped
            print(f"SKIP {spec['id']} {spec['resolution']}: {type(error).__name__}: {error}", flush=True)
            with (out / f"skipped-shard{args.shard:02d}.jsonl").open("a") as handle:
                handle.write(json.dumps({"id": spec["id"], "resolution": spec["resolution"], "error": str(error)}) +
                             "\n")
            continue
        bucket = re.search(r"bucket=[^/]+", spec["v4_parquet"]).group(0)  # type: ignore[union-attr]
        digest = hashlib.sha256(f"{spec['id']}|{spec['resolution']}".encode()).hexdigest()[:24]
        path = out / "parquet" / spec["case"] / bucket / f"{digest}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist([record], schema=pyarrow_schema_ref2va), path)
        pq.ParquetFile(path).read()  # round trip
        entry = {"case": spec["case"], "id": spec["id"], "parquet": str(path), "resolution": spec["resolution"],
                 "source_case": spec["source_case"]}
        with manifest.open("a") as handle:
            handle.write(json.dumps(entry) + "\n")
        print(f"encoded {index + 1}/{len(specs)} {spec['case']} {spec['resolution']} {spec['id']}", flush=True)


def run_plan(args: argparse.Namespace, overrides: list[str]) -> None:
    from fastvideo.train.models.minimax_h3.omniref_data import load_eval_rows

    retier: set[tuple[str, str]] = set()
    for config in args.plan_config:
        retier |= qad_plan_ids(config, overrides)
        for frames in args.plan_max_frames:
            retier |= qad_plan_ids(config, [*overrides, "--models.student.data.max_frames", str(frames)])
    extra = {item.split("=")[0]: int(item.split("=")[1]) for item in args.extra_per_group}
    exclude = {row.source for row in load_eval_rows(args.eval_manifest, V4_CASES, ("480p", "768p"), 99, 0)}
    specs = plan_rows(Path(args.v4_root), Path(args.v4_assets), retier, extra, args.seed, exclude)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "plan.json").write_text(json.dumps(specs))
    counts: dict[str, int] = defaultdict(int)
    for spec in specs:
        counts[f"{spec['case'] if spec['case'] in NEW_CASES else 'retier/' + spec['case']}/{spec['resolution']}"] += 1
    print(f"{len(specs)} rows: {dict(sorted(counts.items()))}")


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
            key = (f"{item['case']}:{item['source']}" if item["case"] in IMAGE_CASES else item["source"],
                   item["resolution"])
            if key in encoded:
                item["parquet"] = encoded[key]
                replaced += 1
        (out / "eval-manifest.json").write_text(json.dumps(items, indent=1))
        print(f"eval-manifest.json: {replaced} of {len(items)} held-out rows retiered")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=("plan", "encode", "finalize"))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--v4-root", help="OmniRef v4 run dir (manifests/, encoded/full/MANIFEST.jsonl)")
    parser.add_argument("--v4-assets", help="OmniRef v4 media root (target_clips/, continuation_clips/, keyframes/)")
    parser.add_argument("--plan-config", nargs="*", default=[], help="QAD configs whose 480p image rows to retier")
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
