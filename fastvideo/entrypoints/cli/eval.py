"""``fastvideo eval`` CLI: list registered eval metrics and run them
against a set of videos.

This is a thin wrapper around :mod:`fastvideo.eval`. Heavy lifting
(metric loading, GPU handling, batching) lives in
:func:`fastvideo.eval.create_evaluator`.

Examples::

    fastvideo eval list
    fastvideo eval list --group vbench
    fastvideo eval run --videos path/to/videos/*.mp4 \\
        --metrics common.ssim --reference path/to/refs/
    fastvideo eval run --videos clip.mp4 --metrics vbench.aesthetic_quality \\
        --output scores.json
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import cast

from fastvideo.entrypoints.cli.cli_types import CLISubcommand
from fastvideo.logger import init_logger
from fastvideo.utils import FlexibleArgumentParser

logger = init_logger(__name__)


class EvalSubcommand(CLISubcommand):
    """The ``eval`` subcommand — entry point for the eval suite."""

    def __init__(self) -> None:
        self.name = "eval"
        super().__init__()

    def cmd(self, args: argparse.Namespace) -> None:
        action = getattr(args, "eval_action", None)
        if action == "list":
            _cmd_list(args)
        elif action == "run":
            _cmd_run(args)
        elif action == "v2a":
            _cmd_v2a(args)
        else:
            # Re-print help if no action was given.
            self._parser.print_help()  # type: ignore[attr-defined]

    def validate(self, args: argparse.Namespace) -> None:
        action = getattr(args, "eval_action", None)
        if action == "v2a":
            if args.backend != "av-benchmark":
                raise SystemExit(f"Unsupported V2A evaluation backend: {args.backend}")
            return
        if action != "run":
            return
        if not args.manifest and not args.videos and not args.audios:
            raise SystemExit("`fastvideo eval run` requires --manifest, --videos, or --audios")
        if args.manifest and (args.videos or args.audios or args.reference or args.reference_audio):
            raise SystemExit("--manifest cannot be combined with media path arguments")

    def subparser_init(self, subparsers: argparse._SubParsersAction) -> FlexibleArgumentParser:
        eval_parser = subparsers.add_parser(
            "eval",
            help="Run video and V2A evaluation metrics",
            usage="fastvideo eval {list,run,v2a} [...]",
        )
        sub = eval_parser.add_subparsers(dest="eval_action", required=False)

        # `eval list`
        list_p = sub.add_parser("list", help="List registered metrics")
        list_p.add_argument("--group", type=str, default=None, help="Filter to a metric group (e.g. 'vbench').")

        # `eval run`
        run_p = sub.add_parser("run", help="Evaluate videos against one or more metrics")
        run_p.add_argument(
            "--manifest",
            type=str,
            default=None,
            help="JSON/JSONL canonical sample manifest (recommended for benchmarks).",
        )
        run_p.add_argument("--videos",
                           type=str,
                           nargs="+",
                           required=False,
                           help="Path, glob, or directory of generated videos.")
        run_p.add_argument("--audios",
                           type=str,
                           nargs="+",
                           required=False,
                           help="Path, glob, or directory of generated audio files.")
        run_p.add_argument("--reference",
                           type=str,
                           default=None,
                           help="Path / glob / dir of reference videos (for paired metrics).")
        run_p.add_argument(
            "--reference-audio",
            type=str,
            nargs="+",
            default=None,
            help="Path, glob, or directory of reference audio files.",
        )
        run_p.add_argument("--metrics", type=str, default="all", help="Comma-separated metric names, or 'all'.")
        run_p.add_argument("--device", type=str, default="cuda", help="Torch device (e.g. 'cuda', 'cuda:0', 'cpu').")
        run_p.add_argument("--num-gpus", type=int, default=1, help="Number of metric replicas for dataset evaluation.")
        run_p.add_argument("--compile", action="store_true", help="Compile supported metric models.")
        run_p.add_argument(
            "--skip-missing-deps",
            action="store_true",
            help="Skip explicitly requested metrics whose optional dependencies are missing.",
        )
        run_p.add_argument(
            "--extract-audio",
            nargs="?",
            const=True,
            default=False,
            help="Extract missing audio tracks; optionally provide a persistent cache directory.",
        )
        run_p.add_argument("--extract-workers", type=int, default=4)
        run_p.add_argument(
            "--text-prompt",
            type=str,
            nargs="*",
            default=None,
            help="Prompt(s) for text-conditioned metrics. One per video.",
        )
        run_p.add_argument("--fps", type=float, default=None, help="Frame-rate annotation passed to fps-aware metrics.")
        run_p.add_argument("--output",
                           type=str,
                           default=None,
                           help="Write results as JSON to this path (default: stdout).")
        run_p.add_argument(
            "--output-format",
            choices=("legacy", "full"),
            default="legacy",
            help="'full' includes corpus metrics and sample identities.",
        )

        # `eval v2a`: isolated corpus-level backends for video-to-audio.
        v2a_p = sub.add_parser("v2a", help="Evaluate V2A audio with a corpus benchmark backend")
        v2a_p.add_argument("--backend", choices=("av-benchmark", ), default="av-benchmark")
        v2a_p.add_argument("--audio-dir", required=True, help="Directory containing generated WAV/FLAC files.")
        v2a_p.add_argument("--gt-cache", required=True, help="Official ground-truth feature cache directory.")
        v2a_p.add_argument(
            "--prediction-cache",
            default=None,
            help="Generated-audio feature cache (default: <audio-dir>/../av_benchmark_cache).",
        )
        v2a_p.add_argument(
            "--output",
            default=None,
            help="Result JSON (default: <audio-dir>/../av_benchmark_results.json).",
        )
        v2a_p.add_argument(
            "--python-executable",
            default=None,
            help="Python from an isolated av-benchmark environment.",
        )
        v2a_p.add_argument("--device", default="cuda")
        v2a_p.add_argument("--batch-size", type=int, default=32)
        v2a_p.add_argument("--num-workers", type=int, default=0)
        v2a_p.add_argument("--audio-length", type=float, default=8.0)
        v2a_p.add_argument("--recompute", action="store_true", help="Re-extract the prediction cache.")
        v2a_p.add_argument("--skip-video-related", action="store_true", help="Skip ImageBind and DeSync.")
        v2a_p.add_argument("--skip-clap", action="store_true", help="Skip LAION/MS CLAP extraction and scores.")
        v2a_p.add_argument(
            "--align-prediction-keys",
            action="store_true",
            help="Align uniquely underscore-prefixed prediction ids to official GT cache ids.",
        )

        # Stash the parser so cmd() can re-print help on no-action.
        self._parser = eval_parser  # type: ignore[attr-defined]
        return cast(FlexibleArgumentParser, eval_parser)


def _cmd_list(args: argparse.Namespace) -> None:
    from fastvideo.eval import list_metrics

    names = list_metrics()
    if args.group:
        prefix = args.group.rstrip(".") + "."
        names = [n for n in names if n == args.group or n.startswith(prefix)]
    if not names:
        print(f"(no metrics matched group {args.group!r})")
        return
    for name in names:
        print(name)


def _cmd_run(args: argparse.Namespace) -> None:
    from fastvideo.eval import create_evaluator, samples_from, samples_from_manifest

    if args.manifest:
        samples = samples_from_manifest(
            args.manifest,
            extract_audio=args.extract_audio,
            extract_workers=args.extract_workers,
        )
    else:
        video_paths = _expand_paths(args.videos, _VIDEO_EXTS) if args.videos else None
        audio_paths = _expand_paths(args.audios, _AUDIO_EXTS) if args.audios else None
        if video_paths is not None and not video_paths:
            raise SystemExit(f"No videos matched: {args.videos}")
        if audio_paths is not None and not audio_paths:
            raise SystemExit(f"No audio files matched: {args.audios}")
        generated_count = len(video_paths if video_paths is not None else audio_paths or [])
        ref_paths = (_fill_reference_paths(_expand_paths([args.reference], _VIDEO_EXTS), generated_count)
                     if args.reference else None)
        ref_audio_paths = (_fill_reference_paths(_expand_paths(args.reference_audio, _AUDIO_EXTS), generated_count)
                           if args.reference_audio else None)
        text_prompts = args.text_prompt
        if text_prompts is not None and len(text_prompts) == 1:
            text_prompts = text_prompts * generated_count
        samples = samples_from(
            video=video_paths,
            audio=audio_paths,
            reference=ref_paths,
            reference_audio=ref_audio_paths,
            text_prompts=text_prompts,
            fps=args.fps,
            extract_audio=args.extract_audio,
            extract_workers=args.extract_workers,
        )

    metrics_arg: list[str] | str = ("all" if args.metrics == "all" else
                                    [m.strip() for m in args.metrics.split(",") if m.strip()])

    evaluator = create_evaluator(
        metrics=metrics_arg,
        device=args.device,
        num_gpus=max(1, args.num_gpus),
        compile=args.compile,
        skip_missing_deps=args.skip_missing_deps,
    )
    logger.info("Evaluating %d samples on %d GPU worker(s)", len(samples), evaluator.num_gpus)
    results = evaluator.evaluate(samples=samples)
    corpus = _serialize_results(results.corpus)
    output_value: dict | list
    if args.output_format == "full":
        sample_results = [{
            **_sample_identity(sample, index),
            "scores": _serialize_results(result),
        } for index, (sample, result) in enumerate(zip(samples, results, strict=True))]
        output_value = {
            "samples": sample_results,
            "corpus": corpus,
            "num_samples": len(samples),
        }
    else:
        if corpus:
            logger.warning("Corpus results are omitted in legacy output format; pass --output-format full to keep them")
        output_value = [{
            **_legacy_sample_identity(sample),
            "scores": _serialize_results(result),
        } for sample, result in zip(samples, results, strict=True)]

    payload = json.dumps(output_value, indent=2, default=_jsonable)
    if args.output:
        Path(args.output).write_text(payload)
        logger.info("Wrote results to %s", args.output)
    else:
        print(payload)


def _cmd_v2a(args: argparse.Namespace) -> None:
    """Run an isolated corpus-level backend for generated V2A audio."""
    from fastvideo.eval.v2a import run_av_benchmark

    output = run_av_benchmark(
        audio_dir=args.audio_dir,
        gt_cache=args.gt_cache,
        prediction_cache=args.prediction_cache,
        output=args.output,
        python_executable=args.python_executable,
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        audio_length=args.audio_length,
        recompute=args.recompute,
        skip_video_related=args.skip_video_related,
        skip_clap=args.skip_clap,
        align_prediction_keys=args.align_prediction_keys,
    )
    logger.info("Official av-benchmark results: %s", output)


_VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".gif", ".webm")
_AUDIO_EXTS = (".wav", ".mp3", ".flac", ".m4a", ".ogg", ".opus")


def _expand_paths(patterns: list[str], extensions: tuple[str, ...]) -> list[str]:
    out: list[str] = []
    for pat in patterns:
        p = Path(pat)
        if p.is_dir():
            out.extend(sorted(str(f) for f in p.iterdir() if f.suffix.lower() in extensions))
        elif any(c in pat for c in "*?["):
            out.extend(sorted(glob.glob(pat)))
        else:
            out.append(pat)
    # de-dup, preserve order
    seen: set[str] = set()
    deduped: list[str] = []
    for x in out:
        if x not in seen:
            seen.add(x)
            deduped.append(x)
    return deduped


def _fill_reference_paths(paths: list[str], count: int) -> list[str]:
    if not paths:
        return paths
    if len(paths) >= count:
        return paths
    return paths + [paths[0]] * (count - len(paths))


def _legacy_sample_identity(sample: dict) -> dict:
    """Preserve the original ``fastvideo eval run`` output schema."""
    from fastvideo.eval.types import Video

    video = sample.get("video")
    if isinstance(video, Video) and video.source is not None:
        return {"video": str(video.source)}
    audio = sample.get("audio")
    if isinstance(audio, str | Path):
        return {"audio": str(audio)}
    return {}


def _sample_identity(sample: dict, index: int) -> dict:
    from fastvideo.eval.types import Video

    identity: dict = {"index": index}
    if sample.get("id") is not None:
        identity["id"] = sample["id"]
    for key in ("video", "audio", "reference", "reference_audio", "text_prompt"):
        value = sample.get(key)
        if isinstance(value, Video):
            if value.source is not None:
                identity[key] = str(value.source)
        elif isinstance(value, str | Path):
            identity[key] = str(value)
    return identity


def _serialize_results(results) -> dict | list:
    """Turn evaluator output (dict or list-of-dicts) into JSON-friendly form."""
    if isinstance(results, list):
        return [_serialize_results(r) for r in results]
    if isinstance(results, dict):
        return {k: _serialize_metric_result(v) for k, v in results.items()}
    return _serialize_metric_result(results)


def _serialize_metric_result(mr) -> dict:
    return {
        "name": getattr(mr, "name", None),
        "score": getattr(mr, "score", None),
        "details": getattr(mr, "details", None),
    }


def _jsonable(obj):
    """``json.dumps(default=...)`` coercer for metric outputs.

    Metrics frequently land numpy scalars / arrays, torch tensors, and
    pathlib paths inside ``MetricResult.details`` (e.g. ``optical_flow``
    populates ``per_frame_metrics`` with numpy floats). The stdlib JSON
    encoder rejects all of those by default — this callback walks the
    leaves and coerces them to native Python types.
    """
    import numpy as np
    import torch

    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def cmd_init() -> list[CLISubcommand]:
    return [EvalSubcommand()]
