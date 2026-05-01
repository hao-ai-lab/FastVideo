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
        else:
            # Re-print help if no action was given.
            self._parser.print_help()  # type: ignore[attr-defined]

    def validate(self, args: argparse.Namespace) -> None:
        action = getattr(args, "eval_action", None)
        if action == "run" and not args.videos:
            raise SystemExit("`fastvideo eval run` requires --videos")

    def subparser_init(self, subparsers: argparse._SubParsersAction) -> FlexibleArgumentParser:
        eval_parser = subparsers.add_parser(
            "eval",
            help="Run video-gen evaluation metrics",
            usage="fastvideo eval {list,run} [...]",
        )
        sub = eval_parser.add_subparsers(dest="eval_action", required=False)

        # `eval list`
        list_p = sub.add_parser("list", help="List registered metrics")
        list_p.add_argument("--group", type=str, default=None,
                            help="Filter to a metric group (e.g. 'vbench').")

        # `eval run`
        run_p = sub.add_parser("run", help="Evaluate videos against one or more metrics")
        run_p.add_argument("--videos", type=str, nargs="+", required=False,
                           help="Path, glob, or directory of generated videos.")
        run_p.add_argument("--reference", type=str, default=None,
                           help="Path / glob / dir of reference videos (for paired metrics).")
        run_p.add_argument("--metrics", type=str, default="all",
                           help="Comma-separated metric names, or 'all'.")
        run_p.add_argument("--device", type=str, default="cuda",
                           help="Torch device (e.g. 'cuda', 'cuda:0', 'cpu').")
        run_p.add_argument("--text-prompt", type=str, nargs="*", default=None,
                           help="Prompt(s) for text-conditioned metrics. One per video.")
        run_p.add_argument("--fps", type=float, default=None,
                           help="Frame-rate annotation passed to fps-aware metrics.")
        run_p.add_argument("--output", type=str, default=None,
                           help="Write results as JSON to this path (default: stdout).")

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
    from fastvideo.eval import create_evaluator
    from fastvideo.eval.io import load_video

    video_paths = _expand_paths(args.videos)
    if not video_paths:
        raise SystemExit(f"No videos matched: {args.videos}")
    ref_paths = _expand_paths([args.reference]) if args.reference else None

    metrics_arg: list[str] | str
    if args.metrics == "all":
        metrics_arg = "all"
    else:
        metrics_arg = [m.strip() for m in args.metrics.split(",") if m.strip()]

    evaluator = create_evaluator(metrics=metrics_arg, device=args.device)

    all_results: list[dict] = []
    for i, vp in enumerate(video_paths):
        logger.info("Evaluating %s (%d/%d)", vp, i + 1, len(video_paths))
        kwargs: dict = {"video": load_video(vp)}
        if ref_paths is not None:
            ref = ref_paths[i] if i < len(ref_paths) else ref_paths[0]
            kwargs["reference"] = load_video(ref)
        if args.text_prompt is not None:
            prompt = (args.text_prompt[i] if i < len(args.text_prompt)
                      else args.text_prompt[0])
            kwargs["text_prompt"] = [prompt]
        if args.fps is not None:
            kwargs["fps"] = args.fps

        results = evaluator.evaluate(**kwargs)
        all_results.append({
            "video": str(vp),
            "scores": _serialize_results(results),
        })

    payload = json.dumps(all_results, indent=2)
    if args.output:
        Path(args.output).write_text(payload)
        logger.info("Wrote results to %s", args.output)
    else:
        print(payload)


def _expand_paths(patterns: list[str]) -> list[str]:
    out: list[str] = []
    for pat in patterns:
        p = Path(pat)
        if p.is_dir():
            for ext in (".mp4", ".avi", ".mov", ".mkv", ".gif"):
                out.extend(sorted(str(f) for f in p.iterdir() if f.suffix.lower() == ext))
        elif any(c in pat for c in "*?["):
            out.extend(sorted(glob.glob(pat)))
        else:
            out.append(pat)
    # de-dup, preserve order
    seen: set[str] = set()
    return [x for x in out if not (x in seen or seen.add(x))]


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


def cmd_init() -> list[CLISubcommand]:
    return [EvalSubcommand()]
