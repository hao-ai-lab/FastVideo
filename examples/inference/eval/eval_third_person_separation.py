"""Run ``judge.third_person_separation`` (needs ``.[eval-judge]`` + a Gemini key)
over each baseline and print the candidate's win-rate table — from a ``--manifest``
of pairs, or by pairing ``--candidate-dir`` against each ``--reference`` dir by
filename stem.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from fastvideo.eval import create_evaluator

METRIC = "judge.third_person_separation"
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


def _by_stem(directory: Path, exts: set[str]) -> dict[str, Path]:
    """Map filename stem -> path for files with the given extensions."""
    return {p.stem: p for p in sorted(directory.iterdir()) if p.suffix.lower() in exts}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--candidate-dir", type=Path, default=None,
                   help="Directory of candidate clips (directory mode).")
    p.add_argument("--reference", action="append", default=[], metavar="NAME=DIR",
                   help="Baseline directory, repeatable: 'name=dir' or bare 'dir'.")
    p.add_argument("--image-dir", type=Path, default=None,
                   help="Optional first-frame images, matched to clips by stem.")
    p.add_argument("--prompts-json", type=Path, default=None,
                   help="Optional {stem: control-signal text} JSON.")
    p.add_argument("--actions-json", type=Path, default=None,
                   help="Optional {stem: action-label} JSON for the per-action breakdown.")
    p.add_argument("--manifest", type=Path, default=None,
                   help="JSON list of {baseline, video_path, reference_path, ...} rows.")
    p.add_argument("--output", type=Path, default=None)
    args = p.parse_args()

    # Group path-only samples per baseline: {baseline: [sample dict, ...]}.
    by_baseline: dict[str, list[dict]] = defaultdict(list)
    if args.manifest is not None:
        for row in json.loads(args.manifest.read_text()):
            by_baseline[row.get("baseline", "baseline")].append(
                {k: v for k, v in row.items() if k != "baseline"})
    elif args.candidate_dir is not None and args.reference:
        cands = _by_stem(args.candidate_dir, VIDEO_EXTS)
        images = _by_stem(args.image_dir, IMAGE_EXTS) if args.image_dir else {}
        prompts = json.loads(args.prompts_json.read_text()) if args.prompts_json else {}
        actions = json.loads(args.actions_json.read_text()) if args.actions_json else {}
        for spec in args.reference:
            name, sep, ref_dir = spec.partition("=")
            if not sep:
                name, ref_dir = Path(spec).name, spec
            refs = _by_stem(Path(ref_dir), VIDEO_EXTS)
            for stem in sorted(cands.keys() & refs.keys()):
                sample = {"video_path": str(cands[stem]), "reference_path": str(refs[stem])}
                if stem in images:
                    sample["image_path"] = str(images[stem])
                if stem in prompts:
                    sample["text_prompt"] = prompts[stem]
                if stem in actions:
                    sample["action"] = actions[stem]
                by_baseline[name].append(sample)
    else:
        p.error("provide either --manifest, or --candidate-dir with at least one --reference")

    ev = create_evaluator(metrics=[METRIC], device="cpu")
    print("\n| Baseline | Candidate win-rate (excl. ties) | W / L / T | n |")
    print("|---|---|---|---|")
    rows = {}
    for baseline, samples in by_baseline.items():
        res = ev.evaluate(samples=samples).corpus[METRIC]
        rows[baseline] = res
        d = res.details
        if res.score is None:
            print(f"| {baseline} | — | — | 0 |")
        else:
            print(f"| {baseline} | {100 * res.score:.1f}% | {d['wins']}/{d['losses']}/{d['ties']} | {d['n']} |")

    if args.output is not None:
        payload = {b: {"score": r.score, "details": r.details} for b, r in rows.items()}
        args.output.write_text(json.dumps(payload, indent=2))
        print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
