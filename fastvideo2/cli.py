"""CLI: describe / generate / verify — the three agent-facing verbs.

  python -m fastvideo2 describe wan2.1-t2v-1.3b
  python -m fastvideo2 generate wan2.1-t2v-1.3b --prompt "a cat surfing" --out cat.mp4
  python -m fastvideo2 verify   wan2.1-t2v-1.3b --tier 2 [--bless]

``describe`` prints the card as JSON plus its digest — machine-readable
capability discovery. ``verify`` appends typed results to the evidence ledger
and exits non-zero on any failed gate.
"""
from __future__ import annotations

import argparse
import json
import sys


def _describe(args) -> int:
    from fastvideo2.registry import resolve
    card, _ = resolve(args.model)
    print(card.to_json())
    print(f'// digest: {card.digest()}', file=sys.stderr)
    return 0


def _generate(args) -> int:
    import fastvideo2
    kwargs = {k: getattr(args, k) for k in
              ("seed", "num_steps", "guidance_scale", "height", "width", "num_frames", "shift")
              if getattr(args, k) is not None}
    model = fastvideo2.load(args.model, root=args.root, device=args.device)
    result = model.generate(args.prompt, **kwargs)
    result.save(args.out, fps=args.fps)
    steps = [t for t in result.trace if "/denoise." in t["label"]]
    print(f"video {result.video.shape} -> {args.out}")
    print(f"total {result.seconds:.1f}s; denoise steps {len(steps)}, "
          f"mean {sum(t['seconds'] for t in steps) / max(len(steps), 1):.2f}s/step")
    return 0


def _verify(args) -> int:
    from fastvideo2.verify import LEDGER, verify
    results = verify(args.model, tier=args.tier, root=args.root, device=args.device,
                     bless=args.bless, anchor=args.anchor)
    for r in results:
        mark = {"pass": "PASS ", "blessed": "BLESS", "fail": "FAIL "}[r.status]
        print(f"  {mark} {r.gate:14s} {r.detail or json.dumps(r.metrics)[:120]}")
    print(f"ledger: {LEDGER}")
    return 0 if all(r.ok for r in results) else 1


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="fastvideo2", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("describe", help="print a card as JSON + digest")
    d.add_argument("model")
    d.set_defaults(fn=_describe)

    g = sub.add_parser("generate", help="run one request, save an mp4")
    g.add_argument("model")
    g.add_argument("--prompt", required=True)
    g.add_argument("--out", default="out.mp4")
    g.add_argument("--root", default=None, help="local checkpoint dir (else HF cache)")
    g.add_argument("--device", default=None)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--num-steps", dest="num_steps", type=int, default=None)
    g.add_argument("--guidance-scale", dest="guidance_scale", type=float, default=None)
    g.add_argument("--height", type=int, default=None)
    g.add_argument("--width", type=int, default=None)
    g.add_argument("--num-frames", dest="num_frames", type=int, default=None)
    g.add_argument("--shift", type=float, default=None)
    g.add_argument("--fps", type=int, default=16)
    g.set_defaults(fn=_generate)

    v = sub.add_parser("verify", help="run tiered gates; append to the evidence ledger")
    v.add_argument("model")
    v.add_argument("--tier", type=int, default=3, choices=(0, 1, 2, 3))
    v.add_argument("--root", default=None)
    v.add_argument("--device", default=None)
    v.add_argument("--bless", action="store_true",
                   help="write the T1 fingerprint baseline for this environment")
    v.add_argument("--anchor", action="store_true",
                   help="also certify components against the official Wan2.1 goldens")
    v.set_defaults(fn=_verify)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
