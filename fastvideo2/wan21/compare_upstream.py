"""Three-way Wan2.1 numerics comparison against the official goldens.

Runs any subset of implementations over the captured official probes and
writes one markdown table (rel L2 vs official per component) plus ledger
records. Implementations:

  fastvideo2      this repo's diffusers-backed components (our loading path)
  fastvideo_main  fastvideo main-branch native components (requires the main
                  environment, e.g. /mnt/FastVideo/.venv on the cluster)

    python -m fastvideo2.wan21.compare_upstream \\
        --impls fastvideo2,fastvideo_main --root <diffusers ckpt dir> \\
        --report fastvideo2/evidence/wan21_numerics_report.md
"""
from __future__ import annotations

import argparse
import json


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--impls", default="fastvideo2")
    p.add_argument("--root", default=None, help="diffusers-layout checkpoint dir (else HF cache)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--goldens-set", default=None)
    p.add_argument("--report", default=None, help="write the markdown table here")
    args = p.parse_args()

    from fastvideo2.loading import resolve_weights
    from fastvideo2.verify import GateResult, append_ledger, env_fingerprint
    from fastvideo2.wan21 import anchor as A
    from fastvideo2.wan21 import goldens as G
    from fastvideo2.wan21.card import WAN21_T2V_1_3B as CARD

    gdir = G.golden_dir(args.goldens_set) if args.goldens_set else G.golden_dir()
    manifest = G.load_manifest(gdir)
    root = resolve_weights(CARD, args.root)
    env = env_fingerprint()

    adapters = {"fastvideo2": A.fastvideo2_adapter, "fastvideo_main": A.fastvideo_main_adapter}
    all_records: dict[str, list[dict]] = {}
    ledger: list[GateResult] = []
    for impl in [s.strip() for s in args.impls.split(",") if s.strip()]:
        print(f"== {impl} ==")
        records = A.run_anchor(adapters[impl](root, args.device), gdir)
        all_records[impl] = records
        for rec in records:
            r = dict(rec)
            name, status = r.pop("name"), r.pop("status")
            ledger.append(GateResult(f"anchor[{impl}].{name}", status, CARD.model_id,
                                     CARD.digest(), metrics=r, env=env,
                                     tolerances={"rel_l2": r.get("tol_rel")},
                                     detail=r.pop("detail", "")))
            v = rec.get("rel_l2")
            print(f"  {rec['status'].upper():4s} {rec['name']:24s} "
                  f"rel_l2={v:.3e}" if v is not None else f"  {rec['name']}: {rec}")
    append_ledger(ledger)

    if args.report:
        md = A.report_markdown(all_records, manifest)
        with open(args.report, "w") as f:
            f.write(md)
        print(f"report -> {args.report}")
    print(json.dumps({i: sum(r["status"] == "fail" for r in rs) for i, rs in all_records.items()},
                     indent=None), "failures per impl")


if __name__ == "__main__":
    main()
