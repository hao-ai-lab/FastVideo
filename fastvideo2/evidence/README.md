# Evidence ledger

Typed verification records, committed with the code they vouch for.

- `ledger.jsonl` — append-only `GateResult` records: gate, status, card digest,
  metrics, tolerances, environment fingerprint, timestamp. Written only by
  `python -m fastvideo2 verify`; never edited by hand.
- `<model_id>.fingerprints.json` — the blessed T1 component baseline for one
  card digest in one environment.
- `sample_*.mp4` — eyeballable artifacts from full-scale runs (e.g.
  `sample_wan21_seed7.mp4`, 50 steps / 81 frames on GB200, byte-identical
  between the production pipeline and `reference.py` at the same seed). Re-bless deliberately (`verify --bless`)
  when the card or environment legitimately changes; a digest mismatch is a
  failure, not a skip.

Ownership rule: baselines and gate tolerances are human-owned. Agents run the
gates and append evidence; they do not re-bless baselines to make a failure
disappear.
