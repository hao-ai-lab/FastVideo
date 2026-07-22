# Repository guidelines (branch `will/v2.1`)

This branch is the fastvideo2 MVP — one package, one model (Wan2.1), four
surfaces. Read `README.md` first.

## Layout

| Path | Role |
|---|---|
| `fastvideo2/card.py` | Frozen data cards, `derive()`, digest, validation (stdlib-only) |
| `fastvideo2/loop.py` | Driven-loop protocol + `LoopRunner` (stdlib; NVTX lazily) |
| `fastvideo2/pipeline.py` | Stage list with enforced `reads`/`writes` (stdlib-only) |
| `fastvideo2/loading.py` | Checkpoint → modules, standalone; component fingerprints |
| `fastvideo2/layers/` | Shared model layers (norms, MLP, rotary, attention) — torch-only, checkpoint-key compatible, cast semantics preserved (anchor-proven) |
| `fastvideo2/engine.py` | One-shot runner: request → outputs + identity-chained trace |
| `fastvideo2/verify.py` | Gates T0–T3 + evidence ledger |
| `fastvideo2/registry.py` | The only catalog: name → (card, pipeline builder) |
| `fastvideo2/wan21/` | The model family: card constant, loop, pipeline, `reference.py` |
| `fastvideo2/evidence/` | Append-only ledger + blessed baselines (see its README) |
| `fastvideo2/tests/` | T0 contract tests — CPU, no torch, no weights |

## Invariants (enforced by review; violating them is the bug)

1. **Cards are pure data.** No callables, no live objects, no deploy-local
   paths. If it can't round-trip through JSON, it doesn't belong on a card.
2. **Import direction is one-way:** `card` → `loop`/`pipeline` → `engine` →
   `verify`. Family packages depend on core, never the reverse.
   `wan21/reference.py` imports only the vendored official model file
   (`wan21/model.py`, itself standalone) — never core/runtime modules — and
   nothing outside `verify.py` may import the reference.
3. **Loop modules import torch-free** (torch inside methods) so contracts
   validate anywhere; `import fastvideo2` must work without torch installed.
4. **Model-specific inputs are typed** (`WanForwardInputs`); never add an
   untyped passthrough kwarg to a forward call.
5. **Evidence is append-only and human-owned.** Agents run `verify` and commit
   the records; agents do not edit tolerances, re-bless baselines, or touch
   `reference.py` to make a failing gate pass — say so instead.
6. **One catalog.** New servable ⇒ card constant + registry entry. No parallel
   model lists.
7. **Official implementations are the numerics authority.** Where fidelity is
   the requirement, run the authors' modeling code: the Wan DiT is vendored
   from the pinned official commit (`wan21/model.py`, provenance in its
   header). Restructuring vendored code (e.g. extracting `layers/`) is
   allowed ONLY when the anchor stays bitwise 0.0 — the gate, not "verbatim",
   is the equivalence guarantee. Two invariants for any extraction:
   checkpoint keys unchanged (Sequential indices preserved) and cast/dtype
   semantics unchanged (fp32 islands, promotion order, fp64 RoPE). Ports
   (diffusers, etc.) may serve components only with anchor certification,
   never on trust; the official repo never becomes a dependency or submodule;
   when conventions conflict, official wins.
8. **One environment for goldens and gates.** The supported env is python 3.12
   + torch 2.12 (the fastvideo cluster venv). Goldens are captured with that
   same env — official code rides `PYTHONPATH`, its extra deps go to a pip
   `--target` dir — so anchor deltas measure implementation differences, never
   torch/kernel version differences.

## Commands

```bash
pytest                                             # T0, runs on a laptop
python -m fastvideo2 verify <model> --tier N       # gates; appends evidence
python -m fastvideo2 describe <model>              # card JSON + digest
python -m fastvideo2 generate <model> --prompt ... # one request
```

GPU work runs on dlcluster via the `run-fastvideo-dlcluster` skill from the
main FastVideo checkout (sync this branch with `git push origin HEAD`, then
run inside the branch clone at `/mnt/fv21` — do not disturb `/mnt/FastVideo`'s
checkout). Cluster runs append to `fastvideo2/evidence/` and those files get
fetched and committed locally, so before every cluster `git pull`, reset that
tree or the pull conflicts:

```bash
git checkout -- fastvideo2/evidence; git clean -qfd fastvideo2/evidence; git pull
```

## Commit style

Short subject with a tag prefix (`[feat]: ...`, `[fix]: ...`, `[docs]: ...`).
Do not add AI co-author trailers.
