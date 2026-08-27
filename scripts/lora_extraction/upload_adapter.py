"""Publish a finalized adapter to the Hugging Face Hub, with a card that matches it.

The card is generated from the file rather than written by hand, so the tensor counts,
rank, and sizes it advertises cannot drift from what is actually in the repo. Run
``finalize_adapter.py`` first: this script refuses an adapter that still carries bare
``<param>.weight`` keys, because those are the ones a loader silently drops.

    python scripts/lora_extraction/upload_adapter.py \\
        --adapter-dir /models/fasth3-loras-publish/FastH3-4-step-v1.1 \\
        --repo-id FastVideo/FastH3-4-step-v1.1-LoRA --dry-run

Drop ``--dry-run`` to actually push. Authentication comes from the usual
``huggingface-cli login`` / ``HF_TOKEN`` sources; no token is read from the command line
so it cannot end up in a shell history.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from safetensors import safe_open

LOG = logging.getLogger("upload_adapter")

CARD = """---
license: other
license_name: minimax-h3-community
license_link: LICENSE
base_model: MiniMaxAI/MiniMax-H3
tags:
- video-generation
- minimax-h3
- lora
- distillation
- few-step
- fastvideo
---

# {title}

LoRA extraction of [{finetuned}](https://huggingface.co/{finetuned}), a four-step DMD2
distillation of [MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) under video
sparse attention (VSA).

## What is in the file

{variant_table}

Each adapter is three kinds of payload, because a distillation of this model is not
purely low-rank:

| kind | keys | what it is |
|---|---|---|
| low-rank factors | `<module>.lora_A/.lora_B` | attention, feed-forward and AdaLN projections |
| exact deltas | `<module>.diff`, `<module>.diff_b` | norms and biases, where a rank-r factorization would cost more than the tensor |
| whole parameters | `attn.to_gate_compress.set_weight` | the VSA compression gate, which **does not exist in base MiniMax-H3** |

The `.set_weight` keys are why this is not an ordinary LoRA. `to_gate_compress` is
created only under the VSA attention backend and is zero-initialized when a checkpoint
does not carry it, so a loader that ignores those keys produces a model with the
compression branch switched off — quietly, and without an error.

## Usage (FastVideo)

```bash
python examples/inference/lora/minimax_h3_lora_inference.py \\
    --model-path MiniMaxAI/MiniMax-H3 \\
    --lora-path {repo_id} \\
    --prompt "your prompt here"
```

The adapter must be supplied when the pipeline is constructed, not swapped in later: a
parameter the base model lacks has to arrive while weights are still unsharded.
Sample with **4 steps** (5 sigma-grid points), **cfg 1.0** — the checkpoint is
guidance-distilled — and the VSA backend enabled.

## Fidelity

Reconstruction against the checkpoint this was extracted from, as relative Frobenius
error per parameter family:

{fidelity}

`.diff` and `.set_weight` families are stored exactly. The low-rank families carry the
SVD truncation tail, which is what choosing a rank buys or costs.

## License

MiniMax H3 Community License, inherited from the base model. Review its territory and
acceptable-use terms before use or redistribution.
"""


def inspect(path: Path) -> dict:
    """Counts, rank, and size of one adapter file, read from its header."""
    with safe_open(path, framework="pt") as handle:
        meta = handle.metadata() or {}
        keys = list(handle.keys())
        ranks = {handle.get_slice(k).get_shape()[0] for k in keys if ".lora_A" in k}
    return {
        "path": path,
        "metadata": meta,
        "n_low_rank": sum(1 for k in keys if ".lora_" in k),
        "n_diff": sum(1 for k in keys if k.endswith((".diff", ".diff_b"))),
        "n_set": sum(1 for k in keys if k.endswith(".set_weight")),
        "n_bare": sum(1 for k in keys if k.endswith((".weight", ".bias")) and ".lora_" not in k),
        "rank": max(ranks) if ranks else None,
        "size_gib": path.stat().st_size / 2**30,
    }


def build_card(repo_id: str, variants: list[dict], fidelity: str) -> str:
    finetuned = next((v["metadata"].get("finetuned_model") for v in variants if v["metadata"].get("finetuned_model")),
                     "")
    rows = ["| file | rank | low-rank | .diff | .set_weight | size |", "|---|---|---|---|---|---|"]
    for v in sorted(variants, key=lambda x: x["rank"] or 0):
        rel = v["path"].relative_to(v["path"].parents[1])
        rows.append(f"| `{rel}` | {v['rank']} | {v['n_low_rank']} | {v['n_diff']} | "
                    f"{v['n_set']} | {v['size_gib']:.2f} GiB |")
    return CARD.format(
        title=repo_id.split("/")[-1],
        finetuned=finetuned,
        repo_id=repo_id,
        variant_table="\n".join(rows),
        fidelity=fidelity or "_not measured for this build_",
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--adapter-dir", required=True, help="directory holding rank-*/adapter_model.safetensors")
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--fidelity-json", default=None, help="verification summary to quote in the card")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="write the card locally and stop")
    parser.add_argument("--allow-bare-weights",
                        action="store_true",
                        help="upload even when the adapter still carries unconvertible bare .weight keys")
    args = parser.parse_args()

    root = Path(args.adapter_dir)
    files = sorted(root.glob("rank-*/adapter_model.safetensors"))
    if not files:
        raise SystemExit(f"no rank-*/adapter_model.safetensors under {root}")

    variants = [inspect(f) for f in files]
    for v in variants:
        LOG.info("%s: rank=%s low_rank=%d diff=%d set_weight=%d bare=%d %.2f GiB", v["path"].parent.name, v["rank"],
                 v["n_low_rank"], v["n_diff"], v["n_set"], v["n_bare"], v["size_gib"])
        if v["n_bare"] and not args.allow_bare_weights:
            raise SystemExit(f"{v['path']} still has {v['n_bare']} bare .weight/.bias keys. Those are the keys "
                             f"loaders drop silently -- run finalize_adapter.py first, or pass "
                             f"--allow-bare-weights if you really mean it.")

    fidelity = ""
    if args.fidelity_json:
        fidelity = "```\n" + Path(args.fidelity_json).read_text().strip() + "\n```"

    card_path = root / "README.md"
    card_path.write_text(build_card(args.repo_id, variants, fidelity))
    LOG.info("wrote %s", card_path)

    if args.dry_run:
        LOG.info("dry run: not uploading. Review %s, then rerun without --dry-run.", card_path)
        return

    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(repo_id=args.repo_id, repo_type="model", private=args.private, exist_ok=True)
    LOG.info("uploading %s -> %s", root, args.repo_id)
    api.upload_folder(
        folder_path=str(root),
        repo_id=args.repo_id,
        repo_type="model",
        # Bookkeeping from the finalize step; the card already carries the numbers.
        ignore_patterns=["*.json.tmp", "finalize_summary.json"],
    )
    LOG.info("done: https://huggingface.co/%s", args.repo_id)


if __name__ == "__main__":
    main()
