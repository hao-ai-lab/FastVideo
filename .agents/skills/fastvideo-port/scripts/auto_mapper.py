#!/usr/bin/env python3
"""Auto-mapper: generate first-pass param_names_mapping from two state_dicts.

Loads the official checkpoint and a FastVideo model, computes string similarity
between key names, and outputs candidate regex rules to stdout.

Usage:
    # With safetensors checkpoints:
    python auto_mapper.py \
        --official official_weights/model/transformer/diffusion_pytorch_model.safetensors \
        --fastvideo_class fastvideo.models.dits.mymodel.MyTransformer \
        --fastvideo_args '{"hidden_size": 1024, "num_layers": 28}'

    # With .pt checkpoints:
    python auto_mapper.py \
        --official official_weights/model/model.pt \
        --fastvideo_class fastvideo.models.dits.mymodel.MyTransformer \
        --fastvideo_args '{}'

Output:
    Prints Python code for param_names_mapping that can be pasted into the
    arch config. Also prints a coverage report: how many official keys are
    covered by the generated rules.

Strategy:
    1. Exact matches (same key in both): no rule needed.
    2. Prefix differences: detect common prefix strips/swaps.
    3. Token-level edit distance: align key segments greedily.
    4. Shape-based matching: as a fallback, align by tensor shape.
"""
import argparse
import importlib
import json
import re
import sys
from collections import defaultdict
from difflib import SequenceMatcher


def load_official(path: str) -> dict:
    if path.endswith(".safetensors"):
        import safetensors.torch as st
        return dict(st.load_file(path))
    else:
        import torch
        sd = torch.load(path, map_location="cpu", weights_only=True)
        # Handle nested state_dict wrappers
        if "state_dict" in sd:
            sd = sd["state_dict"]
        if "model" in sd and isinstance(sd["model"], dict):
            sd = sd["model"]
        return sd


def load_fastvideo_keys(class_path: str, init_args: dict) -> list[str]:
    """Instantiate the FastVideo model class and return its state_dict keys."""
    module_path, cls_name = class_path.rsplit(".", 1)
    mod = importlib.import_module(module_path)
    cls = getattr(mod, cls_name)
    import torch
    with torch.no_grad():
        model = cls(**init_args)
    return list(model.state_dict().keys())


def tokenize_key(key: str) -> list[str]:
    """Split a state_dict key into segments for comparison."""
    return re.split(r"[.\-_]|\d+", key)


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def find_best_match(official_key: str, fv_keys: list[str],
                    used: set[str]) -> tuple[str | None, float]:
    """Find the most similar FastVideo key for an official key."""
    best_key, best_score = None, 0.0
    for fk in fv_keys:
        if fk in used:
            continue
        score = similarity(official_key, fk)
        if score > best_score:
            best_score = score
            best_key = fk
    return best_key, best_score


def key_to_regex(key: str) -> str:
    """Convert a state_dict key into a regex pattern, escaping dots and
    replacing numeric indices with \\d+ wildcards."""
    parts = key.split(".")
    pattern_parts = []
    for p in parts:
        if p.isdigit():
            pattern_parts.append(r"\d+")
        else:
            pattern_parts.append(re.escape(p))
    return r"\.".join(pattern_parts)


def extract_prefix_rules(official_keys: list[str],
                          fv_keys: list[str]) -> list[tuple[str, str]]:
    """Detect simple prefix transformations covering many keys."""
    o_prefixes = defaultdict(int)
    fv_prefixes = defaultdict(int)
    for k in official_keys:
        o_prefixes[k.split(".")[0]] += 1
    for k in fv_keys:
        fv_prefixes[k.split(".")[0]] += 1

    rules = []
    for op, oc in sorted(o_prefixes.items(), key=lambda x: -x[1]):
        if op not in fv_prefixes:
            # Find best match for this prefix in FV
            best_fp, best_score = None, 0.0
            for fp in fv_prefixes:
                s = similarity(op, fp)
                if s > best_score:
                    best_score = s
                    best_fp = fp
            if best_fp and best_score > 0.5:
                rules.append((f"^{re.escape(op)}\\.(.*)$",
                               f"{best_fp}.\\1",
                               oc, best_score))
    return rules


def generate_rules(official_keys: list[str],
                   fv_keys: list[str],
                   min_score: float = 0.6) -> list[tuple[str, str, str]]:
    """Generate (pattern, replacement, comment) tuples."""
    rules = []
    used_fv = set()

    # First pass: exact matches (no rule needed, just note)
    exact = set(official_keys) & set(fv_keys)
    if exact:
        rules.append(("# EXACT MATCHES (no rule needed): "
                      f"{len(exact)} keys match directly", "", ""))

    # Second pass: prefix rules
    prefix_rules = extract_prefix_rules(
        [k for k in official_keys if k not in exact],
        [k for k in fv_keys if k not in exact])

    for pattern, replacement, count, score in prefix_rules:
        rules.append((pattern, replacement,
                      f"prefix rule covers ~{count} keys (score={score:.2f})"))

    # Third pass: key-by-key similarity for remaining
    remaining_official = [k for k in official_keys if k not in exact]
    covered_by_prefix = set()
    for k in remaining_official:
        for pat, rep, _ in rules:
            if pat.startswith("#"):
                continue
            if re.match(pat, k):
                covered_by_prefix.add(k)
                break

    for ok in remaining_official:
        if ok in covered_by_prefix:
            continue
        fk, score = find_best_match(ok, fv_keys, used_fv)
        if fk and score >= min_score:
            used_fv.add(fk)
            # Try to extract a generalizable pattern
            pat = key_to_regex(ok)
            rep = fk
            # If they share the same structure with different segment names,
            # try to build a capturing rule
            ok_parts = ok.split(".")
            fk_parts = fk.split(".")
            if len(ok_parts) == len(fk_parts):
                captures = []
                replacements = []
                for i, (op, fp) in enumerate(zip(ok_parts, fk_parts)):
                    if op == fp:
                        captures.append(re.escape(op))
                        replacements.append(fp)
                    elif op.isdigit() and fp.isdigit():
                        captures.append(r"(\d+)")
                        replacements.append(f"\\{len([x for x in captures if '(' in x])}")
                    else:
                        captures.append(re.escape(op))
                        replacements.append(fp)
                pat = r"^" + r"\." .join(captures) + r"$"
                rep = ".".join(replacements)

            rules.append((pat, rep, f"score={score:.2f} ({ok} → {fk})"))

    return rules


def coverage_report(official_keys: list[str],
                    rules: list[tuple[str, str, str]]) -> float:
    covered = set()
    for ok in official_keys:
        for pat, rep, comment in rules:
            if pat.startswith("#"):
                continue
            try:
                if re.match(pat, ok):
                    covered.add(ok)
                    break
            except re.error:
                pass
    pct = len(covered) / len(official_keys) * 100 if official_keys else 0
    return pct, covered


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--official", required=True,
                        help="Path to official safetensors or .pt checkpoint")
    parser.add_argument("--fastvideo_class", required=True,
                        help="Dotted class path, e.g. fastvideo.models.dits.mymodel.MyTransformer")
    parser.add_argument("--fastvideo_args", default="{}",
                        help="JSON dict of __init__ kwargs for the FastVideo model")
    parser.add_argument("--min_score", type=float, default=0.6,
                        help="Minimum similarity score to emit a rule (default 0.6)")
    parser.add_argument("--output_py", default=None,
                        help="Write param_names_mapping Python snippet to this file")
    args = parser.parse_args()

    print("[auto_mapper] Loading official state dict...", file=sys.stderr)
    official_sd = load_official(args.official)
    official_keys = list(official_sd.keys())
    print(f"[auto_mapper] {len(official_keys)} official keys", file=sys.stderr)

    print("[auto_mapper] Loading FastVideo model keys...", file=sys.stderr)
    fv_init = json.loads(args.fastvideo_args)
    fv_keys = load_fastvideo_keys(args.fastvideo_class, fv_init)
    print(f"[auto_mapper] {len(fv_keys)} FastVideo keys", file=sys.stderr)

    print("[auto_mapper] Generating rules...", file=sys.stderr)
    rules = generate_rules(official_keys, fv_keys, args.min_score)

    pct, covered = coverage_report(official_keys, rules)
    print(f"\n# Coverage: {len(covered)}/{len(official_keys)} keys "
          f"({pct:.1f}%) covered by generated rules\n")

    # Emit Python code
    lines = ["param_names_mapping = ["]
    for pat, rep, comment in rules:
        if pat.startswith("#"):
            lines.append(f"    {pat}")
        elif rep:
            lines.append(f"    (r\"{pat}\", r\"{rep}\"),  # {comment}")
    lines.append("]")
    output = "\n".join(lines)

    print(output)
    if args.output_py:
        with open(args.output_py, "w") as f:
            f.write(output + "\n")
        print(f"\n[auto_mapper] Written to {args.output_py}", file=sys.stderr)

    print(f"\n# Next step: load state_dict with these rules and check for "
          f"missing/unexpected keys:\n"
          f"#   python - <<'PY'\n"
          f"#   from fastvideo.models.dits.mymodel import MyTransformer\n"
          f"#   model = MyTransformer(...)\n"
          f"#   result = model.load_state_dict(mapped_sd, strict=False)\n"
          f"#   print('missing:', result.missing_keys[:10])\n"
          f"#   print('unexpected:', result.unexpected_keys[:10])\n"
          f"#   PY")


if __name__ == "__main__":
    main()
