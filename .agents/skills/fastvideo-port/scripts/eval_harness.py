#!/usr/bin/env python3
"""Eval harness for the fastvideo-port skill training loop.

Workflow:
  1. Reset to pre-port commit (or use a saved branch).
  2. Run the port skill (agent generates the port).
  3. Diff agent output vs ground-truth branch.
  4. Format the delta as lesson candidates for human review.

Usage:
    # Full eval against Cosmos 2.5 ground truth:
    python eval_harness.py \
        --ground_truth_branch feat/cosmos25-training \
        --pre_port_commit abc1234 \
        --model_name cosmos2_5 \
        --output_dir eval_results/cosmos25/

    # Just diff two branches (skip agent run, diff only):
    python eval_harness.py \
        --ground_truth_branch feat/cosmos25-training \
        --agent_branch eval/cosmos25-port-test \
        --diff_only \
        --output_dir eval_results/cosmos25/

Output files in --output_dir:
    delta.diff              Raw unified diff
    delta_summary.md        Human-readable summary with lesson candidates
    coverage_report.json    Which ground-truth files are present/missing/wrong
"""
import argparse
import json
import os
import re
import subprocess
import sys
from datetime import date
from pathlib import Path


# Files expected in a complete port (adjust per model)
EXPECTED_FILE_PATTERNS = [
    r"fastvideo/models/dits/\w+\.py",
    r"fastvideo/configs/models/dits/\w+\.py",
    r"fastvideo/configs/pipelines/\w+\.py",
    r"fastvideo/training/\w+_training_pipeline\.py",
    r"examples/training/finetune/\w+/",
    r"tests/local_tests/\w+/",
]

LESSON_CATEGORY_HINTS = {
    "ReplicatedLinear": "porting",
    "nn.Linear": "porting",
    "dtype": "porting",
    "float32": "porting",
    "bfloat16": "porting",
    "shape": "porting",
    "hardcoded": "porting",
    "preprocessing": "porting",
    "tokenizer": "porting",
    "scheduler": "porting",
    "shift": "porting",
    "param_names_mapping": "porting",
    "learning_rate": "hyperparameter",
    "batch_size": "hyperparameter",
    "ssim": "evaluation",
    "alignment": "evaluation",
}


def run(cmd: list[str], cwd: str | None = None, check: bool = True) -> str:
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)
    if check and result.returncode != 0:
        print(f"[error] Command failed: {' '.join(cmd)}", file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        sys.exit(1)
    return result.stdout.strip()


def get_diff(repo: str, base: str, head: str) -> str:
    return run(["git", "diff", f"{base}...{head}", "--", "fastvideo/", "examples/", "tests/"],
               cwd=repo)


def get_changed_files(repo: str, base: str, head: str) -> list[str]:
    out = run(["git", "diff", "--name-only", f"{base}...{head}"], cwd=repo)
    return [f for f in out.splitlines() if f.strip()]


def coverage_report(gt_files: list[str], agent_files: list[str]) -> dict:
    gt_set = set(gt_files)
    agent_set = set(agent_files)
    return {
        "ground_truth_files": sorted(gt_files),
        "agent_files": sorted(agent_files),
        "present_in_both": sorted(gt_set & agent_set),
        "missing_from_agent": sorted(gt_set - agent_set),
        "extra_in_agent": sorted(agent_set - gt_set),
        "coverage_pct": round(len(gt_set & agent_set) / len(gt_set) * 100, 1) if gt_set else 0,
    }


def classify_diff_hunk(hunk: str) -> str:
    """Guess the lesson category for a diff hunk based on content."""
    for keyword, category in LESSON_CATEGORY_HINTS.items():
        if keyword.lower() in hunk.lower():
            return category
    return "other"


def extract_lesson_candidates(diff: str, missing_files: list[str]) -> list[dict]:
    """Parse the diff and produce lesson candidates."""
    candidates = []

    # Missing files → lesson about what the agent forgot
    for mf in missing_files:
        candidates.append({
            "title": f"Agent did not create {mf}",
            "category": "porting",
            "severity": "important",
            "what_happened": f"The ground-truth port includes `{mf}` but the agent did not create it.",
            "hint": "Check if the fastvideo-port skill Step 1–4 explicitly mentions this file type.",
        })

    # Significant diff hunks
    current_file = None
    current_hunk = []
    for line in diff.splitlines():
        if line.startswith("diff --git"):
            if current_file and current_hunk:
                hunk_text = "\n".join(current_hunk)
                if len(current_hunk) > 5:  # ignore trivial diffs
                    candidates.append({
                        "title": f"Diff in {current_file}",
                        "category": classify_diff_hunk(hunk_text),
                        "severity": "minor",
                        "what_happened": f"Agent implementation differs from ground truth in `{current_file}`.",
                        "diff_preview": "\n".join(current_hunk[:20]),
                    })
            m = re.search(r"^diff --git a/.+ b/(.+)$", line)
            current_file = m.group(1) if m else line
            current_hunk = []
        else:
            current_hunk.append(line)

    return candidates


def format_lesson_md(candidate: dict, model_name: str) -> str:
    today = date.today().isoformat()
    slug = re.sub(r"[^a-z0-9]+", "-", candidate["title"].lower())[:50]
    filename = f"{today}_{slug}.md"

    body = f"""---
date: {today}
experiment: {model_name} port (eval harness candidate — needs human review)
category: {candidate["category"]}
severity: {candidate["severity"]}
---

# {candidate["title"]}

## What Happened
{candidate["what_happened"]}

## Root Cause
TODO: fill in after investigation.

## Fix / Workaround
TODO: fill in.

## Prevention
{candidate.get("hint", "TODO: add to fastvideo-port SKILL.md or .agents/lessons/.")}
"""
    if "diff_preview" in candidate:
        body += f"\n## Diff Preview\n```diff\n{candidate['diff_preview']}\n```\n"
    return filename, body


def write_summary(output_dir: Path, candidates: list[dict],
                  coverage: dict, model_name: str) -> None:
    summary = [f"# Eval Harness Report: {model_name} port\n"]
    summary.append(f"**Coverage**: {coverage['coverage_pct']}% of ground-truth files "
                   f"({len(coverage['present_in_both'])}/{len(coverage['ground_truth_files'])})\n")

    if coverage["missing_from_agent"]:
        summary.append("## Missing files (agent didn't create)\n")
        for f in coverage["missing_from_agent"]:
            summary.append(f"- `{f}`\n")

    if coverage["extra_in_agent"]:
        summary.append("\n## Extra files (agent created but not in ground truth)\n")
        for f in coverage["extra_in_agent"]:
            summary.append(f"- `{f}`\n")

    summary.append(f"\n## Lesson Candidates ({len(candidates)} total)\n")
    for i, c in enumerate(candidates, 1):
        summary.append(f"\n### {i}. {c['title']}\n")
        summary.append(f"- **Category**: {c['category']}\n")
        summary.append(f"- **Severity**: {c['severity']}\n")
        summary.append(f"- {c['what_happened']}\n")
        if "hint" in c:
            summary.append(f"- *Hint*: {c['hint']}\n")

    summary.append("\n---\n*Generated by eval_harness.py — review and promote "
                   "candidates to .agents/lessons/ manually.*\n")

    (output_dir / "delta_summary.md").write_text("".join(summary))
    print(f"[eval] Summary written to {output_dir / 'delta_summary.md'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ground_truth_branch", required=True,
                        help="Branch with the correct port (e.g. feat/cosmos25-training)")
    parser.add_argument("--pre_port_commit", default=None,
                        help="Commit hash before the port (for creating eval branch)")
    parser.add_argument("--agent_branch", default=None,
                        help="Branch where agent output lives (skips agent run if set)")
    parser.add_argument("--model_name", default="model",
                        help="Short name used in output filenames")
    parser.add_argument("--output_dir", default="eval_results/",
                        help="Directory for output files")
    parser.add_argument("--repo", default=".", help="Path to FastVideo repo")
    parser.add_argument("--diff_only", action="store_true",
                        help="Skip agent run, just diff ground_truth vs agent_branch")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.diff_only:
        if not args.agent_branch:
            print("[error] --agent_branch required with --diff_only", file=sys.stderr)
            sys.exit(1)
        base = args.agent_branch
    else:
        if not args.pre_port_commit:
            print("[error] --pre_port_commit required unless --diff_only", file=sys.stderr)
            sys.exit(1)
        # Create eval branch
        eval_branch = f"eval/{args.model_name}-port-test"
        print(f"[eval] Creating eval branch {eval_branch} from {args.pre_port_commit}")
        run(["git", "checkout", "-b", eval_branch, args.pre_port_commit], cwd=args.repo)
        print(f"[eval] Agent should now run /fastvideo-port on this branch.")
        print(f"[eval] After the agent completes, re-run with:")
        print(f"  --agent_branch {eval_branch} --diff_only --ground_truth_branch {args.ground_truth_branch}")
        sys.exit(0)

    # Diff
    print(f"[eval] Diffing {base} vs {args.ground_truth_branch}...")
    diff = get_diff(args.repo, base, args.ground_truth_branch)
    (output_dir / "delta.diff").write_text(diff)
    print(f"[eval] Diff written to {output_dir / 'delta.diff'} ({len(diff)} chars)")

    gt_files = get_changed_files(args.repo, base, args.ground_truth_branch)
    agent_files = get_changed_files(args.repo, args.ground_truth_branch, base)

    cov = coverage_report(gt_files, agent_files)
    (output_dir / "coverage_report.json").write_text(json.dumps(cov, indent=2))

    candidates = extract_lesson_candidates(diff, cov["missing_from_agent"])
    write_summary(output_dir, candidates, cov, args.model_name)

    # Write lesson candidate files
    lessons_dir = output_dir / "lesson_candidates"
    lessons_dir.mkdir(exist_ok=True)
    for c in candidates:
        fname, body = format_lesson_md(c, args.model_name)
        (lessons_dir / fname).write_text(body)
    print(f"[eval] {len(candidates)} lesson candidates written to {lessons_dir}/")
    print(f"[eval] Coverage: {cov['coverage_pct']}% ({len(cov['present_in_both'])}/{len(gt_files)} files)")


if __name__ == "__main__":
    main()
