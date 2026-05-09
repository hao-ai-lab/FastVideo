#!/usr/bin/env python3
"""Clone an official reference repo without overwriting existing paths."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clone a reference repo for FastVideo parity tests."
    )
    parser.add_argument("repo_url", help="Official reference repository URL")
    parser.add_argument("target_dir", help="Directory to clone into")
    parser.add_argument("--branch", help="Branch or tag to clone")
    parser.add_argument("--commit", help="Commit SHA to check out after clone")
    parser.add_argument(
        "--update-gitignore",
        action="store_true",
        help="Add the target directory to .gitignore if missing",
    )
    parser.add_argument(
        "--gitignore",
        default=".gitignore",
        help="Path to gitignore file when --update-gitignore is used",
    )
    return parser.parse_args()


def run(command: list[str], check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        check=check,
        text=True,
        capture_output=True,
    )


def print_existing_repo_info(target: Path) -> int:
    print(f"target_exists: {target}")
    if not (target / ".git").exists():
        print("error: target exists but is not a git repo", file=sys.stderr)
        return 1

    remote = run(["git", "-C", str(target), "remote", "-v"], check=False)
    head = run(["git", "-C", str(target), "rev-parse", "HEAD"], check=False)
    if remote.stdout:
        print("remote_v:")
        print(remote.stdout.rstrip())
    if head.stdout:
        print(f"head: {head.stdout.strip()}")
    print("not_overwritten: true")
    return 0


def gitignore_entry_for(target: Path) -> str:
    root = Path.cwd().resolve()
    resolved = target.resolve()
    try:
        relative = resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            "--update-gitignore requires target_dir to be under the current directory"
        ) from exc

    text = relative.as_posix().rstrip("/")
    return "/" + text + "/"


def update_gitignore(path: Path, target: Path) -> bool:
    entry = gitignore_entry_for(target)
    existing = path.read_text().splitlines() if path.exists() else []
    if entry in existing:
        return False

    new_text = "\n".join(existing).rstrip("\n")
    if new_text:
        new_text += "\n"
    new_text += entry + "\n"
    path.write_text(new_text)
    return True


def main() -> int:
    args = parse_args()
    target = Path(args.target_dir)

    if target.exists():
        return print_existing_repo_info(target)

    command = ["git", "clone", "--depth", "1"]
    if args.branch:
        command.extend(["--branch", args.branch])
    command.extend([args.repo_url, str(target)])

    try:
        run(command)
        if args.commit:
            run(["git", "-C", str(target), "fetch", "--depth", "1", "origin", args.commit])
            run(["git", "-C", str(target), "checkout", args.commit])
    except subprocess.CalledProcessError as exc:
        if exc.stdout:
            print(exc.stdout, end="")
        if exc.stderr:
            print(exc.stderr, end="", file=sys.stderr)
        return exc.returncode

    head = run(["git", "-C", str(target), "rev-parse", "HEAD"])
    print(f"cloned: {target}")
    print(f"head: {head.stdout.strip()}")

    if args.update_gitignore:
        changed = update_gitignore(Path(args.gitignore), target)
        print(f"gitignore_updated: {str(changed).lower()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
