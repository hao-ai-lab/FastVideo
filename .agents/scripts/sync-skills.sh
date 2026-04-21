#!/usr/bin/env bash
# Sync .agents/skills/ into .claude/skills/ via per-skill symlinks.
#
# Why: Claude Code only scans .claude/skills/ and ~/.claude/skills/ for
# user-invocable skills (no skillsPath config exists — see
# https://code.claude.com/docs/en/skills.md). This repo's skills live
# in .agents/skills/ so they travel with the repo and stay under git.
# Run this once after cloning (or after adding/removing a skill) to
# expose them to Claude Code without maintaining a parallel tree.
#
# Usage:
#   .agents/scripts/sync-skills.sh
#
# Idempotent and safe to re-run. Prunes stale symlinks whose source
# has been removed from .agents/skills/. Leaves hand-written
# .claude/skills/<name>/ directories untouched (only symlinks are
# managed).

set -euo pipefail

REPO_ROOT="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
SRC_DIR="$REPO_ROOT/.agents/skills"
DST_DIR="$REPO_ROOT/.claude/skills"

if [[ ! -d "$SRC_DIR" ]]; then
    echo "Error: $SRC_DIR does not exist." >&2
    exit 1
fi

mkdir -p "$DST_DIR"

linked=0
unchanged=0
skipped=0
pruned=0

link_skill() {
    local name="$1"
    local src="$SRC_DIR/$name"
    local dst="$DST_DIR/$name"
    # Relative target keeps symlinks portable across clones.
    local rel="../../.agents/skills/$name"

    if [[ -L "$dst" ]]; then
        if [[ "$(readlink "$dst")" == "$rel" ]]; then
            unchanged=$((unchanged + 1))
            return
        fi
        rm "$dst"
    elif [[ -e "$dst" ]]; then
        echo "Skipped (not a symlink): .claude/skills/$name" >&2
        skipped=$((skipped + 1))
        return
    fi

    ln -s "$rel" "$dst"
    echo "Linked: .claude/skills/$name -> $rel"
    linked=$((linked + 1))
}

prune_stale() {
    local link="$1"
    local target
    target="$(readlink "$link")"
    case "$target" in
        ../../.agents/skills/*) ;;
        *) return ;;
    esac
    local name="${target##*/}"
    if [[ ! -d "$SRC_DIR/$name" ]]; then
        rm "$link"
        echo "Pruned stale: .claude/skills/$(basename "$link")"
        pruned=$((pruned + 1))
    fi
}

for src in "$SRC_DIR"/*/; do
    [[ -d "$src" ]] || continue
    name="$(basename "$src")"
    # Only treat directories that actually contain a SKILL.md as skills.
    [[ -f "$src/SKILL.md" ]] || continue
    link_skill "$name"
done

shopt -s nullglob
for link in "$DST_DIR"/*; do
    [[ -L "$link" ]] || continue
    prune_stale "$link"
done
shopt -u nullglob

printf "\nSummary: %d linked, %d unchanged, %d pruned" "$linked" "$unchanged" "$pruned"
if [[ "$skipped" -gt 0 ]]; then
    printf ", %d skipped (non-symlink collision)" "$skipped"
fi
printf "\n"
