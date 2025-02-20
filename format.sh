#!/usr/bin/env bash
# YAPF formatter, adapted from fastvideo.
#
# Usage:
#    # Do work and commit your work.

#    # Format files that differ from origin/main.
#    bash format.sh

#    # Commit changed files with message 'Run yapf and ruff'
#
#
# This script formats all changed files from the last mergebase.
# You are encouraged to run this locally before pushing changes for review.

# Cause the script to exit if a single command fails
set -eo pipefail

# this stops git rev-parse from failing if we run this from the .git directory
builtin cd "$(dirname "${BASH_SOURCE:-$0}")"
ROOT="$(git rev-parse --show-toplevel)"
builtin cd "$ROOT" || exit 1

check_command() {
    if ! command -v "$1" &> /dev/null; then
        echo "❓❓$1 is not installed, please run \`bash env_setup.sh\`"
        exit 1
    fi
}

check_command yapf
check_command ruff
check_command codespell
check_command isort

YAPF_VERSION=$(yapf --version | awk '{print $2}')
RUFF_VERSION=$(ruff --version | awk '{print $2}')
CODESPELL_VERSION=$(codespell --version)
ISORT_VERSION=$(isort --vn)
SPHINX_LINT_VERSION=$(sphinx-lint --version | awk '{print $2}')


# # params: tool name, tool version, required version
tool_version_check() {
    expected=$(grep "$1" requirements-lint.txt | cut -d'=' -f3)
    if [[ "$2" != "$expected" ]]; then
        echo "❓❓Wrong $1 version installed: $expected is required, not $2."
        exit 1
    fi
}

tool_version_check "yapf" "$YAPF_VERSION"
tool_version_check "ruff" "$RUFF_VERSION"
tool_version_check "isort" "$ISORT_VERSION"
tool_version_check "codespell" "$CODESPELL_VERSION"
tool_version_check "sphinx-lint" "$SPHINX_LINT_VERSION"

YAPF_FLAGS=(
    '--recursive'
    '--parallel'
)

YAPF_EXCLUDES=(
    '--exclude' 'data/**'
)

# Format specified files
format() {
    yapf --in-place "${YAPF_FLAGS[@]}" "$@"
}

# Format files that differ from main branch. Ignores dirs that are not slated
# for autoformat yet.
format_changed() {
    # The `if` guard ensures that the list of filenames is not empty, which
    # could cause yapf to receive 0 positional arguments, making it hang
    # waiting for STDIN.
    #
    # `diff-filter=ACM` and $MERGEBASE is to ensure we only format files that
    # exist on both branches.
    MERGEBASE="$(git merge-base origin/main HEAD)"

    if ! git diff --diff-filter=ACM --quiet --exit-code "$MERGEBASE" -- '*.py' '*.pyi' &>/dev/null; then
        git diff --name-only --diff-filter=ACM "$MERGEBASE" -- '*.py' '*.pyi' | xargs -P 5 \
             yapf --in-place "${YAPF_EXCLUDES[@]}" "${YAPF_FLAGS[@]}"
    fi

}

# Format all files
format_all() {
    yapf --in-place "${YAPF_FLAGS[@]}" "${YAPF_EXCLUDES[@]}" .
}

## This flag formats individual files. --files *must* be the first command line
## arg to use this option.
if [[ "$1" == '--files' ]]; then
   format "${@:2}"
   # If `--all` is passed, then any further arguments are ignored and the
   # entire python directory is formatted.
elif [[ "$1" == '--all' ]]; then
   format_all
else
   # Format only the files that changed in last commit.
   format_changed
fi
echo 'FastVideo yapf: Done'


# If git diff returns a file that is in the skip list, the file may be checked anyway:
# https://github.com/codespell-project/codespell/issues/1915
# Avoiding the "./" prefix and using "/**" globs for directories appears to solve the problem
CODESPELL_EXCLUDES=(
    '--skip' 'data/**,
            fastvideo/distill.py,
            fastvideo/models/hunyuan/modules/models.py,
            fastvideo/models/mochi_hf/modeling_mochi.py,
            fastvideo/utils/env_utils.py,
            ./csrc/sliding_tile_attention/tk/**'

)

# check spelling of specified files
spell_check() {
    codespell "$@"
}

spell_check_all(){
  codespell --toml pyproject.toml
}

# Spelling check of files that differ from main branch.
spell_check_changed() {
    # The `if` guard ensures that the list of filenames is not empty, which
    # could cause ruff to receive 0 positional arguments, making it hang
    # waiting for STDIN.
    #
    # `diff-filter=ACM` and $MERGEBASE is to ensure we only lint files that
    # exist on both branches.
    MERGEBASE="$(git merge-base origin/main HEAD)"
    if ! git diff --diff-filter=ACM --quiet --exit-code "$MERGEBASE" -- '*.py' '*.pyi' &>/dev/null; then
        git diff --name-only --diff-filter=ACM "$MERGEBASE" -- '*.py' '*.pyi' | xargs \
            codespell "${CODESPELL_EXCLUDES[@]}"
    fi
}

# Run Codespell
## This flag runs spell check of individual files. --files *must* be the first command line
# arg to use this option.
if [[ "$1" == '--files' ]]; then
   spell_check "${@:2}"
   # If `--all` is passed, then any further arguments are ignored and the
   # entire python directory is linted.
elif [[ "$1" == '--all' ]]; then
   spell_check_all
else
   # Check spelling only of the files that changed in last commit.
   spell_check_changed
fi
echo 'FastVideo codespell: Done'


# Lint specified files
lint() {
    ruff check "$@"
}

# Lint files that differ from main branch. Ignores dirs that are not slated
# for autolint yet.
lint_changed() {
    # The `if` guard ensures that the list of filenames is not empty, which
    # could cause ruff to receive 0 positional arguments, making it hang
    # waiting for STDIN.
    #
    # `diff-filter=ACM` and $MERGEBASE is to ensure we only lint files that
    # exist on both branches.
    MERGEBASE="$(git merge-base origin/main HEAD)"

    if ! git diff --diff-filter=ACM --quiet --exit-code "$MERGEBASE" -- '*.py' '*.pyi' &>/dev/null; then
        git diff --name-only --diff-filter=ACM "$MERGEBASE" -- '*.py' '*.pyi' | xargs \
             ruff check
    fi

}

# Run Ruff
### This flag lints individual files. --files *must* be the first command line
### arg to use this option.
if [[ "$1" == '--files' ]]; then
   lint "${@:2}"
   # If `--all` is passed, then any further arguments are ignored and the
   # entire python directory is linted.
elif [[ "$1" == '--all' ]]; then
   lint fastvideo scripts
else
   # Format only the files that changed in last commit.
   lint_changed
fi
echo 'FastVideo ruff: Done'

ISORT_EXCLUDES=(
    '--skip' './csrc/sliding_tile_attention/tk'
)

# check spelling of specified files
isort_check() {
    isort "$@"
}

isort_check_all(){
  isort "${ISORT_EXCLUDES[@]}" .
}

# Spelling  check of files that differ from main branch.
isort_check_changed() {
    # The `if` guard ensures that the list of filenames is not empty, which
    # could cause ruff to receive 0 positional arguments, making it hang
    # waiting for STDIN.
    #
    # `diff-filter=ACM` and $MERGEBASE is to ensure we only lint files that
    # exist on both branches.
    MERGEBASE="$(git merge-base origin/main HEAD)"

    if ! git diff --diff-filter=ACM --quiet --exit-code "$MERGEBASE" -- '*.py' '*.pyi' &>/dev/null; then
        git diff --name-only --diff-filter=ACM "$MERGEBASE" -- '*.py' '*.pyi' | xargs \
             isort
    fi
}

# Run Isort
# This flag runs spell check of individual files. --files *must* be the first command line
# arg to use this option.
if [[ "$1" == '--files' ]]; then
   isort_check "${@:2}"
   # If `--all` is passed, then any further arguments are ignored and the
   # entire python directory is linted.
elif [[ "$1" == '--all' ]]; then
   isort_check_all
else
   # Check spelling only of the files that changed in last commit.
   isort_check_changed
fi
echo 'FastVideo isort: Done'
