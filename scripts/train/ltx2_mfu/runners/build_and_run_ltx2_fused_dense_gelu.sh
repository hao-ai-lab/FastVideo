#!/usr/bin/env bash
set -euo pipefail

PINNED_REV=82d6441
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
BENCHMARK="$SCRIPT_DIR/bench_ltx2_fused_dense_gelu.py"

if [[ ! -f "$BENCHMARK" ]]; then
    echo "missing benchmark: $BENCHMARK" >&2
    exit 1
fi

if [[ -n "${PYTHON:-}" ]]; then
    PYTHON_BIN=$PYTHON
elif [[ -n "${VIRTUAL_ENV:-}" && -x "$VIRTUAL_ENV/bin/python" ]]; then
    PYTHON_BIN="$VIRTUAL_ENV/bin/python"
elif [[ -x "$PWD/.venv/bin/python" ]]; then
    PYTHON_BIN="$PWD/.venv/bin/python"
else
    echo "activate the existing FastVideo venv, run from its repo root, or set PYTHON" >&2
    exit 1
fi

if [[ -n "${FLASH_ATTN_SRC:-}" ]]; then
    SOURCE=$FLASH_ATTN_SRC
else
    if ! command -v uv >/dev/null 2>&1; then
        echo "set FLASH_ATTN_SRC to the pinned flash-attention checkout" >&2
        exit 1
    fi
    UV_CACHE=$(uv cache dir)
    EXTENSION_DIR=$(find "$UV_CACHE/git-v0/checkouts" -type d \
        -path "*/$PINNED_REV/csrc/fused_dense_lib" -print -quit 2>/dev/null || true)
    if [[ -z "$EXTENSION_DIR" ]]; then
        echo "could not locate pinned flash-attention source; set FLASH_ATTN_SRC" >&2
        exit 1
    fi
    SOURCE=$(CDPATH= cd -- "$EXTENSION_DIR/../.." && pwd)
fi

if [[ -d "$SOURCE/csrc/fused_dense_lib" ]]; then
    SOURCE=$(CDPATH= cd -- "$SOURCE" && pwd)
    EXTENSION_DIR="$SOURCE/csrc/fused_dense_lib"
elif [[ -f "$SOURCE/setup.py" && "${SOURCE##*/}" == fused_dense_lib ]]; then
    EXTENSION_DIR=$(CDPATH= cd -- "$SOURCE" && pwd)
    SOURCE=$(CDPATH= cd -- "$EXTENSION_DIR/../.." && pwd)
else
    echo "FLASH_ATTN_SRC must be a flash-attention root or its csrc/fused_dense_lib" >&2
    exit 1
fi

if git -C "$SOURCE" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    REVISION=$(git -C "$SOURCE" rev-parse HEAD)
    if [[ "$REVISION" != "$PINNED_REV"* ]]; then
        echo "expected flash-attention $PINNED_REV, found $REVISION" >&2
        exit 1
    fi
    echo "FLASH_ATTN_REVISION $REVISION"
else
    echo "warning: source has no git metadata; could not verify revision $PINNED_REV" >&2
fi

"$PYTHON_BIN" -c 'import torch; assert torch.__version__.startswith("2.12."), torch.__version__; assert torch.version.cuda and torch.version.cuda.startswith("13."), torch.version.cuda; print("TORCH_ENV", torch.__version__, torch.version.cuda)'

(
    cd "$EXTENSION_DIR"
    TORCH_CUDA_ARCH_LIST=10.0 "$PYTHON_BIN" setup.py build_ext --inplace
)

EXTENSION=$(find "$EXTENSION_DIR" -maxdepth 1 -type f -name 'fused_dense_lib*.so' -print -quit)
if [[ -z "$EXTENSION" ]]; then
    echo "fused_dense_lib build produced no extension" >&2
    exit 1
fi

hash_file() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1"
    else
        shasum -a 256 "$1"
    fi
}

echo "BENCHMARK_SHA256 $(hash_file "$BENCHMARK")"
echo "RUNNER_SHA256 $(hash_file "$0")"
echo "EXTENSION_SHA256 $(hash_file "$EXTENSION")"

export PYTHONPATH="$EXTENSION_DIR${PYTHONPATH:+:$PYTHONPATH}"
exec "$PYTHON_BIN" "$BENCHMARK" "$@"
