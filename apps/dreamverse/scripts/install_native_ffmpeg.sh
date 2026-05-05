#!/usr/bin/env bash
#
# Build & install a perf-optimized ffmpeg (LTO + libx264 + native arch).
# Mirrors the team playbook's flags verbatim per stage, with three
# deliberate deviations made necessary by our build host:
#
#   1. --disable-libxcb --disable-xlib   ffmpeg's auto-detect linked
#                                        libxcb at build time → binary
#                                        wouldn't even start at runtime.
#   2. LIBRARY_PATH / LD_LIBRARY_PATH    conda-forge gcc wrappers search
#      + -L$INSTALL_PREFIX/lib early     $CONDA_PREFIX/lib implicitly,
#                                        leaking an older libx264 into
#                                        ffmpeg's link. Forcing our prefix
#                                        first makes the resolver pick
#                                        our just-built lib.
#   3. MAKE_JOBS cap (default 16)        very high -j (e.g. nproc=96 on
#                                        NVL72) tripped a race in
#                                        ffmpeg's recursive recipes.
#
# Usage:
#     bash scripts/install_native_ffmpeg.sh
#
# Knobs (env vars, all optional):
#     INSTALL_PREFIX  install destination     default: $HOME/opt/ffmpeg-native
#     SOURCE_DIR      build workspace         default: $HOME/src/ffmpeg-native
#     X264_REF        x264 git ref            default: stable
#     FFMPEG_REF      FFmpeg git ref          default: n7.1
#     MAKE_JOBS       parallel make jobs      default: min(nproc, 16)
#     CC, CXX, AS     compiler / assembler    default: conda-forge triplet
#
set -euo pipefail

# ─── Defaults (override via env) ──────────────────────────────────────────
INSTALL_PREFIX="${INSTALL_PREFIX:-$HOME/opt/ffmpeg-native}"
SOURCE_DIR="${SOURCE_DIR:-$HOME/src/ffmpeg-native}"
X264_REF="${X264_REF:-stable}"
FFMPEG_REF="${FFMPEG_REF:-n7.1}"
NPROC="$(nproc)"
MAKE_JOBS="${MAKE_JOBS:-$(( NPROC < 16 ? NPROC : 16 ))}"

# ─── Per-platform, per-stage flags (verbatim from the playbook) ───────────
# TODO: edit the CC/CXX defaults below if you are NOT using the conda-forge
#       gcc_linux-{64,aarch64} toolchain. e.g.: CC=gcc CXX=g++.
#       The probe step below probes whatever CC/CXX resolve to, so this
#       block is the single source of truth for compiler choice.
case "$(uname -m)" in
  x86_64)
    : "${CC:=x86_64-conda-linux-gnu-cc}"
    : "${CXX:=x86_64-conda-linux-gnu-c++}"
    : "${AS:=nasm}"
    X264_CFLAGS="-O3 -march=native -mtune=native -fPIC -flto"
    X264_LDFLAGS="-flto -fuse-linker-plugin"
    FFMPEG_CFLAGS="-O3 -march=native -mtune=native -fPIC -flto"
    FFMPEG_LDFLAGS="-flto -Wl,-rpath,$INSTALL_PREFIX/lib"
    ;;
  aarch64)
    : "${CC:=aarch64-conda-linux-gnu-cc}"
    : "${CXX:=aarch64-conda-linux-gnu-c++}"
    unset AS  # GNU as on ARM
    X264_CFLAGS="-O3 -mcpu=native -fPIC -flto"
    X264_LDFLAGS="-flto -fuse-linker-plugin"
    FFMPEG_CFLAGS="-O3 -mcpu=native -fPIC -flto -fno-tree-vectorize"
    FFMPEG_LDFLAGS="-flto -Wl,-rpath,$INSTALL_PREFIX/lib"
    ;;
  *)
    echo "[install_native_ffmpeg] unsupported arch: $(uname -m)" >&2
    exit 1
    ;;
esac
export CC CXX
[[ -n "${AS:-}" ]] && export AS

# ─── Step 0: probe required tools ─────────────────────────────────────────
required=("$CC" "$CXX" make pkg-config git)
[[ "$(uname -m)" == "x86_64" ]] && required+=(nasm)
missing=()
for cmd in "${required[@]}"; do
  command -v "$cmd" >/dev/null 2>&1 || missing+=("$cmd")
done
if (( ${#missing[@]} > 0 )); then
  echo "[install_native_ffmpeg] missing required tools: ${missing[*]}" >&2
  echo "[install_native_ffmpeg] install them, or edit CC/CXX at the top of this script." >&2
  exit 1
fi

# ─── Step 0.5: destructive-path guards ────────────────────────────────────
guard_path() {
  local name="$1" value="$2"
  case "$value" in
    "")        echo "[install_native_ffmpeg] $name is empty"                 >&2; exit 1 ;;
    "/")       echo "[install_native_ffmpeg] refusing to wipe '/'"           >&2; exit 1 ;;
    "$HOME")   echo "[install_native_ffmpeg] refusing to wipe \$HOME"        >&2; exit 1 ;;
  esac
  [[ "$value" == /* ]] || {
    echo "[install_native_ffmpeg] $name must be absolute, got: $value" >&2; exit 1; }
  [[ "$value" == *ffmpeg-native* ]] || {
    echo "[install_native_ffmpeg] $name must contain 'ffmpeg-native' for safety, got: $value" >&2
    exit 1; }
}
guard_path INSTALL_PREFIX "$INSTALL_PREFIX"
guard_path SOURCE_DIR     "$SOURCE_DIR"

# ─── Step 1: clean ────────────────────────────────────────────────────────
echo "[install_native_ffmpeg] cleaning prior install + sources"
rm -rf "$INSTALL_PREFIX" "$SOURCE_DIR/x264" "$SOURCE_DIR/ffmpeg"
mkdir -p "$SOURCE_DIR" "$INSTALL_PREFIX/lib"

# Deviation 2: force our prefix to win over conda-forge gcc's implicit
# library search (otherwise an older libx264 from conda's lib dir leaks
# into ffmpeg's link, producing a binary that needs *two* x264 SONAMEs).
export LIBRARY_PATH="$INSTALL_PREFIX/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
export LD_LIBRARY_PATH="$INSTALL_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# ─── Step 2: build x264 (LTO, shared lib, no CLI) ─────────────────────────
echo "[install_native_ffmpeg] cloning x264 ($X264_REF)"
git clone --depth 1 --branch "$X264_REF" \
    https://code.videolan.org/videolan/x264.git "$SOURCE_DIR/x264"
(
  cd "$SOURCE_DIR/x264"
  export CFLAGS="$X264_CFLAGS"
  export CXXFLAGS="$X264_CFLAGS"
  export LDFLAGS="$X264_LDFLAGS"
  echo "[install_native_ffmpeg] configuring x264"
  ./configure --prefix="$INSTALL_PREFIX" --enable-shared --enable-pic --disable-cli
  echo "[install_native_ffmpeg] building x264 (-j$MAKE_JOBS)"
  make -j"$MAKE_JOBS"
  make install
)

# ─── Step 3: build ffmpeg (LTO, libx264, shared) ──────────────────────────
echo "[install_native_ffmpeg] cloning FFmpeg ($FFMPEG_REF)"
git clone --depth 1 --branch "$FFMPEG_REF" \
    https://github.com/FFmpeg/FFmpeg.git "$SOURCE_DIR/ffmpeg"
(
  cd "$SOURCE_DIR/ffmpeg"
  export PKG_CONFIG_PATH="$INSTALL_PREFIX/lib/pkgconfig"
  export CFLAGS="$FFMPEG_CFLAGS"
  export CXXFLAGS="$FFMPEG_CFLAGS"
  export LDFLAGS="$FFMPEG_LDFLAGS"
  # Informational sanity prints (matches playbook).
  [[ "$(uname -m)" == "x86_64" ]] && which nasm
  pkg-config --modversion x264
  echo "[install_native_ffmpeg] configuring ffmpeg"
  # Deviation 1 (--disable-libxcb --disable-xlib) and the leading -L of
  # deviation 2 are inserted here. All other flags are verbatim playbook.
  ./configure \
    --prefix="$INSTALL_PREFIX" \
    --enable-gpl \
    --enable-libx264 \
    --enable-lto \
    --enable-shared \
    --disable-static \
    --disable-debug \
    --disable-doc \
    --disable-ffplay \
    --disable-libxcb \
    --disable-xlib \
    --extra-cflags="$CFLAGS" \
    --extra-cxxflags="$CXXFLAGS" \
    --extra-ldflags="-L$INSTALL_PREFIX/lib $LDFLAGS"
  echo "[install_native_ffmpeg] building ffmpeg (-j$MAKE_JOBS)"
  make -j"$MAKE_JOBS"
  make install
)

# ─── Step 4: sanity check (verbatim playbook checks) ──────────────────────
ffmpeg_bin="$INSTALL_PREFIX/bin/ffmpeg"
echo "[install_native_ffmpeg] verifying $ffmpeg_bin"
"$ffmpeg_bin" -hide_banner -buildconf | grep -i -E 'libx264|lto'
"$ffmpeg_bin" -hide_banner -encoders  | grep -i libx264
"$ffmpeg_bin" -hide_banner -h encoder=libx264 2>&1 | grep -i preset

# ─── Step 5: emit env file (matches playbook exports) ─────────────────────
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
env_file="$script_dir/ffmpeg-env.sh"
{
  printf '#!/usr/bin/env bash\n'
  printf 'export FASTVIDEO_FFMPEG_BIN=%q\n' "$ffmpeg_bin"
  printf 'export FASTVIDEO_VIDEO_CODEC=libx264\n'
} > "$env_file"
chmod +x "$env_file"

echo
echo "[install_native_ffmpeg] ✓ done."
echo "[install_native_ffmpeg]   binary:  $ffmpeg_bin"
echo "[install_native_ffmpeg]   env:     $env_file"
echo "[install_native_ffmpeg]   source it before running the demo:"
echo "[install_native_ffmpeg]     source scripts/ffmpeg-env.sh"
