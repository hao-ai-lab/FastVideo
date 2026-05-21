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
#     NV_CODEC_REF    nv-codec-headers ref    default: master
#     CUDA_PREFIX     CUDA toolkit root       default: /usr/local/cuda
#     ENABLE_NVENC    build with NVENC/NVDEC  default: 1 (1|0)
#     MAKE_JOBS       parallel make jobs      default: min(nproc, 16)
#     FFMPEG_NATIVE_CC  explicit C compiler command for native builds
#     FFMPEG_NATIVE_CXX explicit C++ compiler command for native builds
#
# Toolchain selection (precedence, highest first):
#   1. FFMPEG_NATIVE_CC / FFMPEG_NATIVE_CXX, if set — explicit override.
#   2. The conda-forge triplet matching `uname -m`
#      (`x86_64-conda-linux-gnu-cc` or `aarch64-conda-linux-gnu-cc`),
#      if present on PATH. Pinning the triplet sidesteps a bug where
#      conda envs with BOTH `gcc_linux-64` and `gcc_linux-aarch64`
#      installed export the cross-compiler triplet on every
#      `conda activate` (the `aarch64` activation script sorts later
#      and wins), silently breaking x264's compiler probe on the
#      opposite host.
#   3. System `gcc` / `g++` — fallback so the script also works in a
#      plain venv or bare shell with no conda toolchain installed.
# Inherited bare CC / CXX from the caller environment are ignored in
# cases (2) and (3); use FFMPEG_NATIVE_CC/CXX to inject a non-host
# toolchain.
set -euo pipefail

# ─── Defaults (override via env) ──────────────────────────────────────────
INSTALL_PREFIX="${INSTALL_PREFIX:-$HOME/opt/ffmpeg-native}"
SOURCE_DIR="${SOURCE_DIR:-$HOME/src/ffmpeg-native}"
X264_REF="${X264_REF:-stable}"
FFMPEG_REF="${FFMPEG_REF:-n7.1}"
NV_CODEC_REF="${NV_CODEC_REF:-master}"
CUDA_PREFIX="${CUDA_PREFIX:-/usr/local/cuda}"
ENABLE_NVENC="${ENABLE_NVENC:-1}"
case "${ENABLE_NVENC}" in
  0|1) ;;
  *) echo "[install_native_ffmpeg] ENABLE_NVENC must be 0 or 1, got '${ENABLE_NVENC}'" >&2; exit 1 ;;
esac
NPROC="$(nproc)"
MAKE_JOBS="${MAKE_JOBS:-$(( NPROC < 16 ? NPROC : 16 ))}"

# ─── Per-platform, per-stage flags (verbatim from the playbook) ───────────
ARCH="$(uname -m)"
case "$ARCH" in
  x86_64)
    CONDA_CC=x86_64-conda-linux-gnu-cc
    CONDA_CXX=x86_64-conda-linux-gnu-c++
    AS=nasm
    X264_CFLAGS="-O3 -march=native -mtune=native -fPIC -flto"
    X264_LDFLAGS="-flto -fuse-linker-plugin"
    FFMPEG_CFLAGS="-O3 -march=native -mtune=native -fPIC -flto"
    FFMPEG_LDFLAGS="-flto -Wl,-rpath,$INSTALL_PREFIX/lib"
    ;;
  aarch64)
    CONDA_CC=aarch64-conda-linux-gnu-cc
    CONDA_CXX=aarch64-conda-linux-gnu-c++
    unset AS  # GNU as on ARM
    X264_CFLAGS="-O3 -mcpu=native -fPIC -flto"
    X264_LDFLAGS="-flto -fuse-linker-plugin"
    FFMPEG_CFLAGS="-O3 -mcpu=native -fPIC -flto -fno-tree-vectorize"
    FFMPEG_LDFLAGS="-flto -Wl,-rpath,$INSTALL_PREFIX/lib"
    ;;
  *)
    echo "[install_native_ffmpeg] unsupported arch: $ARCH" >&2
    exit 1
    ;;
esac

# Prefer the conda-forge triplet when it's actually on PATH; otherwise fall
# back to system gcc/g++ so a plain venv works too. FFMPEG_NATIVE_CC/CXX
# overrides both.
if command -v -- "$CONDA_CC" >/dev/null 2>&1 \
   && command -v -- "$CONDA_CXX" >/dev/null 2>&1; then
  DEFAULT_CC="$CONDA_CC"
  DEFAULT_CXX="$CONDA_CXX"
  default_source="conda-forge ($ARCH triplet)"
else
  DEFAULT_CC=gcc
  DEFAULT_CXX=g++
  default_source="system gcc/g++"
fi

CC="${FFMPEG_NATIVE_CC:-$DEFAULT_CC}"
CXX="${FFMPEG_NATIVE_CXX:-$DEFAULT_CXX}"
if [[ -n "${FFMPEG_NATIVE_CC:-}" || -n "${FFMPEG_NATIVE_CXX:-}" ]]; then
  toolchain_source="FFMPEG_NATIVE_CC/CXX override"
else
  toolchain_source="$default_source"
fi

require_compiler() {
  local name="$1" compiler="$2"
  if [[ -z "$compiler" ]]; then
    echo "[install_native_ffmpeg] $name is empty" >&2
    exit 1
  fi
  if ! command -v -- "$compiler" >/dev/null 2>&1; then
    echo "[install_native_ffmpeg] $name is unavailable: $compiler" >&2
    echo "[install_native_ffmpeg] install a compiler, or set FFMPEG_NATIVE_CC/FFMPEG_NATIVE_CXX" \
      "to compiler commands on PATH." >&2
    exit 1
  fi
  if ! "$compiler" --version >/dev/null 2>&1; then
    echo "[install_native_ffmpeg] $name failed sanity check: $compiler --version" >&2
    exit 1
  fi
}

require_compiler CC "$CC"
require_compiler CXX "$CXX"
export CC CXX
[[ -n "${AS:-}" ]] && export AS
echo "[install_native_ffmpeg] toolchain: CC=$CC CXX=$CXX AS=${AS:-<gnu-as>} (uname -m=$ARCH, source: $toolchain_source)"

# ─── Step 0: probe required tools ─────────────────────────────────────────
required=("$CC" "$CXX" make pkg-config git)
[[ "$(uname -m)" == "x86_64" ]] && required+=(nasm)
missing=()
for cmd in "${required[@]}"; do
  command -v -- "$cmd" >/dev/null 2>&1 || missing+=("$cmd")
done
if (( ${#missing[@]} > 0 )); then
  echo "[install_native_ffmpeg] missing required tools: ${missing[*]}" >&2
  echo "[install_native_ffmpeg] install the missing tools. FFMPEG_NATIVE_CC/FFMPEG_NATIVE_CXX" \
    "only override compiler selection; they do not provide make, pkg-config, git, or nasm." >&2
  if [[ " ${missing[*]} " == *" nasm "* ]]; then
    echo "[install_native_ffmpeg] nasm is required on x86_64 for x264 SIMD; install nasm" \
      "(for example via apt or conda-forge) and retry." >&2
  fi
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
rm -rf "$INSTALL_PREFIX" "$SOURCE_DIR/x264" "$SOURCE_DIR/ffmpeg" "$SOURCE_DIR/nv-codec-headers"
mkdir -p "$SOURCE_DIR" "$INSTALL_PREFIX/lib"

if [[ "$ENABLE_NVENC" == "1" ]]; then
  if [[ ! -f "$CUDA_PREFIX/include/cuda.h" ]]; then
    echo "[install_native_ffmpeg] ENABLE_NVENC=1 but cuda.h not at $CUDA_PREFIX/include/cuda.h" >&2
    echo "[install_native_ffmpeg] set CUDA_PREFIX env to your CUDA toolkit root, or ENABLE_NVENC=0 to skip" >&2
    exit 1
  fi
  echo "[install_native_ffmpeg] NVENC build enabled (CUDA_PREFIX=$CUDA_PREFIX)"
fi

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

# ─── Step 2.5: nv-codec-headers (NVENC/NVDEC API headers, no CUDA libs) ──
# Required for ffmpeg's --enable-cuda --enable-nvenc --enable-cuvid configure
# flags. Installs ffnvcodec.pc + headers into $INSTALL_PREFIX so ffmpeg's
# pkg-config picks them up alongside libx264. The runtime libraries
# (libcuda.so, libnvcuvid.so, libnvidia-encode.so) come from the NVIDIA
# driver, not from these headers.
if [[ "$ENABLE_NVENC" == "1" ]]; then
  echo "[install_native_ffmpeg] cloning nv-codec-headers ($NV_CODEC_REF)"
  git clone --depth 1 --branch "$NV_CODEC_REF" \
      https://git.videolan.org/git/ffmpeg/nv-codec-headers.git \
      "$SOURCE_DIR/nv-codec-headers"
  (
    cd "$SOURCE_DIR/nv-codec-headers"
    make PREFIX="$INSTALL_PREFIX" install
  )
fi

# ─── Step 3: build ffmpeg (LTO, libx264, shared, optional NVENC) ──────────
echo "[install_native_ffmpeg] cloning FFmpeg ($FFMPEG_REF)"
git clone --depth 1 --branch "$FFMPEG_REF" \
    https://github.com/FFmpeg/FFmpeg.git "$SOURCE_DIR/ffmpeg"
(
  cd "$SOURCE_DIR/ffmpeg"
  export PKG_CONFIG_PATH="$INSTALL_PREFIX/lib/pkgconfig"
  export CFLAGS="$FFMPEG_CFLAGS"
  export CXXFLAGS="$FFMPEG_CFLAGS"
  export LDFLAGS="$FFMPEG_LDFLAGS"
  [[ "$(uname -m)" == "x86_64" ]] && which nasm
  pkg-config --modversion x264
  echo "[install_native_ffmpeg] configuring ffmpeg"
  ffmpeg_configure_flags=(
    --prefix="$INSTALL_PREFIX"
    --enable-gpl
    --enable-libx264
    --enable-lto
    --enable-shared
    --disable-static
    --disable-debug
    --disable-doc
    --disable-ffplay
    --disable-libxcb
    --disable-xlib
    --extra-cflags="$CFLAGS"
    --extra-cxxflags="$CXXFLAGS"
    --extra-ldflags="-L$INSTALL_PREFIX/lib $LDFLAGS"
  )
  if [[ "$ENABLE_NVENC" == "1" ]]; then
    pkg-config --modversion ffnvcodec
    ffmpeg_configure_flags+=(
      --enable-cuda
      --enable-nvenc
      --enable-cuvid
      --enable-nvdec
      --extra-cflags="-I$CUDA_PREFIX/include"
      --extra-ldflags="-L$CUDA_PREFIX/lib64"
    )
  fi
  ./configure "${ffmpeg_configure_flags[@]}"
  echo "[install_native_ffmpeg] building ffmpeg (-j$MAKE_JOBS)"
  make -j"$MAKE_JOBS"
  make install
)

# ─── Step 4: sanity check ──────────────────────────────────────────────────
ffmpeg_bin="$INSTALL_PREFIX/bin/ffmpeg"
echo "[install_native_ffmpeg] verifying $ffmpeg_bin"
"$ffmpeg_bin" -hide_banner -buildconf | grep -i -E 'libx264|lto'
"$ffmpeg_bin" -hide_banner -encoders  | grep -i libx264
"$ffmpeg_bin" -hide_banner -h encoder=libx264 2>&1 | grep -i preset
if [[ "$ENABLE_NVENC" == "1" ]]; then
  "$ffmpeg_bin" -hide_banner -encoders | grep -E 'h264_nvenc|hevc_nvenc' || {
    echo "[install_native_ffmpeg] ENABLE_NVENC=1 but built ffmpeg has no h264_nvenc/hevc_nvenc encoder" >&2
    exit 1
  }
fi

# ─── Step 5: emit env file ─────────────────────────────────────────────────
# FASTVIDEO_VIDEO_CODEC stays at libx264 by default to preserve runtime
# behavior; NVENC is selected at deploy time via dreamverse-deploy.sh
# --nvenc, which exports FASTVIDEO_VIDEO_CODEC=h264_nvenc.
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
