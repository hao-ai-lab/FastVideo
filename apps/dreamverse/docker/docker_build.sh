#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
IMAGE="${DREAMVERSE_IMAGE:-dreamverse:dev}"
ROOT_DOCKERIGNORE="${REPO_ROOT}/.dockerignore"
DOCKERFILE_DOCKERIGNORE="${SCRIPT_DIR}/Dockerfile.dockerignore"
CREATED_ROOT_DOCKERIGNORE_SYMLINK=0

cleanup_root_dockerignore_symlink() {
  if [[ "${CREATED_ROOT_DOCKERIGNORE_SYMLINK}" == "1" && -L "${ROOT_DOCKERIGNORE}" ]] && \
    [[ "$(readlink "${ROOT_DOCKERIGNORE}")" == "${DOCKERFILE_DOCKERIGNORE}" ]]; then
    rm -- "${ROOT_DOCKERIGNORE}"
  fi
}
trap cleanup_root_dockerignore_symlink EXIT INT TERM

if [[ -z "${DOCKER_BUILDKIT:-}" ]] && docker buildx version >/dev/null 2>&1; then
  export DOCKER_BUILDKIT=1
fi

build_args=()
cuda_version="${CUDA_VERSION:-}"
torch_backend="${UV_TORCH_BACKEND:-}"

# CUDA_TAG was the Dockerfile's original override and contains the complete
# nvidia/cuda tag (for example, 12.6.3-cudnn-devel-ubuntu22.04). Keep accepting
# it while translating it to the parameterized Dockerfile inputs.
if [[ -n "${CUDA_TAG:-}" ]]; then
  if [[ -n "${CUDA_VERSION:-}" ]]; then
    printf 'CUDA_TAG and CUDA_VERSION cannot both be set. Use CUDA_VERSION for new builds.\n' >&2
    exit 2
  fi

  cuda_version="${CUDA_TAG%%-*}"
  if [[ ! "${cuda_version}" =~ ^[0-9]+\.[0-9]+(\.[0-9]+)?$ ]]; then
    printf 'Cannot infer CUDA_VERSION from CUDA_TAG=%s. Use CUDA_VERSION and UV_TORCH_BACKEND instead.\n' \
      "${CUDA_TAG}" >&2
    exit 2
  fi
  build_args+=(--build-arg "BUILD_BASE_IMAGE=nvidia/cuda:${CUDA_TAG}")
fi

if [[ -n "${cuda_version}" ]]; then
  build_args+=(--build-arg "CUDA_VERSION=${cuda_version}")
fi

if [[ -z "${torch_backend}" && -n "${cuda_version}" ]]; then
  case "${cuda_version}" in
    12.*) torch_backend=cu126 ;;
    13.*) torch_backend=cu130 ;;
    *)
      printf 'No default UV_TORCH_BACKEND for CUDA_VERSION=%s. Set UV_TORCH_BACKEND explicitly.\n' \
        "${cuda_version}" >&2
      exit 2
      ;;
  esac
fi

if [[ -n "${torch_backend}" ]]; then
  build_args+=(--build-arg "UV_TORCH_BACKEND=${torch_backend}")
fi

[[ -n "${BUILD_FASTVIDEO_KERNEL_FROM_SOURCE:-}" ]] && \
  build_args+=(--build-arg "BUILD_FASTVIDEO_KERNEL_FROM_SOURCE=${BUILD_FASTVIDEO_KERNEL_FROM_SOURCE}")
build_args+=(--build-arg "BUILD_DREAMVERSE_UI=${BUILD_DREAMVERSE_UI:-0}")

if [[ "${DOCKER_BUILDKIT:-}" == "1" ]]; then
  if [[ -L "${ROOT_DOCKERIGNORE}" ]] && \
    [[ "$(readlink "${ROOT_DOCKERIGNORE}")" == "${DOCKERFILE_DOCKERIGNORE}" ]]; then
    rm -- "${ROOT_DOCKERIGNORE}"
    printf 'Removed stale temporary root .dockerignore symlink: %s\n' "${ROOT_DOCKERIGNORE}"
  fi
elif [[ -e "${ROOT_DOCKERIGNORE}" || -L "${ROOT_DOCKERIGNORE}" ]]; then
  printf 'Using existing root .dockerignore: %s\n' "${ROOT_DOCKERIGNORE}"
else
  ln -s -- "${DOCKERFILE_DOCKERIGNORE}" "${ROOT_DOCKERIGNORE}"
  CREATED_ROOT_DOCKERIGNORE_SYMLINK=1
  printf 'Created temporary root .dockerignore symlink for legacy Docker builder: %s -> %s\n' \
    "${ROOT_DOCKERIGNORE}" "${DOCKERFILE_DOCKERIGNORE}"
fi

docker build \
  -f "${SCRIPT_DIR}/Dockerfile" \
  -t "${IMAGE}" \
  "${build_args[@]}" \
  "${REPO_ROOT}"
