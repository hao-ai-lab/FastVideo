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
[[ -n "${CUDA_TAG:-}"        ]] && build_args+=(--build-arg "CUDA_TAG=${CUDA_TAG}")
[[ -n "${BUILD_FASTVIDEO_KERNEL_FROM_SOURCE:-}" ]] && \
  build_args+=(--build-arg "BUILD_FASTVIDEO_KERNEL_FROM_SOURCE=${BUILD_FASTVIDEO_KERNEL_FROM_SOURCE}")

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
