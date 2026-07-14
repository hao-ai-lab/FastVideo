#!/usr/bin/env bash
# Replace the stale XOR-op fork FA4 with the OFFICIAL Dao-AILab flash_attn/cute,
# which pins the matching cutlass-dsl==4.6.0.dev0 + quack>=0.5.3 (coherent set).
set +e
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:/mnt/lustre/vlm-s4duan/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export UV_CACHE_DIR=/mnt/lustre/vlm-s4duan/.uv/cache
export UV_PYTHON_INSTALL_DIR=/mnt/lustre/vlm-s4duan/.uv/python
cd /mnt/lustre/vlm-s4duan/FastVideo || exit 1
source .venv/bin/activate

echo "==================== remove fork FA4 + its cutlass/quack ===================="
uv pip uninstall flash-attn-cute quack-kernels nvidia-cutlass-dsl 2>&1 | tail -6

echo "==================== install OFFICIAL Dao-AILab flash_attn/cute (real name: flash-attn-4) ===================="
GIT="git+https://github.com/Dao-AILab/flash-attention.git#subdirectory=flash_attn/cute"
uv pip install --prerelease allow "$GIT" 2>&1 | tail -35
if [ "${PIPESTATUS[0]}" != "0" ]; then
  echo ">> retry with --no-build-isolation + setuptools_scm"
  uv pip install setuptools_scm setuptools wheel 2>&1 | tail -3
  uv pip install --prerelease allow --no-build-isolation "$GIT" 2>&1 | tail -35
fi

echo "==================== resolved versions ===================="
python - <<'PY'
import importlib.metadata as md
for p in ["flash-attn-cute","nvidia-cutlass-dsl","nvidia-cutlass-dsl-libs-base",
          "nvidia-cutlass-dsl-libs-core","nvidia-cutlass-dsl-libs-cu12",
          "quack-kernels","torch-c-dlpack-ext","torch"]:
    try: print(" ", p, md.version(p))
    except Exception: print(" ", p, "absent")
PY

echo "==================== import tests ===================="
python -c "import cutlass.cute.core as c; print('ThrMma?', hasattr(c,'ThrMma'))" 2>&1
python -c "from flash_attn.cute.interface import _flash_attn_fwd,_flash_attn_bwd; print('FA4_CUTE_INTERFACE_OK')" 2>&1
python -c "import importlib; importlib.import_module('fastvideo.attention.backends.flash_attn'); print('FASTVIDEO_FLASH_ATTN_BACKEND_OK')" 2>&1
echo "==================== DONE fa4_official ===================="
