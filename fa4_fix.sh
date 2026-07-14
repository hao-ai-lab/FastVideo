#!/usr/bin/env bash
set +e
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:/mnt/lustre/vlm-s4duan/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export UV_CACHE_DIR=/mnt/lustre/vlm-s4duan/.uv/cache
export UV_PYTHON_INSTALL_DIR=/mnt/lustre/vlm-s4duan/.uv/python
cd /mnt/lustre/vlm-s4duan/FastVideo || exit 1
source .venv/bin/activate

echo "==================== BEFORE ===================="
python test_fa4.py

echo "==================== try A: coherent cutlass-dsl 4.5.2, drop quack ===================="
uv pip uninstall quack-kernels 2>&1 | tail -3
uv pip install --reinstall "nvidia-cutlass-dsl==4.5.2" 2>&1 | tail -20
echo "---- test ----"; python test_fa4.py

if ! python -c "from flash_attn.cute.interface import _flash_attn_fwd" 2>/dev/null; then
  echo "==================== try B: coherent cutlass-dsl 4.6.0 (quack-consistent) ===================="
  uv pip install "quack-kernels==0.6.1" 2>&1 | tail -3
  uv pip install --reinstall "nvidia-cutlass-dsl==4.6.0" 2>&1 | tail -15
  echo "---- test ----"; python test_fa4.py
fi
echo "==================== DONE fa4_fix ===================="
