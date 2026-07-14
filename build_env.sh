#!/usr/bin/env bash
# FastVideo (trackwan_bidir) env — CORRECTED build. CUDA 12.9 container, aarch64 + GB200 (sm_100a).
# Ordering fix: install repo-pinned torch (2.11.0 cu128) FIRST, build the local Blackwell kernel
# against it, then core deps (relaxed kernel pin so the local 0.3.0 satisfies), then FA4.
set +e
say(){ echo; echo "==================== $* ===================="; }

export DEBIAN_FRONTEND=noninteractive
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export PATH=$CUDA_HOME/bin:/mnt/lustre/vlm-s4duan/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export UV_CACHE_DIR=/mnt/lustre/vlm-s4duan/.uv/cache
export UV_PYTHON_INSTALL_DIR=/mnt/lustre/vlm-s4duan/.uv/python
export TORCH_CUDA_ARCH_LIST=10.0a
export MAX_JOBS=${MAX_JOBS:-64}
cd /mnt/lustre/vlm-s4duan/FastVideo || { echo NO_REPO; exit 1; }

say "ENSURE toolchain"
if ! command -v g++ >/dev/null || ! command -v cmake >/dev/null || ! command -v ninja >/dev/null; then
  apt-get update -y >/tmp/apt.log 2>&1 && apt-get install -y --no-install-recommends g++ gcc make cmake ninja-build git ca-certificates >>/tmp/apt.log 2>&1 && echo "apt ok" || { echo APT_FAIL; tail -20 /tmp/apt.log; }
fi
nvcc --version | tail -2

say "FRESH venv (python 3.12 on Lustre)"
rm -rf .venv
uv venv --python 3.12 --seed .venv || exit 1
source .venv/bin/activate
python --version

say "STAGE 1: torch 2.11.0 cu128 (repo pin, Blackwell-capable)"
uv pip install --torch-backend=cu128 torch==2.11.0 torchvision torchaudio 2>&1 | tail -15
python -c "import torch;print('TORCH',torch.__version__,torch.version.cuda,'avail',torch.cuda.is_available())" || echo TORCH_FAIL

say "STAGE 2: submodules + build local fastvideo-kernel (sm_100a) against torch 2.11"
git submodule update --init --recursive fastvideo-kernel/include/cutlass fastvideo-kernel/include/tk 2>&1 | tail -5
( cd fastvideo-kernel && TORCH_CUDA_ARCH_LIST=10.0a bash ./build.sh ) 2>&1 | tail -20
echo "KERNEL_EXIT=$?"
python -c "import fastvideo_kernel;print('kernel import OK')" || echo KERNEL_IMPORT_FAIL

say "STAGE 3: core deps  uv pip install -e .[dev]  (local kernel already satisfies relaxed pin)"
uv pip install -e ".[dev]" 2>&1 | tail -40
echo "CORE_EXIT=$?"
python -c "import fastvideo;print('fastvideo import OK')" || echo FASTVIDEO_IMPORT_FAIL

say "STAGE 4: FA4 — OFFICIAL Dao-AILab flash_attn/cute (NOT the XOR-op fork; fork is stale/broken)"
# Package name is flash-attn-4; pass the bare git URL (name-qualified spec is rejected).
# --prerelease allow lets it pull the matching nvidia-cutlass-dsl==4.6.0.dev0 + quack>=0.5.3.
uv pip install --prerelease allow \
  "git+https://github.com/Dao-AILab/flash-attention.git#subdirectory=flash_attn/cute" 2>&1 | tail -25
python -c "from flash_attn.cute.interface import _flash_attn_fwd,_flash_attn_bwd; print('FA4_CUTE_OK')" 2>&1

say "VERIFY"
python - <<'PY'
import importlib, torch
print("torch", torch.__version__, torch.version.cuda, "cuda_avail", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device0", torch.cuda.get_device_name(0), "cc", torch.cuda.get_device_capability(0))
    x = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)
    y = (x @ x); torch.cuda.synchronize()
    print("bf16 matmul on GPU OK", tuple(y.shape))
for m in ["fastvideo", "fastvideo_kernel", "flash_attn"]:
    try: importlib.import_module(m); print("import OK:", m)
    except Exception as e: print("import FAIL:", m, repr(e)[:180])
try:
    from flash_attn.cute.interface import _flash_attn_fwd; print("FA4 flash_attn.cute.interface OK")
except Exception as e:
    print("FA4 cute interface:", repr(e)[:180])
PY

say "DONE build_env.sh (v2)"
