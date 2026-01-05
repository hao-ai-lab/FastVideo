# FastVideo Attention Kernels

FastVideo provides highly optimized custom attention kernels to accelerate video generation.

## Supported Kernels

* **[Video Sparse Attention (VSA)](vsa/index.md)**: Sparse attention mechanism selecting top-k blocks.
* **[Sliding Tile Attention (STA)](sta/index.md)**: Optimized attention for window-based video generation.

## General Build Instructions

These instructions apply to building the `fastvideo-kernel` package from source, which includes both STA and VSA kernels.

### Prerequisites

* **PyTorch**: 2.5.0+
* **CUDA**: 12.4+ (12.8 recommended for best performance)
* **C++ Compiler**: GCC 11+ (C++20 support required for ThunderKittens)

Install system dependencies:

```bash
sudo apt update
sudo apt install -y gcc-11 g++-11 clang-11 ninja-build

# Set gcc-11 as default
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 100 --slave /usr/bin/g++ g++ /usr/bin/g++-11
```

Set up your CUDA environment variables (adjust version as needed):

```bash
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=${CUDA_HOME}/bin:${PATH}
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:$LD_LIBRARY_PATH
```

### Compile and Install

Clone the repository and build the kernel:

```bash
# Clone recursively to get ThunderKittens submodule
git clone https://github.com/hao-ai-lab/FastVideo.git
cd FastVideo/fastvideo-kernel

# Build and install
./build.sh
```

The build script automatically detects your GPU architecture:
* **H100 (sm_90a)**: Compiles optimized C++ ThunderKittens kernels.
* **Other (A100, etc.)**: Skips C++ compilation; installs Python package with Triton kernels.
