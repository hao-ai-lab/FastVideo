(vsa-installation)=

# 🔧 Installation
You can install the Video Sparse Attention package using

```bash
pip install vsa
```

# Building from Source
We support H100 (via ThunderKittens) and any other GPU (via Triton) for VSA.

First, install C++20 for ThunderKittens (if using H100):

```bash
sudo apt update
sudo apt install gcc-11 g++-11

sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 100 --slave /usr/bin/g++ g++ /usr/bin/g++-11

sudo apt update
sudo apt install clang-11
```

Set up CUDA environment (if using CUDA 12.8):

```bash
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=${CUDA_HOME}/bin:${PATH} 
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:$LD_LIBRARY_PATH
```

Install VSA:

```bash
cd csrc/attn/video_sparse_attn/
git submodule update --init --recursive
python setup.py install
```

# 🧪 Test

```bash
python csrc/attn/tests/test_vsa.py
```

# 📋 Usage

```python
from vsa import video_sparse_attn

# q, k, v: [batch_size, num_heads, seq_len, head_dim]
# variable_block_sizes: [num_blocks] - number of valid tokens in each block
# topk: int - number of top-k blocks to attend to
# block_size: int or tuple of 3 ints - size of each block (default: 64 tokens)
# compress_attn_weight: optional weight for compressed attention branch

output = video_sparse_attn(q, k, v, variable_block_sizes, topk, block_size, compress_attn_weight)

``` 

# 🚀Inference

```bash
bash scripts/inference/v1_inference_wan_VSA.sh
```
