(sta-installation)=

# 🔧 Installation
You can install the Sliding Tile Attention package using

```
pip install st_attn==0.0.4
``` 

# Building from Source
We test our code on Pytorch 2.5.0 and CUDA>=12.4. Currently we only have implementation on H100.
First, install C++20 for ThunderKittens:

```bash
cd csrc/sliding_tile_attention/
sudo apt update
sudo apt install gcc-11 g++-11

sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 100 --slave /usr/bin/g++ g++ /usr/bin/g++-11

sudo apt update
sudo apt install clang-11
```

Install STA:

```bash
export CUDA_HOME=/usr/local/cuda-12.4
export PATH=${CUDA_HOME}/bin:${PATH} 
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:$LD_LIBRARY_PATH
git submodule update --init --recursive
python setup.py install
```

# 🧪 Test

```bash
python test/test_sta.py
```

# 📋 Usage

```python
from st_attn import sliding_tile_attention
# assuming video size (T, H, W) = (30, 48, 80), text tokens = 256 with padding. 
# q, k, v: [batch_size, num_heads, seq_length, head_dim], seq_length = T*H*W + 256
# a tile is a cube of size (6, 8, 8)
# window_size in tiles: [(window_t, window_h, window_w), (..)...]. For example, window size (3, 3, 3) means a query can attend to (3x6, 3x8, 3x8) = (18, 24, 24) tokens out of the total 30x48x80 video.
# text_length: int ranging from 0 to 256
# If your attention contains text token (Hunyuan)
out = sliding_tile_attention(q, k, v, window_size, text_length)
# If your attention does not contain text token (StepVideo)
out = sliding_tile_attention(q, k, v, window_size, 0, False)

```
