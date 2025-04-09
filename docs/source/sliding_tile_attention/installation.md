(sta-installation)=

# Installation
We test our code on Pytorch 2.5.0 and CUDA>=12.4. Currently we only have implementation on H100.
First, install C++20 for ThunderKittens:

```bash
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
