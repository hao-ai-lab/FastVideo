(fastvideo-installation)=

# 🔧 Installation

FastVideo currently only supports Linux and NVIDIA CUDA GPUs.

## Requirements

- **OS: Linux**
- **Python: 3.10-3.12**
- **CUDA 12.4**
- **At least 1 NVIDIA GPU**

## Quick start
We recommend using an environment manager such as [Conda](#miniconda-recommended)

```bash
pip install fastvideo
```

Also optionally install flash-attn:

```bash
pip install flash-attn==2.7.4.post1 --no-build-isolation
```

[Click here to install from source](#installation-from-source)

## Environment Setup

We highly recommend using a `conda` or `virtualenv` environment:
### Miniconda (Recommended)
#### 1. Install Miniconda (if not already installed)

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc
```

#### 2. Create and activate a Conda environment for FastVideo

```bash
conda create -n fastvideo python=3.10 -y
conda activate fastvideo
```

## Installation from Source

### 1. Clone the FastVideo repository

```bash
git clone https://github.com/hao-ai-lab/FastVideo.git && cd FastVideo
```

#### 2. Install FastVideo

Basic installation:

```bash
pip install -e .
```

## Optional Dependencies

### Flash Attention

```bash
pip install flash-attn==2.7.4.post1 --no-build-isolation
```

### Sliding Tile Attention (STA) (Requires CUDA 12.4 and H100)

To try Sliding Tile Attention (optional), please follow the instructions in [csrc/sliding_tile_attention/README.md](#sta-installation) to install STA.

## Docker Images
We also have prebuilt docker images with FastVideo dependencies pre-installed:
[Docker Images](#docker)
## Development Environment Setup

If you're planning to contribute to FastVideo please see the following page:
[Contributor Guide](#developer-overview)

## Hardware Requirements

### For Basic Inference
- NVIDIA GPU with CUDA 12.4support

### For Lora Finetuning
- 40GB GPU memory each for 2 GPUs with lora
- 30GB GPU memory each for 2 GPUs with CPU offload and lora

### For Full Finetuning/Distillation
- Multiple high-memory GPUs recommended (e.g., H100)

## Troubleshooting

If you encounter any issues during installation, please open an issue on our [GitHub repository](https://github.com/hao-ai-lab/FastVideo).

You can also join our [Slack community](https://join.slack.com/t/fastvideo/shared_invite/zt-2zf6ru791-sRwI9lPIUJQq1mIeB_yjJg) for additional support.
