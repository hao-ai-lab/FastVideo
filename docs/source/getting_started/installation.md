(fastvideo-installation)=

# 🔧 Installation

FastVideo currently only supports Linux and CUDA GPUs. The code is tested on Python 3.10.0 and CUDA 12.4, primarily with NVIDIA H100 GPUs.

## Prerequisites

- CUDA 12.4 installed and supported
- Linux operating system

## Installation Options

### Option 1: Quick Install

```bash
pip install fastvideo
```

### Option 2: Installation from Source

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

#### 3. Clone the FastVideo repository

```bash
git clone https://github.com/hao-ai-lab/FastVideo.git && cd FastVideo
```

#### 4. Install FastVideo

Basic installation:

```bash
pip install -e .
```

## Optional Dependencies

### Flash Attention

```bash
pip install flash-attn==2.7.0.post2 --no-build-isolation
```

### Sliding Tile Attention (STA)

To try Sliding Tile Attention (optional), please follow the instructions in [csrc/sliding_tile_attention/README.md](#sta-installation) to install STA.

## Development Environment Setup

If you're planning to contribute to FastVideo please see the following page:
[Contributor Guide](#developer-guide)

## Hardware Requirements

### For Basic Inference
- NVIDIA GPU with CUDA support
- Minimum 20GB VRAM for quantized models (e.g., single RTX 4090)

### For Lora Finetuning
- 40GB GPU memory each for 2 GPUs with lora
- 30GB GPU memory each for 2 GPUs with CPU offload and lora

### For Full Finetuning/Distillation
- Multiple high-memory GPUs recommended (e.g., H100)

## Troubleshooting

If you encounter any issues during installation, please open an issue on our [GitHub repository](https://github.com/hao-ai-lab/FastVideo).

You can also join our [Slack community](https://join.slack.com/t/fastvideo/shared_invite/zt-2zf6ru791-sRwI9lPIUJQq1mIeB_yjJg) for additional support.
